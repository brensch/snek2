#!/usr/bin/env bash
set -euo pipefail

# Local (non-Docker) orchestrator.
# Runs: train -> export onnx -> generate -> repeat.
# Uses Makefile targets for the heavy lifting.

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p logs
start_ts=$(date -u +"%Y%m%d_%H%M%S")
log_file="logs/orchestrate_local_${start_ts}.log"
ln -sf "$(basename "${log_file}")" logs/orchestrate_local_latest.log
exec > >(tee -a "${log_file}") 2>&1

# When output is piped through tee, Python defaults to block-buffering.
# Force unbuffered mode so epoch logs show up promptly.
export PYTHONUNBUFFERED=1

: "${GENERATED_DIR:=data/generated}"
: "${SCRAPED_DIR:=data/scraped}"
: "${PROCESSED_DIR:=processed}"
: "${MODEL_DIR:=models}"
: "${HISTORY_DIR:=${MODEL_DIR}/history}"

: "${GENERATE_GAMES:=256}"
: "${WORKERS:=512}"
: "${GAMES_PER_FLUSH:=50}"
: "${ONNX_SESSIONS:=1}"
: "${ONNX_BATCH_SIZE:=512}"
: "${ONNX_BATCH_TIMEOUT:=5ms}"
: "${MCTS_SIMS:=800}"
: "${TRACE:=false}"

: "${TRAIN_EPOCHS:=5}"
: "${TRAIN_BATCH_SIZE:=256}"
: "${TRAIN_LR:=0.001}"

: "${ARCHIVE_EXISTING_ON_START:=0}"
: "${SLEEP_BETWEEN_CYCLES:=0}"
: "${MAX_CYCLES:=0}"             # 0 = infinite

mkdir -p "${GENERATED_DIR}" "${SCRAPED_DIR}" "${PROCESSED_DIR}/generated" "${PROCESSED_DIR}/scraped" "${HISTORY_DIR}"

# Always rebuild Go binaries to pick up code changes
echo "[startup] rebuilding executor binary..."
go build -o bin/executor ./executor

# Ensure we have a model and ONNX to start.
if [[ ! -f "${MODEL_DIR}/latest.pt" ]]; then
  echo "[startup] missing ${MODEL_DIR}/latest.pt; initializing"
  make init-ckpt
fi

if [[ "${ARCHIVE_EXISTING_ON_START}" == "1" ]]; then
  shopt -s nullglob
  old_gen=("${GENERATED_DIR}"/*.parquet)
  old_scr=("${SCRAPED_DIR}"/*.parquet)
  shopt -u nullglob
  if [[ ${#old_gen[@]} -gt 0 || ${#old_scr[@]} -gt 0 ]]; then
    echo "[startup] archiving existing shards: ${#old_gen[@]} generated + ${#old_scr[@]} scraped"
    for f in "${old_gen[@]}"; do mv -f "${f}" "${PROCESSED_DIR}/generated/" || true; done
    for f in "${old_scr[@]}"; do mv -f "${f}" "${PROCESSED_DIR}/scraped/" || true; done
  fi
fi

cycle=0
while true; do
  cycle=$((cycle+1))
  ts=$(date -u +"%Y%m%d_%H%M%S")
  ckpt_path="${HISTORY_DIR}/model_${ts}.pt"
  onnx_path="${HISTORY_DIR}/model_${ts}.onnx"

  # Freeze the training set for this cycle (symlinks), so we can archive consumed shards.
  train_dir="$(pwd)/.train_sets/snek2_train_${ts}"
  mkdir -p "${train_dir}"

  mapfile -t gen_files < <(find "${GENERATED_DIR}" -maxdepth 1 -type f -name "*.parquet" ! -path "*/tmp/*" 2>/dev/null | sort)
  mapfile -t scr_files < <(find "${SCRAPED_DIR}" -maxdepth 1 -type f -name "*.parquet" ! -path "*/tmp/*" 2>/dev/null | sort)

  if [[ ${#gen_files[@]} -eq 0 && ${#scr_files[@]} -eq 0 ]]; then
    echo "[cycle ${cycle}] no eligible parquet shards yet; skipping training"
    rm -rf "${train_dir}" || true
  else
    i=0
    for f in "${gen_files[@]}"; do
      ln -sf "$(realpath "${f}")" "${train_dir}/generated_${i}.parquet"
      i=$((i+1))
    done
    j=0
    for f in "${scr_files[@]}"; do
      ln -sf "$(realpath "${f}")" "${train_dir}/scraped_${j}.parquet"
      j=$((j+1))
    done

    echo "[cycle ${cycle}] training on ${#gen_files[@]} generated + ${#scr_files[@]} scraped shards"
    # trainer/train.py uses --model-path as both input and output.
    # Seed a new history checkpoint from the current latest model so we don't
    # re-initialize randomly each cycle.
    if [[ ! -f "${ckpt_path}" ]]; then
      cp -f "${MODEL_DIR}/latest.pt" "${ckpt_path}"
    fi
    make train ARGS="--data-dir ${train_dir} --model-path ${ckpt_path} --epochs ${TRAIN_EPOCHS} --batch-size ${TRAIN_BATCH_SIZE} --lr ${TRAIN_LR}"

    echo "[cycle ${cycle}] exporting onnx (${onnx_path})"
    make export-onnx-ckpt CKPT="${ckpt_path}" OUT="${onnx_path}"

    # Archive consumed shards only after a successful train+export.
    for f in "${gen_files[@]}"; do mv -f "${f}" "${PROCESSED_DIR}/generated/" || true; done
    for f in "${scr_files[@]}"; do mv -f "${f}" "${PROCESSED_DIR}/scraped/" || true; done
    rm -rf "${train_dir}" || true

    # Update pointers for next generation round.
    ln -sf "$(realpath --relative-to="${MODEL_DIR}" "${ckpt_path}" 2>/dev/null || echo "${ckpt_path}")" "${MODEL_DIR}/latest.pt"
    ln -sf "$(realpath --relative-to="${MODEL_DIR}" "${onnx_path}" 2>/dev/null || echo "${onnx_path}")" "${MODEL_DIR}/snake_net.onnx"
  fi

  # Ensure we have an ONNX to generate with.
  # (We do this here, after the optional train+export, to avoid exporting
  # before training on cycle 1.)
  if [[ ! -e "${MODEL_DIR}/snake_net.onnx" ]]; then
    echo "[cycle ${cycle}] missing ${MODEL_DIR}/snake_net.onnx; exporting from latest.pt"
    make export-onnx
  fi

  echo "[cycle ${cycle}] generating ${GENERATE_GAMES} games"
  make generate \
    OUT_DIR="${GENERATED_DIR}" \
    WORKERS="${WORKERS}" \
    GAMES_PER_FLUSH="${GAMES_PER_FLUSH}" \
    ONNX_SESSIONS="${ONNX_SESSIONS}" \
    ONNX_BATCH_SIZE="${ONNX_BATCH_SIZE}" \
    ONNX_BATCH_TIMEOUT="${ONNX_BATCH_TIMEOUT}" \
    MCTS_SIMS="${MCTS_SIMS}" \
    MAX_GAMES="${GENERATE_GAMES}" \
    TRACE="${TRACE}"

  echo "[cycle ${cycle}] done"

  if [[ "${MAX_CYCLES}" =~ ^[0-9]+$ ]] && [[ "${MAX_CYCLES}" -gt 0 ]] && [[ "${cycle}" -ge "${MAX_CYCLES}" ]]; then
    echo "[cycle ${cycle}] reached MAX_CYCLES=${MAX_CYCLES}; exiting"
    exit 0
  fi

  [[ "${SLEEP_BETWEEN_CYCLES}" != "0" ]] && sleep "${SLEEP_BETWEEN_CYCLES}"

done
