#!/usr/bin/env bash
set -euo pipefail

# Orchestrates: selfplay generate -> train -> export onnx -> repeat.
# Scraper runs as a separate compose service and continuously writes to data/scraped.

: "${GENERATED_DIR:=data/generated}"
: "${SCRAPED_DIR:=data/scraped}"
: "${PROCESSED_DIR:=processed}"
: "${MODEL_DIR:=models}"
: "${HISTORY_DIR:=${MODEL_DIR}/history}"

: "${GENERATE_GAMES:=100}"      # per cycle
: "${WORKERS:=128}"
: "${GAMES_PER_FLUSH:=50}"
: "${ONNX_SESSIONS:=1}"
: "${ONNX_BATCH_SIZE:=0}"        # 0 = executor default
: "${ONNX_BATCH_TIMEOUT:=2ms}"

: "${TRAIN_EPOCHS:=10}"
: "${TRAIN_BATCH_SIZE:=256}"
: "${TRAIN_LR:=0.01}"

: "${SLEEP_BETWEEN_CYCLES:=0}"   # seconds
: "${MIN_FILE_AGE_SECONDS:=30}"  # only consume shards older than this (avoid racing writers)

# If running as a non-root user in a container, ensure we have a writable HOME/cache.
# Some environments set HOME=/, which is not writable.
if [[ -z "${HOME:-}" || "${HOME}" == "/" || ! -w "${HOME}" ]]; then
  export HOME="/tmp/home"
fi
mkdir -p "${HOME}" || true

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export GOCACHE="${GOCACHE:-/tmp/go-build}"
export GOPATH="${GOPATH:-${HOME}/go}"
export GOMODCACHE="${GOMODCACHE:-${GOPATH}/pkg/mod}"
mkdir -p "${XDG_CACHE_HOME}" "${GOCACHE}" "${GOMODCACHE}" || true

mkdir -p "${GENERATED_DIR}" "${SCRAPED_DIR}" "${PROCESSED_DIR}/generated" "${PROCESSED_DIR}/scraped" "${HISTORY_DIR}"

cycle=0
while true; do
  cycle=$((cycle+1))
  ts=$(date -u +"%Y%m%d_%H%M%S")
  ckpt_path="${HISTORY_DIR}/model_${ts}.pt"
  onnx_path="${HISTORY_DIR}/model_${ts}.onnx"

  # Make sure ORT can locate CUDA/Torch shared libraries inside the container venv.
  # (nvidia-*-cu12 wheels ship .so files under site-packages/nvidia/*/lib)
  extra_ld=()
  for d in \
    /opt/venv/lib/python*/site-packages/nvidia/*/lib \
    /opt/venv/lib/python*/site-packages/triton/backends/nvidia/lib \
    /opt/venv/lib/python*/site-packages/torch/lib
  do
    if [[ -d "$d" ]]; then
      extra_ld+=("$d")
    fi
  done
  if [[ ${#extra_ld[@]} -gt 0 ]]; then
    export LD_LIBRARY_PATH="/workspace:$(IFS=:; echo "${extra_ld[*]}"):${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="/workspace:${LD_LIBRARY_PATH:-}"
  fi

  echo "[cycle ${cycle}] generating ${GENERATE_GAMES} games using ${MODEL_DIR}/snake_net.onnx"

  gen_args=(
    -out-dir "${GENERATED_DIR}"
    -workers "${WORKERS}"
    -games-per-flush "${GAMES_PER_FLUSH}"
    -onnx-sessions "${ONNX_SESSIONS}"
    -onnx-batch-timeout "${ONNX_BATCH_TIMEOUT}"
    -max-games "${GENERATE_GAMES}"
  )
  if [[ "${ONNX_BATCH_SIZE}" != "0" ]]; then
    gen_args+=( -onnx-batch-size "${ONNX_BATCH_SIZE}" )
  fi

  go run ./executor "${gen_args[@]}"

  # Build an explicit per-cycle training set (symlinks) so we can archive consumed shards.
  train_dir="/tmp/snek2_train_${ts}"
  mkdir -p "${train_dir}"

  # Only include fully-written parquet shards (exclude tmp and very recent files).
  # Use find -mmin so we don't depend on GNU stat flags.
  age_mins=$(python3 - <<'PY'
import os
sec = int(os.environ.get('MIN_FILE_AGE_SECONDS','30'))
print(max(1, (sec + 59)//60))
PY
)

  mapfile -t gen_files < <(find "${GENERATED_DIR}" -maxdepth 1 -type f -name "*.parquet" ! -path "*/tmp/*" -mmin "+${age_mins}" 2>/dev/null | sort)
  mapfile -t scr_files < <(find "${SCRAPED_DIR}" -maxdepth 1 -type f -name "*.parquet" ! -path "*/tmp/*" -mmin "+${age_mins}" 2>/dev/null | sort)

  if [[ ${#gen_files[@]} -eq 0 && ${#scr_files[@]} -eq 0 ]]; then
    echo "[cycle ${cycle}] no eligible parquet shards yet; skipping training"
    rm -rf "${train_dir}" || true
    if [[ "${SLEEP_BETWEEN_CYCLES}" != "0" ]]; then
      sleep "${SLEEP_BETWEEN_CYCLES}"
    fi
    continue
  fi

  i=0
  for f in "${gen_files[@]}"; do
    ln -sf "${f}" "${train_dir}/generated_${i}.parquet"
    i=$((i+1))
  done
  j=0
  for f in "${scr_files[@]}"; do
    ln -sf "${f}" "${train_dir}/scraped_${j}.parquet"
    j=$((j+1))
  done

  echo "[cycle ${cycle}] training on ${#gen_files[@]} generated + ${#scr_files[@]} scraped shards (ckpt=${ckpt_path})"
  python3 trainer/train.py \
    --data-dir "${train_dir}" \
    --model-path "${ckpt_path}" \
    --epochs "${TRAIN_EPOCHS}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --lr "${TRAIN_LR}"

  echo "[cycle ${cycle}] exporting onnx (${onnx_path})"
  python3 trainer/export_onnx.py --ckpt "${ckpt_path}" --out "${onnx_path}"

  # Archive consumed shards only after a successful train+export.
  for f in "${gen_files[@]}"; do
    base=$(basename "${f}")
    mv -f "${f}" "${PROCESSED_DIR}/generated/${base}"
  done
  for f in "${scr_files[@]}"; do
    base=$(basename "${f}")
    mv -f "${f}" "${PROCESSED_DIR}/scraped/${base}"
  done
  rm -rf "${train_dir}" || true

  # Update "latest" pointers for next generation round.
  rel_ckpt="$(realpath --relative-to="${MODEL_DIR}" "${ckpt_path}" 2>/dev/null || true)"
  rel_onnx="$(realpath --relative-to="${MODEL_DIR}" "${onnx_path}" 2>/dev/null || true)"
  if [[ -n "${rel_ckpt}" ]]; then
    ln -sf "${rel_ckpt}" "${MODEL_DIR}/latest.pt"
  else
    ln -sf "${ckpt_path}" "${MODEL_DIR}/latest.pt"
  fi
  if [[ -n "${rel_onnx}" ]]; then
    ln -sf "${rel_onnx}" "${MODEL_DIR}/snake_net.onnx"
  else
    ln -sf "${onnx_path}" "${MODEL_DIR}/snake_net.onnx"
  fi

  echo "[cycle ${cycle}] done (latest.pt -> ${ckpt_path}, snake_net.onnx -> ${onnx_path})"

  if [[ "${SLEEP_BETWEEN_CYCLES}" != "0" ]]; then
    sleep "${SLEEP_BETWEEN_CYCLES}"
  fi

done
