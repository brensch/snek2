#!/usr/bin/env bash
set -euo pipefail

# Orchestrates: train -> export onnx -> selfplay generate -> repeat.
# Scraper runs as a separate compose service and continuously writes to data/scraped.

: "${GENERATED_DIR:=data/generated}"
: "${SCRAPED_DIR:=data/scraped}"
: "${PROCESSED_DIR:=processed}"
: "${MODEL_DIR:=models}"
: "${HISTORY_DIR:=${MODEL_DIR}/history}"
: "${WORKSPACE_DIR:=$(pwd)}"       # repo root; /workspace in container

: "${GENERATE_GAMES:=256}"        # per cycle
: "${WORKERS:=512}"
: "${GAMES_PER_FLUSH:=50}"
: "${MCTS_SIMS:=800}"
: "${ONNX_SESSIONS:=1}"
: "${ONNX_BATCH_SIZE:=512}"
: "${ONNX_BATCH_TIMEOUT:=5ms}"
: "${TRACE:=true}"

: "${TRAIN_EPOCHS:=10}"
: "${TRAIN_BATCH_SIZE:=256}"
: "${TRAIN_LR:=0.001}"

: "${SLEEP_BETWEEN_CYCLES:=0}"   # seconds
: "${MIN_FILE_AGE_SECONDS:=0}"   # 0 disables age gating (writers use atomic tmp+rename)
: "${ARCHIVE_EXISTING_ON_START:=0}"  # if 1, move existing shards to processed/* before first cycle
: "${MAX_CYCLES:=0}"             # 0 = infinite

# Python interpreter to use for training/export utilities.
# Prefer local repo venv when running on-host.
: "${PYTHON_BIN:=}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${WORKSPACE_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${WORKSPACE_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[startup] error: python not found (set PYTHON_BIN or install python3)" >&2
    exit 1
  fi
fi

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

# Ensure we have a model and ONNX before starting cycle 1.
if [[ ! -f "${MODEL_DIR}/latest.pt" ]]; then
  echo "[startup] missing ${MODEL_DIR}/latest.pt; initializing"
  "${PYTHON_BIN}" trainer/init_ckpt.py --out "${MODEL_DIR}/latest.pt" --in-channels 10
fi
if [[ ! -f "${MODEL_DIR}/snake_net.onnx" ]]; then
  echo "[startup] missing ${MODEL_DIR}/snake_net.onnx; exporting from latest.pt"
  "${PYTHON_BIN}" trainer/export_onnx.py --ckpt "${MODEL_DIR}/latest.pt" --out "${MODEL_DIR}/snake_net.onnx"
fi

if [[ "${ARCHIVE_EXISTING_ON_START}" == "1" ]]; then
  # Treat any pre-existing shards as "old" and move them out of the active dirs
  # so cycle 1 trains only on newly created data.
  shopt -s nullglob
  old_gen=("${GENERATED_DIR}"/*.parquet)
  old_scr=("${SCRAPED_DIR}"/*.parquet)
  shopt -u nullglob

  if [[ ${#old_gen[@]} -gt 0 || ${#old_scr[@]} -gt 0 ]]; then
    echo "[startup] archiving existing shards: ${#old_gen[@]} generated + ${#old_scr[@]} scraped"
    for f in "${old_gen[@]}"; do
      mv -f "${f}" "${PROCESSED_DIR}/generated/" || true
    done
    for f in "${old_scr[@]}"; do
      mv -f "${f}" "${PROCESSED_DIR}/scraped/" || true
    done
  fi
fi

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
    export LD_LIBRARY_PATH="${WORKSPACE_DIR}:$(IFS=:; echo "${extra_ld[*]}"):${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="${WORKSPACE_DIR}:${LD_LIBRARY_PATH:-}"
  fi

  # Build an explicit per-cycle training set (symlinks) so we can archive consumed shards.
  train_dir="${WORKSPACE_DIR}/.train_sets/snek2_train_${ts}"
  mkdir -p "${train_dir}"

  # Only include fully-written parquet shards (exclude tmp).
  # If MIN_FILE_AGE_SECONDS>0, additionally skip very recent files.
  if [[ "${MIN_FILE_AGE_SECONDS}" =~ ^[0-9]+$ ]] && [[ "${MIN_FILE_AGE_SECONDS}" -gt 0 ]]; then
    cutoff_epoch=$(date -d "-${MIN_FILE_AGE_SECONDS} seconds" +%s)
    mapfile -t gen_files < <(find "${GENERATED_DIR}" -maxdepth 1 -type f -name "*.parquet" ! -path "*/tmp/*" ! -newermt "@${cutoff_epoch}" 2>/dev/null | sort)
    mapfile -t scr_files < <(find "${SCRAPED_DIR}" -maxdepth 1 -type f -name "*.parquet" ! -path "*/tmp/*" ! -newermt "@${cutoff_epoch}" 2>/dev/null | sort)
  else
    mapfile -t gen_files < <(find "${GENERATED_DIR}" -maxdepth 1 -type f -name "*.parquet" ! -path "*/tmp/*" 2>/dev/null | sort)
    mapfile -t scr_files < <(find "${SCRAPED_DIR}" -maxdepth 1 -type f -name "*.parquet" ! -path "*/tmp/*" 2>/dev/null | sort)
  fi

  if [[ ${#gen_files[@]} -eq 0 && ${#scr_files[@]} -eq 0 ]]; then
    echo "[cycle ${cycle}] no eligible parquet shards yet; skipping training"
    rm -rf "${train_dir}" || true
  else
    i=0
    for f in "${gen_files[@]}"; do
      f_abs="$(realpath "${f}" 2>/dev/null || readlink -f "${f}" 2>/dev/null || echo "${f}")"
      ln -sf "${f_abs}" "${train_dir}/generated_${i}.parquet"
      i=$((i+1))
    done
    j=0
    for f in "${scr_files[@]}"; do
      f_abs="$(realpath "${f}" 2>/dev/null || readlink -f "${f}" 2>/dev/null || echo "${f}")"
      ln -sf "${f_abs}" "${train_dir}/scraped_${j}.parquet"
      j=$((j+1))
    done

    if [[ ! -f "${ckpt_path}" ]]; then
      cp -f "${MODEL_DIR}/latest.pt" "${ckpt_path}"
    fi

    echo "[cycle ${cycle}] training on ${#gen_files[@]} generated + ${#scr_files[@]} scraped shards (ckpt=${ckpt_path})"
    "${PYTHON_BIN}" trainer/train.py \
      --data-dir "${train_dir}" \
      --model-path "${ckpt_path}" \
      --epochs "${TRAIN_EPOCHS}" \
      --batch-size "${TRAIN_BATCH_SIZE}" \
      --lr "${TRAIN_LR}"

    echo "[cycle ${cycle}] exporting onnx (${onnx_path})"
    "${PYTHON_BIN}" trainer/export_onnx.py --ckpt "${ckpt_path}" --out "${onnx_path}"

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
  fi

  echo "[cycle ${cycle}] generating ${GENERATE_GAMES} games using ${MODEL_DIR}/snake_net.onnx"

  gen_args=(
    -out-dir "${GENERATED_DIR}"
    -workers "${WORKERS}"
    -games-per-flush "${GAMES_PER_FLUSH}"
    -trace="${TRACE}"
    -mcts-sims "${MCTS_SIMS}"
    -onnx-sessions "${ONNX_SESSIONS}"
    -onnx-batch-timeout "${ONNX_BATCH_TIMEOUT}"
    -max-games "${GENERATE_GAMES}"
  )
  if [[ "${ONNX_BATCH_SIZE}" != "0" ]]; then
    gen_args+=( -onnx-batch-size "${ONNX_BATCH_SIZE}" )
  fi

  go run ./executor "${gen_args[@]}"

  echo "[cycle ${cycle}] done"

  if [[ "${MAX_CYCLES}" =~ ^[0-9]+$ ]] && [[ "${MAX_CYCLES}" -gt 0 ]] && [[ "${cycle}" -ge "${MAX_CYCLES}" ]]; then
    echo "[cycle ${cycle}] reached MAX_CYCLES=${MAX_CYCLES}; exiting"
    exit 0
  fi

  if [[ "${SLEEP_BETWEEN_CYCLES}" != "0" ]]; then
    sleep "${SLEEP_BETWEEN_CYCLES}"
  fi

done
