#!/usr/bin/env bash
set -euo pipefail

# Simple Mac-friendly training script for nanoGPT.
# Default: Apple Silicon MPS if available, otherwise CPU.

PYTHON="${PYTHON:-python3}"
PRINT_STEPS="${PRINT_STEPS:-0}"

step() {
  echo ">>> STEP: $*"
}

if [[ "$PRINT_STEPS" -eq 1 && -f "steps.md" ]]; then
  step "print steps.md"
  echo "===== steps.md ====="
  cat "steps.md"
  echo "===================="
fi

DEVICE="${DEVICE:-}"
if [[ -z "${DEVICE}" ]]; then
  step "default device: mps"
  DEVICE="mps"
fi

if [[ "${DEVICE}" == "mps" ]]; then
  step "validate mps runtime"
  if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import torch
torch.zeros(1, device="mps")
PY
  then
    echo "MPS is unavailable or unusable in this environment. Exiting because DEVICE=mps is required."
    exit 1
  fi
fi

DEFAULT_CONFIG="config/train_dolly15k_scratch.py"

# args: [config] [duration]
ARG1="${1:-}"
ARG2="${2:-}"

if [[ "$ARG1" =~ ^(10m|2h|1d)$ ]]; then
  step "parse args: duration-only"
  CONFIG="$DEFAULT_CONFIG"
  DURATION="${ARG1}"
else
  step "parse args: config + optional duration"
  CONFIG="${ARG1:-$DEFAULT_CONFIG}"
  if [[ "$ARG2" =~ ^(10m|2h|1d)$ ]]; then
    DURATION="${ARG2}"
  fi
fi

DURATION="${DURATION:-}"

# Derive dataset/init_from/out_dir from config.
step "read config: dataset/init_from/out_dir"
CONFIG_INFO="$(CONFIG_PATH="$CONFIG" "$PYTHON" - <<'PY'
import os

config = os.environ.get("CONFIG_PATH", "config/finetune_shakespeare_gpt2.py")
g = {}
g["__file__"] = config
with open(config, "r", encoding="utf-8") as f:
    code = f.read()
exec(compile(code, config, "exec"), g, g)
dataset = g.get("dataset", "")
init_from = g.get("init_from", "scratch")
out_dir = g.get("out_dir", "out")
print(f"DATASET={dataset}")
print(f"INIT_FROM={init_from}")
print(f"OUT_DIR_FROM_CONFIG={out_dir}")
PY
)"
eval "$CONFIG_INFO"

# Decide out_dir (env wins).
OUT_DIR="${OUT_DIR:-${OUT_DIR_FROM_CONFIG:-out}}"

# Ensure required deps exist (no venv). Install to user site-packages if missing.
SKIP_PIP="${SKIP_PIP:-0}"
NEED_TRANSFORMERS=0
if [[ "${INIT_FROM:-scratch}" == gpt2* ]]; then
  NEED_TRANSFORMERS=1
fi

if [[ "$NEED_TRANSFORMERS" -eq 1 ]]; then
  step "check deps: numpy/torch/requests/tiktoken/transformers"
  if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import numpy, torch, requests, tiktoken, transformers  # noqa: F401
PY
  then
    if [[ "$SKIP_PIP" -eq 1 ]]; then
      echo "Missing deps (numpy/torch/requests/tiktoken/transformers) and SKIP_PIP=1; please install them first."
      exit 1
    fi
    echo "Installing Python deps (numpy, torch, requests, tiktoken, transformers) to user site-packages..."
    # Homebrew Python enforces PEP 668; allow user-site installs explicitly.
    "$PYTHON" -m pip install --user --break-system-packages -U pip numpy torch requests tiktoken transformers
    if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import numpy, torch, requests, tiktoken, transformers  # noqa: F401
PY
    then
      echo "Failed to import required deps after install. Exiting."
      exit 1
    fi
  fi
else
  step "check deps: numpy/torch/requests/tiktoken"
  if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import numpy, torch, requests, tiktoken  # noqa: F401
PY
  then
    if [[ "$SKIP_PIP" -eq 1 ]]; then
      echo "Missing deps (numpy/torch/requests/tiktoken) and SKIP_PIP=1; please install them first."
      exit 1
    fi
    echo "Installing Python deps (numpy, torch, requests, tiktoken) to user site-packages..."
    # Homebrew Python enforces PEP 668; allow user-site installs explicitly.
    "$PYTHON" -m pip install --user --break-system-packages -U pip numpy torch requests tiktoken
    if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import numpy, torch, requests, tiktoken  # noqa: F401
PY
    then
      echo "Failed to import required deps after install. Exiting."
      exit 1
    fi
  fi
fi

# Conservative defaults for Mac.
step "check dataset availability"
if CONFIG_PATH="$CONFIG" "$PYTHON" - <<'PY'
import os
import sys

config = os.environ.get("CONFIG_PATH", "config/finetune_shakespeare_gpt2.py")
g = {}
g["__file__"] = config
with open(config, "r", encoding="utf-8") as f:
    code = f.read()
exec(compile(code, config, "exec"), g, g)
dataset = g.get("dataset", "")
if dataset:
    data_dir = os.path.join("data", dataset)
    train_bin = os.path.join(data_dir, "train.bin")
    if not os.path.exists(train_bin):
        sys.exit(2)
sys.exit(0)
PY
then
  NEED_PREP=0
else
  status=$?
  if [[ $status -eq 2 ]]; then
    NEED_PREP=1
  else
    exit $status
  fi
fi

if [[ ${NEED_PREP:-0} -eq 1 ]]; then
  if [[ "${DATASET:-}" == "shakespeare" ]]; then
    step "prepare dataset: shakespeare (BPE)"
    echo "Preparing Shakespeare BPE dataset..."
    "$PYTHON" data/shakespeare/prepare.py
  elif [[ "${DATASET:-}" == "shakespeare_char" ]]; then
    step "prepare dataset: shakespeare_char"
    echo "Preparing Shakespeare char dataset..."
    "$PYTHON" data/shakespeare_char/prepare.py
  elif [[ "${DATASET:-}" == "dolly" ]]; then
    step "prepare dataset: dolly"
    echo "Preparing Dolly 15k dataset..."
    "$PYTHON" data/dolly/prepare.py
  else
    step "prepare dataset: unknown (error)"
    echo "Dataset ${DATASET:-<unknown>} missing train.bin; please prepare it manually."
    exit 1
  fi
fi

step "print device"
echo "Using device: $DEVICE"

LOG_INPLACE="${LOG_INPLACE:-0}"
LOG_INPLACE_ARG=()
if [[ "$LOG_INPLACE" -eq 1 ]]; then
  LOG_INPLACE_ARG=(--log_inplace=True)
fi

# Optional duration presets (rough estimates; actual time depends on hardware/settings)
MAX_ITERS_OVERRIDE="${MAX_ITERS_OVERRIDE:-}"
MAX_ITERS_FROM_PRESET=0
MAX_TIME_SECONDS_OVERRIDE="${MAX_TIME_SECONDS_OVERRIDE:-}"
if [[ -n "$DURATION" && -z "$MAX_ITERS_OVERRIDE" ]]; then
  step "apply duration preset: $DURATION"
  case "$DURATION" in
    10m)
      # The tiny Shakespeare BPE scratch config runs very fast on MPS.
      # Bump the preset so "10m" is closer to ~8-12 minutes on recent Macs.
      if [[ "$DEVICE" == "mps" && "$CONFIG" == "config/train_shakespeare_bpe_scratch.py" ]]; then
        MAX_ITERS_OVERRIDE=50000
      else
        MAX_ITERS_OVERRIDE=2000
      fi
      ;;
    2h)  MAX_ITERS_OVERRIDE=24000 ;;
    1d)  MAX_ITERS_OVERRIDE=288000 ;;
  esac
  MAX_ITERS_FROM_PRESET=1
  MAX_ITERS_ESTIMATE="$MAX_ITERS_OVERRIDE"
  if [[ -z "$MAX_TIME_SECONDS_OVERRIDE" ]]; then
    case "$DURATION" in
      10m) MAX_TIME_SECONDS_OVERRIDE=600 ;;
      2h)  MAX_TIME_SECONDS_OVERRIDE=7200 ;;
      1d)  MAX_TIME_SECONDS_OVERRIDE=86400 ;;
    esac
  fi
  # Ensure time-based limit can take effect even on fast hardware.
  if [[ -n "$MAX_TIME_SECONDS_OVERRIDE" && "$MAX_ITERS_FROM_PRESET" -eq 1 ]]; then
    MAX_ITERS_OVERRIDE=100000000
  fi
  if [[ -n "${MAX_ITERS_ESTIMATE:-}" && -n "$MAX_TIME_SECONDS_OVERRIDE" ]]; then
    echo "Duration preset: $DURATION -> max_time_seconds=$MAX_TIME_SECONDS_OVERRIDE, max_iters_estimate=$MAX_ITERS_ESTIMATE, max_iters_cap=$MAX_ITERS_OVERRIDE"
  else
    echo "Duration preset: $DURATION -> max_iters=$MAX_ITERS_OVERRIDE (estimate)"
  fi
fi

LOG_FILE="${LOG_FILE:-train.log}"
LOG_TS="${LOG_TS:-1}"

TRAIN_CMD=( env PYTHONUNBUFFERED=1 "$PYTHON" -u train.py "$CONFIG"
  --device="$DEVICE"
  --out_dir="$OUT_DIR"
  ${LOG_INPLACE_ARG[@]:+"${LOG_INPLACE_ARG[@]}"}
  ${MAX_ITERS_OVERRIDE:+--max_iters="$MAX_ITERS_OVERRIDE"}
  ${MAX_TIME_SECONDS_OVERRIDE:+--max_time_seconds="$MAX_TIME_SECONDS_OVERRIDE"}
  --compile=False
)

if [[ "$LOG_TS" -eq 1 ]]; then
  step "run training (timestamped log)"
  "${TRAIN_CMD[@]}" 2>&1 | "$PYTHON" -u -c '
import sys
from datetime import datetime

log_path = sys.argv[1]
with open(log_path, "a", encoding="utf-8") as f:
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out = f"[{ts}] {line}"
        print(out, flush=True)
        f.write(out + "\n")
' "$LOG_FILE"
else
  step "run training (stdout)"
  "${TRAIN_CMD[@]}"
fi

RUN_SAMPLE="${RUN_SAMPLE:-1}"
if [[ "$RUN_SAMPLE" -eq 1 ]]; then
  step "run sample generation"
  echo "Running a quick sample..."
  "$PYTHON" sample.py \
    --out_dir="$OUT_DIR" \
    --device="$DEVICE" \
    --num_samples=1 \
    --max_new_tokens=200 \
    --start="To be"
fi
