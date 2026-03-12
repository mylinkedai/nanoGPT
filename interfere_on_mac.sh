#!/usr/bin/env bash
set -euo pipefail

# Simple Mac-friendly chat script for nanoGPT.
# Default: Apple Silicon MPS if available, otherwise CPU.

PYTHON="${PYTHON:-python3}"

# Ensure required deps exist (no venv). Install to user site-packages if missing.
if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import numpy, torch, requests, tiktoken  # noqa: F401
PY
then
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

DEVICE="${DEVICE:-}"
if [[ -z "${DEVICE}" ]]; then
  DEVICE="$("$PYTHON" - <<'PY'
import platform
import torch

def macos_major():
    ver = platform.mac_ver()[0]
    try:
        return int(ver.split(".")[0])
    except Exception:
        return 0

mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
if mps_ok and macos_major() >= 14:
    print("mps")
else:
    print("cpu")
PY
  )"
fi

OUT_DIR="${OUT_DIR:-${1:-out-shakespeare-bpe}}"

OUT_DIR="$OUT_DIR" DEVICE="$DEVICE" "$PYTHON" - "$OUT_DIR" <<'PY'
import os
import sys
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT

out_dir = os.environ.get("OUT_DIR", "out-shakespeare-char")
if len(sys.argv) > 1:
    out_dir = sys.argv[1]

device = os.environ.get("DEVICE", "cpu")
max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "200"))
temperature = float(os.environ.get("TEMPERATURE", "0.8"))
top_k = int(os.environ.get("TOP_K", "200"))
seed = int(os.environ.get("SEED", "1337"))
max_turns = int(os.environ.get("HISTORY_TURNS", "6"))
system_prompt = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")

ckpt_path = os.path.join(out_dir, "ckpt.pt")
if not os.path.exists(ckpt_path):
    print(f"Checkpoint not found: {ckpt_path}")
    print("Train first, or set OUT_DIR to a directory with ckpt.pt")
    sys.exit(1)

print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

torch.manual_seed(seed)
if "cuda" in device:
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# model
if "model_args" not in checkpoint:
    print("Invalid checkpoint: missing model_args")
    sys.exit(1)

gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]

# Unwrap DDP prefix if present
unwanted_prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

# dtype/ctx
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = torch.float16 if device_type == "cuda" else torch.float32
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if "config" in checkpoint and "dataset" in checkpoint["config"]:
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

history = []

print("Commands: /help, /reset, /exit")

# Ensure we can read interactive input even if stdin is closed/non-tty.
in_stream = sys.stdin
if not in_stream.isatty():
    try:
        in_stream = open("/dev/tty", "r", encoding="utf-8", errors="ignore")
    except Exception:
        in_stream = sys.stdin

while True:
    try:
        print("You> ", end="", flush=True)
        user = in_stream.readline()
        if user == "":
            raise EOFError()
        user = user.strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not user:
        continue
    if user in {"/exit", "/quit"}:
        print("Exiting.")
        break
    if user == "/reset":
        history.clear()
        print("History cleared.")
        continue
    if user == "/help":
        print("/reset clears history, /exit quits.")
        continue

    parts = [system_prompt, ""]
    for u, a in history:
        parts.append(f"User: {u}\nAssistant: {a}")
    parts.append(f"User: {user}\nAssistant:")
    prompt = "\n".join(parts).strip() + " "

    prompt_ids = encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    gen_ids = y[0].tolist()
    reply = decode(gen_ids[len(prompt_ids):])

    # Stop at common separators
    for stop in ["\nUser:", "\nUser", "\n\n", "\nAssistant:"]:
        idx = reply.find(stop)
        if idx != -1:
            reply = reply[:idx]
            break

    reply = reply.strip()
    if not reply:
        reply = "(no output)"

    print(f"Assistant> {reply}")

    history.append((user, reply))
    if max_turns > 0 and len(history) > max_turns:
        history = history[-max_turns:]
PY
