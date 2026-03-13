import json
import os
import random
import requests
import tiktoken
import numpy as np

# Databricks Dolly 15k (instruction-following) dataset.
# Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k

DATA_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
    "databricks-dolly-15k.jsonl"
)

data_dir = os.path.dirname(__file__)
input_file_path = os.path.join(data_dir, "databricks-dolly-15k.jsonl")

if not os.path.exists(input_file_path):
    print(f"Downloading Dolly 15k dataset to {input_file_path}...")
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

examples = []
with open(input_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        instruction = (obj.get("instruction") or "").strip()
        context = (obj.get("context") or "").strip()
        response = (obj.get("response") or "").strip()
        if not instruction or not response:
            continue

        if context:
            text = (
                "Instruction:\n"
                f"{instruction}\n\n"
                "Context:\n"
                f"{context}\n\n"
                "Response:\n"
                f"{response}\n\n"
            )
        else:
            text = (
                "Instruction:\n"
                f"{instruction}\n\n"
                "Response:\n"
                f"{response}\n\n"
            )
        examples.append(text)

if not examples:
    raise RuntimeError("No valid examples found in Dolly dataset.")

# Shuffle before split for a more representative validation set.
random.seed(1337)
random.shuffle(examples)

n = len(examples)
split = int(n * 0.9)
train_text = "".join(examples[:split])
val_text = "".join(examples[split:])

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_text)
val_ids = enc.encode_ordinary(val_text)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, "train.bin"))
val_ids.tofile(os.path.join(data_dir, "val.bin"))
