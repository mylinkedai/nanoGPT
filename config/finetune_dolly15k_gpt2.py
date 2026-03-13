# Finetune GPT-2 (124M) on Databricks Dolly 15k (English instructions).
# This is a practical baseline for "common-sense-like" behavior on Mac.

out_dir = 'out-dolly15k-gpt2'

dataset = 'dolly'
init_from = 'gpt2'

# evaluation/logging
log_interval = 1
eval_interval = 200
eval_iters = 50

# batch/sequence (Mac-friendly)
block_size = 512
batch_size = 2
gradient_accumulation_steps = 16

# training length
max_iters = 4000

# optimizer
learning_rate = 5e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = False

# regularization
dropout = 0.1
