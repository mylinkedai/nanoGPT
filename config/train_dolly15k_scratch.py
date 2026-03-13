# Finetune GPT-2 (124M) on Databricks Dolly 15k (instruction-following).
# Mac-friendly defaults; adjust as needed.

out_dir = 'out-dolly15k'

dataset = 'dolly'
init_from = 'scratch'

# evaluation/logging
log_interval = 1
eval_interval = 200
eval_iters = 50

# batch/sequence
block_size = 512
batch_size = 2
gradient_accumulation_steps = 16

# training length
max_iters = 3000

# optimizer
learning_rate = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = False

# regularization
dropout = 0.1
