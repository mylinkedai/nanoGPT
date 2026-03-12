# Train from scratch on tiny Shakespeare with GPT-2 BPE (no pretrained weights).

out_dir = 'out-shakespeare-bpe-scratch'

dataset = 'shakespeare'
init_from = 'scratch'

# evaluation/logging
log_interval = 1
eval_interval = 200
eval_iters = 50

# batch/sequence
block_size = 256
batch_size = 12
gradient_accumulation_steps = 4

# training length (adjust with --max_iters or duration presets in train_on_mac.sh)
max_iters = 5000

# optimizer
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 3e-5

# regularization
# dropout helps avoid overfitting on tiny data
# (can set to 0.0 for max memorization)
dropout = 0.1
