# Finetune GPT-2 (124M) on tiny Shakespeare with BPE.
# Good default for Mac overnight runs.

out_dir = 'out-shakespeare-bpe'

dataset = 'shakespeare'
init_from = 'gpt2'

# evaluation/logging
log_interval = 1
eval_interval = 200
eval_iters = 50

# batch/sequence
block_size = 256
batch_size = 4
gradient_accumulation_steps = 8

# training length
max_iters = 5000

# optimizer
learning_rate = 1e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = False

# regularization
# dropout helps avoid overfitting on tiny data
# (can set to 0.0 for max memorization)
dropout = 0.1
