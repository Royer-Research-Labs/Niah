# A small, fast, NIAH-capable demo run.
#
# Trains a ~10M param GPT on the tiny-shakespeare corpus encoded with the GPT-2
# BPE tokenizer (NOT char-level: NIAH keyword choices must be single GPT-2 tokens).
# Runs the during-training NIAH eval so you can watch retrieval accuracy evolve.
#
#   python data/shakespeare/prepare.py
#   python train.py config/train_niah_demo.py            # GPU
#   python train.py config/train_niah_demo.py --device=cpu --compile=False   # CPU
#
# This is a toy: a from-scratch 10M model on ~300k tokens will show only modest
# NIAH accuracy. It exists to demonstrate the full loop end to end, not to win.

out_dir = 'out-niah-demo'
eval_interval = 100
eval_iters = 20
log_interval = 10

always_save_checkpoint = True   # tiny run; keep the latest checkpoint

wandb_log = False
wandb_project = 'niah'
wandb_run_name = 'niah-demo'

dataset = 'shakespeare'         # GPT-2 BPE bin files from data/shakespeare/prepare.py
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256                # also caps the NIAH context length

# a small model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# torch.compile recompiles per NIAH input length; off by default so the demo is
# snappy on CPU and avoids recompiles. Turn it on for longer real training runs.
compile = False

# --- NIAH during-training eval ---
niah_eval = True
niah_eval_interval = 500        # run NIAH every 500 steps
niah_context_length = 256       # scored input length, question included; <= block_size
niah_num_needles = 4
niah_samples = 5
niah_use_shared_choices = True
niah_tokenizer = 'gpt2'
