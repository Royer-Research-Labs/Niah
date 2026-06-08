"""Prepare the tiny-shakespeare corpus with the GPT-2 BPE tokenizer.

Vendored from nanoGPT (Karpathy, MIT). Writes train.bin / val.bin next to this
file. We use BPE (not char-level) so that NIAH keyword choices are single tokens
and the eval tokenizer ('gpt2') matches the model's training tokenizer.
"""

import os

import numpy as np
import requests
import tiktoken

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
np.array(train_ids, dtype=np.uint16).tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
np.array(val_ids, dtype=np.uint16).tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
# train.bin has ~301,966 tokens; val.bin has ~36,059 tokens
