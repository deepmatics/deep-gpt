import torch
import torch.nn
from torch.nn import functional as F

torch.manual_seed(1000)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

s_to_i = {ch:i for i, ch in enumerate(chars)}
i_to_s = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join(i_to_s[i] for i in l)

print(encode("is this a GPT tokenizer?"))
print(decode(encode("this is a GPT tokenizer?")))