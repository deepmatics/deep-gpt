import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(2016)
batch_size = 4 # number of independent sequences that will be processed in parallel 
block_size = 8 # maximum context length for the predictions

# Read the input file
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# encode and decode logic
s_to_i = {ch:i for i, ch in enumerate(chars)}
i_to_s = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join(i_to_s[i] for i in l)

print(encode("is this a GPT tokenizer?"))
print(decode(encode("this is a GPT tokenizer?")))

# convert to torch tensor
data = torch.tensor(encode(text), dtype = torch.long)

# train and test split
n = int(len(data)*90)
train = data[:n]
val = data[n:]

# Generate training batches
def get_batch(split):

    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_size, (batch_size, )) #to review
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: block_size+ i + 1] for i in ix])
    return x, y

xb, yb = get_batch('train')

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)

        return logits
    
m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out.shape)
