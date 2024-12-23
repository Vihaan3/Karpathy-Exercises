import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    batch_size: int = 128
    num_digits: int = 1
    block_size = 3*num_digits + 3 
    max_iters: int = 50000
    eval_interval: int = 500
    learning_rate: int = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 20
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

config = Config()

# Vocabulary Setup
vocab = '0123456789+='
vocab_size = len(vocab) + 1
stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }
PAD_TOKEN = vocab_size - 1
stoi['P'] = PAD_TOKEN
itos[PAD_TOKEN] = 'P'
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Problem Generation Function
def generate_addition_problem(digits: int = config.num_digits) -> Tuple[str, str]:
  num_1 = random.randint(0, 10**digits -1)
  num_2 = random.randint(0, 10**digits -1)

  problem = f"{num_1}+{num_2}="
  answer = str(num_1+num_2)
  if len(answer) < digits + 1:
    answer = '0'+answer 
  return problem, answer[::-1]

def encode_problem(problem: str, answer: str):
  y = [-1] * len(problem)
  x = torch.tensor(encode(problem))
  y.extend(encode(answer))
  y = torch.tensor(y)
  return x, y

# Get_Batch Function
def get_batch():
  xs, ys = [], []
  for i in range(config.batch_size):
    x, y = encode_problem(*generate_addition_problem())
    x = F.pad(x, (0, config.block_size - len(x)), value=PAD_TOKEN)
    y = F.pad(y, (0, config.block_size - len(y)), value=-1)
    xs.append(x)
    ys.append(y)
  x_stack = torch.stack(xs)
  y_stack = torch.stack(ys)
  return x_stack.to(config.device), y_stack.to(config.device)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fn(logits, targets)

        return logits, loss

    def generate(self, idx):
        # idx is (B, T) array of indices in the current context
        for _ in range(config.block_size):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Test Function
def test_generations(model, num_evals = 10):
  for i in range(num_evals):
    problem, answer = generate_addition_problem()
    problem_tensified, _ = encode_problem(problem, answer)
    out = model.generate(problem_tensified.unsqueeze(0).to(config.device))
    out = decode([int(x) for x in out[0].tolist()])[len(problem):]
    print(f"Problem: {problem} | Output: {out[:config.num_digits+1][::-1]}")

# Define model
model = GPTLanguageModel()
m = model.to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)

# Train Loop
for iteration in range(config.max_iters):
  if iteration % config.eval_interval == 0:
    model.eval()
    losses = []
    with torch.no_grad():
      for _ in range(config.eval_iters):
        for k in range(config.eval_iters):
            X, Y = get_batch()
            logits, loss = model(X, Y)
            losses.append(loss.item())
    model.train()
    mean_loss = torch.mean(torch.tensor(losses))
    print(f"step {iteration}: loss {mean_loss:.4f}")
    xb, yb = get_batch()

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
