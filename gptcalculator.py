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

def generate_subtraction_problem(digits: int = config.num_digits) -> Tuple[str, str]:
    num_1 = random.randint(10**(digits-1), 10**digits - 1)
    num_2 = random.randint(0, num_1) 

    problem = f"{num_1}-{num_2}="
    answer = str(num_1 - num_2)
    if len(answer) < digits + 1:
        answer = '0' * ((digits+1) - len(answer)) + answer
    return problem, answer[::-1]

def generate_multiplication_problem(digits: int = config.num_digits) -> Tuple[str, str]:
    max_result_digits = digits + 1
    max_input = 10**(max_result_digits // 2) - 1  

    num_1 = random.randint(1, max_input)
    num_2 = random.randint(1, max_input)

    problem = f"{num_1}*{num_2}="
    answer = str(num_1 * num_2)

    if len(answer) < max_result_digits:
        answer = '0' * (max_result_digits - len(answer)) + answer

    return problem, answer[::-1]

def generate_division_problem(digits: int = config.num_digits) -> Tuple[str, str]:
    if digits == 1:
        quotient = random.randint(1, 9)
        num_2 = random.randint(1, 9)
        num_1 = quotient * num_2
    else:
        max_quotient = 10**digits - 1
        quotient = random.randint(1, max_quotient)
        max_divisor = 10**(digits - 1) - 1
        num_2 = random.randint(1, max_divisor)
        num_1 = quotient * num_2
    problem = f"{num_1}/{num_2}="
    answer = str(quotient)

    if len(answer) < digits + 1:
        answer = '0' * ((digits + 1) - len(answer)) + answer

    return problem, answer[::-1]

def encode_problem(problem: str, answer: str):
    y = [-1] * len(problem)
    x = torch.tensor(encode(problem))
    y.extend(encode(answer))
    y = torch.tensor(y)
    return x, y

def get_batch():
    xs, ys = [], []
    operations = [
        generate_addition_problem,
        generate_subtraction_problem,
        generate_multiplication_problem,
        generate_division_problem,
    ]

    for _ in range(config.batch_size):
        operation = random.choice(operations)
        problem, answer = operation()
        x, y = encode_problem(problem, answer)
        x = F.pad(x, (0, config.block_size - len(x)), value=PAD_TOKEN)
        y = F.pad(y, (0, config.block_size - len(y)), value=-1)
        xs.append(x)
        ys.append(y)
    x_stack = torch.stack(xs)
    y_stack = torch.stack(ys)
    return x_stack.to(config.device), y_stack.to(config.device)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
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
# Note: Direct transformer code (next ~70 lines) adapted from Andrej Karpathy's lecture series
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
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
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)

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

       
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device))
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

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
        for _ in range(config.block_size):
            idx_cond = idx[:, -config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# Test Function
def test_generations(model, num_evals=10):
    operations = [
        generate_addition_problem,
        generate_subtraction_problem,
        generate_multiplication_problem,
        generate_division_problem,
    ]

    for _ in range(num_evals):
        operation = random.choice(operations)
        problem, answer = operation()

        problem_tensified, _ = encode_problem(problem, answer)
        out = model.generate(problem_tensified.unsqueeze(0).to(device))

        out_decoded = decode([int(x) for x in out[0].tolist()])
        out_answer = out_decoded[len(problem):len(problem) + num_digits + 1]
        print(f"Problem: {problem} | Output: {out_answer}")

def evaluate_model_accuracy_by_type(model, num_evals=100):
    operations = {
        "addition": generate_addition_problem,
        "subtraction": generate_subtraction_problem,
        "multiplication": generate_multiplication_problem,
        "division": generate_division_problem,
    }
    stats = {op: {"correct": 0, "total": 0} for op in operations}
    for _ in range(num_evals):
        op_name, operation = random.choice(list(operations.items()))
        problem, answer = operation()
        problem_tensified, _ = encode_problem(problem, answer)
        out = model.generate(problem_tensified.unsqueeze(0).to(device))
        out_decoded = decode([int(x) for x in out[0].tolist()])[len(problem):len(problem) + len(answer)]
        stats[op_name]["total"] += 1
        if out_decoded == answer:
            stats[op_name]["correct"] += 1
    overall_accuracy = sum(stat["correct"] for stat in stats.values()) / num_evals
    operation_accuracies = {op: stat["correct"] / stat["total"] for op, stat in stats.items()}
    return overall_accuracy, operation_accuracies

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
