import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.onnx
from torch import nn

# Load Indian cities from local CSV
cities = pd.read_csv("cities.csv")["city"].tolist()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(cities))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0  # start and finish char
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

def encode(s: str):
    return [stoi[c] for c in s]

def decode(ints: list[int]):
    return ''.join(itos[i] for i in ints)

from dataclasses import dataclass
from typing import Literal

@dataclass
class LearningInterval():
    lr: int
    iters: int

def train_model_full(model, schedules, eval_interval=2000):
    for i, sch in enumerate(schedules):
        print(f"SCHEDULE {i+1}/{len(schedules)}: lr={sch.lr}, iters={sch.iters}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=sch.lr)

        for cur_iter in range(sch.iters):
            if cur_iter % eval_interval == 0 or cur_iter == sch.iters - 1:
                losses = db.estimate_loss(model)
                print(f"iter {cur_iter + 1}/{sch.iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = db.get_batch('full')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

class DatasetManager:
    def __init__(self, train_data, val_data, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self.train_dataset = self._build_dataset(train_data)
        self.val_dataset = self._build_dataset(val_data)
        self.full_dataset = self._build_dataset(cities)

    def _build_dataset(self, data):
        X, Y = [], []
        for w in data:
            encoding = encode(w + '.')
            context = encode('.') * self.block_size
            for idx in encoding:
                X.append(context)
                Y.append(idx)
                context = context[1:] + [idx]
        return torch.tensor(X), torch.tensor(Y)

    def get_batch(self, split: Literal["train", "val", "full"]):
        if split == "train":
            data = self.train_dataset
        elif split == "val":
            data = self.val_dataset
        else:
            data = self.full_dataset
        ix = torch.randint(len(data[0]), (self.batch_size,))
        return data[0][ix], data[1][ix]

    def estimate_loss(self, model, eval_iters=200):
        out = {}
        model.eval()
        with torch.no_grad():
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = self.get_batch(split)
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
        model.train()
        return out

# HYPERPARAMETERS
block_size = 15  # Increased for longer Indian city names
batch_size = 64  # Increased batch size for better training
n_embd = 48     # Increased embedding dimension
n_hidden = 256  # Increased hidden layer size

# Randomly split into train/val
indices = torch.randperm(len(cities))
split = int(0.9*len(cities))
train_data = [cities[i] for i in indices[:split]]
val_data = [cities[i] for i in indices[split:]]

db = DatasetManager(train_data, val_data, batch_size=batch_size, block_size=block_size)

class FinalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(vocab_size, n_embd),
            nn.Flatten(start_dim=1),
            nn.Linear(n_embd * block_size, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(n_hidden, n_hidden // 2),  # Added another layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hidden // 2, vocab_size)
        )

        with torch.no_grad():
            self.net[-1].weight *= 0.1

    def forward(self, x, targets=None):
        logits = self.net(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, number_of_cities):
        for _ in range(number_of_cities):
            out = []
            context = [0] * block_size

            while True:
                logits = self.net(torch.tensor([context]))
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break

            print(''.join(itos[i] for i in out))

    def infer(self):
        context = [0] * block_size
        out = []
        
        while True:
            logits = self.net(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
                
        return ''.join(itos[i] for i in out)

if __name__ == '__main__':
    model = FinalMLP()
    total_params = sum(p.numel() for p in model.parameters())
    print("Params: ", total_params)

    final_schedule = [
        LearningInterval(1e-2, 10_000),
        LearningInterval(1e-3, 15_000),
        LearningInterval(1e-4, 15_000),
        LearningInterval(1e-5, 25_000),
    ]

    train_model_full(model, final_schedule)

    model.generate(40)  # Generate 40 example cities

    torch.save(model.state_dict(), "in_cities_mlp.pt") 