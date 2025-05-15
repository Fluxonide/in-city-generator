import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.onnx
from torch import nn

# Load Indian cities from local CSV
df = pd.read_csv("cities.csv")
cities = df['city'].tolist()  # Get city names from the 'city' column

# Clean the city names (remove any NaN or empty values)
cities = [str(city).strip() for city in cities if pd.notna(city) and str(city).strip()]

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(cities))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0  # start and finish char
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

def encode(s: str):
    try:
        return [stoi[c] for c in s]
    except KeyError as e:
        print(f"Error encoding character in string '{s}': {e}")
        raise

def decode(ints: list[int]):
    try:
        return ''.join(itos[i] for i in ints)
    except KeyError as e:
        print(f"Error decoding index in list {ints}: {e}")
        raise

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
        print("Building training dataset...")
        self.train_dataset = self._build_dataset(train_data)
        print("Building validation dataset...")
        self.val_dataset = self._build_dataset(val_data)
        print("Building full dataset...")
        self.full_dataset = self._build_dataset(cities)

    def _build_dataset(self, data):
        X, Y = [], []
        for w in data:
            try:
                # Add start and end tokens
                w = '.' + w + '.'
                # Ensure the context window is filled
                if len(w) < self.block_size:
                    w = '.' * (self.block_size - len(w)) + w
                
                # Encode the string
                encoding = encode(w)
                
                # Create training examples
                for i in range(len(encoding) - self.block_size):
                    context = encoding[i:i + self.block_size]
                    target = encoding[i + self.block_size]
                    X.append(context)
                    Y.append(target)
            except Exception as e:
                print(f"Warning: Skipping city '{w}' due to error: {e}")
                continue
        
        if not X or not Y:
            raise ValueError("No valid training examples could be created")
            
        return torch.tensor(X), torch.tensor(Y)

    def get_batch(self, split: Literal["train", "val", "full"]):
        if split == "train":
            data = self.train_dataset
        elif split == "val":
            data = self.val_dataset
        else:
            data = self.full_dataset
            
        if len(data[0]) == 0:
            raise ValueError(f"No data available for split {split}")
            
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
block_size = 8  # Reduced block size for better training
batch_size = 32  # Reduced batch size for stability
n_embd = 32     # Reduced embedding dimension
n_hidden = 128  # Reduced hidden layer size

# Randomly split into train/val
indices = torch.randperm(len(cities))
split = int(0.9*len(cities))
train_data = [cities[i] for i in indices[:split]]
val_data = [cities[i] for i in indices[split:]]

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

db = DatasetManager(train_data, val_data, batch_size=batch_size, block_size=block_size)

class FinalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(vocab_size, n_embd),
            nn.Flatten(start_dim=1),
            nn.Linear(n_embd * block_size, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_hidden, vocab_size)
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
    print(f"Total cities: {len(cities)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters in vocabulary: {chars}")
    
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