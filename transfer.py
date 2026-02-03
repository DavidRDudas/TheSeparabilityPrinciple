"""
Transfer Experiments for "The Separability Law of Transfer in Grokking"
========================================================================

This script reproduces the core transfer learning experiments from the paper.
It tests positive transfer (add → sub) and negative transfer (mul → sub).

Usage:
    python experiments/transfer.py --prime 97
    python experiments/transfer.py --prime 197
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')

# Operations
def op_add(a, b, p): return (a + b) % p
def op_sub(a, b, p): return (a - b) % p
def op_mul(a, b, p): return (a * b) % p

def generate_data(op_fn, prime):
    """Generate all pairs (a, b) and their labels for a given operation."""
    pairs, labels = [], []
    for a in range(prime):
        for b in range(prime):
            pairs.append((a, b))
            labels.append(op_fn(a, b, prime))
    return torch.tensor(pairs), torch.tensor(labels)

class MLP(nn.Module):
    """Simple 2-layer MLP for modular arithmetic."""
    def __init__(self, prime, hidden_dim):
        super().__init__()
        self.embed_a = nn.Embedding(prime, hidden_dim)
        self.embed_b = nn.Embedding(prime, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prime)
        )
    
    def forward(self, x):
        a, b = x[:, 0], x[:, 1]
        return self.net(self.embed_a(a) + self.embed_b(b))

def train_to_grokking(model, train_data, train_labels, test_data, test_labels,
                      max_epochs=15000, lr=1e-3, weight_decay=1.0):
    """Train model until it groks (>90% test accuracy)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(max_epochs):
        model.train()
        logits = model(train_data)
        loss = criterion(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                test_acc = (model(test_data).argmax(1) == test_labels).float().mean().item()
            if test_acc > 0.90:
                return epoch, True
    
    return max_epochs, False

def run_transfer_experiment(prime, hidden_dim, source_op, target_op, 
                           pretrain_epochs=10000, max_epochs=15000):
    """Run a single transfer experiment."""
    # Generate data
    pairs, labels = generate_data(target_op, prime)
    n = len(pairs)
    idx = torch.randperm(n)
    
    train = pairs[idx[:n//2]].to(device)
    test = pairs[idx[n//2:]].to(device)
    train_labels = labels[idx[:n//2]].to(device)
    test_labels = labels[idx[n//2:]].to(device)
    
    model = MLP(prime, hidden_dim).to(device)
    
    # Pretrain on source task if specified
    if source_op is not None:
        pairs_src, labels_src = generate_data(source_op, prime)
        train_src = pairs_src[idx[:n//2]].to(device)
        train_labels_src = labels_src[idx[:n//2]].to(device)
        test_src_labels = labels_src[idx[n//2:]].to(device)
        
        print(f"  Pretraining on source task...", end=" ", flush=True)
        grok_epoch, grokked = train_to_grokking(
            model, train_src, train_labels_src, test, test_src_labels,
            max_epochs=pretrain_epochs
        )
        print(f"{'grokked at ' + str(grok_epoch) if grokked else 'done'}")
    
    # Train on target task
    print(f"  Training on target task...", end=" ", flush=True)
    # Reset optimizer for fine-tuning
    grok_epoch, grokked = train_to_grokking(
        model, train, train_labels, test, test_labels,
        max_epochs=max_epochs
    )
    print(f"{'grokked at ' + str(grok_epoch) if grokked else 'did not grok'}")
    
    return grok_epoch, grokked

def main():
    parser = argparse.ArgumentParser(description='Transfer experiments')
    parser.add_argument('--prime', type=int, default=97, help='Prime number for modular arithmetic')
    parser.add_argument('--hidden', type=int, default=None, help='Hidden dimension (default: proportional to prime)')
    args = parser.parse_args()
    
    prime = args.prime
    hidden_dim = args.hidden if args.hidden else max(128, int(prime * 1.3))
    
    print("=" * 70)
    print(f"TRANSFER EXPERIMENTS: p={prime}")
    print(f"Hidden dim: {hidden_dim}, Device: {device}")
    print("=" * 70)
    print()
    
    results = {}
    
    # Baseline
    print("1. Baseline (sub from scratch):")
    baseline, bg = run_transfer_experiment(prime, hidden_dim, None, op_sub)
    results['baseline'] = baseline
    print()
    
    # Positive transfer
    print("2. Positive transfer (add → sub):")
    positive, pg = run_transfer_experiment(prime, hidden_dim, op_add, op_sub)
    results['positive'] = positive
    print()
    
    # Negative transfer
    print("3. Negative transfer (mul → sub):")
    negative, ng = run_transfer_experiment(prime, hidden_dim, op_mul, op_sub)
    results['negative'] = negative
    print()
    
    # Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline (sub):  {results['baseline']:,} epochs")
    print(f"add → sub:       {results['positive']:,} epochs")
    print(f"mul → sub:       {results['negative']:,} epochs")
    print()
    
    if bg and results['baseline'] > 0:
        pos_effect = (results['baseline'] - results['positive']) / results['baseline'] * 100
        neg_effect = (results['baseline'] - results['negative']) / results['baseline'] * 100
        print(f"Positive transfer: {pos_effect:+.0f}%")
        print(f"Negative transfer: {neg_effect:+.0f}%")

if __name__ == "__main__":
    main()
