"""
Scaling Experiments for "The Separability Law of Transfer in Grokking"
======================================================================

This script reproduces the scaling experiments (p=97, p=197, p=509).
Shows that positive transfer remains stable (~80-88%) while negative 
transfer amplifies with scale (-49% to -300%).

Usage:
    python experiments/scaling.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

device = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')

def op_add(a, b, p): return (a + b) % p
def op_sub(a, b, p): return (a - b) % p
def op_mul(a, b, p): return (a * b) % p

def generate_data(op_fn, prime):
    pairs, labels = [], []
    for a in range(prime):
        for b in range(prime):
            pairs.append((a, b))
            labels.append(op_fn(a, b, prime))
    return torch.tensor(pairs), torch.tensor(labels)

class MLP(nn.Module):
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

def run_experiment(prime, hidden_dim, source_op, target_op, 
                   max_epochs=15000, pretrain_epochs=10000, batch_size=2048):
    """Run transfer experiment with batched training for large primes."""
    criterion = nn.CrossEntropyLoss()
    
    pairs, labels = generate_data(target_op, prime)
    n = len(pairs)
    idx = torch.randperm(n)
    
    train = pairs[idx[:n//2]].to(device)
    test = pairs[idx[n//2:]].to(device)
    train_labels = labels[idx[:n//2]].to(device)
    test_labels = labels[idx[n//2:]].to(device)
    
    model = MLP(prime, hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    # Pretrain if source provided
    if source_op is not None:
        pairs_src, labels_src = generate_data(source_op, prime)
        train_src = pairs_src[idx[:n//2]].to(device)
        train_labels_src = labels_src[idx[:n//2]].to(device)
        test_src_labels = labels_src[idx[n//2:]].to(device)
        
        for epoch in range(pretrain_epochs):
            for i in range(0, len(train_src), batch_size):
                batch_x = train_src[i:i+batch_size]
                batch_y = train_labels_src[i:i+batch_size]
                model.train()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 1000 == 0 and epoch > 0:
                model.eval()
                with torch.no_grad():
                    acc = (model(test).argmax(1) == test_src_labels).float().mean().item()
                if acc > 0.90:
                    break
        
        del pairs_src, labels_src, train_src, train_labels_src
        gc.collect()
    
    # Train on target
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    for epoch in range(max_epochs):
        for i in range(0, len(train), batch_size):
            batch_x = train[i:i+batch_size]
            batch_y = train_labels[i:i+batch_size]
            model.train()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(test).argmax(1) == test_labels).float().mean().item()
            if acc > 0.90:
                return epoch, True
    
    return max_epochs, False

def main():
    print("=" * 70)
    print("SCALING EXPERIMENTS")
    print(f"Device: {device}")
    print("=" * 70)
    print()
    
    # Test configurations: (prime, hidden_dim)
    configs = [
        (97, 128),   # Original
        (197, 256),  # Medium scale
        (509, 512),  # Large scale
    ]
    
    results = []
    
    for prime, hidden in configs:
        print(f"\n{'='*70}")
        print(f"PRIME = {prime}, HIDDEN = {hidden}")
        print(f"{'='*70}\n")
        
        # Baseline
        print("Baseline (sub)...", end=" ", flush=True)
        baseline, bg = run_experiment(prime, hidden, None, op_sub)
        print(f"{baseline} epochs")
        gc.collect()
        
        # Positive
        print("add → sub...", end=" ", flush=True)
        positive, pg = run_experiment(prime, hidden, op_add, op_sub)
        print(f"{positive} epochs")
        gc.collect()
        
        # Negative
        print("mul → sub...", end=" ", flush=True)
        negative, ng = run_experiment(prime, hidden, op_mul, op_sub)
        print(f"{negative} epochs")
        gc.collect()
        
        # Calculate effects
        if bg and baseline > 0:
            pos_effect = (baseline - positive) / baseline * 100 if pg else None
            neg_effect = (baseline - negative) / baseline * 100 if ng else None
        else:
            pos_effect, neg_effect = None, None
        
        results.append({
            'prime': prime,
            'baseline': baseline,
            'positive': positive,
            'negative': negative,
            'pos_effect': pos_effect,
            'neg_effect': neg_effect
        })
    
    # Summary table
    print("\n" + "=" * 70)
    print("SCALING RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Prime':<10} {'Baseline':<12} {'Pos Transfer':<15} {'Neg Transfer':<15}")
    print("-" * 52)
    for r in results:
        pos = f"{r['pos_effect']:+.0f}%" if r['pos_effect'] is not None else "N/A"
        neg = f"{r['neg_effect']:+.0f}%" if r['neg_effect'] is not None else "CATASTROPHIC"
        print(f"p={r['prime']:<7} {r['baseline']:<12} {pos:<15} {neg:<15}")

if __name__ == "__main__":
    main()
