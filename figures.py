"""
Figure Generation for "The Separability Law of Transfer in Grokking"
====================================================================

This script generates the key figures used in the paper.

Usage:
    python experiments/figures.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

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

def generate_learning_curves(save_path='results/learning_curves.png'):
    """Generate learning curves showing grokking phenomenon."""
    print("Generating learning curves...", end=" ", flush=True)
    
    prime, hidden = 97, 128
    pairs, labels = generate_data(op_add, prime)
    n = len(pairs)
    idx = torch.randperm(n)
    
    train = pairs[idx[:n//2]].to(device)
    test = pairs[idx[n//2:]].to(device)
    train_labels = labels[idx[:n//2]].to(device)
    test_labels = labels[idx[n//2:]].to(device)
    
    model = MLP(prime, hidden).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    
    train_accs, test_accs, epochs_list = [], [], []
    
    for epoch in range(6001):
        model.train()
        logits = model(train)
        loss = criterion(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (model(train).argmax(1) == train_labels).float().mean().item()
                test_acc = (model(test).argmax(1) == test_labels).float().mean().item()
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epochs_list.append(epoch)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_accs, label='Train', linewidth=2)
    plt.plot(epochs_list, test_accs, label='Test', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Grokking: Delayed Generalization', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved to {save_path}")

def generate_transfer_comparison(save_path='results/paper_figure_1.png'):
    """Generate main transfer comparison figure."""
    print("Generating transfer comparison...", end=" ", flush=True)
    
    # Results from experiments
    conditions = ['Baseline\n(sub)', 'Positive\n(add→sub)', 'Negative\n(mul→sub)']
    epochs = [4500, 500, 6700]
    colors = ['#666666', '#2ecc71', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conditions, epochs, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    
    # Add labels
    for bar, ep in zip(bars, epochs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{ep:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add effect annotations
    ax.annotate('+88%', xy=(1, 500), xytext=(1, 2000),
                fontsize=16, color='#2ecc71', fontweight='bold',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    ax.annotate('-49%', xy=(2, 6700), xytext=(2, 5500),
                fontsize=16, color='#e74c3c', fontweight='bold',
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    
    ax.set_ylabel('Epochs to Grok', fontsize=14)
    ax.set_title('The Separability Law of Transfer', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 8000)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved to {save_path}")

def generate_scaling_figure(save_path='results/scaling_law.png'):
    """Generate scaling results figure."""
    print("Generating scaling figure...", end=" ", flush=True)
    
    primes = [97, 197, 509]
    pos_transfer = [88, 83, 0]
    neg_transfer = [-49, -117, -300]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Positive transfer
    ax1.bar(range(len(primes)), pos_transfer, color='#2ecc71', edgecolor='black')
    ax1.set_xticks(range(len(primes)))
    ax1.set_xticklabels([f'p={p}' for p in primes])
    ax1.set_ylabel('Transfer Effect (%)')
    ax1.set_title('Positive Transfer (add→sub)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylim(-10, 100)
    
    # Negative transfer
    ax2.bar(range(len(primes)), neg_transfer, color='#e74c3c', edgecolor='black')
    ax2.set_xticks(range(len(primes)))
    ax2.set_xticklabels([f'p={p}' for p in primes])
    ax2.set_ylabel('Transfer Effect (%)')
    ax2.set_title('Negative Transfer (mul→sub)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylim(-350, 50)
    
    plt.suptitle('Scaling: Negative Transfer Amplifies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved to {save_path}")

def main():
    os.makedirs('results', exist_ok=True)
    
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)
    print()
    
    generate_learning_curves()
    generate_transfer_comparison()
    generate_scaling_figure()
    
    print()
    print("Done! Figures saved to results/")

if __name__ == "__main__":
    main()
