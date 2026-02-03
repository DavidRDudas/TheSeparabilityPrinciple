# The Separability Principle of Transfer in Grokking

This repository contains the code and experiments for the paper "The Separability Law of Transfer in Grokking".

## Key Finding

Neural networks exhibit predictable transfer patterns during grokking:
- **Positive transfer (+88%)**: Between tasks with compatible functional structure (e.g., add → sub)
- **Negative transfer (-49% to -300%)**: Between tasks with incompatible structure (e.g., mul → sub)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy matplotlib scikit-learn
```

## Reproducing Results

### Core Transfer Experiments
```bash
# Original experiments at p=97
python experiments/transfer.py --prime 97

# Scaling to larger primes
python experiments/transfer.py --prime 197
```

### Scaling Experiments
```bash
# Run all scaling experiments (p=97, p=197, p=509)
python experiments/scaling.py
```

### Analysis (Algebraic Degree + ICC)
```bash
# Reproduce predictability analysis
python experiments/analysis.py
```

### Generate Figures
```bash
python experiments/figures.py
```

## Repository Structure

```
grokking_experiments/
├── experiments/
│   ├── transfer.py      # Core transfer experiments
│   ├── scaling.py       # Scaling to larger primes
│   ├── analysis.py      # Algebraic degree & ICC analysis
│   └── figures.py       # Generate paper figures
├── results/             # Generated figures
├── paper.tex            # LaTeX source
└── README.md
```

## Main Results

| Experiment | Positive Transfer | Negative Transfer |
|------------|-------------------|-------------------|
| p=97       | +88%              | -49%              |
| p=197      | +83%              | -117%             |
| p=509      | 0% (floor)        | -300%             |

**Key insight**: Negative transfer amplifies with scale, while positive transfer remains stable.

## Citation

```bibtex
@article{dudas2026separability,
  title={The Separability Law of Transfer in Grokking},
  author={Dudas, David R.},
  year={2026}
}
```

## Contact

Correspondence: daviddudas@hotmail.com
