# Interpreting Scaled Dot-Product and Multi-Head Attention

## Overview

This project provides a hands-on, from-scratch exploration of scaled dot-product attention and multi-head self-attention. Rather than training a full Transformer, the goal is to isolate and understand the mechanics of attention—how similarity scores are computed, normalized, and used to mix information across tokens.

Through small synthetic examples and visualization, this project demonstrates how attention behaves mathematically and why key design choices (like sqrt(d_k) scaling and multiple heads) are necessary.

---

## Methods

The project implements attention from first principles using PyTorch:

- **Scaled Dot-Product Attention**
  - Compute similarity scores using `QK^T`
  - Normalize using `sqrt(d_k)`
  - Apply softmax row-wise
  - Form weighted sums of value vectors

- **Scaling Experiment**
  - Increase `d_k` and observe how score magnitudes affect softmax behavior

- **Self-Attention (Q = K = V)**
  - Analyze clustering behavior when tokens attend based purely on similarity

- **Multi-Head Self-Attention (Minimal)**
  - Split the embedding space into subspaces
  - Compute attention independently per head
  - Concatenate outputs

Attention heatmaps are used throughout to visualize token-to-token influence patterns.

---

## Key Results

- Attention acts as a structured similarity-based mixing mechanism.
- The `sqrt(d_k)` scaling term stabilizes softmax behavior as dimensionality increases.
- Multi-head attention enables different similarity structures to be learned in parallel.
- Heatmaps provide a concrete, interpretable view of token influence.

---

## How to Run

1. Clone the repository
2. Run pip install -r requirements.txt (There's only two packages and the import commands are included in the notebook)
3. It's a jupyternotebook so you can either select run all at the top or run each cell individually