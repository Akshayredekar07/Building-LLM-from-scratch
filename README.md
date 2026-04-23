# **Building GPT-2 from Scratch**

A clean, end-to-end implementation of the GPT-2 architecture in PyTorch covering every layer from raw text preprocessing to pretraining, sampling, and evaluation. Built by reading the original paper and implementing each component incrementally.

---

## Overview

This project implements GPT-2 from first principles, without using Hugging Face `transformers` or any high-level model library. Every component tokenization, self-attention, causal masking, multi-head attention, transformer blocks, and the pretraining loop is coded and explained from scratch.

The goal is not just to reproduce results, but to deeply understand *why* each design decision exists in the original [GPT-2 paper (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

---

## Architecture

```
Input tokens
     │
     ▼
Token Embedding + Positional Embedding
     │
     ▼
┌─────────────────────────────────┐
│         Transformer Block × N   │
│                                 │
│  ┌──────────────────────────┐   │
│  │   Layer Normalization    │   │
│  └──────────┬───────────────┘   │
│             │                   │
│  ┌──────────▼───────────────┐   │
│  │  Multi-Head Causal       │   │
│  │  Self-Attention          │   │
│  └──────────┬───────────────┘   │
│             │  (+ residual)     │
│  ┌──────────▼───────────────┐   │
│  │   Layer Normalization    │   │
│  └──────────┬───────────────┘   │
│             │                   │
│  ┌──────────▼───────────────┐   │
│  │  Feed-Forward (GELU)     │   │
│  └──────────┬───────────────┘   │
│             │  (+ residual)     │
└─────────────┼───────────────────┘
              │
              ▼
       Linear + Softmax
              │
              ▼
         Next Token Logits
```

**GPT-2 Small configuration** (default):

| Hyperparameter    | Value   |
|-------------------|---------|
| Vocab size        | 50,257  |
| Context length    | 1,024   |
| Embedding dim     | 768     |
| Attention heads   | 12      |
| Transformer layers| 12      |
| Feed-forward dim  | 3,072   |
| Parameters        | ~124M   |

---

## Project Structure

```
Building-LLM-from-scratch/
│
├── data/
│   ├── raw/                        # Raw corpus text files
│   │   └── the-verdict.txt
│   └── processed/                  # Tokenized tensors (.pt files)
│
├── notebooks/                      # Incremental exploration notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_self_attention.ipynb
│   ├── 03_trainable_attention.ipynb
│   ├── 04_causal_attention.ipynb
│   ├── 05_multihead_attention.ipynb
│   └── 06_architecture_overview.ipynb
│
├── src/
│   ├── data/
│   │   ├── tokenizer.py            # BPE tokenizer (tiktoken wrapper)
│   │   └── dataset.py              # GPTDataset, DataLoader setup
│   │
│   ├── model/
│   │   ├── attention.py            # SimpleSelfAttention, CausalAttention, MultiHeadAttention
│   │   ├── layers.py               # LayerNorm, GELU, FeedForward
│   │   ├── transformer.py          # TransformerBlock with residual connections
│   │   └── gpt.py                  # Full GPTModel — assembles all components
│   │
│   ├── training/
│   │   ├── trainer.py              # Pretraining loop, gradient clipping, logging
│   │   ├── loss.py                 # Cross-entropy loss, perplexity calculation
│   │   └── scheduler.py            # Cosine LR schedule with warmup
│   │
│   └── inference/
│       ├── generate.py             # Greedy, temperature scaling, top-k sampling
│       └── evaluate.py             # Token-level accuracy, loss curves
│
├── configs/
│   ├── gpt2_small.yaml             # 124M — default training config
│   └── gpt2_medium.yaml            # 355M — scaled-up config
│
├── docs/
│   ├── notes/                      # Concept notes (.docx) per topic
│   └── papers/                     # Reference PDFs (original paper, etc.)
│
├── checkpoints/                    # Saved model weights (epoch_*.pt)
├── outputs/                        # Generated text samples, loss plots
├── tests/
│   ├── test_attention.py
│   └── test_model.py
│
├── requirements.txt
└── README.md
```


## Getting Started

### Prerequisites

```bash
Python >= 3.10
PyTorch >= 2.1
CUDA 12.x (optional, for GPU training)
```

### Installation

```bash
git clone https://github.com/Akshayredekar07/Building-LLM-from-scratch.git
cd Building-LLM-from-scratch

pip install -r requirements.txt
```

---

## Training

### 1. Preprocess the data

```python
from src.data.tokenizer import get_tokenizer
from src.data.dataset import create_dataloader

tokenizer = get_tokenizer()  # wraps tiktoken gpt2 encoding
train_loader = create_dataloader(
    txt_file="data/raw/the-verdict.txt",
    tokenizer=tokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
)
```

### 2. Instantiate the model

```python
from src.model.gpt import GPTModel
import yaml

with open("configs/gpt2_small.yaml") as f:
    config = yaml.safe_load(f)

model = GPTModel(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# → Parameters: 123,653,888
```

### 3. Run pretraining

```python
from src.training.trainer import pretrain

pretrain(
    model=model,
    train_loader=train_loader,
    num_epochs=10,
    lr=3e-4,
    checkpoint_dir="checkpoints/",
)
```

---

## Inference & Text Generation

```python
from src.inference.generate import generate

output = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="The quick brown fox",
    max_new_tokens=50,
    strategy="top_k",   # "greedy" | "temperature" | "top_k"
    temperature=0.8,
    top_k=40,
)

print(output)
```

### Sampling strategies implemented

| Strategy | Description | Key param |
|----------|-------------|-----------|
| Greedy | Always pick argmax token | N/A |
| Temperature | Soften/sharpen the distribution | `temperature` ∈ (0, ∞) |
| Top-k | Sample from top-k most likely tokens | `top_k` ∈ [1, vocab_size] |

---

## Concepts Covered

**Attention mechanism**
- Simplified self-attention (dot-product, no learnable weights)
- Trainable self-attention with Q/K/V projections
- Causal (masked) attention, preventing lookahead
- Multi-head attention with parallel heads

**Architecture components**
- Layer normalization (pre-norm variant, GPT-2 style)
- GELU activation (vs ReLU, with smoother gradient flow)
- Shortcut (residual) connections, which stabilize deep network training
- Transformer block composition

**Training**
- Cross-entropy loss on next-token prediction
- Perplexity as evaluation metric
- Full pretraining loop with gradient clipping
- Cosine LR schedule with linear warmup

**Inference**
- Temperature scaling, which controls output randomness
- Top-k sampling, which avoids low-probability tail tokens
- Measuring LLM loss and evaluating generation quality

---

## Key Design Decisions

**Why pre-norm instead of post-norm?**
GPT-2 uses Layer Norm *before* the attention and FFN sublayers (pre-norm), unlike the original Transformer paper (post-norm). Pre-norm stabilizes training at large scale because gradients flow more evenly through residual paths.

**Why GELU over ReLU?**
GELU (Gaussian Error Linear Unit) is non-zero for small negative inputs, giving smoother gradients and consistently better empirical performance on language tasks than the hard zero cutoff of ReLU.

**Why causal masking?**
GPT-2 is a *decoder-only* model trained to predict the next token. The causal mask (upper-triangular block of −∞) ensures that attention at position `i` can only attend to positions `0..i`, preserving the autoregressive property.

---

## References

- [Language Models are Unsupervised Multitask Learners, Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need, Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [Build a Large Language Model (From Scratch), Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- [Building a Large Language Model (LLM) from Scratch. Nothing will be assumed. Everything will be spelled out., Vizuara.ai](https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&si=iqMtK2sbM14l-7gW)
- [The Annotated Transformer, Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/)

---

## License

MIT License, see [LICENSE](LICENSE) for details.
