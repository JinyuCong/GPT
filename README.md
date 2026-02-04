# GPT

A minimal GPT (Generative Pre-trained Transformer) implementation from scratch using PyTorch. The project includes a custom BPE tokenizer, decoder-only transformer with RoPE (Rotary Positional Embeddings), training pipeline on the Alpaca dataset, and text generation with top-k sampling.

## Project Structure

```
.
├── config.py                 # Hyperparameters and configuration
├── layers.py                 # Model architecture (attention, FFN, decoder, GPT)
├── tokenizer.py              # Byte-Pair Encoding tokenizer
├── dataset.py                # Alpaca dataset loading and preprocessing
├── train.py                  # Training loop with early stopping
├── generate.py               # Text generation with sampling strategies
├── Alan_Turing_corpus.txt    # Corpus for BPE tokenizer training
├── logs.json                 # Pretrained tokenizer vocabulary and merge rules
└── gpt_weights.pth           # Trained model weights
```

## Model Architecture

Decoder-only Transformer with the following components:

- **Token Embedding** with weight tying (shared with output head)
- **RoPE** (Rotary Positional Embeddings) for position encoding
- **Masked Multi-Head Attention** with causal masking
- **Feed-Forward Network** with ReLU activation (4x expansion)
- **Residual connections + LayerNorm** at each sublayer

Default configuration:

| Parameter | Value |
|-----------|-------|
| Embedding dim | 256 |
| Sequence length | 64 |
| Attention heads | 3 |
| Decoder layers | 12 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |

## Dependencies

```
torch
transformers
datasets
```

Install:

```bash
pip install torch transformers datasets
```

## Usage

### Train

```bash
python train.py
```

Trains the model on the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) using the Meta-Llama-3-8B tokenizer. Supports early stopping (patience=2). Trained weights are saved to `gpt_weights.pth`.

### Generate

```bash
python generate.py
```

Loads trained weights and generates text with temperature scaling (1.5) and top-k sampling (k=1000).

### Train BPE Tokenizer

```bash
python tokenizer.py
```

Trains a custom BPE tokenizer on `Alan_Turing_corpus.txt` and saves vocabulary/merge rules to `logs.json`.

## Configuration

All hyperparameters can be modified in `config.py`.
