# Custom GPT2 Implementation for CLIF

A clean, memory-efficient GPT2 implementation based on nanoGPT for training on clinical hospitalization narratives.

## Overview

This custom GPT2 implementation provides an alternative to the HuggingFace-based pipeline (AR/gpt2_hf/) with:

- **nanoGPT Architecture** - Clean, understandable PyTorch implementation (easy to modify)
- **Polars Streaming** - Memory-efficient processing for massive datasets (never loads full data)
- **Registry-Based Vocabulary** - Includes ALL tokens from token_registry.json (better generalization)
- **Simple Sequential Chunking** - Filter sequences > 8K tokens (97% coverage)
- **Vocabulary Lock System** - SHA256 hash validation prevents drift across experiments

## Key Features

### 🎯 Simplicity
- **784 lines** of clean model code (vs thousands in HuggingFace)
- Easy to understand and modify
- Direct PyTorch implementation

### 💾 Memory Efficiency
- Polars streaming: Never loads full dataset into memory
- Can process 100GB+ parquet files
- 10-100x faster aggregations than pandas

### 🔬 Better Generalization
- Registry-based vocabulary includes rare events (e.g., ECMO flow rates)
- Ensures consistent vocabulary across all experiments
- Prevents catastrophic incompatibility in federated learning

### ⚡ Performance
- Flash Attention support (2-4x faster)
- Gradient checkpointing (~60% memory reduction)
- Packed sequence collation (1.5-2x throughput)

## Quick Start

### 1. Install Dependencies

```bash
# Already installed if you have CLIFATRON environment
pip install torch transformers datasets polars pyarrow
```

### 2. Full Pipeline (8K Context)

```bash
# Step 0: Convert narratives to sentences
uv run AR/gpt2/00_convert_narrative_to_sentences.py \
    --train-val OutputTokens/narratives/train_val_sequences.parquet \
    --test OutputTokens/narratives/test_sequences.parquet \
    --output-dir ./models/gpt2/data

# Step 1: Build vocabulary from registry
uv run AR/gpt2/01a_build_vocab_from_registry.py \
    --token-registry OutputTokens/token_registry.json \
    --output-dir ./models/gpt2/vocab

# Step 2: Create splits and tokenize
uv run AR/gpt2/03_create_splits.py \
    --presplit \
    --train-val ./models/gpt2/data/clif_sentences_train_val.parquet \
    --test ./models/gpt2/data/clif_sentences_test.parquet \
    --vocab-dir ./models/gpt2/vocab \
    --output-dir ./models/gpt2/splits \
    --max-length 8192

# Step 3: Train model (L40 profile)
uv run python AR/gpt2/04_train_gpt2.py \
    --config-profile l40 \
    --data-dir ./models/gpt2/splits \
    --vocab-dir ./models/gpt2/vocab \
    --output-dir ./models/gpt2/checkpoints \
    --model-size small \
    --epochs 5 \
    --context-size 8192 \
    --gradient-checkpointing \
    --wandb \
    --run-name gpt2-small-8k

# Step 4: Evaluate on test set
uv run python AR/gpt2/05_evaluate_test.py \
    --model models/gpt2/checkpoints/gpt2-small-8k/final_model \
    --data-dir models/gpt2/splits \
    --vocab models/gpt2/vocab/vocab.gzip \
    --output models/gpt2/checkpoints/gpt2-small-8k/test_results.json \
    --batch-size 4
```

### 3. Training a Quick Test Model (1 Epoch)

```bash
# Fast training for testing (completes in ~1 hour on 2x L40)
uv run python AR/gpt2/04_train_gpt2.py \
    --config-profile l40 \
    --data-dir ./models/gpt2/splits \
    --vocab-dir ./models/gpt2/vocab \
    --output-dir ./models/gpt2/checkpoints \
    --model-size small \
    --epochs 1 \
    --context-size 8192 \
    --gradient-checkpointing \
    --run-name gpt2-small-test
```

## Model Sizes

With CLIF vocabulary (1,373 tokens):

| Model | Parameters | Context | Memory (8K) | L40 (2x 48GB) | A100 (8x 40GB) |
|-------|------------|---------|-------------|---------------|----------------|
| **small** | 92.4M | 8,192 | ~15GB/GPU | ✅ Recommended | ✅ Recommended |
| **medium** | 340M | 8,192 | ~18GB/GPU | ✅ Supported | ✅ Recommended |
| **large** | 760M | 8,192 | ~30GB/GPU | ✅ Tight fit | ✅ Recommended |
| **xl** | 1.48B | 8,192 | ~55GB/GPU | ❌ Too large | ⚠️ A100-80GB only |

## Differences from AR/gpt2_hf/

| Feature | AR/gpt2/ (Custom) | AR/gpt2_hf/ (HuggingFace) |
|---------|-------------------|---------------------------|
| **Model** | nanoGPT (custom) | HuggingFace GPT2 |
| **Vocabulary** | Registry-based (ALL tokens) | Data-based (frequency) |
| **Data** | Polars streaming | Pandas + custom dataset |
| **Special Tokens** | TL_START/TL_END | BOS/EOS |
| **Chunking** | Filter > 8K | Sequential 8,190 boundaries |
| **Tokenization** | Vectorized replace | Element-wise lookup |
| **Hub** | Not compatible | Full HuggingFace integration |

### When to Use AR/gpt2/ (This Implementation)

✅ You need maximum control and code clarity  
✅ Memory efficiency is critical (huge datasets)  
✅ You want to modify the architecture  
✅ You don't need HuggingFace model hub integration  

### When to Use AR/gpt2_hf/ (HuggingFace)

✅ You need HuggingFace ecosystem compatibility  
✅ You want to share models on HuggingFace Hub  
✅ You need rich tokenizer features  
✅ You need day-boundary aware chunking  

## Documentation

- **Quick Start**: This README
- **Comprehensive**: [docs/TOKENIZATION_AND_TRAINING.md](docs/TOKENIZATION_AND_TRAINING.md)
- **Model Architecture**: [model.py](model.py)
- **Training Config**: [config.py](config.py)
- **Dataset**: [dataset.py](dataset.py)
- **Vocabulary**: [vocabulary.py](vocabulary.py)

## Example Training Results

**GPT2-small (8K context, 1 epoch on 2x L40)**:
```
Training time: 52 minutes
Final training loss: 2.31
Parameters: 92.4M
Memory per GPU: ~15GB
Throughput: 3.1 samples/sec
```

---

**Last Updated**: 2025-10-31  
**Version**: 1.0
