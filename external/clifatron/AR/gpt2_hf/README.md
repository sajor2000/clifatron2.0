# GPT2 Clinical Narrative Generation

Complete Hugging Face implementation for training GPT2 models on clinical narratives with extended context (8192 tokens) and sequence packing.

## Overview

This implementation trains GPT2 models from scratch on clinical event sequences using:
- **Extended context**: 8192 tokens (vs standard 1024)
- **Sequence packing**: Reduces padding waste from ~46% to <5%
- **Fast iteration**: 2-step pipeline (preprocess once, train many times)
- **Custom tokenizer**: 1,380 clinical tokens (whitespace-based)
- **Multi-GPU training**: DeepSpeed ZeRO-2/ZeRO-3 with auto-configuration

## Quick Start

### 1. Preprocess Data (Run Once)

```bash
# PRIMARY MODE (first site - can build tokenizer)
uv run AR/gpt2_hf/01_preprocess_data.py \
    --model-size small \
    --split-mode temporal \
    --mode primary

# This creates cached datasets in models/gpt2_hf/preprocessed/small_temporal_len8192/
# Takes 15-30 minutes, but only needs to be done once!
```

### 2. Train Model (Fast Restarts)

```bash
# Train GPT2-small (124M parameters)
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --mode primary

# Train GPT2-medium (355M parameters)
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size medium \
    --preprocessed-dir models/gpt2_hf/preprocessed/medium_temporal_len8192 \
    --mode primary

# Training starts in <2 minutes!
```

### 3. Evaluate Model

```bash
# Evaluate on test set
uv run AR/gpt2_hf/03_evaluate_test.py \
    --checkpoint models/gpt2_hf/checkpoints/clif-gpt2-small/final_model \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --mode primary
```

## Vocabulary Lock System (Multi-Site Training)

This implementation includes a **vocabulary lock system** for safe multi-site training.

### PRIMARY vs SECONDARY Mode

**PRIMARY MODE** (First Site):
- Can build tokenizer if it doesn't exist
- Creates the master vocabulary (1,380 tokens)
- Generates vocabulary hash for validation
- Use this for initial setup

**SECONDARY MODE** (Other Sites):
- **Cannot** build tokenizer - must use existing vocabulary
- Requires vocabulary from primary site
- Validates vocabulary hash matches primary
- **Prevents model corruption** from vocabulary mismatches

### Why Vocabulary Locking?

**Problem without locking:**
```
Site A: "age_56_65" → token ID 234
Site B: "age_56_65" → token ID 567  ❌ DIFFERENT!
Result: Model from Site A corrupted when finetuned at Site B
```

**Solution with locking:**
```
Primary Site: Creates vocab, hash = a1b2c3d4...
Secondary Sites: Use same vocab, validate hash = a1b2c3d4 ✓
Result: Token IDs identical everywhere, safe model transfer
```

### Multi-Site Workflow

**1. Primary Site (First Site):**
```bash
# Build tokenizer and preprocess
uv run AR/gpt2_hf/01_preprocess_data.py \
    --model-size small \
    --mode primary \
    --split-mode temporal

# Note the vocabulary hash displayed (e.g., "a1b2c3d4...")

# Train model
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --mode primary \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192
```

**2. Secondary Sites (All Other Sites):**
```bash
# Copy tokenizer from primary site
rsync -avz primary:/path/AR/qwen2/tokenizer/clinical_tokenizer/ \
            AR/qwen2/tokenizer/clinical_tokenizer/

# Preprocess with LOCKED vocabulary
uv run AR/gpt2_hf/01_preprocess_data.py \
    --model-size small \
    --mode secondary \  # Cannot build tokenizer!
    --split-mode temporal

# Vocabulary hash must match primary site or preprocessing fails

# Train/finetune model
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --mode secondary \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192
```

### Vocabulary Validation

All scripts automatically validate:
1. **Vocabulary size**: Must be exactly 1,380 tokens
2. **Vocabulary hash**: SHA256 hash must match across sites
3. **Consistency**: Cached data must match current tokenizer

**If validation fails:**
```
❌ VOCABULARY MISMATCH!
   Tokenizer vocab hash: a1b2c3d4...
   Cached data vocab hash: e5f6g7h8...

→ Copy correct tokenizer from primary site
→ Rerun preprocessing with correct tokenizer
```

## Model Sizes

| Model | Params | Layers | Hidden | Heads | Memory/GPU | Time (2x L40) |
|-------|--------|--------|--------|-------|------------|---------------|
| **small** | 124M | 12 | 768 | 12 | ~8 GB | ~1-2 days |
| **medium** | 355M | 24 | 1024 | 16 | ~12 GB | ~2-3 days |

## Architecture Details

### GPT2 vs Qwen2

| Feature | GPT2 | Qwen2 |
|---------|------|-------|
| **Attention** | Standard multi-head | Grouped Query Attention (GQA) |
| **Position Encoding** | Absolute positional embeddings | Rotary Position Embeddings (RoPE) |
| **Activation** | GELU | SwiGLU |
| **Normalization** | LayerNorm | RMSNorm |
| **Max Context** | 8192 (extended) | 8192 (native) |

### Key Features

**Sequence Packing**
- Packs multiple hospitalizations into 8192-token sequences
- Industry-standard pattern: `[BOS] hosp1 [EOS] hosp2 [EOS] hosp3 [EOS]`
- Causal attention prevents cross-contamination
- Reduces padding waste from ~46% to <5%
- Increases training throughput by 1.5-2x

**Simple Sequential Chunking**
- Long hospitalizations split at 8,190-token boundaries
- Zero overlap, zero waste
- Each chunk gets [BOS] and [EOS] tokens

**Custom Clinical Tokenizer**
- Whitespace-based tokenization
- 1,380 tokens: 5 special + 1,375 clinical
- Reuses tokenizer from qwen2 implementation

## Hardware Requirements

### GPT2-small (124M)
- **Minimum**: 2x 16GB GPUs
- **Recommended**: 2x L40 48GB
- **Training time**: ~1-2 days (4 epochs)
- **Batch size**: 8 per GPU, gradient accumulation 12

### GPT2-medium (355M)
- **Minimum**: 2x 24GB GPUs
- **Recommended**: 2x L40 48GB or 1x A100 40GB
- **Training time**: ~2-3 days (3 epochs)
- **Batch size**: 6 per GPU, gradient accumulation 16

## Directory Structure

```
AR/gpt2_hf/
├── 01_preprocess_data.py       # Tokenize & cache datasets
├── 02_train_gpt2.py             # Fast training script
├── 03_evaluate_test.py          # Evaluation script
├── config/
│   ├── training_config.yaml     # Hyperparameters
│   ├── ds_config_zero2.json     # DeepSpeed ZeRO-2
│   ├── ds_config_zero3.json     # DeepSpeed ZeRO-3
│   └── gpu_profiles.yaml        # GPU auto-detection
├── data/
│   ├── narrative_dataset.py     # Dataset loader
│   └── data_collator.py         # Packing collator
├── models/
│   └── gpt2_configs.py          # Model configs
├── utils/
│   ├── cache_utils.py           # Caching utilities
│   ├── gpu_detector.py          # GPU auto-config
│   └── metrics.py               # Evaluation metrics
└── docs/
    ├── README.md                # This file
    ├── GPT2_SETUP_GUIDE.md      # Setup instructions
    └── GPT2_ARCHITECTURE.md     # Architecture details
```

## Configuration

### Training Config (`config/training_config.yaml`)

```yaml
# Global settings
vocab_size: 1380
max_length: 8192
enable_packing: true

# Model-specific hyperparameters
models:
  small:
    num_epochs: 4
    batch_size: 8
    gradient_accumulation_steps: 12
    learning_rate: 3.0e-4

  medium:
    num_epochs: 3
    batch_size: 6
    gradient_accumulation_steps: 16
    learning_rate: 2.5e-4
```

### GPU Profiles (`config/gpu_profiles.yaml`)

Automatically detects:
- GPU model (L40, A100, V100, H100, RTX)
- VRAM per GPU
- Compute capability (for BF16 support)
- Optimal DeepSpeed stage (ZeRO-2 or ZeRO-3)

## Advanced Usage

### Resume Training

```bash
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --resume-from checkpoint-1000
```

### Custom Configuration

```bash
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --train-config my_custom_config.yaml
```

### Disable Packing

Edit `config/training_config.yaml`:
```yaml
enable_packing: false  # Standard padding (no packing)
```

### Single GPU Training

```bash
# Auto-selects ZeRO-3 for single GPU
python AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce batch size in `config/training_config.yaml`
2. Increase gradient accumulation steps to maintain effective batch size
3. Enable gradient checkpointing (already enabled by default)
4. Use ZeRO-3 instead of ZeRO-2

### Slow Data Loading
- Make sure you ran `01_preprocess_data.py` first
- Check cache exists: `models/gpt2_hf/preprocessed/{model_size}_{split_mode}_len8192/`

### W&B Login Issues
- Add `wandb_api_key` to `clif_config.json`
- Or use `--no-wandb` flag to disable

## Performance Tips

1. **Use packing**: Reduces training time by 1.5-2x
2. **Use BF16**: Faster than FP16, more stable than FP32 (requires modern GPU)
3. **Multi-GPU**: Near-linear scaling up to 8 GPUs
4. **Gradient checkpointing**: Enabled by default (30-50% memory savings)
5. **Flash Attention**: Auto-enabled via PyTorch SDPA on modern GPUs

## Comparison with Qwen2 Implementation

| Aspect | GPT2 | Qwen2 |
|--------|------|-------|
| **Architecture** | Standard transformer | Modern transformer (GQA, RoPE) |
| **Context length** | 8192 (extended) | 8192 (native) |
| **Packing** | ✓ Same | ✓ Same |
| **Chunking** | ✓ Same | ✓ Same |
| **Tokenizer** | ✓ Reuses qwen2 | ✓ Custom clinical |
| **Data pipeline** | ✓ Same | ✓ Same |
| **Training speed** | Baseline | ~10-15% faster |
| **Performance** | Good | Better |

**When to use GPT2:**
- Baseline comparisons
- Understanding standard transformers
- Maximum compatibility

**When to use Qwen2:**
- Best performance
- More efficient attention
- Better long-context modeling

## Citation

If you use this implementation, please cite:

```bibtex
@software{gpt2_clinical_narratives,
  title={GPT2 for Clinical Narrative Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/CLIFATRON}
}
```

## License

[Your License]

## Contact

For questions or issues, please open an issue on GitHub.
