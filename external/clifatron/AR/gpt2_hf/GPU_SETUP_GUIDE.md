# GPT2 Setup Guide

Complete setup instructions for training GPT2 models on clinical narratives.

## Prerequisites

### Required Software
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- DeepSpeed 0.12+
- CUDA 11.8+

### Hardware
- **Minimum**: 2x 16GB GPUs for small, 2x 24GB for medium
- **Recommended**: 2x L40 48GB or 2x A100 40GB

## Installation

### 1. Clone Repository
```bash
cd /path/to/CLIFATRON
```

### 2. Install Dependencies
```bash
# Install with uv (recommended)
uv pip install torch transformers deepspeed accelerate wandb pyyaml polars
```

## Data Preparation

### 1. Generate Narratives
First, generate clinical narratives using the tokenETL pipeline:

```bash
uv run tokenETL/assemble_narratives.py
```

This creates:
- `OutputTokens/narratives/train_val_sequences.parquet` (2018-2023)
- `OutputTokens/narratives/test_sequences.parquet` (2024)

### 2. Build Tokenizer
The tokenizer is shared with the qwen2 implementation. If not already built:

```bash
uv run AR/qwen2/tokenizer/build_tokenizer.py
```

This creates the clinical tokenizer in `AR/qwen2/tokenizer/clinical_tokenizer/`.

## Preprocessing

Run preprocessing once to create cached datasets:

**Note:** All commands require `--mode` argument for vocabulary lock system. Use `primary` for first/single site, `secondary` for additional sites.

```bash
# For GPT2-small with temporal splits (PRIMARY MODE)
uv run AR/gpt2_hf/01_preprocess_data.py \
    --model-size small \
    --split-mode temporal \
    --mode primary

# For GPT2-medium with temporal splits (PRIMARY MODE)
uv run AR/gpt2_hf/01_preprocess_data.py \
    --model-size medium \
    --split-mode temporal \
    --mode primary
```

**Time**: 15-30 minutes
**Output**: `models/gpt2_hf/preprocessed/{model_size}_temporal_len8192/`

### Multi-Site Training (Secondary Mode)

If training at a **secondary site** (not the first site):

```bash
# 1. Copy tokenizer from primary site first
rsync -avz primary_site:/path/AR/qwen2/tokenizer/clinical_tokenizer/ \
            AR/qwen2/tokenizer/clinical_tokenizer/

# 2. Preprocess with SECONDARY mode (cannot build tokenizer)
uv run AR/gpt2_hf/01_preprocess_data.py \
    --model-size small \
    --split-mode temporal \
    --mode secondary
```

**IMPORTANT:** Secondary mode validates vocabulary hash. If mismatch occurs, copy tokenizer from primary site.

## Training

### Single GPU Training

```bash
python AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --mode primary
```

### Multi-GPU Training (Recommended)

```bash
# Auto-detect number of GPUs (PRIMARY MODE)
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --mode primary

# Specify 2 GPUs explicitly (PRIMARY MODE)
uv run torchrun --nproc_per_node=2 AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --mode primary

# SECONDARY SITE (uses locked vocabulary)
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --mode secondary
```

### Monitoring

Training progress is logged to Weights & Biases (W&B). To enable:

1. Add your W&B API key to `clif_config.json`:
```json
{
  "wandb_api_key": "your_key_here",
  "output_dir": "OutputTokens"
}
```

2. Or login via command line:
```bash
wandb login
```

3. Or disable W&B:
```bash
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --no-wandb
```

## Evaluation

After training, evaluate on the test set:

```bash
uv run AR/gpt2_hf/03_evaluate_test.py \
    --checkpoint models/gpt2_hf/checkpoints/clif-gpt2-small/final_model \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --output test_results.json
```

Results include:
- Loss and perplexity
- Token accuracy (top-1)
- Top-5 accuracy
- Category-wise metrics (vitals, labs, medications, etc.)

## GPU Configuration

### Automatic Detection

The system automatically detects:
- Number of GPUs
- GPU model (L40, A100, V100, etc.)
- VRAM per GPU
- BF16 support
- Optimal DeepSpeed stage

### Manual Configuration

Edit `config/gpu_profiles.yaml` to customize GPU settings.

### DeepSpeed Stages

**ZeRO-2** (default for multi-GPU):
- Shards optimizer states
- Replicates model parameters
- Good balance of speed and memory

**ZeRO-3** (default for single GPU):
- Shards both optimizer states and model parameters
- Maximum memory efficiency
- Slightly slower communication

## Troubleshooting

### OOM Errors

1. **Reduce batch size**:
   Edit `config/training_config.yaml`:
   ```yaml
   models:
     small:
       batch_size: 4  # Reduce from 8
       gradient_accumulation_steps: 24  # Increase to maintain effective batch size
   ```

2. **Use ZeRO-3**:
   ```bash
   uv run AR/gpt2_hf/02_train_gpt2.py \
       --model-size small \
       --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
       --deepspeed-config AR/gpt2_hf/config/ds_config_zero3.json
   ```

3. **Enable gradient checkpointing** (already enabled by default)

### Slow Training

1. **Verify packing is enabled**:
   Check `config/training_config.yaml`:
   ```yaml
   enable_packing: true
   ```

2. **Use BF16** (auto-enabled on modern GPUs)

3. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon -i 0,1
   ```

### Data Loading Issues

1. **Verify cache exists**:
   ```bash
   ls models/gpt2_hf/preprocessed/small_temporal_len8192/
   ```
   Should contain: `train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`, `metadata.json`

2. **Rebuild cache if corrupted**:
   ```bash
   rm -rf models/gpt2_hf/preprocessed/small_temporal_len8192/
   uv run AR/gpt2_hf/01_preprocess_data.py --model-size small --split-mode temporal
   ```

## Advanced Configuration

### Custom Hyperparameters

Create a custom config file:

```yaml
# my_config.yaml
models:
  small:
    num_epochs: 5  # Train longer
    batch_size: 10  # Larger batch
    learning_rate: 2.0e-4  # Lower LR
    warmup_steps: 3000  # More warmup
```

Use it:
```bash
uv run torchrun --nproc_per_node=auto AR/gpt2_hf/02_train_gpt2.py \
    --model-size small \
    --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \
    --train-config my_config.yaml
```

### Disable Packing

Edit `config/training_config.yaml`:
```yaml
enable_packing: false
```

**Note**: This will reduce throughput by ~1.5-2x due to increased padding waste.

### Change Context Length

To use a different context length (e.g., 4096):

1. Rerun preprocessing:
   ```bash
   uv run AR/gpt2_hf/01_preprocess_data.py \
       --model-size small \
       --split-mode temporal \
       --max-length 4096
   ```

2. Update `config/training_config.yaml`:
   ```yaml
   max_length: 4096
   pack_to_max_length: 4096
   ```

3. Train as usual

## Performance Benchmarks

### GPT2-small (124M) on 2x L40 48GB

| Configuration | Throughput | Time/Epoch | Total Time |
|---------------|------------|------------|------------|
| Packing OFF, BF16 | 150 samples/sec | 12 hours | ~2 days |
| Packing ON, BF16 | 250 samples/sec | 7 hours | ~1.2 days |

### GPT2-medium (355M) on 2x L40 48GB

| Configuration | Throughput | Time/Epoch | Total Time |
|---------------|------------|------------|------------|
| Packing OFF, BF16 | 90 samples/sec | 20 hours | ~2.5 days |
| Packing ON, BF16 | 150 samples/sec | 12 hours | ~1.5 days |

## Next Steps

1. **Train your model**: Follow the training instructions above
2. **Monitor progress**: Check W&B dashboard for metrics
3. **Evaluate**: Run evaluation on test set
4. **Generate narratives**: Use trained model for inference
5. **Compare with Qwen2**: Train both models to compare performance

## Support

For issues or questions:
- Check existing GitHub issues
- Create a new issue with:
  - Error messages
  - GPU configuration
  - Command used
  - Relevant logs
