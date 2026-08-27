# 2x L40 GPU Training Guide

Quick reference for training on 2x NVIDIA L40 GPUs (48GB VRAM each).

## TL;DR - Recommended Commands

### Small Model (Fast Iteration)
```bash
uv run gpt2/04_train_gpt2.py \
  --model-size small \
  --epochs 10 \
  --batch-size 16 \
  --context-size 2048
```
- **Effective batch**: 16 × 2 GPUs × 2 grad_accum = **64**
- **Training time**: ~6-8 hours
- **Memory per GPU**: ~15GB

### Medium Model (Balanced)
```bash
uv run gpt2/04_train_gpt2.py \
  --model-size medium \
  --epochs 10 \
  --batch-size 12 \
  --context-size 2048 \
  --gradient-accumulation-steps 4
```
- **Effective batch**: 12 × 2 GPUs × 4 grad_accum = **96**
- **Training time**: ~15-20 hours
- **Memory per GPU**: ~30GB

### Large Model (Maximum Quality)
```bash
uv run gpt2/04_train_gpt2.py \
  --model-size large \
  --epochs 8 \
  --batch-size 8 \
  --context-size 2048 \
  --gradient-accumulation-steps 8
```
- **Effective batch**: 8 × 2 GPUs × 8 grad_accum = **128**
- **Training time**: ~30-40 hours
- **Memory per GPU**: ~42GB

### XL Model (Experimental - Tight Memory)
```bash
uv run gpt2/04_train_gpt2.py \
  --model-size xl \
  --epochs 5 \
  --batch-size 4 \
  --context-size 1024 \
  --gradient-accumulation-steps 16
```
- **Effective batch**: 4 × 2 GPUs × 16 grad_accum = **128**
- **Training time**: ~60+ hours
- **Memory per GPU**: ~46GB (very tight!)

## GPU Monitoring

### Real-time Monitoring
```bash
# Simple
watch -n 1 nvidia-smi

# Detailed
nvidia-smi dmon -s pucvmet -d 1
```

### Check GPU Usage
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

## Performance Optimization Tips

### 1. Maximize Batch Size
Start with recommended batch size, then increase until you hit OOM:
```bash
# Try incrementing batch size
uv run gpt2/04_train_gpt2.py --model-size medium --batch-size 16  # If OOM, reduce
```

### 2. Use Gradient Accumulation
If batch size is limited by memory, increase gradient accumulation:
```bash
# Same effective batch (16×2=32 vs 8×4=32)
--batch-size 16 --gradient-accumulation-steps 2  # Uses more memory
--batch-size 8 --gradient-accumulation-steps 4   # Uses less memory
```

### 3. Flash Attention Benefits
With L40 (Ampere architecture):
- ✅ Flash Attention: **Automatically enabled**
- ✅ Bfloat16: **Automatically enabled**
- ⚡ **2-4x faster** attention vs standard implementation

### 4. Optimal Context Size
L40 has enough memory for larger contexts:
```bash
# Standard
--context-size 1024  # ~20GB

# Recommended for L40
--context-size 2048  # ~30GB (2x tokens per batch!)

# Experimental
--context-size 4096  # ~45GB (tight, reduce batch size)
```

## Troubleshooting

### OOM (Out of Memory)
```bash
# Solution 1: Reduce batch size
--batch-size 4  # Instead of 8

# Solution 2: Increase gradient accumulation (keeps effective batch size)
--batch-size 4 --gradient-accumulation-steps 8

# Solution 3: Reduce context size
--context-size 1024  # Instead of 2048

# Solution 4: Use mixed precision (auto-enabled on L40)
# No action needed - already using bfloat16
```

### NCCL Errors (Multi-GPU Communication)
```bash
# Disable P2P
NCCL_P2P_DISABLE=1 uv run gpt2/04_train_gpt2.py ...

# Enable debug logging
NCCL_DEBUG=INFO uv run gpt2/04_train_gpt2.py ...
```

### GPU Not Detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU count
python -c "import torch; print(torch.cuda.device_count())"

# Verify visible devices
echo $CUDA_VISIBLE_DEVICES
```

## Expected Performance Metrics

### L40 Specs
- **VRAM**: 48GB
- **Architecture**: Ada Lovelace (Ampere-class)
- **TFLOPs (FP32)**: 90.5
- **TFLOPs (BF16)**: 181
- **Memory Bandwidth**: 864 GB/s
- **NVLINK**: No (PCIe only)

### Training Speed (2x L40)
| Model Size | Batch | Context | Speed (tokens/sec) | GPU Util |
|------------|-------|---------|-------------------|----------|
| Small      | 16    | 1024    | ~250K             | 85-95%   |
| Small      | 16    | 2048    | ~200K             | 90-98%   |
| Medium     | 12    | 2048    | ~120K             | 90-98%   |
| Large      | 8     | 2048    | ~60K              | 92-99%   |

*Note: Actual speeds vary based on data pipeline and GPU-GPU communication overhead*

## Advanced: Manual torchrun

For explicit DDP control:

```bash
# 2 GPUs
torchrun --nproc_per_node=2 gpt2/04_train_gpt2.py \
  --model-size medium \
  --batch-size 12

# With specific GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 gpt2/04_train_gpt2.py \
  --model-size large \
  --batch-size 8
```

## Logging & Monitoring

### Weights & Biases
```bash
uv run gpt2/04_train_gpt2.py \
  --model-size medium \
  --epochs 10 \
  --wandb \
  --run-name "2xL40-medium-2048ctx"
```

### View Logs
```bash
# Training logs
tail -f gpt2_output/models/clif-gpt2-*/runs/*/logs/events.out.tfevents.*

# Or use TensorBoard
tensorboard --logdir gpt2_output/models/
```

## Memory Budget Breakdown (Medium Model, 2048 context)

| Component           | Memory (GB) |
|---------------------|-------------|
| Model parameters    | 1.4         |
| Optimizer states    | 2.8         |
| Gradients          | 1.4         |
| Activations (batch=12) | 20-25    |
| **Total per GPU**   | **~30GB**   |

## Checklist Before Long Training Run

- [ ] Data prepared with correct context size
- [ ] Vocabulary built
- [ ] Splits created
- [ ] Test batch forward pass
- [ ] Check GPU memory usage
- [ ] Setup monitoring (nvidia-smi/wandb)
- [ ] Configure checkpointing
- [ ] Test resume from checkpoint
- [ ] Estimate total training time

## Contact / Issues

If you run into issues specific to L40s:
1. Check CUDA compatibility (should be CUDA 12.0+)
2. Verify PyTorch sees both GPUs
3. Monitor GPU temperature (should be < 80°C)
4. Check PCIe bandwidth (should be Gen4 x16)
