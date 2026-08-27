# Task 1: Discharged Home Prediction Benchmark

Benchmark AR models to predict whether ICU patients will be discharged home using **only the first 24 hours of ICU data**.

---

## Table of Contents
- [Overview](#overview)
- [Data Requirements](#data-requirements)
- [Model Requirements](#model-requirements)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Parallel Execution](#parallel-execution)
- [Generated Outputs](#generated-outputs)
- [All Parameters](#all-parameters)
- [Troubleshooting](#troubleshooting)

---

## Overview

**What This Benchmark Does**:
- Evaluates AR models (gpt2, gpt2_hf, qwen2) on binary classification
- **Task**: Predict if a patient will be discharged home (yes/no)
- **Input**: Only tokens from first 24 hours of ICU
- **Output**: Classification metrics (accuracy, F1, AUROC, etc.)

**Two Evaluation Methods**:
1. **Method 1 (XGBoost on Embeddings)**: Extract model embeddings → train XGBoost classifier
2. **Method 2 (Monte Carlo Sliding Window)**: Generate trajectories until disposition_home token appears

---

## Data Requirements

### Where Data Comes From

Your CLIFATRON project must have already generated the following data files:

| File Path | Description | Created By |
|-----------|-------------|------------|
| `OutputTokens/narratives/train_val_sequences.parquet` | Training/validation sequences | `tokenETL/assemble_narratives.py` |
| `OutputTokens/narratives/test_sequences.parquet` | Test sequences | `tokenETL/assemble_narratives.py` |
| `OutputTokens/tokentables/cohort.parquet` | Cohort with ICU timing | `tokenETL/builders/cohort_builder.py` |
| `OutputTokens/token_registry.json` | Vocabulary mapping (1,373 tokens) | `tokenETL/main.py` |

### Required Cohort Column

The `cohort.parquet` file **must** contain the column:
- `first_icu_24hr_completion_time` - Timestamp marking 24 hours after ICU admission

This column is automatically created by the cohort builder if hospitalizations have ICU stays ≥ 24 hours.

### Prerequisites

Before running this benchmark, you must have:
1. ✅ Run `tokenETL/main.py` to generate `OutputTokens/` directory
2. ✅ Run `tokenETL/assemble_narratives.py` to create narrative sequences
3. ✅ Cohort includes hospitalizations with ICU stay ≥ 24 hours

---

## Model Requirements

### What You Need to Provide

You must specify:
1. **Model Type**: `gpt2`, `gpt2_hf`, or `qwen2`
2. **Model Size**: `small`, `medium`, etc.
3. **Checkpoint Path** (optional - uses defaults if not specified)

### Default Model Checkpoint Locations

If you don't specify `--checkpoint-path`, the benchmark looks for models here:

```
models/gpt2_hf/checkpoints/clif-gpt2_hf-small/final_model/
models/gpt2_hf/checkpoints/clif-gpt2_hf-medium/final_model/
models/qwen2/checkpoints/clif-qwen2-small/final_model/
models/gpt2/checkpoints/model_final.pt
```

### Custom Checkpoint Location

If your model is elsewhere, use `--checkpoint-path`:

```bash
--checkpoint-path /path/to/your/model/checkpoint
```

---

## Quick Start

### Complete End-to-End Example

```bash
# Step 1: Build benchmark dataset (run once)
uv run benchmark/task1-discharged-home/build_benchmark.py

# Step 2a: Evaluate with Method 1 (XGBoost on embeddings)
uv run benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --model-size small

# Step 2b: Evaluate with Method 2 (Monte Carlo sampling)
uv run benchmark/task1-discharged-home/method2-montecarlo/run_montecarlo_benchmark.py \
    --model-type gpt2_hf \
    --model-size small \
    --num-samples 100

# Results saved to benchmark/task1-discharged-home/results/
```

**What Gets Generated**:
1. `data/train_val_processed.parquet` - Training data (truncated at 24hr ICU)
2. `data/test_processed.parquet` - Test data (truncated at 24hr ICU)
3. `data/dataset_statistics.json` - Label distribution stats
4. `results/method1_embedding_gpt2_hf_small_results.json` - Method 1 metrics
5. `results/method2_montecarlo_gpt2_hf_small_results.json` - Method 2 metrics

---

## Detailed Usage

### Step 1: Build Benchmark Dataset

**Command**:
```bash
uv run benchmark/task1-discharged-home/build_benchmark.py \
    --input-dir OutputTokens/narratives \
    --cohort-file OutputTokens/tokentables/cohort.parquet \
    --vocab-path OutputTokens/token_registry.json \
    --output-dir benchmark/task1-discharged-home/data
```

**What It Does**:
1. Loads cohort data with ICU timing
2. Filters to hospitalizations with ICU stay ≥ 24 hours
3. Truncates sequences to **only tokens before 24hr ICU mark**
4. Extracts disposition labels (from full sequences)
5. Creates binary labels: 1 = discharged home, 0 = other dispositions

**Generated Files**:
- `data/train_val_processed.parquet` - Truncated training sequences
- `data/test_processed.parquet` - Truncated test sequences
- `data/dataset_statistics.json` - Class distribution statistics

**Example Output**:
```
Dataset Statistics:
Train/Val:
  Total examples:     32,145
  Positive (home):    18,234 (56.7%)
  Negative (other):   13,911 (43.3%)

Test:
  Total examples:     7,141
  Positive (home):    4,023 (56.3%)
  Negative (other):   3,118 (43.7%)
```

---

### Step 2a: Method 1 - XGBoost on Embeddings

**Command (Default Checkpoint)**:
```bash
uv run benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --model-size small \
    --input-dir benchmark/task1-discharged-home/data \
    --output-dir benchmark/task1-discharged-home/results
```

**Command (Custom Checkpoint)**:
```bash
uv run benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --model-size small \
    --checkpoint-path /path/to/your/checkpoint \
    --input-dir benchmark/task1-discharged-home/data \
    --output-dir benchmark/task1-discharged-home/results
```

**What It Does**:
1. Loads AR model (auto-detects GPU, falls back to CPU)
2. Loads truncated test sequences (first 24hr ICU only)
3. Extracts hidden state embeddings for all examples
4. Trains XGBoost classifier on train/val embeddings
5. Tests on test embeddings
6. Computes metrics and saves results

**Generated Files**:
- `results/method1_embedding_<model>_<size>_results.json`

**Example Result**:
```json
{
  "method": "embedding",
  "model_type": "gpt2_hf",
  "model_size": "small",
  "classifier": "xgboost",
  "metrics": {
    "accuracy": 0.847,
    "precision": 0.821,
    "recall": 0.789,
    "f1": 0.805,
    "auroc": 0.883,
    "tp": 3175,
    "tn": 2872,
    "fp": 246,
    "fn": 848
  }
}
```

---

### Step 2b: Method 2 - Monte Carlo Sliding Window

**Command (Default Checkpoint)**:
```bash
uv run benchmark/task1-discharged-home/method2-montecarlo/run_montecarlo_benchmark.py \
    --model-type gpt2_hf \
    --model-size small \
    --num-samples 100 \
    --max-new-tokens 8192 \
    --sliding-window-step 512 \
    --input-dir benchmark/task1-discharged-home/data \
    --output-dir benchmark/task1-discharged-home/results
```

**Command (Custom Checkpoint)**:
```bash
uv run benchmark/task1-discharged-home/method2-montecarlo/run_montecarlo_benchmark.py \
    --model-type qwen2 \
    --model-size small \
    --checkpoint-path /path/to/your/qwen2/checkpoint \
    --num-samples 100 \
    --input-dir benchmark/task1-discharged-home/data \
    --output-dir benchmark/task1-discharged-home/results
```

**What It Does**:
1. Loads AR model (auto-detects GPU, falls back to CPU)
2. For each test example:
   - Starts with truncated input (24hr ICU tokens)
   - Generates 100 trajectories with sliding window
   - Each trajectory generates until `disposition_home` appears OR 8192 tokens reached
   - Aggregates: majority vote across trajectories
3. Computes metrics with uncertainty estimates

**Sliding Window Process**:
```
1. Generate 512 tokens from current position
2. Check if disposition_home token found → STOP if yes
3. If not found: slide window (remove 512 tokens from start)
4. Repeat until disposition_home found OR 8192 total tokens generated
```

**Generated Files**:
- `results/method2_montecarlo_<model>_<size>_results.json`

**Example Result**:
```json
{
  "method": "montecarlo_sliding_window",
  "model_type": "gpt2_hf",
  "model_size": "small",
  "sampling_config": {
    "num_samples": 100,
    "max_new_tokens": 8192,
    "sliding_window_step": 512
  },
  "metrics": {
    "accuracy": 0.831,
    "f1": 0.792,
    "auroc": 0.867
  },
  "uncertainty": {
    "mean_probability": 0.543,
    "std_probability": 0.312
  }
}
```

---

## Parallel Execution

The benchmark supports automatic GPU detection with CPU fallback, single-GPU, multi-GPU (single-node), and multi-node multi-GPU execution. All distributed features are **automatic** - no code changes needed.

### Auto GPU Detection (Recommended)

Scripts automatically detect GPU and fall back to CPU if:
- No GPU available
- GPU out of memory (CUDA OOM)
- Model too large for GPU

```bash
# Automatically uses GPU if available, else CPU
uv run benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf

# You can explicitly set device
uv run benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --device cuda  # or 'cpu' or 'auto' (default)
```

**What Happens**:
1. Script detects available GPUs
2. Tries to load model on GPU
3. If CUDA OOM error occurs → automatically moves to CPU
4. Logs device being used (GPU or CPU)

---

### Multi-GPU (Single Node)

Distribute work across **all GPUs on one machine** using PyTorch's `torchrun`:

#### Method 1: Parallel Embedding Extraction

```bash
# Auto-detect number of GPUs
torchrun --nproc_per_node=auto \
    benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --model-size small

# Or specify exactly 4 GPUs
torchrun --nproc_per_node=4 \
    benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --model-size small
```

**What Happens**:
1. Dataset split across GPUs using `DistributedSampler`
2. Each GPU processes its subset (extracts embeddings in parallel)
3. Embeddings gathered from all ranks to rank 0
4. Rank 0 trains XGBoost classifier
5. Rank 0 saves results

**Speed**: ~4x faster with 4 GPUs (linear scaling for embedding extraction)

#### Method 2: Parallel Monte Carlo Sampling

```bash
# Auto-detect GPUs
torchrun --nproc_per_node=auto \
    benchmark/task1-discharged-home/method2-montecarlo/run_montecarlo_benchmark.py \
    --model-type gpt2_hf \
    --model-size small \
    --num-samples 100

# Specify GPUs and custom parameters
torchrun --nproc_per_node=4 \
    benchmark/task1-discharged-home/method2-montecarlo/run_montecarlo_benchmark.py \
    --model-type qwen2 \
    --num-samples 100 \
    --max-new-tokens 8192 \
    --temperature 1.0
```

**What Happens**:
1. Test examples split across GPUs
2. Each GPU generates 100 trajectories per assigned example (in parallel)
3. Predictions gathered and sorted by original index
4. Rank 0 computes metrics and saves results

**Speed**: ~4x faster with 4 GPUs for large test sets

---

### Multi-Node Multi-GPU

Run across **multiple machines** with multiple GPUs each. Requires network connectivity between nodes.

#### Setup

**Prerequisites**:
- All nodes must have network access to each other
- Same conda/uv environment on all nodes
- Same codebase path on all nodes
- Firewall allows traffic on chosen port (default: 29500)

#### Example: 2 Nodes with 4 GPUs Each (Total: 8 GPUs)

**Node 0** (Master, IP: `10.0.0.1`):
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    benchmark/task1-discharged-home/method2-montecarlo/run_montecarlo_benchmark.py \
    --model-type gpt2_hf \
    --num-samples 100
```

**Node 1** (Worker):
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    benchmark/task1-discharged-home/method2-montecarlo/run_montecarlo_benchmark.py \
    --model-type gpt2_hf \
    --num-samples 100
```

**Parameters**:
- `--nproc_per_node=4`: 4 GPUs per node
- `--nnodes=2`: Total 2 nodes
- `--node_rank=0/1`: Node index (0 for master, 1+ for workers)
- `--master_addr=10.0.0.1`: IP of node 0
- `--master_port=29500`: Port for communication

**Result**: Test set split across 8 GPUs (4 per node), ~8x speedup

#### Example: 3 Nodes with 8 GPUs Each (Total: 24 GPUs)

**Node 0** (IP: 192.168.1.100):
```bash
torchrun --nproc_per_node=8 --nnodes=3 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=29500 \
    benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf
```

**Node 1** (IP: 192.168.1.101):
```bash
torchrun --nproc_per_node=8 --nnodes=3 --node_rank=1 \
    --master_addr=192.168.1.100 --master_port=29500 \
    benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf
```

**Node 2** (IP: 192.168.1.102):
```bash
torchrun --nproc_per_node=8 --nnodes=3 --node_rank=2 \
    --master_addr=192.168.1.100 --master_port=29500 \
    benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf
```

---

### Distributed Execution Summary

| Configuration | Command | Use Case |
|--------------|---------|----------|
| **Single GPU** | `uv run script.py` | Default, automatic detection |
| **Single CPU** | `uv run script.py --device cpu` | No GPU available |
| **Multi-GPU (1 node)** | `torchrun --nproc_per_node=auto script.py` | Fastest for most users |
| **Multi-node** | `torchrun --nnodes=N --node_rank=R ...` | Very large models/datasets |

**Performance**:
- **Method 1** (embedding): Linear scaling with GPUs (4 GPUs = ~4x faster)
- **Method 2** (Monte Carlo): Near-linear scaling for large test sets

---

## Generated Outputs

### Directory Structure

```
benchmark/task1-discharged-home/
├── data/
│   ├── train_val_processed.parquet       # Truncated training sequences (24hr ICU)
│   ├── test_processed.parquet            # Truncated test sequences (24hr ICU)
│   └── dataset_statistics.json           # Class distribution stats
│
└── results/
    ├── method1_embedding_gpt2_hf_small_results.json
    ├── method1_embedding_gpt2_hf_medium_results.json
    ├── method1_embedding_qwen2_small_results.json
    ├── method2_montecarlo_gpt2_hf_small_results.json
    ├── method2_montecarlo_gpt2_hf_medium_results.json
    └── method2_montecarlo_qwen2_small_results.json
```

### Result JSON Structure

**Method 1 (Embedding)**:
```json
{
  "method": "embedding",
  "model_type": "gpt2_hf",
  "model_size": "small",
  "classifier": "xgboost",
  "layer": "last",
  "metrics": {
    "accuracy": 0.847,
    "precision": 0.821,
    "recall": 0.789,
    "f1": 0.805,
    "auroc": 0.883,
    "auprc": 0.856,
    "specificity": 0.921,
    "npv": 0.772,
    "tp": 3175,
    "tn": 2872,
    "fp": 246,
    "fn": 848,
    "total": 7141,
    "positive_examples": 4023,
    "negative_examples": 3118
  },
  "metadata": {
    "timestamp": "2025-10-31T14:23:45",
    "checkpoint_path": "models/gpt2_hf/checkpoints/clif-gpt2_hf-small/final_model",
    "num_train_examples": 32145,
    "num_test_examples": 7141,
    "embedding_dim": 768
  }
}
```

**Method 2 (Monte Carlo)**:
```json
{
  "method": "montecarlo_sliding_window",
  "model_type": "gpt2_hf",
  "model_size": "small",
  "sampling_config": {
    "num_samples": 100,
    "max_new_tokens": 8192,
    "sliding_window_step": 512,
    "temperature": 1.0,
    "top_k": null,
    "top_p": null,
    "aggregation_method": "majority_vote"
  },
  "metrics": {
    "accuracy": 0.831,
    "precision": 0.798,
    "recall": 0.765,
    "f1": 0.792,
    "auroc": 0.867
  },
  "uncertainty": {
    "mean_probability": 0.543,
    "std_probability": 0.312,
    "min_probability": 0.01,
    "max_probability": 0.99,
    "median_probability": 0.51
  },
  "metadata": {
    "timestamp": "2025-10-31T16:45:12",
    "num_test_examples": 7141
  }
}
```

---

## All Parameters

### build_benchmark.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | `OutputTokens/narratives` | Directory with narrative parquet files |
| `--output-dir` | `benchmark/task1-discharged-home/data` | Where to save processed data |
| `--cohort-file` | `OutputTokens/tokentables/cohort.parquet` | Path to cohort.parquet |
| `--vocab-path` | `OutputTokens/token_registry.json` | Path to vocabulary JSON |
| `--config` | `config.yaml` | Path to config file |
| `--min-length` | `10` | Minimum sequence length |
| `--max-length` | `8192` | Maximum sequence length |

### method1-embedding/run_embedding_benchmark.py

| Parameter | Default | Required | Description |
|-----------|---------|----------|-------------|
| `--model-type` | - | ✅ | Model type: `gpt2`, `gpt2_hf`, or `qwen2` |
| `--model-size` | `small` | | Model size: `small`, `medium`, etc. |
| `--checkpoint-path` | (default locations) | | Custom checkpoint path |
| `--input-dir` | `benchmark/.../data` | | Processed benchmark data |
| `--output-dir` | `benchmark/.../results` | | Where to save results |
| `--classifier` | `xgboost` | | Classifier: `xgboost` or `logistic_regression` |
| `--batch-size` | `16` | | Batch size for embedding extraction |
| `--layer` | `last` | | Embedding layer: `last` or `mean` |
| `--device` | `auto` | | Device: `auto`, `cuda`, or `cpu` |
| `--config` | `config.yaml` | | Path to config file |

### method2-montecarlo/run_montecarlo_benchmark.py

| Parameter | Default | Required | Description |
|-----------|---------|----------|-------------|
| `--model-type` | - | ✅ | Model type: `gpt2`, `gpt2_hf`, or `qwen2` |
| `--model-size` | `small` | | Model size: `small`, `medium`, etc. |
| `--checkpoint-path` | (default locations) | | Custom checkpoint path |
| `--input-dir` | `benchmark/.../data` | | Processed benchmark data |
| `--output-dir` | `benchmark/.../results` | | Where to save results |
| `--num-samples` | `100` | | Number of trajectory samples per example |
| `--max-new-tokens` | `8192` | | Max tokens to generate per trajectory |
| `--sliding-window-step` | `512` | | Tokens to remove when sliding window |
| `--temperature` | `1.0` | | Sampling temperature |
| `--top-k` | `null` | | Top-k sampling parameter |
| `--top-p` | `null` | | Nucleus sampling parameter |
| `--aggregation` | `majority_vote` | | Aggregation: `majority_vote` or `probability_threshold` |
| `--threshold` | `0.5` | | Probability threshold (for probability_threshold mode) |
| `--device` | `auto` | | Device: `auto`, `cuda`, or `cpu` |
| `--config` | `config.yaml` | | Path to config file |

---

## Troubleshooting

### Data Issues

**Error**: `Cohort file not found`
```
Solution: Ensure OutputTokens/tokentables/cohort.parquet exists
Check: ls OutputTokens/tokentables/cohort.parquet
```

**Error**: `Missing column: first_icu_24hr_completion_time`
```
Solution: Regenerate cohort with updated cohort_builder.py
This column is automatically created for ICU stays ≥ 24 hours
```

**Error**: `No sequences after truncation`
```
Solution: Check cohort has hospitalizations with ICU stay ≥ 24 hours
Query: How many rows have first_icu_24hr_completion_time not null?
```

### Model Issues

**Error**: `Checkpoint not found`
```
Solution 1: Specify custom path with --checkpoint-path
Solution 2: Check default location: models/<model_type>/checkpoints/
```

**Error**: `CUDA out of memory`
```
Solution: Script automatically falls back to CPU
Manual: Use --device cpu to force CPU
Alternative: Reduce --batch-size (Method 1) or --num-samples (Method 2)
```

**Error**: `Model type not recognized`
```
Solution: Use --model-type with: gpt2, gpt2_hf, or qwen2
Example: --model-type gpt2_hf
```

### Performance Issues

**Method 2 is too slow**
```
Solution 1: Reduce --num-samples (e.g., from 100 to 50)
Solution 2: Increase --sliding-window-step (e.g., from 512 to 1024)
Solution 3: Use multi-GPU: torchrun --nproc_per_node=auto
```

**Out of memory during embedding extraction**
```
Solution: Reduce --batch-size
Example: --batch-size 8 (instead of default 16)
```

### Distributed Issues

**Error**: `torch.distributed not initialized`
```
Solution: Use torchrun for multi-GPU execution
Example: uv run torchrun --nproc_per_node=auto script.py
```

**Multi-node not working**
```
Check: All nodes can reach master_addr:master_port
Check: Same NCCL version on all nodes
Check: Same code on all nodes
```

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{clifatron_benchmark2025,
  title={CLIFATRON Benchmark: Discharged Home Prediction},
  author={Your Team},
  year={2025},
  url={https://github.com/your-org/CLIFATRON}
}
```

---

## Support

For issues or questions:
1. Check this README first
2. Check `config.yaml` for configuration options
3. Review error messages carefully
4. Open an issue on GitHub with:
   - Full command you ran
   - Complete error message
   - Output of: `uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
