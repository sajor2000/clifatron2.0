# CLIFATRON Benchmark Quickstart Guide

This guide provides all commands to run embedding generation, data processing, and Method 1 (XGBoost on embeddings) benchmarks.

---

## Prerequisites

- Trained AR model checkpoints (GPT2, GPT2-HF, or Qwen2)
- CLIF data access and tokenized narratives
- Python environment with required dependencies

---

## Step 1: Data Processing Pipeline

### 1.1 Tokenize Clinical Data (if not already done)

```bash
# Tokenize raw CLIF data into token tables
uv run tokenETL/main.py

# Assemble tokenized data into chronological narratives
uv run tokenETL/assemble_narratives.py
```

**Output:**
- `OutputTokens/tokentables/` - Individual tokenized tables
- `OutputTokens/narratives/train_val_sequences.parquet` - Training narratives
- `OutputTokens/narratives/test_sequences.parquet` - Test narratives

### 1.2 Build Benchmark Datasets

```bash
# Create multi-task benchmark datasets with 24-hour truncation
uv run benchmark/build_benchmark.py \
    --input-dir OutputTokens/narratives \
    --output-dir benchmark/data \
    --cohort-file OutputTokens/tokentables/cohort.parquet
```

**Output:**
- `benchmark/data/task1_task2_disposition_train_val.parquet`
- `benchmark/data/task1_task2_disposition_test.parquet`
- `benchmark/data/task3_task4_respiratory_train_val.parquet`
- `benchmark/data/task3_task4_respiratory_test.parquet`

---

## Step 2: Generate Embeddings

Generate embeddings from trained AR models for all tasks.

### 2.1 Generate Embeddings for Tasks 1 & 2 (Disposition)

```bash
# GPT2-HF model
uv run benchmark/generate_embeddings.py \
    --model-type gpt2_hf \
    --checkpoint models/gpt2_hf/model_weights \
    --task task1_task2 \
    --batch-size 32 \
    --layer mean \
    --device cuda

# Qwen2 model
uv run benchmark/generate_embeddings.py \
    --model-type qwen2 \
    --checkpoint models/qwen2/model_weights \
    --task task1_task2 \
    --batch-size 32 \
    --layer mean \
    --device cuda
```

### 2.2 Generate Embeddings for Tasks 3 & 4 (Respiratory)

```bash
# GPT2-HF model
uv run benchmark/generate_embeddings.py \
    --model-type gpt2_hf \
    --checkpoint models/gpt2_hf/model_weights \
    --task task3_task4 \
    --batch-size 32 \
    --layer mean \
    --device cuda

# Qwen2 model
uv run benchmark/generate_embeddings.py \
    --model-type qwen2 \
    --checkpoint models/qwen2/model_weights \
    --task task3_task4 \
    --batch-size 32 \
    --layer mean \
    --device cuda
```

**Output:**
- `benchmark/embeddings/task1_task2_disposition/embeddings_{model}_mean_train_val.npz`
- `benchmark/embeddings/task1_task2_disposition/embeddings_{model}_mean_test.npz`
- `benchmark/embeddings/task3_task4_respiratory/embeddings_{model}_mean_train_val.npz`
- `benchmark/embeddings/task3_task4_respiratory/embeddings_{model}_mean_test.npz`

### 2.3 Multi-GPU Embedding Generation (Optional)

For faster embedding generation with multiple GPUs:

```bash
# Use torchrun for distributed processing
torchrun --nproc_per_node=2 benchmark/generate_embeddings.py \
    --model-type qwen2 \
    --checkpoint models/qwen2/model_weights \
    --task task1_task2 \
    --batch-size 32 \
    --layer mean \
    --device cuda
```

---

## Step 3: Run Method 1 Benchmarks (XGBoost on Embeddings)

Run all Method 1 benchmarks for each task. These use the pre-generated embeddings.

### 3.1 Task 1: Discharged Home Prediction

```bash
# GPT2-HF model
uv run benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data

# Qwen2 model
uv run benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type qwen2 \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data
```

**Output:** `benchmark/results/task1-discharged-home/{model}/method1-embedding/summary_metrics.json`

### 3.2 Task 2: Discharged to LTACH Prediction

```bash
# GPT2-HF model
uv run benchmark/task2-discharged-ltach/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data

# Qwen2 model
uv run benchmark/task2-discharged-ltach/method1-embedding/run_embedding_benchmark.py \
    --model-type qwen2 \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data
```

**Output:** `benchmark/results/task2-discharged-ltach/{model}/method1-embedding/summary_metrics.json`

### 3.3 Task 3: Outcome (72-hour) Prediction

```bash
# GPT2-HF model
uv run benchmark/task3-outcome-72hr/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data

# Qwen2 model
uv run benchmark/task3-outcome-72hr/method1-embedding/run_embedding_benchmark.py \
    --model-type qwen2 \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data
```

**Output:** `benchmark/results/task3-outcome-72hr/{model}/method1-embedding/summary_metrics.json`

### 3.4 Task 4: Hypoxic Proportion Prediction

```bash
# GPT2-HF model
uv run benchmark/task4-hypoxic-proportion/method1-embedding/run_embedding_benchmark.py \
    --model-type gpt2_hf \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data

# Qwen2 model
uv run benchmark/task4-hypoxic-proportion/method1-embedding/run_embedding_benchmark.py \
    --model-type qwen2 \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data
```

**Output:** `benchmark/results/task4-hypoxic-proportion/{model}/method1-embedding/summary_metrics.json`

---

## Step 4: Run All Tasks at Once (Batch Script)

Create a simple batch script to run all Method 1 benchmarks:

```bash
#!/bin/bash
# run_all_method1.sh

MODEL_TYPE="qwen2"  # Change to "gpt2_hf" for GPT2-HF
CHECKPOINT="models/qwen2/model_weights"  # Update path as needed

echo "Running all Method 1 benchmarks for $MODEL_TYPE..."

# Task 1
echo "Task 1: Discharged Home"
uv run benchmark/task1-discharged-home/method1-embedding/run_embedding_benchmark.py \
    --model-type $MODEL_TYPE \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data

# Task 2
echo "Task 2: Discharged to LTACH"
uv run benchmark/task2-discharged-ltach/method1-embedding/run_embedding_benchmark.py \
    --model-type $MODEL_TYPE \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data

# Task 3
echo "Task 3: Outcome (72-hour)"
uv run benchmark/task3-outcome-72hr/method1-embedding/run_embedding_benchmark.py \
    --model-type $MODEL_TYPE \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data

# Task 4
echo "Task 4: Hypoxic Proportion"
uv run benchmark/task4-hypoxic-proportion/method1-embedding/run_embedding_benchmark.py \
    --model-type $MODEL_TYPE \
    --batch-size 32 \
    --layer mean \
    --input-dir benchmark/data

echo "All Method 1 benchmarks complete!"
```

Run it:
```bash
chmod +x run_all_method1.sh
./run_all_method1.sh
```

---

## Common Options

### Embedding Generation Options

- `--model-type`: Model architecture (`gpt2`, `gpt2_hf`, `qwen2`)
- `--checkpoint`: Path to model checkpoint directory
- `--task`: Task type (`task1_task2` or `task3_task4`)
- `--batch-size`: Batch size for embedding extraction (default: 32)
- `--layer`: Which layer to extract (`last` or `mean`)
- `--device`: Device to use (`cuda` or `cpu`)

### Method 1 Benchmark Options

- `--model-type`: Model architecture
- `--batch-size`: Batch size for loading embeddings
- `--layer`: Which layer embeddings to use (`mean` or `last`)
- `--input-dir`: Directory containing benchmark data

Note: Output directory is automatically determined as `benchmark/results/{task}/{model}/method1-embedding/`

---

## Expected Results

Each Method 1 benchmark produces a JSON file `summary_metrics.json` with:
- **Scalar Metrics**: Accuracy, F1, AUROC, AUPRC, Precision, Recall, Specificity, NPV
- **ROC/PR Curves**: Full ROC curve (fpr, tpr, thresholds) and PR curve (precision, recall, thresholds) for Tasks 1 & 2
- **Confusion Matrix**: TP, TN, FP, FN
- **Model Info**: Model type, embedding layer, classifier hyperparameters
- **Metadata**: Timestamp, checkpoint info

Example output (Task 1):
```json
{
  "method": "embedding",
  "model_type": "qwen2",
  "metrics": {
    "accuracy": 0.7222,
    "f1": 0.7739,
    "auroc": 0.7848,
    "auprc": 0.8445,
    "precision": 0.7598,
    "recall": 0.7887,
    "roc_curve": {
      "fpr": [0.0, 0.01, ...],
      "tpr": [0.0, 0.15, ...],
      "thresholds": [1.0, 0.95, ...]
    },
    "pr_curve": {
      "precision": [0.8, 0.82, ...],
      "recall": [1.0, 0.95, ...],
      "thresholds": [0.1, 0.2, ...]
    }
  }
}
```

---

## Troubleshooting

### Issue: Embeddings not found

**Error:** `FileNotFoundError: Embedding file not found`

**Solution:** Run embedding generation (Step 2) before Method 1 benchmarks (Step 3)

### Issue: Out of memory during embedding generation

**Solution:** Reduce batch size:
```bash
uv run benchmark/generate_embeddings.py \
    --model-type qwen2 \
    --batch-size 16 \
    --device cuda
```

### Issue: CUDA out of memory

**Solution:** Use CPU instead:
```bash
uv run benchmark/generate_embeddings.py \
    --model-type qwen2 \
    --device cpu
```

---

## Directory Structure

After running all steps:

```
CLIFATRON/
├── benchmark/
│   ├── data/                                # Benchmark datasets (Step 1)
│   │   ├── task1_task2_disposition_train_val.parquet
│   │   ├── task1_task2_disposition_test.parquet
│   │   ├── task3_task4_respiratory_train_val.parquet
│   │   └── task3_task4_respiratory_test.parquet
│   ├── embeddings/                          # Cached embeddings (Step 2)
│   │   ├── task1_task2_disposition/
│   │   │   ├── embeddings_qwen2_mean_train_val.npz
│   │   │   ├── embeddings_qwen2_mean_test.npz
│   │   │   ├── embeddings_gpt2_hf_mean_train_val.npz
│   │   │   └── embeddings_gpt2_hf_mean_test.npz
│   │   └── task3_task4_respiratory/
│   │       ├── embeddings_qwen2_mean_train_val.npz
│   │       └── embeddings_qwen2_mean_test.npz
│   └── results/                             # Benchmark results (Step 3)
│       ├── task1-discharged-home/
│       │   ├── qwen2/method1-embedding/
│       │   │   └── summary_metrics.json
│       │   └── gpt2_hf/method1-embedding/
│       │       └── summary_metrics.json
│       ├── task2-discharged-ltach/
│       │   ├── qwen2/method1-embedding/
│       │   │   └── summary_metrics.json
│       │   └── gpt2_hf/method1-embedding/
│       │       └── summary_metrics.json
│       ├── task3-outcome-72hr/
│       │   └── {model}/method1-embedding/
│       │       └── summary_metrics.json
│       └── task4-hypoxic-proportion/
│           └── {model}/method1-embedding/
│               └── summary_metrics.json
└── OutputTokens/
    ├── tokentables/                         # Tokenized data
    └── narratives/                          # Assembled narratives
```

---

## Quick Reference

**Full pipeline (all steps):**
```bash
# 1. Build benchmark data
uv run benchmark/build_benchmark.py \
    --input-dir OutputTokens/narratives \
    --output-dir benchmark/data \
    --cohort-file OutputTokens/tokentables/cohort.parquet

# 2. Generate embeddings (Tasks 1&2)
uv run benchmark/generate_embeddings.py \
    --model-type qwen2 \
    --checkpoint models/qwen2/model_weights \
    --task task1_task2

# 3. Generate embeddings (Tasks 3&4)
uv run benchmark/generate_embeddings.py \
    --model-type qwen2 \
    --checkpoint models/qwen2/model_weights \
    --task task3_task4

# 4. Run all Method 1 benchmarks
./run_all_method1.sh
```

---

## Next Steps

After running Method 1:
- **Visualize results**: `uv run benchmark/visualize_all_tasks.py`
- **Run Method 2**: See Method 2 benchmarks (Monte Carlo simulation)
- **Compare models**: Compare GPT2-HF vs Qwen2 performance

For questions or issues, check the main documentation or contact the development team.
