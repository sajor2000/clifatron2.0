# CLIFATRON Tokenization and Training Pipeline (Custom GPT2)

This document provides a comprehensive overview of CLIFATRON's custom GPT2 implementation - a clean, memory-efficient alternative to the HuggingFace-based pipeline.

---

## ⚠️ IMPORTANT: Custom Implementation

**This is a custom GPT2 implementation based on nanoGPT**, not HuggingFace Transformers.

**Key Differences from AR/gpt2_hf/**:
- ✅ **nanoGPT Architecture** - Clean, understandable PyTorch implementation
- ✅ **Polars Streaming** - Memory-efficient data processing for huge datasets
- ✅ **Registry-Based Vocabulary** - Includes ALL tokens from token_registry.json
- ✅ **Simple Sequential Chunking** - 8,190-token boundaries, zero overlap
- ❌ **No HuggingFace Model Hub** - Not directly compatible with HuggingFace ecosystem

**When to use this implementation:**
- You need maximum control and code clarity
- Memory efficiency is critical (huge datasets)
- You want to modify the architecture
- You don't need HuggingFace model hub integration

**When to use AR/gpt2_hf/ instead:**
- You need HuggingFace compatibility
- You want to share models on HuggingFace Hub
- You need day-boundary aware chunking

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Special Tokens](#2-special-tokens)
3. [Context Window Handling](#3-context-window-handling)
4. [Model Architecture](#4-model-architecture)
5. [Training Configuration](#5-training-configuration)
6. [Dataset Implementation](#6-dataset-implementation)
7. [Vocabulary System](#7-vocabulary-system)
8. [Code Examples](#8-code-examples)
9. [Summary Statistics](#9-summary-statistics)

---

## 1. Pipeline Overview

The custom GPT2 pipeline consists of 6 stages:

### Stage 0: Convert Narratives to Sentences

**Script**: `00_convert_narrative_to_sentences.py`

**Purpose**: Transform tokenETL output (one token per row) into GPT2 training format (one hospitalization per row with space-separated tokens).

**Input**:
- `OutputTokens/narratives/train_val_sequences.parquet` (2018-2023)
- `OutputTokens/narratives/test_sequences.parquet` (2024)

**Process**:
```python
# Group tokens by hospitalization
# Sort by event_time and sequence_order
# Concatenate into space-separated strings
# Calculate sequence length statistics
```

**Output**:
- `clif_sentences_train_val.parquet`: [hospitalization_id, clif_sentence, seq_length]
- `clif_sentences_test.parquet`: same schema
- Context coverage statistics (4K vs 8K)

**Example**:
```bash
uv run AR/gpt2/00_convert_narrative_to_sentences.py \
    --train-val OutputTokens/narratives/train_val_sequences.parquet \
    --test OutputTokens/narratives/test_sequences.parquet \
    --output-dir ./models/gpt2/data
```

**Output Statistics**:
```
Train/Val: 43,279 hospitalizations (84.9M tokens)
Test:       6,932 hospitalizations (13.2M tokens)

Context Coverage:
  4096 tokens: 90.0% of sequences
  8192 tokens: 97.0% of sequences
```

### Stage 1: Build Vocabulary from Registry

**Script**: `01a_build_vocab_from_registry.py`

**Purpose**: Build complete vocabulary from token_registry.json (NOT from actual data).

**Key Feature**: **Registry-Based Vocabulary**
- Includes ALL tokens from token registry, even those with count=0
- Better generalization to rare clinical events
- Ensures consistency across all experiments

**Input**: `OutputTokens/token_registry.json`

**Process**:
```python
1. Load token registry with usage statistics
2. Extract all tokens from 9 categories
3. Sort by category, then frequency
4. Create Vocabulary object with special tokens
5. Freeze vocabulary (training=False)
6. Generate SHA256 hash for validation
7. Save vocabulary files + metadata
```

**Output**:
- `vocab.gzip`: Pickled Vocabulary object (1,373 tokens)
- `vocabulary.csv`: Human-readable token→ID mapping
- `vocab_stats.txt`: Statistics by category
- `vocab_metadata.json`: Hash and validation info

**Example**:
```bash
uv run AR/gpt2/01a_build_vocab_from_registry.py \
    --token-registry OutputTokens/token_registry.json \
    --output-dir ./models/gpt2/vocab
```

**Vocabulary Structure**:
```
Total: 1,373 tokens
  - Special tokens (5): PAD, TL_START, TL_END, UNK, TRUNC
  - Clinical tokens (1,368): Across 9 categories

Categories:
  1. cohort_adt (26 tokens): Demographics, transfers, disposition
  2. elixhauser (32 tokens): Comorbidities
  3. assessment (23 tokens): GCS, RASS scores
  4. vitals (165 tokens): Vital signs
  5. labs (609 tokens): Laboratory values
  6. respiratory_support (217 tokens): Respiratory devices/parameters
  7. medications (294 tokens): Medication administration
  8. crrt_therapy (1 token): CRRT therapy
  9. ecmo_mcs (1 token): ECMO/MCS therapy
```

### Stage 2: Create Splits and Tokenize

**Script**: `03_create_splits.py`

**Purpose**: Tokenize sequences and create train/val/test splits.

**Modes**:
- **Standard**: 80/10/10 split from single file
- **Presplit**: Handle pre-split train_val + test files (used for narratives)

**Process**:
```python
1. Load vocabulary (frozen mode)
2. Tokenize sequences:
   - Split clif_sentence by spaces
   - Replace tokens with IDs using vectorized Polars replace_strict
   - Add TL_START (BOS) and TL_END (EOS)
   - Filter sequences > max_length
3. Create splits:
   - Presplit mode: Split train_val 90/10 → train/val
   - Deterministic shuffling using hash-based indexing
   - Direct sink to parquet (pure lazy operations)
```

**Output**:
- `train/data.parquet`
- `val/data.parquet`
- `test/data.parquet`
- `splits_summary.txt`

**Example (8K context)**:
```bash
uv run AR/gpt2/03_create_splits.py \
    --presplit \
    --train-val ./models/gpt2/data/clif_sentences_train_val.parquet \
    --test ./models/gpt2/data/clif_sentences_test.parquet \
    --vocab-dir ./models/gpt2/vocab \
    --output-dir ./models/gpt2/splits \
    --max-length 8192
```

**Output Statistics**:
```
Dataset sizes (8K context):
  Train: 37,765 sequences
  Val:    4,196 sequences
  Test:   6,743 sequences
```

### Stage 3: Train Model

**Script**: `04_train_gpt2.py`

**Purpose**: Train GPT2 from scratch on CLIF vocabulary.

**Key Features**:
- Hardware-optimized profiles (L40, A100-40GB, A100-80GB)
- 4 model sizes (small/medium/large/xl)
- Auto-detects GPU and precision (fp16/bf16)
- Gradient checkpointing for 8K context
- Weights & Biases integration

**Example (GPT2-small on L40)**:
```bash
uv run python AR/gpt2/04_train_gpt2.py \
    --config-profile l40 \
    --data-dir ./models/gpt2/splits \
    --vocab-dir ./models/gpt2/vocab \
    --output-dir ./models/gpt2/checkpoints \
    --model-size small \
    --epochs 1 \
    --context-size 8192 \
    --gradient-checkpointing \
    --wandb \
    --run-name gpt2-small-8k
```

**Training Configuration (L40 Profile, Small)**:
```python
- Batch size: 6 per device
- Gradient accumulation: 4 steps
- Effective batch size: 48 (6 × 4 × 2 GPUs)
- Learning rate: 2e-4 with cosine schedule
- Warmup: 3% of training
- Precision: bfloat16 (Ampere+)
- Gradient checkpointing: Enabled (60% memory reduction)
```

**Output**:
- `{run_name}/checkpoint-*/`: Training checkpoints
- `{run_name}/final_model/`: Final model with vocabulary

### Stage 4: Evaluate Model

**Script**: `05_evaluate_test.py`

**Purpose**: Evaluate trained model on test set.

**Metrics**:
- Loss
- Perplexity
- Token accuracy (next-token prediction)
- Top-5 accuracy

**Example**:
```bash
uv run python AR/gpt2/05_evaluate_test.py \
    --model models/gpt2/checkpoints/gpt2-small-8k/final_model \
    --data-dir models/gpt2/splits \
    --vocab models/gpt2/vocab/vocab.gzip \
    --output models/gpt2/checkpoints/gpt2-small-8k/test_results.json \
    --batch-size 4
```

**Output**: JSON file with metrics and configuration

---

## 2. Special Tokens

### 2.1 Definition

CLIFATRON uses 5 special tokens:

| Token | ID | Purpose | Usage |
|-------|----|---------| ------|
| `PAD` | 0 | Padding | Fill shorter sequences in batch |
| `TL_START` | 1 | Beginning of sequence | Mark start of hospitalization (BOS) |
| `TL_END` | 2 | End of sequence | Mark end of hospitalization (EOS) |
| `UNK` | 3 | Unknown | Handle out-of-vocabulary tokens |
| `TRUNC` | 4 | Truncation marker | Indicate sequence was truncated |

### 2.2 Special Token Usage

#### TL_START (BOS) - Token ID 1

**Purpose**: Signals the beginning of a hospitalization

**When applied**: Automatically added during tokenization

**Example**:
```
TL_START age_56_65 sex_female day_1 hour_11 labs_lactate_(0.5,0.9] ...
```

#### TL_END (EOS) - Token ID 2

**Purpose**: Signals the end of a hospitalization

**When applied**: Automatically added after all tokens

**Example**:
```
... vitals_heart_rate_(77.0,83.0] disposition_home TL_END
```

#### PAD - Token ID 0

**Purpose**: Pads shorter sequences to match batch length

**When applied**: During batching by data collator

**Example batch with padding**:
```python
# Variable-length sequences
Sequence 1: TL_START ... 2000 tokens ... TL_END  (2002 tokens)
Sequence 2: TL_START ... 1500 tokens ... TL_END  (1502 tokens)

# After padding to longest (2002)
Sequence 1: TL_START ... 2000 tokens ... TL_END
Sequence 2: TL_START ... 1500 tokens ... TL_END PAD PAD ... (500 padding)

# Attention mask
Sequence 1: [1] × 2002
Sequence 2: [1] × 1502 + [0] × 500

# Labels for loss
Sequence 1: [label_ids] × 2002
Sequence 2: [label_ids] × 1502 + [-100] × 500 (padding ignored)
```

#### UNK - Token ID 3

**Purpose**: Fallback for tokens not in vocabulary

**When applied**: Rarely, since vocabulary is pre-built from token registry

**Usage scenarios**:
- New token patterns not in registry
- Corrupted data
- Edge cases in binning logic

#### TRUNC - Token ID 4

**Purpose**: Indicates sequence was truncated to fit context window

**When applied**: When sequence exceeds max_length during padding

**Example**:
```python
# Original sequence: 10,000 tokens
# Max length: 8,192

# After truncation
TL_START ... 8,188 tokens ... TRUNC TL_END  (8,192 total)
```

### 2.3 Token ID to Clinical Token Mapping

```
ID 0:      PAD
ID 1:      TL_START (BOS)
ID 2:      TL_END (EOS)
ID 3:      UNK
ID 4:      TRUNC
ID 5-1372: Clinical tokens (sorted by category, then frequency)
```

**Total vocabulary size**: 1,373 tokens

---

## 3. Context Window Handling

CLIFATRON uses **8,192 token context window** with simple sequential chunking.

**Note**: GPT2's standard context is 1024 tokens. This implementation extends it to 8192 to handle long clinical narratives.

### 3.1 Configuration

**Effective context**:
- Total: 8,192 tokens
- Available for clinical tokens: 8,190 tokens (reserves 2 for TL_START and TL_END)
- **No overlap**: Each token appears in exactly one chunk

### 3.2 Three Scenarios

#### Scenario 1: Sequence SHORTER than Context (< 8,190 tokens)

**What happens**:
- Entire hospitalization kept as single sequence
- TL_START and TL_END added automatically
- No padding in dataset (padding happens during batching)

**Example**:
```python
# Hospitalization with 500 clinical tokens
tokens = ["age_56_65", "sex_female", ..., "disposition_home"]  # 500 tokens

# After tokenization
input_ids = [1, 5, 6, ..., 499, 2]  # TL_START + 500 + TL_END = 502 tokens
```

**Result**: Single training example with 502 tokens

#### Scenario 2: Sequence EQUAL to Context (≈ 8,190 tokens)

**What happens**:
- Entire hospitalization fits exactly
- TL_START and TL_END added
- Total: 8,192 tokens (perfect fit)

**Example**:
```python
# Hospitalization with exactly 8,190 clinical tokens
tokens = ["age_56_65", ..., "disposition_home"]  # 8,190 tokens

# After tokenization
input_ids = [1, ...8190 tokens..., 2]  # 8,192 tokens total
```

**Result**: Single training example using full context window

#### Scenario 3: Sequence LONGER than Context (> 8,190 tokens)

**What happens**: **Filtered out during split creation**

**Strategy**:
- Sequences > max_length are filtered during `03_create_splits.py`
- Not included in training data
- Ensures all sequences fit in context window

**Why not chunk?**:
- Simplicity: No need to track chunk metadata
- Packing handles efficiency: Multiple short sequences packed together
- 97% coverage at 8K: Only 3% of sequences exceed 8,192 tokens

**Alternative**: To include long sequences, you could implement chunking similar to AR/gpt2_hf/, but this implementation prioritizes simplicity.

### 3.3 Padding Strategy

**When padding happens**: During batching by the data collator

**How padding works**:
1. Collator receives batch of sequences with variable lengths
2. Finds longest sequence in batch
3. Pads all shorter sequences to this length
4. Sets attention mask: 1 for real tokens, 0 for padding
5. Sets labels: -100 for padding (ignored in loss)

**Example batch**:
```python
# Input batch (variable lengths)
Sequence 1: TL_START ... 2000 tokens ... TL_END  # Length: 2002
Sequence 2: TL_START ... 3500 tokens ... TL_END  # Length: 3502
Sequence 3: TL_START ... 1200 tokens ... TL_END  # Length: 1202
Sequence 4: TL_START ... 800 tokens ... TL_END   # Length: 802

# Determine padding length
max_length = 3502

# After padding (all sequences → 3502 tokens)
Sequence 1: TL_START ... 2000 tokens ... TL_END PAD×1500
Sequence 2: TL_START ... 3500 tokens ... TL_END
Sequence 3: TL_START ... 1200 tokens ... TL_END PAD×2300
Sequence 4: TL_START ... 800 tokens ... TL_END PAD×2700

# Attention masks (1 = attend, 0 = ignore)
Sequence 1: [1]×2002 + [0]×1500
Sequence 2: [1]×3502
Sequence 3: [1]×1202 + [0]×2300
Sequence 4: [1]×802 + [0]×2700

# Labels for loss (clinical tokens + special tokens, -100 for padding)
Sequence 1: [label_ids]×2002 + [-100]×1500
Sequence 2: [label_ids]×3502
Sequence 3: [label_ids]×1202 + [-100]×2300
Sequence 4: [label_ids]×802 + [-100]×2700
```

### 3.4 Sequence Packing

**Does this implementation use packing?** YES

**Implementation**: `ClifDataset.chunk_iterable()`

**Pattern**:
```
TL_START hosp1_tokens TL_END PAD×N hosp2_tokens TL_END PAD×M hosp3_tokens TL_END PAD ...
```

**Key features**:
- Packs multiple hospitalizations into fixed-length sequences
- Random Poisson padding (λ=7) between hospitalizations
- Prevents sequence boundary learning
- Reduces padding waste from ~46% to <5%
- 1.5-2x throughput improvement

**Why use packing?**
- Efficiency: Maximizes GPU utilization
- Simplicity: No attention mask needed (causal masking handles it)
- Privacy: Causal attention prevents cross-contamination

**Configuration** (`config.py`):
```python
collation: Literal["padded", "packed"] = "packed"  # Default
```

---

## 4. Model Architecture

### 4.1 Overview

Based on **nanoGPT** by Andrej Karpathy - a clean, understandable GPT implementation in PyTorch.

**Key components**:
- Token Embeddings (wte): vocab_size × n_embd
- Position Embeddings (wpe): block_size × n_embd
- Transformer Blocks (h): n_layer blocks
- Final LayerNorm (ln_f)
- LM Head: n_embd → vocab_size (weight tied with wte)

### 4.2 GPTConfig

```python
@dataclass
class GPTConfig:
    block_size: int = 4096          # Context window (4x GPT2 default)
    vocab_size: int = 50304         # Set from vocabulary (1373 for CLIF)
    n_layer: int = 12               # Transformer layers
    n_head: int = 12                # Attention heads
    n_embd: int = 768               # Embedding dimension
    dropout: float = 0.0            # Dropout (0 for pretraining)
    bias: bool = True               # Use bias in Linear/LayerNorm
    gradient_checkpointing: bool = False  # Memory optimization

    # Special token IDs (set from vocabulary)
    bos_token_id: int = None  # TL_START (1)
    eos_token_id: int = None  # TL_END (2)
    pad_token_id: int = None  # PAD (0)
```

### 4.3 Model Sizes

From `config.py`:

```python
GPT2_CONFIGS = {
    "small": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_params": "124M"
    },
    "medium": {
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "n_params": "355M"
    },
    "large": {
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20,
        "n_params": "774M"
    },
    "xl": {
        "n_embd": 1600,
        "n_layer": 48,
        "n_head": 25,
        "n_params": "1.5B"
    }
}
```

Note: `block_size` is overrideable at runtime (commonly 4096 or 8192 for narratives)

With CLIF vocabulary (1373 tokens):
- **GPT2-small**: ~92.4M parameters (vs 124M with full vocab)
- **GPT2-medium**: ~340M parameters
- **GPT2-large**: ~760M parameters
- **GPT2-xl**: ~1.48B parameters

### 4.4 Architecture Features

#### 1. Flash Attention

```python
# Uses torch.nn.functional.scaled_dot_product_attention (PyTorch ≥2.0)
# 2-4x faster than manual attention
# Automatically detected and enabled
```

#### 2. Weight Tying

```python
self.transformer.wte.weight = self.lm_head.weight
```
- Shares weights between token embeddings and output projection
- Reduces parameters, often improves performance

#### 3. Gradient Checkpointing

```python
# Enabled via --gradient-checkpointing flag
# Trades computation for memory (~60% reduction)
# Essential for 8K context windows
# Applied at block level
```

#### 4. Causal Self-Attention

```python
# Multi-head attention with causal masking
# QKV projection: 3 * n_embd
# Flash Attention support with fallback
# Attention dropout + residual dropout
```

#### 5. MLP (Feed-Forward)

```python
# 2-layer: n_embd → 4*n_embd → n_embd
# GELU activation
# Dropout for regularization
```

#### 6. Block (Transformer Layer)

```python
# Pre-LayerNorm architecture
# Residual connections: x + Attention(LN(x)) + MLP(LN(x))
# Gradient checkpointing support
```

---

## 5. Training Configuration

### 5.1 Training Parameters

From `config.py`:

```python
@dataclass
class TrainingConfig:
    # Training
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # Optimization
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True

    # Data
    collation: Literal["padded", "packed"] = "packed"

    # Mixed precision
    fp16: bool = False  # Auto-set based on GPU
    bf16: bool = False  # Auto-set based on GPU
```

### 5.2 Hardware Profiles

#### L40 Profile (2x NVIDIA L40, 48GB each)

```python
# GPT2-small (92.4M params)
batch_size: 6
gradient_accumulation_steps: 4
effective_batch_size: 48  # 6 × 4 × 2 GPUs
memory_per_gpu: ~15GB
precision: bfloat16
gradient_checkpointing: True

# GPT2-medium (340M params)
batch_size: 4
gradient_accumulation_steps: 8
effective_batch_size: 64  # 4 × 8 × 2 GPUs
memory_per_gpu: ~18GB
precision: bfloat16
gradient_checkpointing: True

# GPT2-large (760M params)
batch_size: 2
gradient_accumulation_steps: 16
effective_batch_size: 64  # 2 × 16 × 2 GPUs
memory_per_gpu: ~30GB
precision: bfloat16
gradient_checkpointing: True
```

#### A100 Profile (8x A100, 40GB or 80GB)

```python
# GPT2-small/medium/large
effective_batch_size: 192
higher_batch_sizes: True
supports_xl: True (80GB variant)
auto_scaled_lr: True
```

### 5.3 Training Workflow

1. **Initialization**:
   ```python
   - Load vocabulary (frozen)
   - Initialize GPT model from scratch (random weights)
   - Set special token IDs (bos/eos/pad)
   - Move model to device (GPU/CPU)
   - Enable gradient checkpointing if configured
   ```

2. **Dataset Loading**:
   ```python
   - ClifDataset wraps train/val/test parquet files
   - Streaming mode for memory efficiency
   - Packed or padded collation
   - Random Poisson padding (λ=7) between sequences
   ```

3. **Training Loop** (HuggingFace Trainer):
   ```python
   - Automatic mixed precision (fp16/bf16)
   - Gradient accumulation
   - Cosine LR schedule with warmup
   - Early stopping on validation loss
   - Checkpoint saving every N steps
   ```

4. **Optimization**:
   ```python
   - AdamW optimizer (fused version on CUDA)
   - Weight decay on 2D parameters only
   - Max gradient norm clipping (1.0)
   - Learning rate warmup (3% of training)
   ```

### 5.4 Automatic Device Detection

```python
# CUDA → bf16 if Ampere+, else fp16
# MPS (Apple Silicon) → fp32
# CPU → fp32 (with warnings)
# Multi-GPU → automatic DDP
```

---

## 6. Dataset Implementation

### 6.1 ClifDataset Class

From `dataset.py`:

```python
class ClifDataset:
    """
    Dataset handler for CLIF tokenized sequences

    Features:
    - Streaming mode (low memory)
    - Packed or padded collation
    - Configurable context length
    - Polars-based row counting
    """

    def __init__(
        self,
        data_dir: pathlib.Path,
        vocab_path: pathlib.Path,
        collation: Literal["padded", "packed"] = "packed",
        max_seq_length: int = 4096,
        shuffle_buffer_size: int = 1024,
    )
```

### 6.2 Key Methods

#### `_load_datasets()`

```python
# Uses datasets.load_dataset with streaming=True
# Never loads full dataset into memory
# Gets row counts from parquet metadata (Polars)
# Returns IterableDataset objects
```

#### `generate_padding()`

```python
# Creates random padding between sequences
# Poisson distribution (λ=7)
# Uses PAD token (ID 0)
# Prevents sequence boundary learning
```

#### `chunk_iterable()`

```python
# Packs multiple sequences into max_seq_length chunks
# Concatenates sequences with random padding
# Yields fixed-length tensors
# Used for packed collation
```

### 6.3 Collation Strategies

**Padded Collation**:
- Each sequence padded to max_seq_length
- Adds PAD tokens (ID 0) at end
- Adds TRUNC token if sequence too long
- Attention mask: 1 for real tokens, 0 for padding
- Labels: -100 for padding (ignored in loss)

**Packed Collation** (Recommended):
- Multiple sequences concatenated into max_seq_length chunks
- Random Poisson padding between sequences
- No attention masks needed (causal masking handles it)
- More efficient: ~46% padding → <5% padding
- 1.5-2x throughput improvement

### 6.4 Memory Optimizations

1. **Streaming Mode**:
   ```python
   # datasets.load_dataset(streaming=True)
   # Never materializes full dataset
   # Processes in configurable chunks
   ```

2. **Polars Row Counting**:
   ```python
   # Reads parquet metadata only
   # No data loading for counts
   # Fast and memory-efficient
   ```

3. **On-the-fly Processing**:
   ```python
   # Tokenization happens during iteration
   # Padding/packing happens in data loader
   # No intermediate storage
   ```

---

## 7. Vocabulary System

### 7.1 Vocabulary Class

From `vocabulary.py`:

```python
class Vocabulary:
    """
    Bidirectional token<->ID mapping with auxiliary metadata

    Attributes:
    - lookup: Dict[str, int]  # token → ID
    - reverse: Dict[int, str]  # ID → token
    - aux: Dict[str, Any]      # token → metadata
    - _is_training: bool       # Freeze after building
    """
```

### 7.2 Key Features

#### Registry-Based Vocabulary

**Unique to this implementation**:
- Builds from `token_registry.json` (includes ALL tokens)
- Even tokens with count=0 are included
- Better generalization to rare clinical events
- Ensures consistency across experiments

**Why registry-based?**
- Rare events matter in clinical data (e.g., ECMO flow rates)
- Prevents vocabulary drift across experiments
- Enables model to generalize to unseen events

#### Vocabulary Lock System

```python
Purpose: Prevent vocabulary drift across experiments

1. Build vocabulary from token_registry.json
2. Generate SHA256 hash of token→ID mapping
3. Save hash to vocab_metadata.json
4. Validate size (1373 tokens)
5. Warn against rebuilding

Why: Different vocabularies = incompatible models
```

### 7.3 Vocabulary Structure

```
ID 0:     PAD
ID 1:     TL_START (BOS)
ID 2:     TL_END (EOS)
ID 3:     UNK
ID 4:     TRUNC
ID 5+:    Clinical tokens (sorted by category, then frequency)
```

### 7.4 Token Categories

From token registry:

| Category | Count | Examples |
|----------|-------|----------|
| cohort_adt | 26 | age_56_65, sex_male, transfer_to_icu, disposition_home |
| elixhauser | 32 | elix_congestive_heart_failure, elix_diabetes |
| assessment | 23 | assessment_rass_0, assessment_gcs_total_15 |
| vitals | 165 | vitals_heart_rate_(77.0,83.0], vitals_sbp_(95.0,100.0] |
| labs | 609 | labs_lactate_(0.5,0.9], labs_creatinine_(2.1,2.5] |
| respiratory_support | 217 | respiratory_support_fio2_set_(0.4,0.5] |
| medications | 294 | medications_norepinephrine_mcg_kg_min_(0.06,0.12] |
| crrt_therapy | 1 | crrt_therapy_rate_100_200 |
| ecmo_mcs | 1 | ecmo_mcs_flow_2_3 |

**Total**: 1,368 clinical tokens + 5 special = 1,373 tokens

### 7.5 Key Methods

#### `__call__(word)`

```python
# Returns token ID for given word
# If training mode: adds new tokens dynamically
# If frozen: returns UNK for unseen tokens
# Used as: vocab("age_56_65") → 42
```

#### `get_vocab_hash()`

```python
# SHA256 hash of sorted token→ID mapping
# Used for vocabulary lock system
# Ensures consistency across training runs
```

#### `validate_vocab_size()`

```python
# Checks vocabulary has expected size (1373)
# Raises ValueError if mismatch
# Prevents training with wrong vocabulary
```

#### `save_metadata()`

```python
# Saves JSON with vocab hash and size
# Documents special token IDs
# Used for vocabulary lock validation
```

---

## 8. Code Examples

### 8.1 Convert Narratives to Sentences

From `00_convert_narrative_to_sentences.py`:

```python
def convert_narratives_to_sentences(parquet_path, output_path):
    """
    Convert token-per-row narratives to space-separated sentences.

    Uses Polars streaming for memory efficiency.
    """
    # Load with streaming
    df = pl.scan_parquet(parquet_path)

    # Group by hospitalization and concatenate tokens
    sentences = (
        df
        .sort(["hospitalization_id", "event_time", "sequence_order"])
        .group_by("hospitalization_id")
        .agg([
            pl.col("token").str.concat(" ").alias("clif_sentence"),
            pl.col("token").count().alias("seq_length")
        ])
    )

    # Sink to parquet (never loads full dataset)
    sentences.sink_parquet(output_path)
```

### 8.2 Build Vocabulary from Registry

From `01a_build_vocab_from_registry.py`:

```python
def build_vocabulary_from_registry(registry_path):
    """
    Build complete vocabulary from token registry.

    Includes ALL tokens (even count=0) for generalization.
    """
    # Load token registry
    with open(registry_path) as f:
        registry = json.load(f)

    # Extract all tokens
    all_tokens = []
    for category, tokens in registry.items():
        for token_name in tokens:
            all_tokens.append(token_name)

    # Sort by category, then frequency
    all_tokens.sort()

    # Create vocabulary with special tokens
    special_tokens = ["PAD", "TL_START", "TL_END", "UNK", "TRUNC"]
    vocab_words = special_tokens + all_tokens

    # Create Vocabulary object
    vocab = Vocabulary(words=tuple(vocab_words), is_training=False)

    # Generate hash for validation
    vocab_hash = vocab.get_vocab_hash()

    return vocab, vocab_hash
```

### 8.3 Tokenize with Polars

From `03_create_splits.py`:

```python
def tokenize_sequences(df, vocab):
    """
    Vectorized tokenization using Polars replace_strict.

    10-100x faster than element-wise lookup.
    """
    # Split clif_sentence by spaces
    df = df.with_columns(
        pl.col("clif_sentence").str.split(" ").alias("tokens")
    )

    # Create mapping: token → ID
    token_to_id = {token: idx for token, idx in vocab.lookup.items()}

    # Vectorized replace (fast!)
    df = df.with_columns(
        pl.col("tokens")
        .list.eval(pl.element().replace_strict(token_to_id, default=3))  # 3 = UNK
        .alias("input_ids")
    )

    # Add special tokens
    df = df.with_columns(
        pl.concat_list([
            pl.lit([1]),  # TL_START
            pl.col("input_ids"),
            pl.lit([2])   # TL_END
        ]).alias("input_ids")
    )

    return df
```

### 8.4 Packed Data Collation

From `dataset.py`:

```python
def chunk_iterable(dataset, vocab, max_seq_length):
    """
    Pack multiple sequences into fixed-length chunks.

    Adds random Poisson padding between sequences.
    """
    buffer = []
    current_length = 0

    for example in dataset:
        input_ids = example['input_ids']

        # Add random padding before this sequence
        pad_length = np.random.poisson(7)  # λ=7
        padding = [vocab("PAD")] * pad_length

        # Check if we can fit this sequence
        total_length = current_length + pad_length + len(input_ids)

        if total_length <= max_seq_length:
            buffer.extend(padding)
            buffer.extend(input_ids)
            current_length = total_length
        else:
            # Yield current chunk
            if current_length > 0:
                # Pad to max_seq_length
                buffer.extend([vocab("PAD")] * (max_seq_length - current_length))
                yield {"input_ids": torch.tensor(buffer, dtype=torch.long)}

            # Start new chunk
            buffer = input_ids.copy()
            current_length = len(input_ids)

    # Yield final chunk
    if current_length > 0:
        buffer.extend([vocab("PAD")] * (max_seq_length - current_length))
        yield {"input_ids": torch.tensor(buffer, dtype=torch.long)}
```

---

## 9. Summary Statistics

### 9.1 Vocabulary

- **Total vocabulary size:** 1,373 tokens
  - 5 special tokens: PAD, TL_START, TL_END, UNK, TRUNC
  - 1,368 clinical tokens across 9 categories

### 9.2 Context Window

- **Maximum context:** 8,192 tokens
- **Effective for clinical tokens:** 8,190 tokens (reserves 2 for TL_START and TL_END)
- **Chunking strategy:** Filter sequences > max_length (97% fit in 8K)
- **No overlap:** Not applicable (sequences either fit or filtered)

### 9.3 Training Data (Temporal Split)

- **Training set:** 2018-2023 data
  - Train: 37,765 sequences
  - Validation: 4,196 sequences
- **Test set:** 2024 data
  - Test: 6,743 sequences

### 9.4 Sequence Length Distribution

Based on processed data (8K context):

| Length Category | Token Range | Percentage | Treatment |
|----------------|-------------|------------|-----------|
| Very short | 50-500 | ~15% | Single sequence (packing reduces waste) |
| Short | 500-2000 | ~35% | Single sequence (packing reduces waste) |
| Medium | 2000-5000 | ~30% | Single sequence, minimal padding |
| Long | 5000-8190 | ~15% | Single sequence (fits exactly) |
| Very long | 8190+ | ~3% | Filtered out |

### 9.5 Model Sizes (with CLIF vocabulary)

```python
GPT2-small:  ~92.4M parameters
GPT2-medium: ~340M parameters
GPT2-large:  ~760M parameters
GPT2-xl:     ~1.48B parameters
```

### 9.6 Training Configuration (L40 Profile)

**GPT2-small (recommended)**:
```yaml
batch_size: 6
gradient_accumulation_steps: 4
effective_batch_size: 48  # 6 × 4 × 2 GPUs
learning_rate: 2e-4
context_size: 8192
precision: bfloat16
gradient_checkpointing: true
num_epochs: 5
```

**Estimated training time:** ~4-6 hours on 2x L40 (48GB each)

---

## Appendix: Comparison with AR/gpt2_hf/

### Key Differences

| Feature | AR/gpt2/ (Custom) | AR/gpt2_hf/ (HuggingFace) |
|---------|-------------------|---------------------------|
| **Model Source** | nanoGPT (custom) | HuggingFace Transformers |
| **Vocabulary** | Registry-based (ALL tokens) | Data-based (frequency) |
| **Data Processing** | Polars streaming | Pandas + custom dataset |
| **Special Tokens** | TL_START/TL_END | BOS/EOS |
| **Chunking** | Filter > max_length | Sequential 8190 boundaries |
| **Tokenization** | Vectorized Polars replace | Element-wise lookup |
| **Model Hub** | Not compatible | Full HuggingFace integration |

### When to Use Which?

**Use AR/gpt2/ (Custom) when:**
- You need maximum control and code clarity
- Memory efficiency is critical (huge datasets)
- You want to modify the architecture
- You don't need HuggingFace model hub integration

**Use AR/gpt2_hf/ (HuggingFace) when:**
- You need HuggingFace ecosystem compatibility
- You want to share models on HuggingFace Hub
- You need rich tokenizer features
- You need day-boundary aware chunking

---

## References

- Model architecture: `AR/gpt2/model.py`
- Training script: `AR/gpt2/04_train_gpt2.py`
- Dataset implementation: `AR/gpt2/dataset.py`
- Vocabulary system: `AR/gpt2/vocabulary.py`
- Configuration: `AR/gpt2/config.py`
- Token registry: `OutputTokens/token_registry.json`

---

**Document Version:** 1.0 (Custom GPT2)
**Last Updated:** 2025-10-31
**Author:** CLIFATRON Development Team
