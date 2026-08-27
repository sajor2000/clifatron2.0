# Sequence Packing for GPT2 HF Training

## Overview

Sequence packing is an optimization that packs multiple short hospitalizations into a single 8192-token training sequence, maximizing GPU utilization.

## Why Pack Sequences?

**Without packing:**
- Short hospitalization (500 tokens) wastes 7692 tokens of GPU memory
- Only ~6% GPU utilization per example
- Training is slow and inefficient

**With packing:**
- Pack ~16 short hospitalizations into one 8192-token sequence
- ~100% GPU utilization
- **16x faster training** for short documents

## How It Works

### 1. Packing Strategy

```
Sequence 1: [BOS] hosp1_tokens [SEP] hosp2_tokens [SEP] hosp3_tokens [EOS]
Sequence 2: [BOS] hosp4_tokens [SEP] hosp5_tokens [EOS]
...
```

- Greedily pack hospitalizations until reaching 8190 tokens (leave 2 for [BOS]/[EOS])
- Insert `[SEP]` token between different hospitalizations
- Track document boundaries for attention masking

### 2. Document-Aware Attention Mask

Standard causal attention would allow tokens from `hosp2` to attend to tokens from `hosp1`. We don't want this!

**Solution:** Create a custom attention mask that:
- Allows causal attention **within** each hospitalization
- **Blocks** attention across `[SEP]` boundaries
- Prevents information leakage between documents

Example for 2 documents:
```
Positions:  0    1-100    101     102-200    201
Tokens:    [BOS] doc1... [SEP]   doc2...   [EOS]

Attention mask:
- Tokens 1-100 (doc1): Can attend to [BOS] and positions 1-100 only
- Token 101 ([SEP]): Can attend to [BOS] and positions 1-101
- Tokens 102-200 (doc2): Can attend to [BOS], [SEP], and positions 102-200 only
- Token 201 ([EOS]): Can attend to all tokens in doc2
```

### 3. Implementation

**File: `AR/gpt2_hf/data/sequence_packer.py`**

Key functions:
- `pack_sequences()`: Greedily packs hospitalizations
- `create_document_attention_mask()`: Creates block-diagonal attention mask
- `pack_and_create_batch()`: End-to-end packing with tokenization

### 4. Usage

**Enable packing (default):**
```python
dataset = load_narrative_dataset(
    config_path='clif_config.json',
    tokenizer=tokenizer,
    split='train',
    pack_sequences=True  # Enable packing
)
```

**Disable packing (one hospitalization per sequence):**
```python
dataset = load_narrative_dataset(
    config_path='clif_config.json',
    tokenizer=tokenizer,
    split='train',
    pack_sequences=False  # Disable packing
)
```

## Benefits

1. **16x faster training** for short documents
2. **Better GPU utilization** (~100% vs ~6%)
3. **Lower memory overhead** per example
4. **No information leakage** between documents (blocked attention)

## Trade-offs

1. **Slightly more complex** preprocessing
2. **Custom attention masks** (larger memory footprint: `batch_size x seq_len x seq_len`)
3. **Mixed document lengths** in a batch (may need gradient accumulation tuning)

## Verification

To verify packing works correctly:

```python
# Check that attention masks block cross-document attention
import torch
from AR.gpt2_hf.data.sequence_packer import create_document_attention_mask

# Two documents: positions [0, 50) and [51, 100)
# Position 50 is [SEP]
boundaries = [(0, 50), (51, 100)]
mask = create_document_attention_mask(100, boundaries)

# Verify doc2 tokens can't attend to doc1
assert not mask[60, 30].item()  # Token 60 (doc2) cannot attend to token 30 (doc1)
assert mask[60, 70].item()       # Token 60 (doc2) CAN attend to token 70 (doc2)
```

## Expected Improvements

- **Training speed**: 5-16x faster (depends on avg hospitalization length)
- **GPU memory efficiency**: 90-95% utilization vs 5-20%
- **Convergence**: Similar or slightly better (more diverse examples per batch)
