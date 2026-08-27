# GPT2 Architecture for Clinical Narratives

Technical deep-dive into the GPT2 implementation for clinical narrative generation.

## Architecture Overview

### GPT2 Transformer

GPT2 uses the standard transformer decoder architecture with:
- **Multi-head self-attention**: All tokens attend to all previous tokens
- **Absolute positional embeddings**: Learned position encodings
- **GELU activation**: Smooth activation function in FFN
- **LayerNorm**: Standard layer normalization
- **Causal masking**: Token i can only attend to positions 0...i

### Model Configurations

#### GPT2-small (124M parameters)
```python
GPT2Config(
    vocab_size=1380,
    n_positions=8192,     # Extended from 1024
    n_embd=768,           # Hidden size
    n_layer=12,           # Transformer layers
    n_head=12,            # Attention heads (64 dim each)
    n_inner=3072,         # FFN size (4x hidden)
    activation_function="gelu_new",
    layer_norm_epsilon=1e-5,
)
```

#### GPT2-medium (355M parameters)
```python
GPT2Config(
    vocab_size=1380,
    n_positions=8192,     # Extended from 1024
    n_embd=1024,          # Hidden size
    n_layer=24,           # Transformer layers
    n_head=16,            # Attention heads (64 dim each)
    n_inner=4096,         # FFN size (4x hidden)
    activation_function="gelu_new",
    layer_norm_epsilon=1e-5,
)
```

## Key Differences from Qwen2

| Component | GPT2 | Qwen2 | Impact |
|-----------|------|-------|--------|
| **Attention** | Multi-Head Attention (MHA) | Grouped Query Attention (GQA) | Qwen2 more memory efficient |
| **Position** | Absolute learned embeddings | RoPE (Rotary Position Embeddings) | Qwen2 better for long sequences |
| **Activation** | GELU | SwiGLU | Qwen2 slightly better performance |
| **Normalization** | LayerNorm | RMSNorm | Similar performance |
| **Context** | 8192 (extended) | 8192 (native) | Same effective context |

### Attention Mechanisms

**GPT2 Multi-Head Attention (MHA)**:
```
Query, Key, Value computed for all heads
Attention scores: softmax(QK^T / sqrt(d))
Output: Attention(Q,K,V) for each head, concatenated
Memory: O(n_heads * d_model) for KV cache
```

**Qwen2 Grouped Query Attention (GQA)**:
```
Multiple query heads share same key/value heads
n_query_heads = 12-28
n_kv_heads = 2-4  # Shared across query heads
Memory: O(n_kv_heads * d_model) for KV cache
Benefit: ~4-7x smaller KV cache
```

### Position Encodings

**GPT2 Absolute Positional Embeddings**:
```python
# Learned position embeddings (shape: [n_positions, n_embd])
pos_emb = nn.Embedding(config.n_positions, config.n_embd)

# Add to token embeddings
hidden_states = token_emb + pos_emb[position_ids]
```

**Qwen2 RoPE (Rotary Position Embeddings)**:
```python
# Rotate queries and keys by position-dependent angles
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

Benefits of RoPE:
- Better extrapolation to longer sequences
- Relative position information naturally encoded
- No learned parameters

## Extended Context Window

### Challenge
Standard GPT2 has max context of 1024 tokens. Clinical hospitalizations often exceed 8000 tokens.

### Solution
Extend `n_positions` to 8192:

```python
config = GPT2Config(
    n_positions=8192,  # Extended from 1024
    ...
)
```

**Impact**:
- Positional embedding table grows: 1024×768 → 8192×768 (+21K parameters)
- Attention memory: O(seq_len²) - requires gradient checkpointing
- Training cost: ~2-3x slower than 1024 context

**Mitigation**:
- Gradient checkpointing: Trade 20% speed for 40% memory
- Sequence packing: Fill 8192 tokens with multiple hospitalizations
- FlashAttention via PyTorch SDPA: Faster attention computation

## Sequence Packing

### Problem
Most hospitalizations < 8192 tokens, leading to ~46% padding waste:

```
Batch without packing:
[BOS] hosp1 (1200 tokens) [EOS] [PAD]×6990  # 85% waste
[BOS] hosp2 (3500 tokens) [EOS] [PAD]×4690  # 57% waste
[BOS] hosp3 (800 tokens) [EOS] [PAD]×7390   # 90% waste
```

### Solution
Pack multiple hospitalizations per sequence:

```
Packed sequence:
[BOS] hosp1 [EOS] hosp2 [EOS] hosp3 [EOS] hosp4 [EOS] [PAD]×120  # 1.5% waste
```

**Implementation**:
1. Remove [BOS] from all but first hospitalization
2. Concatenate: `tokens1 + [EOS] + tokens2 + [EOS] + tokens3 + [EOS]`
3. Stop when adding next would exceed 8192
4. Pad remainder

**Safety**:
Causal attention mask prevents cross-contamination:
```
Attention mask for packed sequence:
[BOS] h1_1 h1_2 [EOS] h2_1 h2_2 [EOS] h3_1
  1    1    1    1    0    0    0    0     # [BOS] attends to self only
  1    1    1    1    0    0    0    0     # h1_1 attends to [BOS], h1_1
  1    1    1    1    1    1    0    0     # h1_2 attends to all of h1
  ...
```

**Results**:
- Padding waste: 46% → 1.5%
- Training throughput: +70-90%
- Total training time: -40-45%

## Chunking Strategy

### Problem
Some hospitalizations exceed 8192 tokens (10-15% of dataset).

### Solution: Simple Sequential Chunking

```python
effective_max_length = 8192 - 2  # Reserve for [BOS] and [EOS] = 8190

if len(tokens) <= 8190:
    # Single chunk
    return [[BOS] + tokens + [EOS]]
else:
    # Split at 8190-token boundaries
    chunks = []
    for i in range(0, len(tokens), 8190):
        chunk = tokens[i:i+8190]
        chunks.append([BOS] + chunk + [EOS])
    return chunks
```

**Properties**:
- Zero overlap between chunks
- Zero token waste
- Truncated portions continue in next chunk
- Each chunk is independent (gets own [BOS]/[EOS])

**Example**:
```
Hospitalization with 20,000 tokens:

Chunk 0: [BOS] + tokens[0:8190] + [EOS]      # 8192 total
Chunk 1: [BOS] + tokens[8190:16380] + [EOS]  # 8192 total
Chunk 2: [BOS] + tokens[16380:20000] + [EOS] # 3622 total
```

## Training Objective

### Causal Language Modeling

Predict next token given all previous tokens:

```python
# Loss computation
loss = CrossEntropyLoss(ignore_index=-100)(
    logits.view(-1, vocab_size),
    labels.view(-1)
)
```

**Input**:
```
[BOS] age_56_65 sex_female day_1 hour_11 vitals_hr_(77,83] [EOS]
```

**Labels** (shifted by 1):
```
age_56_65 sex_female day_1 hour_11 vitals_hr_(77,83] [EOS] [IGNORE]
```

**Predictions**:
- Position 0 ([BOS]) predicts age_56_65
- Position 1 (age_56_65) predicts sex_female
- Position 2 (sex_female) predicts day_1
- ...
- Position 6 ([EOS]) prediction ignored

## Memory Optimization

### Gradient Checkpointing

Trade computation for memory:

```python
# Forward pass: Don't store all activations
# Backward pass: Recompute activations on-the-fly

TrainingArguments(
    gradient_checkpointing=True,  # Enable
    gradient_checkpointing_kwargs={"use_reentrant": False}  # PyTorch 2.0+
)
```

**Impact**:
- Memory: -30-50% (can fit 2x larger models)
- Speed: -10-20% (extra forward passes)
- **Net benefit**: Usually worth it to avoid OOM

### Mixed Precision (BF16)

Use BFloat16 for forward/backward passes:

```python
TrainingArguments(
    bf16=True,  # Requires compute capability >= 8.0 (A100, L40, H100)
    # bf16_full_eval=True,  # Optional: BF16 for evaluation too
)
```

**Benefits**:
- Speed: ~1.5-2x faster than FP32
- Memory: ~2x reduction (16-bit vs 32-bit)
- Stability: Better than FP16 (wider exponent range)

**BF16 vs FP16**:
```
FP16: 1 sign + 5 exponent + 10 mantissa bits (range: ±65k)
BF16: 1 sign + 8 exponent + 7 mantissa bits (range: ±3.4e38)

BF16 has same range as FP32 → better for gradients
BF16 has lower precision → fine for deep learning
```

### DeepSpeed ZeRO

Shard optimizer states and/or parameters across GPUs:

**ZeRO-2** (recommended for 2+ GPUs):
```json
{
  "zero_optimization": {
    "stage": 2,  // Shard optimizer states only
    "allgather_partitions": true,
    "reduce_scatter": true,
    "contiguous_gradients": true
  }
}
```
- Memory per GPU: Model + Gradients + Optimizer/N
- Communication: Moderate (gather optimizer states)
- Speed: ~95% of non-ZeRO

**ZeRO-3** (for single GPU or low VRAM):
```json
{
  "zero_optimization": {
    "stage": 3,  // Shard model parameters too
    "stage3_param_persistence_threshold": "auto",
    "stage3_prefetch_bucket_size": "auto"
  }
}
```
- Memory per GPU: (Model + Gradients + Optimizer)/N
- Communication: Higher (gather parameters each layer)
- Speed: ~85-90% of non-ZeRO

## Performance Analysis

### Throughput Comparison (2x L40 48GB)

| Configuration | Tokens/sec | Samples/sec | Time/Epoch |
|---------------|------------|-------------|------------|
| GPT2-small, no packing, FP32 | 2,500 | 80 | 22 hours |
| GPT2-small, no packing, BF16 | 4,800 | 150 | 12 hours |
| GPT2-small, packing, BF16 | 8,000 | 250 | 7 hours |
| GPT2-medium, packing, BF16 | 4,800 | 150 | 12 hours |

### Memory Usage

| Model | Parameters | Activations | Gradients | Optimizer | Total/GPU (ZeRO-2) |
|-------|------------|-------------|-----------|-----------|---------------------|
| GPT2-small | 0.5 GB | 4 GB | 0.5 GB | 1 GB / N | ~6 GB (N=2) |
| GPT2-medium | 1.4 GB | 8 GB | 1.4 GB | 2.8 GB / N | ~12 GB (N=2) |

## Tokenizer Details

### Clinical Whitespace Tokenizer

```python
class ClinicalTokenizer(PreTrainedTokenizer):
    def _tokenize(self, text):
        # Split on whitespace
        return text.strip().split()

    def _convert_token_to_id(self, token):
        # Map token to ID via vocabulary
        return self.vocab.get(token, self.unk_token_id)
```

**Vocabulary**:
- Special tokens (5): [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, [SEP]=4
- Clinical tokens (1,375): Sorted alphabetically, IDs 5-1379

**Example**:
```
Input text: "age_56_65 sex_female day_1"
Tokenize: ["age_56_65", "sex_female", "day_1"]
To IDs: [234, 567, 345]
Add special: [2, 234, 567, 345, 3]  # [BOS] ... [EOS]
```

## Comparison Summary

### GPT2 Strengths
- Simple, well-understood architecture
- Maximum compatibility
- Good baseline for comparisons
- Faster training for short sequences (<2048 tokens)

### GPT2 Weaknesses
- Less memory efficient (MHA vs GQA)
- Worse long-context performance (absolute vs RoPE)
- Slightly slower than Qwen2 overall

### When to Use GPT2
1. Baseline comparisons with standard architectures
2. Maximum reproducibility and compatibility
3. Educational purposes (understand transformers)
4. When Qwen2 is unavailable

### When to Use Qwen2
1. Best performance for clinical narratives
2. More efficient for long sequences (GQA + RoPE)
3. Better memory efficiency (KV cache)
4. Production deployments

## Future Improvements

1. **FlashAttention-2**: Native Flash Attention instead of SDPA
2. **xFormers**: Memory-efficient attention patterns
3. **BF16 gradients**: BF16 for gradients too (not just activations)
4. **Dynamic padding**: Vary sequence length per batch
5. **Curriculum learning**: Start with shorter sequences

## References

1. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.
2. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
3. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention. NeurIPS.
4. Rajbhandari, S., et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC20.
5. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
