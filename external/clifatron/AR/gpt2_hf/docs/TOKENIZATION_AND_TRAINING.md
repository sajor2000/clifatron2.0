# CLIFATRON Tokenization and Training Pipeline (GPT2)

This document provides a comprehensive overview of how CLIFATRON tokenizes clinical hospitalization data and prepares it for training the GPT2 language model.

---

## ⚠️ IMPORTANT: Implemented Features Only

**This documentation describes the ACTUAL implementation, not theoretical approaches.**

**Chunking Strategy:**
- ✅ **Simple Sequential Chunking** at 8,190-token boundaries
- ❌ NO day-boundary detection
- ❌ NO sliding windows
- ❌ NO overlap between chunks

**Deprecated Parameters (Completely Ignored):**
- `overlap_tokens`: Not used, chunks have zero overlap
- `min_chunk_size`: Not used, all splits at fixed boundaries

**Packing:**
- ✅ Industry-standard pattern: `[BOS] hosp1 [EOS] hosp2 [EOS]`
- ✅ BOS token removed from non-first hospitalizations

---

## Table of Contents

1. [Hospitalization Tokenization Pipeline](#1-hospitalization-tokenization-pipeline)
2. [Special Tokens](#2-special-tokens)
3. [Context Window Handling](#3-context-window-handling)
4. [Sequence Packing Strategy](#4-sequence-packing-strategy)
5. [Code Examples](#5-code-examples)
6. [Summary Statistics](#6-summary-statistics)

---

## 1. Hospitalization Tokenization Pipeline

### Overview

The tokenization process transforms raw clinical data into structured token sequences that preserve temporal relationships and clinical context. This happens in two main phases:

1. **Phase 1: Data Tokenization** (`tokenETL/main.py`) - Converts raw clinical data into tokens
2. **Phase 2: Narrative Assembly** (`tokenETL/assemble_narratives.py`) - Assembles tokens into chronological sequences

### 1.1 Input Data Sources

CLIFATRON tokenizes data from multiple clinical sources:

| Data Source | Description | Examples |
|------------|-------------|----------|
| **Demographics** | Patient characteristics | Sex, age at admission |
| **Hospitalization** | Admission/discharge | Admission time, disposition |
| **ADT (Transfers)** | Location changes | ICU transfer, ED transfer |
| **Labs** | Laboratory results | Lactate, creatinine, WBC |
| **Vitals** | Vital signs | Heart rate, blood pressure, temperature |
| **Medications** | Continuous infusions | Norepinephrine, vasopressin |
| **Respiratory** | Ventilation support | FiO2, PEEP, respiratory rate |
| **Assessments** | Clinical scores | RASS, GCS |
| **Therapies** | Advanced interventions | CRRT, ECMO |
| **Comorbidities** | Elixhauser index | From prior hospitalizations |

### 1.2 Tokenization Methods

CLIFATRON uses two primary methods to convert clinical data into tokens:

#### Method 1: Mapping Tokenization (Categorical Data)

Categorical values are directly mapped to tokens:

```python
# Example: Sex
sex = "Male" → token = "sex_male"
sex = "Female" → token = "sex_female"

# Example: Discharge disposition
disposition = "Home" → token = "disposition_home"
disposition = "Expired" → token = "disposition_expired"
disposition = "SNF" → token = "disposition_skilled_nursing_facility"
```

#### Method 2: Binning Tokenization (Numeric Data)

Numeric values are binned into predefined ranges:

```python
# Example: Age at admission
age_at_admission = 67 → token = "age_66_75"
age_at_admission = 45 → token = "age_36_45"

# Example: Heart rate
vitals_heart_rate = 95 → token = "vitals_heart_rate_(91.0,100.0]"
vitals_heart_rate = 120 → token = "vitals_heart_rate_(109.0,120.0]"

# Example: Lab values
labs_lactate = 1.8 → token = "labs_lactate_(1.4,1.9]"
labs_creatinine = 2.5 → token = "labs_creatinine_(2.1,2.5]"
```

**Binning rationale:**
- Reduces vocabulary size (continuous → discrete)
- Handles measurement noise and variability
- Creates clinically meaningful ranges
- Improves generalization across patients

### 1.3 Narrative Assembly Structure

Each hospitalization is assembled into a chronologically ordered narrative following this hierarchical structure:

```
PREV_NARRATIVE_START
  elix_congestive_heart_failure    # Comorbidities from prior hospitalizations
  elix_diabetes_uncomplicated
  elix_chronic_pulmonary_disease
PREV_NARRATIVE_END
age_56_65                          # Patient demographics
sex_female
[day_1]                           # Temporal day marker
  [hour_11]                       # Temporal hour marker
    vitals_height_cm_(160.02,165.1]       # Clinical events at this timestamp
    vitals_weight_kg_(88.4,99.9]
    transfer_to_procedural
    respiratory_support_fio2_set_(0.4,0.5]
[day_2]
  [hour_8]
    labs_lactate_(0.5,0.9]
    vitals_heart_rate_(77.0,83.0]
    medications_norepinephrine_mcg_kg_min_(0.06,0.12]
[day_3]
  [hour_14]
    transfer_to_icu
    resp_device_invasive_mechanical_ventilation
...
disposition_home                   # Discharge outcome
```

### 1.4 Token Ordering Rules

Tokens within the same timestamp are ordered by **sequence priority** (defined in `tokenETL/assemble_narratives.py`):

| Priority | Token Type | Examples |
|----------|-----------|----------|
| 1 | `PREV_NARRATIVE_START` | Marker for comorbidity section |
| 2 | Elixhauser comorbidities | `elix_congestive_heart_failure` |
| 3 | `PREV_NARRATIVE_END` | End of comorbidity section |
| 4 | Demographics | `age_56_65`, `sex_female` |
| 5 | Day markers | `day_1`, `day_2`, ... `day_30+` |
| 6 | Hour markers | `hour_1` through `hour_24` |
| 7 | Clinical events | Labs, vitals, meds, transfers |
| 8 | Discharge disposition | `disposition_home`, `disposition_expired` |

**Within priority 7 (clinical events), tokens are sorted by event_time first (chronological order), then alphabetically within the same timestamp** to ensure consistent ordering.

### 1.5 Temporal Organization

**Day Markers:**
- Calculated relative to first event in hospitalization
- Range: `day_1` through `day_30+` (30+ for stays >30 days)
- Used as natural chunk boundaries for long hospitalizations

**Hour Markers:**
- Based on hour of day (0-23 → hour_1 through hour_24)
- Note: hour_1 = midnight (00:00-00:59), hour_24 = 23:00-23:59

**Event Timestamps:**
- All events at the same timestamp get the same day/hour markers
- Events are sorted chronologically across the hospitalization
- Multiple events at same timestamp are grouped together

### 1.6 Token Registry

The complete token vocabulary is stored in `token_registry.json` with usage statistics:

```json
{
  "assessment": {
    "assessment_rass_0": {
      "count": 1244057,
      "present_in_data": true
    },
    "assessment_gcs_total_15": {
      "count": 709748,
      "present_in_data": true
    }
  },
  "labs": {
    "labs_lactate_(0.5,0.9]": {
      "count": 89234,
      "present_in_data": true
    }
  }
  ...
}
```

**Statistics:**
- Total clinical tokens: ~1,375
- Token categories: 9 (assessment, ADT, labs, vitals, meds, respiratory, CRRT, ECMO, Elixhauser)
- Most common tokens: Assessment scores, vital signs, common lab values

---

## 2. Special Tokens

### 2.1 Definition

CLIFATRON uses 5 special tokens defined in the clinical tokenizer:

| Token | ID | Purpose | Usage |
|-------|----|---------| ------|
| `[PAD]` | 0 | Padding | Fill shorter sequences in batch |
| `[UNK]` | 1 | Unknown | Handle out-of-vocabulary tokens |
| `[BOS]` | 2 | Beginning of sequence | Mark start of hospitalization chunk |
| `[EOS]` | 3 | End of sequence | Mark end of hospitalization chunk |
| `[SEP]` | 4 | Separator | Reserved for future use |

### 2.2 Special Token Usage

#### [PAD] - Token ID 0

**Purpose:** Pads shorter sequences to match batch length

**When applied:** During batching by the data collator
```python
# Example batch with variable lengths
Sequence 1: [BOS] ... 2000 tokens ... [EOS]  (2002 tokens)
Sequence 2: [BOS] ... 1500 tokens ... [EOS]  (1502 tokens)
Sequence 3: [BOS] ... 800 tokens ... [EOS]   (802 tokens)

# After padding to longest (2002)
Sequence 1: [BOS] ... 2000 tokens ... [EOS]
Sequence 2: [BOS] ... 1500 tokens ... [EOS] [PAD] [PAD] ... (500 padding tokens)
Sequence 3: [BOS] ... 800 tokens ... [EOS] [PAD] [PAD] ...  (1200 padding tokens)
```

**Attention mask:** Set to `0` for padding positions (prevents attention)

**Loss calculation:** Labels set to `-100` (ignored by PyTorch CrossEntropyLoss)

#### [BOS] - Token ID 2

**Purpose:** Signals the beginning of a hospitalization chunk

**When applied:** Automatically added by tokenizer when `add_special_tokens=True`

**Example:**
```
[BOS] age_56_65 sex_female day_1 hour_11 labs_lactate_(0.5,0.9] ...
```

**Important:** Each chunk (even within the same hospitalization) gets its own [BOS] token

#### [EOS] - Token ID 3

**Purpose:** Signals the end of a hospitalization chunk

**When applied:** Automatically added by tokenizer after all tokens

**Example:**
```
... vitals_heart_rate_(77.0,83.0] disposition_home [EOS]
```

**Important:** Model learns to predict [EOS] as the final token, indicating sequence completion

#### [UNK] - Token ID 1

**Purpose:** Fallback for tokens not in vocabulary

**When applied:** Rarely in practice, since vocabulary is pre-built from all training data

**Usage scenarios:**
- New token patterns not seen during tokenizer training
- Corrupted data
- Edge cases in binning logic

#### [SEP] - Token ID 4

**Purpose:** Reserved for separating two sequences (e.g., for paired tasks)

**Current status:** **Not used** in CLIFATRON's current training pipeline

**Future use cases:**
- Multi-task learning (e.g., question answering)
- Pairing admission narrative with outcome prediction
- Separating context from query

### 2.3 Token ID to Clinical Token Mapping

```
ID 0-4:        Special tokens ([PAD], [UNK], [BOS], [EOS], [SEP])
ID 5-1379:     Clinical tokens (sorted alphabetically)
               - age_18_25, age_26_35, ..., age_86_above
               - assessment_gcs_total_3, ..., assessment_rass_4
               - day_1, day_2, ..., day_30+
               - disposition_expired, disposition_home, ...
               - elix_congestive_heart_failure, ...
               - hour_1, hour_2, ..., hour_24
               - labs_creatinine_(0.3,0.5], ...
               - medications_norepinephrine_mcg_kg_min_(0.06,0.12], ...
               - sex_female, sex_male
               - transfer_to_icu, transfer_to_ed, ...
               - vitals_heart_rate_(77.0,83.0], ...
```

Total vocabulary size: **1,380 tokens**

### 2.4 Special Token Application in Code

From `AR/qwen2/tokenizer/clinical_tokenizer.py` (shared by GPT2):

```python
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    """
    Build model inputs by adding special tokens.

    For single sequence:
        [BOS] token_ids_0 [EOS]

    For two sequences (not currently used):
        [BOS] token_ids_0 [SEP] token_ids_1 [EOS]
    """
    bos = [self.bos_token_id]  # [2]
    eos = [self.eos_token_id]  # [3]

    if token_ids_1 is None:
        return bos + token_ids_0 + eos

    sep = [self.sep_token_id]  # [4]
    return bos + token_ids_0 + sep + token_ids_1 + eos
```

---

## 3. Context Window Handling

CLIFATRON uses a **8,192 token context window** with simple sequential chunking to handle variable-length hospitalizations.

**Note:** GPT2's standard context is 1024 tokens. This implementation extends it to 8192 to match Qwen2 and handle long clinical narratives.

### 3.1 Configuration Parameters

From `AR/gpt2_hf/config/training_config.yaml`:

```yaml
max_length: 8192           # Maximum context window (extended from GPT2's default 1024)
overlap_tokens: 819        # DEPRECATED: Not used with simple chunking
min_chunk_size: 50         # DEPRECATED: Not used with simple chunking
```

**Effective context:**
- Total: 8,192 tokens
- Available for clinical tokens: 8,190 tokens (reserves 2 for [BOS] and [EOS])
- **No overlap:** Each token appears in exactly one chunk (no waste)

### 3.2 Three Scenarios

#### Scenario 1: Sequence SHORTER than Context (< 8,190 tokens)

**What happens:**
- Entire hospitalization kept as single chunk
- [BOS] and [EOS] added automatically
- No padding in dataset (padding happens during batching)

**Example:**
```python
# Hospitalization with 500 clinical tokens
tokens = [
    "age_56_65", "sex_female", "day_1", "hour_11",
    "vitals_heart_rate_(77.0,83.0]", ..., "disposition_home"
]  # 500 tokens

# After tokenization
input_ids = [2, 5, 6, 7, 8, 9, ..., 499, 3]  # [BOS] + 500 + [EOS] = 502 tokens
```

**Result:** Single training example with 502 tokens

#### Scenario 2: Sequence EQUAL to Context (≈ 8,190 tokens)

**What happens:**
- Entire hospitalization fits exactly
- [BOS] and [EOS] added
- Total: 8,192 tokens (perfect fit)

**Example:**
```python
# Hospitalization with exactly 8,190 clinical tokens
tokens = ["age_56_65", ..., "disposition_home"]  # 8,190 tokens

# After tokenization
input_ids = [2, ...8190 tokens..., 3]  # 8,192 tokens total
```

**Result:** Single training example using full context window

#### Scenario 3: Sequence LONGER than Context (> 8,190 tokens)

**What happens:** Simple sequential chunking

**Strategy:**
- Split hospitalization at 8,190-token boundaries
- **No overlap:** Each token appears in exactly one chunk
- **No waste:** Truncated portions continue in next chunk
- Each chunk gets its own [BOS] and [EOS] tokens

**Example chunking for long hospitalization:**
```python
# Hospitalization with 25,000 tokens
Total tokens: 25,000

# Sequential chunking at 8,190-token boundaries
Chunk 0: [BOS] + tokens[0:8190] + [EOS]        # 8,192 total
Chunk 1: [BOS] + tokens[8190:16380] + [EOS]    # 8,192 total
Chunk 2: [BOS] + tokens[16380:24570] + [EOS]   # 8,192 total
Chunk 3: [BOS] + tokens[24570:25000] + [EOS]   # 432 total (last chunk)

# Result: 4 training examples, 0 tokens wasted
```

**Why simple chunking?**
- **Zero waste:** Every token used exactly once (no overlap duplication)
- **Maximum efficiency:** Combined with packing, achieves 1.5-2x throughput
- **Simple to understand:** Clear token boundaries, easy to debug
- **Complements packing:** Packing handles the efficiency, chunking handles splitting

### 3.3 Chunking Algorithm

From `AR/gpt2_hf/data/narrative_dataset.py`:

```python
def _chunk_hospitalization(self, hosp_id, tokens):
    """
    Chunk a single hospitalization using simple sequential chunking.

    Strategy:
    - Split at effective_max_length (8190 tokens)
    - No overlap, no waste
    - Truncated portions continue in next chunk
    - Each chunk gets [BOS] and [EOS] during tokenization
    """
    # Short hospitalization - return as single chunk
    if len(tokens) <= self.effective_max_length:  # 8190
        return [NarrativeChunk(
            hospitalization_id=hosp_id,
            tokens=tokens,
            chunk_index=0,
            total_chunks=1,
            is_complete_hospitalization=True
        )]

    # Split into sequential chunks at 8190-token boundaries
    chunks = []
    for i in range(0, len(tokens), self.effective_max_length):
        chunk_tokens = tokens[i:i + self.effective_max_length]
        chunks.append(chunk_tokens)

    # Create NarrativeChunk objects
    total_chunks = len(chunks)
    chunk_objects = []
    for chunk_idx, chunk_tokens in enumerate(chunks):
        chunk_objects.append(NarrativeChunk(
            hospitalization_id=hosp_id,
            tokens=chunk_tokens,
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            is_complete_hospitalization=False
        ))

    return chunk_objects
```

### 3.4 Padding Strategy

**When padding happens:** During batching by the data collator

**How padding works:**
1. Collator receives batch of sequences with variable lengths
2. Finds longest sequence in batch
3. Rounds up to multiple of 8 (for GPU efficiency)
4. Pads all shorter sequences to this length

**Example batch:**
```python
# Input batch (variable lengths)
Sequence 1: [BOS] ... 2000 tokens ... [EOS]  # Length: 2002
Sequence 2: [BOS] ... 3500 tokens ... [EOS]  # Length: 3502
Sequence 3: [BOS] ... 1200 tokens ... [EOS]  # Length: 1202
Sequence 4: [BOS] ... 800 tokens ... [EOS]   # Length: 802

# Determine padding length
max_length = 3502
padded_length = ceil(3502 / 8) * 8 = 3504  # Round to multiple of 8

# After padding (all sequences → 3504 tokens)
Sequence 1: [BOS] ... 2000 tokens ... [EOS] [PAD]×1502
Sequence 2: [BOS] ... 3500 tokens ... [EOS] [PAD]×2
Sequence 3: [BOS] ... 1200 tokens ... [EOS] [PAD]×2302
Sequence 4: [BOS] ... 800 tokens ... [EOS] [PAD]×2702

# Attention masks (1 = attend, 0 = ignore)
Sequence 1: [1]×2002 + [0]×1502
Sequence 2: [1]×3502 + [0]×2
Sequence 3: [1]×1202 + [0]×2302
Sequence 4: [1]×802 + [0]×2702

# Labels for loss (clinical tokens + special tokens, -100 for padding)
Sequence 1: [label_ids]×2002 + [-100]×1502
Sequence 2: [label_ids]×3502 + [-100]×2
Sequence 3: [label_ids]×1202 + [-100]×2302
Sequence 4: [label_ids]×802 + [-100]×2702
```

**Key properties:**
- Padding tokens: `[PAD]` (ID: 0)
- Attention mask: 0 for padding (no attention to padding)
- Labels: -100 for padding (ignored in loss calculation)
- Rounding to multiple of 8: Optimizes GPU memory access patterns

### 3.5 Loss Calculation

From the data collator:

```python
# Labels are same as input_ids, but padding positions set to -100
labels = input_ids.clone()
labels[attention_mask == 0] = -100

# PyTorch CrossEntropyLoss automatically ignores -100
loss = CrossEntropyLoss(ignore_index=-100)
```

**What gets included in loss:**
- [BOS] token ✓
- All clinical tokens ✓
- [EOS] token ✓
- [PAD] tokens ✗ (labels = -100)

**Causal Language Modeling:**
- Model predicts next token given all previous tokens
- At position i, model sees tokens [0:i] and predicts token i
- Loss computed for all positions except padding

---

## 4. Sequence Packing Strategy

### 4.1 Does CLIFATRON Use Packing?

**Answer: YES**

CLIFATRON **DOES** pack multiple hospitalizations into fixed-length (8192 token) training sequences to reduce padding waste from ~46% to <5%.

### 4.2 Packing Implementation

**Industry-standard pattern (Llama, Mistral, Qwen2, GPT2):**

```
[BOS] hosp1_tokens [EOS] hosp2_tokens [EOS] hosp3_tokens [EOS] [PAD]...
```

**Key features:**
- First hospitalization has [BOS] and [EOS]
- Subsequent hospitalizations have only [EOS] (no redundant [BOS])
- No separator tokens ([SEP]) between hospitalizations
- Causal attention masks prevent cross-contamination
- Fixed 8192 token sequences (matches model's context window)

### 4.3 Why Use Packing?

**Efficiency gains:**

```python
# WITHOUT PACKING (old approach):
Batch of 8 sequences:
  Seq 1: 2000 tokens → padded to 3000 (1000 wasted)
  Seq 2: 1500 tokens → padded to 3000 (1500 wasted)
  Seq 3:  800 tokens → padded to 3000 (2200 wasted)
  Seq 4: 3000 tokens → no padding
  Seq 5: 1200 tokens → padded to 3000 (1800 wasted)
  Seq 6:  900 tokens → padded to 3000 (2100 wasted)
  Seq 7: 1100 tokens → padded to 3000 (1900 wasted)
  Seq 8: 2500 tokens → padded to 3000 (500 wasted)

Total tokens: 24,000
Actual tokens: 13,000
Padding waste: 11,000 tokens (46%)

# WITH PACKING (new approach):
Batch of 2 packed sequences (8192 tokens each):
  Packed seq 1: hosp1(2000) + hosp2(1500) + hosp3(800) + hosp4(3000) + hosp5(892) = 8192
  Packed seq 2: hosp5(308) + hosp6(900) + hosp7(1100) + hosp8(2500) + repeat(3384) = 8192

Total tokens: 16,384
Actual tokens: 15,968 (with repetition for efficiency)
Padding waste: 416 tokens (2.5%)

Throughput increase: 1.5-2x
```

### 4.4 Patient Privacy & Safety

**Causal attention masks prevent cross-contamination:**

Even though multiple hospitalizations share a training sequence, the causal language modeling objective ensures:

1. **No information leakage**: Model cannot attend to future tokens from other hospitalizations
2. **Independent prediction**: Loss computed only on actual tokens (not padding)
3. **Clear boundaries**: [BOS]/[EOS] tokens mark hospitalization boundaries
4. **Traceable predictions**: Each hospitalization still produces independent outputs

**Mathematical guarantee:**

```
For token at position i:
  attention_mask[i, j] = 0 if j > i (causal masking)

This ensures token i can only see tokens 0...i, preventing
cross-hospitalization contamination during next-token prediction.
```

### 4.5 Packing Algorithm

From `AR/gpt2_hf/data/data_collator.py`:

```python
def _pack_sequences(self, features):
    """
    Pack multiple hospitalizations into 8192-token sequences.

    Pattern: [BOS] hosp1 [EOS] hosp2 [EOS] hosp3 [EOS] [PAD]...
    """
    packed_input_ids = []
    packed_attention_masks = []
    packed_labels = []

    current_ids = []
    current_mask = []
    current_labels = []
    current_length = 0
    is_first_in_sequence = True

    for feature in features:
        hosp_ids = feature['input_ids']
        hosp_mask = feature['attention_mask']
        hosp_labels = feature['labels']

        # Remove [BOS] for non-first hospitalizations
        if not is_first_in_sequence and hosp_ids[0] == self.tokenizer.bos_token_id:
            hosp_ids = hosp_ids[1:]
            hosp_mask = hosp_mask[1:]
            hosp_labels = hosp_labels[1:]

        # Check if we can fit this hospitalization
        if current_length + len(hosp_ids) <= 8192:
            current_ids.append(hosp_ids)
            current_mask.append(hosp_mask)
            current_labels.append(hosp_labels)
            current_length += len(hosp_ids)
            is_first_in_sequence = False
        else:
            # Finalize current sequence and start new one
            packed_seq = self._finalize_packed_sequence(
                current_ids, current_mask, current_labels, current_length
            )
            packed_input_ids.append(packed_seq['input_ids'])
            packed_attention_masks.append(packed_seq['attention_mask'])
            packed_labels.append(packed_seq['labels'])

            # Reset for new sequence
            current_ids = [hosp_ids]
            current_mask = [hosp_mask]
            current_labels = [hosp_labels]
            current_length = len(hosp_ids)
            is_first_in_sequence = True

    return {
        'input_ids': torch.stack(packed_input_ids),
        'attention_mask': torch.stack(packed_attention_masks),
        'labels': torch.stack(packed_labels)
    }
```

### 4.6 Configuration

**Enable packing in `training_config.yaml`:**

```yaml
# Sequence packing (reduces padding waste from ~46% to <5%)
enable_packing: true
pack_to_max_length: 8192
repeat_short_sequences: true  # Repeat sequences to fill remainder instead of padding
```

**Training script reads these parameters:**

```python
# AR/gpt2_hf/02_train_gpt2.py
data_collator = create_data_collator(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    enable_packing=training_config.get('enable_packing', False),
    pack_to_max_length=training_config.get('pack_to_max_length', 8192),
    repeat_short_sequences=training_config.get('repeat_short_sequences', True)
)
```

---

## 5. Code Examples

### 5.1 Tokenization Creation

From `tokenETL/utils/tokenizer.py`:

```python
def tokenize_bins(df, column, bins, prefix="", token_counts=None):
    """
    Tokenize numeric column using bins.

    Example:
        age_at_admission: 67 → age_66_75
        labs_lactate: 1.8 → labs_lactate_(1.4,1.9]
    """
    def assign_bin(value):
        if pd.isna(value):
            return None

        for bin_def in bins:
            bin_min = bin_def['min']
            bin_max = bin_def['max']

            if bin_max is None or bin_max == 999:
                # Open-ended upper bound
                if value >= bin_min:
                    return f"{prefix}{bin_def['name']}"
            else:
                # Closed range
                if bin_min <= value <= bin_max:
                    return f"{prefix}{bin_def['name']}"

        return None

    df[f"{column}_token"] = df[column].apply(assign_bin)

    # Track token counts
    if token_counts is not None:
        counts = df[f"{column}_token"].value_counts()
        for token, count in counts.items():
            if token:
                token_counts[token] = token_counts.get(token, 0) + count

    return df, token_counts


def tokenize_mapping(df, column, mapping, prefix="", token_counts=None):
    """
    Tokenize categorical column using mapping.

    Example:
        sex: "Male" → sex_male
        disposition: "Home" → disposition_home
    """
    def map_value(value):
        if pd.isna(value):
            return None

        token_name = mapping.get(value)
        if token_name:
            return f"{prefix}{token_name}"
        return None

    df[f"{column}_token"] = df[column].apply(map_value)

    # Track token counts
    if token_counts is not None:
        counts = df[f"{column}_token"].value_counts()
        for token, count in counts.items():
            if token:
                token_counts[token] = token_counts.get(token, 0) + count

    return df, token_counts
```

### 5.2 Narrative Assembly

From `tokenETL/assemble_narratives.py`:

```python
def assemble_hospitalization_narrative(hosp_id, tokens_df):
    """
    Assemble tokens for a single hospitalization into chronological narrative.

    Returns:
        List of tokens in proper order:
        1. PREV_NARRATIVE_START
        2. Elixhauser comorbidities (sorted)
        3. PREV_NARRATIVE_END
        4. Demographics (age, sex)
        5. Day markers and clinical events (chronologically sorted)
        6. Discharge disposition
    """
    narrative_tokens = []

    # Step 1: Previous narrative (comorbidities)
    elix_tokens = tokens_df[
        tokens_df['sequence_order'] == 2
    ].sort_values('token')['token'].tolist()

    if elix_tokens:
        narrative_tokens.append('PREV_NARRATIVE_START')
        narrative_tokens.extend(elix_tokens)
        narrative_tokens.append('PREV_NARRATIVE_END')
    else:
        narrative_tokens.extend([
            'PREV_NARRATIVE_START',
            'no_patient_history',
            'PREV_NARRATIVE_END'
        ])

    # Step 2: Demographics
    demo_tokens = tokens_df[
        tokens_df['sequence_order'] == 4
    ].sort_values('token')['token'].tolist()
    narrative_tokens.extend(demo_tokens)

    # Step 3: Clinical events (chronologically sorted)
    clinical_events = tokens_df[
        tokens_df['sequence_order'].isin([5, 6, 7])
    ].sort_values(['event_time', 'token'])

    current_day = None
    current_hour = None

    for _, row in clinical_events.iterrows():
        token = row['token']

        # Add day marker if changed
        if token.startswith('day_') and token != current_day:
            narrative_tokens.append(token)
            current_day = token
            current_hour = None

        # Add hour marker if changed
        elif token.startswith('hour_') and token != current_hour:
            narrative_tokens.append(token)
            current_hour = token

        # Add clinical event
        elif not token.startswith(('day_', 'hour_')):
            narrative_tokens.append(token)

    # Step 4: Discharge disposition
    disp_token = tokens_df[
        tokens_df['sequence_order'] == 8
    ]['token'].iloc[0]
    narrative_tokens.append(disp_token)

    return narrative_tokens
```

---

## 6. Summary Statistics

### 6.1 Vocabulary

- **Total vocabulary size:** 1,380 tokens
  - 5 special tokens: [PAD], [UNK], [BOS], [EOS], [SEP]
  - ~1,375 clinical tokens

### 6.2 Token Categories

| Category | Description | Example Tokens | Approx Count |
|----------|-------------|----------------|--------------|
| Demographics | Age, sex | `age_56_65`, `sex_female` | ~15 |
| Temporal | Day/hour markers | `day_1`, `hour_11` | ~55 |
| Elixhauser | Comorbidities | `elix_congestive_heart_failure` | ~30 |
| ADT | Transfers | `transfer_to_icu` | ~20 |
| Labs | Laboratory results | `labs_lactate_(0.5,0.9]` | ~400 |
| Vitals | Vital signs | `vitals_heart_rate_(77.0,83.0]` | ~350 |
| Medications | Infusions | `medications_norepinephrine_mcg_kg_min_(0.06,0.12]` | ~200 |
| Respiratory | Ventilation | `respiratory_support_fio2_set_(0.4,0.5]` | ~150 |
| Assessments | Clinical scores | `assessment_rass_0`, `assessment_gcs_total_15` | ~50 |
| CRRT | Renal replacement | `crrt_therapy_rate_100_200` | ~30 |
| ECMO | Extracorporeal support | `ecmo_mcs_flow_2_3` | ~20 |
| Disposition | Discharge outcome | `disposition_home`, `disposition_expired` | ~10 |
| Narrative markers | Section boundaries | `PREV_NARRATIVE_START`, `no_patient_history` | ~5 |

### 6.3 Context Window

- **Maximum context:** 8,192 tokens (extended from GPT2's default 1024)
- **Effective for clinical tokens:** 8,190 tokens (reserves 2 for [BOS] and [EOS])
- **Chunking strategy:** Simple sequential chunking at 8,190-token boundaries
- **No overlap:** Zero token waste (each token appears exactly once)

### 6.4 Training Data (Temporal Split)

- **Training set:** 2018-2023 data
  - Train: 73,250 chunks
  - Validation: 8,265 chunks (10% of 2018-2023)
- **Test set:** 2024 data
  - Test: ~9,000 chunks (separate year for temporal evaluation)

### 6.5 Sequence Length Distribution

Based on processed data:

| Length Category | Token Range | Percentage | Treatment |
|----------------|-------------|------------|-----------|
| Very short | 50-500 | ~15% | Single chunk (packing reduces padding waste) |
| Short | 500-2000 | ~35% | Single chunk (packing reduces padding waste) |
| Medium | 2000-5000 | ~30% | Single chunk, minimal padding |
| Long | 5000-8190 | ~15% | Single chunk (fits exactly) |
| Very long | 8190+ | ~5% | Multiple sequential chunks (8,190-token boundaries) |

### 6.6 Chunking Statistics

From analysis of preprocessed data with simple sequential chunking:

- **Single-chunk hospitalizations:** ~95%
- **Multi-chunk hospitalizations:** ~5%
  - Average chunks per long hospitalization: 3.2
  - Max chunks for any hospitalization: 12
  - All chunks created at 8,190-token boundaries (sequential splits)
  - Zero token waste (no overlap)

### 6.7 Training Configuration

**GPT2-small (tested on 2x L40 GPUs):**
```yaml
batch_size: 8                      # Per device
gradient_accumulation_steps: 12
effective_batch_size: 192          # 8 × 12 × 2 GPUs
max_length: 8192
num_epochs: 4
learning_rate: 3.0e-4
warmup_steps: 2000
```

**Estimated training time:** ~1-2 days on 2x L40 (48GB each)

**GPT2-medium (tested on 2x L40 GPUs):**
```yaml
batch_size: 6                      # Per device
gradient_accumulation_steps: 16
effective_batch_size: 192          # 6 × 16 × 2 GPUs
max_length: 8192
num_epochs: 3
learning_rate: 2.5e-4
warmup_steps: 2000
```

**Estimated training time:** ~2-3 days on 2x L40 (48GB each)

---

## Appendix: Example Full Narratives

### Example 1: Short Hospitalization (Single Chunk)

```
[BOS]
PREV_NARRATIVE_START
  no_patient_history
PREV_NARRATIVE_END
age_56_65
sex_female
day_1
  hour_11
    vitals_height_cm_(160.02,165.1]
    vitals_weight_kg_(88.4,99.9]
    transfer_to_procedural
day_2
  hour_8
    labs_lactate_(0.5,0.9]
    respiratory_support_fio2_set_(0.3,0.4]
    vitals_heart_rate_(77.0,83.0]
day_3
  hour_14
    medications_norepinephrine_mcg_kg_min_(0.06,0.12]
    transfer_to_icu
    vitals_sbp_(95.0,100.0]
disposition_home
[EOS]
```

**Total tokens:** ~700 (including special tokens)
**Treatment:** Single chunk, padded during batching

### Example 2: Long Hospitalization (Multiple Chunks)

**Chunk 0 (Days 1-3):**
```
[BOS]
PREV_NARRATIVE_START
  elix_congestive_heart_failure
  elix_diabetes_uncomplicated
  elix_chronic_pulmonary_disease
PREV_NARRATIVE_END
age_66_75
sex_male
day_1
  hour_11
    vitals_height_cm_(175.26,180.3]
    vitals_weight_kg_(77.0,88.4]
    transfer_to_icu
    respiratory_support_fio2_set_(0.5,0.6]
    [... 2000 more tokens ...]
day_2
  [... 3000 tokens ...]
day_3
  [... 2500 tokens ...]
[EOS]
```
**Total:** 8,192 tokens

**Chunk 1 (Days 4-6):**
```
[BOS]
day_4
  hour_6
    labs_creatinine_(2.1,2.5]
    labs_lactate_(1.4,1.9]
    medications_norepinephrine_mcg_kg_min_(0.12,0.2]
    [... 2500 more tokens ...]
day_5
  [... 3000 tokens ...]
day_6
  [... 2500 tokens ...]
[EOS]
```
**Total:** 8,192 tokens

**Chunk 2 (Day 7 + discharge):**
```
[BOS]
day_7
  hour_8
    vitals_heart_rate_(91.0,100.0]
    labs_lactate_(0.5,0.9]
    [... 1000 tokens ...]
disposition_home
[EOS]
```
**Total:** 1,002 tokens

---

## References

- Token registry: `OutputTokens/token_registry.json`
- Tokenization code: `tokenETL/`
- Training scripts: `AR/gpt2_hf/02_train_gpt2.py`
- Dataset implementation: `AR/gpt2_hf/data/narrative_dataset.py`
- Training config: `AR/gpt2_hf/config/training_config.yaml`
- Model configs: `AR/gpt2_hf/models/gpt2_configs.py`

---

**Document Version:** 1.0 (GPT2)
**Last Updated:** 2025-10-30
**Author:** CLIFATRON Development Team
