#!/usr/bin/env python3
"""
pack_test_only.py - Pack only the test split

Quick script to pack test set without reprocessing train/val.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'AR' / 'qwen2'))

from tokenizer.clinical_tokenizer import ClinicalTokenizer
from data.hospitalization_dataset import load_hospitalization_dataset
from pack_sequences import pack_and_save

print("=" * 80)
print("PACKING TEST SPLIT ONLY")
print("=" * 80)
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = ClinicalTokenizer.from_pretrained("AR/qwen2/tokenizer/clinical_tokenizer")
print(f"  ✓ Tokenizer loaded, vocab size: {len(tokenizer)}")
print()

# Output directory
output_dir = Path("models/qwen2/preprocessed/packed_temporal_len8192")
output_dir.mkdir(parents=True, exist_ok=True)

# Pack test split only
print("Loading test dataset...")
test_dataset = load_hospitalization_dataset(
    config_path="clif_config.json",
    split='test',
    tokenizer=tokenizer,
    max_length=8192,
)
print(f"  ✓ Loaded {len(test_dataset)} hospitalizations")
print()

pack_and_save(
    dataset=test_dataset,
    output_path=output_dir / "test_packed.parquet",
    max_seq_length=8192,
    pad_token_id=tokenizer.pad_token_id,
    sep_token_id=tokenizer.sep_token_id,
    num_sep_tokens=1,
    num_workers=128,  # Use all CPU cores
)
print()

print("=" * 80)
print("TEST SET PACKING COMPLETE")
print("=" * 80)
print(f"Test packed sequences saved to: {output_dir / 'test_packed.parquet'}")
print()
