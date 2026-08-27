#!/usr/bin/env python3
"""
test_packing.py - Test packing on small sample to verify 2D masks work

Quick test before full 4-hour packing run.
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
print("TESTING PACKING WITH 100 SAMPLES")
print("=" * 80)
print()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = ClinicalTokenizer.from_pretrained("AR/qwen2/tokenizer/clinical_tokenizer")
print(f"  ✓ Tokenizer loaded, vocab size: {len(tokenizer)}")
print()

# Load small subset of train data
print("Loading 100 train samples...")
train_dataset = load_hospitalization_dataset(
    config_path="clif_config.json",
    split='train',
    tokenizer=tokenizer,
    max_length=8192,
)

# Take only first 100 samples
class SmallDataset:
    def __init__(self, dataset, n=100):
        self.dataset = dataset
        self.n = min(n, len(dataset))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.dataset[idx]

small_train = SmallDataset(train_dataset, 100)
print(f"  ✓ Using {len(small_train)} samples for test")
print()

# Pack with parallel workers
output_dir = Path("models/qwen2/preprocessed/test_packed")
output_dir.mkdir(parents=True, exist_ok=True)

pack_and_save(
    dataset=small_train,
    output_path=output_dir / "train_packed.parquet",
    max_seq_length=8192,
    pad_token_id=tokenizer.pad_token_id,
    sep_token_id=tokenizer.sep_token_id,
    num_sep_tokens=1,
    num_workers=8,  # Use 8 workers for test
)

print()
print("=" * 80)
print("TEST PACKING COMPLETE")
print("=" * 80)
print(f"Test packed file: {output_dir / 'train_packed.parquet'}")
print()
print("Next: Run training test to verify 2D masks work")
print("  uv run AR/qwen2/train_sft.py --config AR/qwen2/config/training_config.yaml --model-size 0.5b --packed-data-dir models/qwen2/preprocessed/test_packed --max-steps 10")
