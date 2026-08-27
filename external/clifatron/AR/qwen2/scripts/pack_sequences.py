#!/usr/bin/env python3
"""
pack_sequences.py - Pre-pack hospitalization sequences offline

Generates packed sequences and saves to parquet for efficient training.
Each packed sequence is exactly max_seq_length tokens with document isolation.
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict
import polars as pl
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count

# Add project root and AR directories to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'AR' / 'qwen2'))

from tokenizer.clinical_tokenizer import ClinicalTokenizer
from data.hospitalization_dataset import load_hospitalization_dataset


class OfflinePacker:
    """Packs pre-tokenized hospitalizations into fixed-length sequences with document isolation."""

    def __init__(
        self,
        max_seq_length: int,
        pad_token_id: int,
        sep_token_id: int,
        num_sep_tokens: int = 1,
    ):
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
        self.num_sep_tokens = num_sep_tokens

        # Buffers for packing
        self.buffer_input_ids = []
        self.buffer_attention_mask = []
        self.buffer_labels = []
        self.buffer_doc_boundaries = []  # Track document boundaries for 2D masks

        # Track packed sequences
        self.packed_sequences = []

    def add_hospitalization(self, example: Dict):
        """Add a hospitalization to the buffer and pack when ready."""
        # Track document start position
        doc_start = len(self.buffer_input_ids)

        # Add document content
        self.buffer_input_ids.extend(example["input_ids"])
        self.buffer_attention_mask.extend(example["attention_mask"])
        self.buffer_labels.extend(example["labels"])

        # Track document end position (before SEP separator)
        doc_end = len(self.buffer_input_ids)
        self.buffer_doc_boundaries.append((doc_start, doc_end))

        # Add separator SEP tokens (typically just 1 [SEP] token)
        separator_ids = [self.sep_token_id] * self.num_sep_tokens
        self.buffer_input_ids.extend(separator_ids)
        self.buffer_attention_mask.extend([1] * self.num_sep_tokens)
        self.buffer_labels.extend([-100] * self.num_sep_tokens)  # Don't learn from SEP

        # Pack if buffer is large enough
        while len(self.buffer_input_ids) >= self.max_seq_length:
            self._pack_sequence()

    def _pack_sequence(self):
        """Extract a full sequence from buffer with 1D document IDs."""
        packed_input_ids = self.buffer_input_ids[:self.max_seq_length]
        packed_attention_mask = self.buffer_attention_mask[:self.max_seq_length]
        packed_labels = self.buffer_labels[:self.max_seq_length]

        # Get document boundaries for this sequence
        current_boundaries = []
        for start, end in self.buffer_doc_boundaries:
            if end <= self.max_seq_length:
                # Document fully in this sequence
                current_boundaries.append((start, end))
            elif start < self.max_seq_length < end:
                # Document split across sequences
                current_boundaries.append((start, self.max_seq_length))

        # Create 1D document ID array (much more memory efficient than 2D mask)
        document_ids = self._create_document_ids(
            seq_length=self.max_seq_length,
            doc_boundaries=current_boundaries
        )

        # Keep overflow for next sequence
        self.buffer_input_ids = self.buffer_input_ids[self.max_seq_length:]
        self.buffer_attention_mask = self.buffer_attention_mask[self.max_seq_length:]
        self.buffer_labels = self.buffer_labels[self.max_seq_length:]

        # Update document boundaries for next sequence
        new_boundaries = []
        for start, end in self.buffer_doc_boundaries:
            if end <= self.max_seq_length:
                # Document fully consumed
                continue
            elif start < self.max_seq_length < end:
                # Document split - add second part
                new_boundaries.append((0, end - self.max_seq_length))
            else:
                # Document entirely in buffer - shift positions
                new_boundaries.append((start - self.max_seq_length, end - self.max_seq_length))
        self.buffer_doc_boundaries = new_boundaries

        # Store packed sequence with 1D document IDs
        self.packed_sequences.append({
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "labels": packed_labels,
            "document_ids": document_ids,  # 1D array: which document each token belongs to
        })

    def flush(self):
        """Pad and pack remaining buffer content with 1D document IDs."""
        if len(self.buffer_input_ids) > 0:
            pad_length = self.max_seq_length - len(self.buffer_input_ids)

            # Pad to max_seq_length
            padded_input_ids = self.buffer_input_ids + [self.pad_token_id] * pad_length
            padded_attention_mask = self.buffer_attention_mask + [0] * pad_length
            padded_labels = self.buffer_labels + [-100] * pad_length

            # Create 1D document ID array
            document_ids = self._create_document_ids(
                seq_length=self.max_seq_length,
                doc_boundaries=self.buffer_doc_boundaries
            )

            self.packed_sequences.append({
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "labels": padded_labels,
                "document_ids": document_ids,  # 1D array: which document each token belongs to
            })

            # Clear buffer
            self.buffer_input_ids = []
            self.buffer_attention_mask = []
            self.buffer_labels = []
            self.buffer_doc_boundaries = []

    def _create_document_ids(
        self,
        seq_length: int,
        doc_boundaries: List[tuple]
    ) -> List[int]:
        """
        Create 1D document ID array - assigns each token to a document ID.

        Much more memory efficient than 2D attention masks:
        - 1D: 8192 ints = 32KB per sequence
        - 2D: 8192×8192 ints = 268MB per sequence

        Args:
            seq_length: Length of the sequence
            doc_boundaries: List of (start, end) tuples for each document

        Returns:
            1D array of shape [seq_len] where each value is the document ID
            Padding tokens get document_id = -1
        """
        # Initialize all positions to -1 (padding/no document)
        document_ids = [-1] * seq_length

        # Assign document IDs based on boundaries
        for doc_id, (doc_start, doc_end) in enumerate(doc_boundaries):
            doc_start = max(0, doc_start)
            doc_end = min(seq_length, doc_end)

            for pos in range(doc_start, doc_end):
                document_ids[pos] = doc_id

        return document_ids

    def get_packed_sequences(self) -> List[Dict]:
        """Return all packed sequences."""
        return self.packed_sequences


def _pack_chunk(args):
    """Helper function to pack a chunk of examples in parallel."""
    examples_chunk, max_seq_length, pad_token_id, sep_token_id, num_sep_tokens = args

    packer = OfflinePacker(
        max_seq_length=max_seq_length,
        pad_token_id=pad_token_id,
        sep_token_id=sep_token_id,
        num_sep_tokens=num_sep_tokens,
    )

    for example in examples_chunk:
        packer.add_hospitalization(example)

    packer.flush()
    return packer.get_packed_sequences()


def pack_and_save(
    dataset,
    output_path: Path,
    max_seq_length: int,
    pad_token_id: int,
    sep_token_id: int,
    num_sep_tokens: int = 1,
    num_workers: int = 1,
):
    """Pack sequences and save to parquet."""
    print(f"Packing {len(dataset)} hospitalizations...")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  SEP tokens between documents: {num_sep_tokens}")
    print(f"  Workers: {num_workers}")

    if num_workers > 1:
        # Parallel pre-tokenization (dataset already tokenized, so just fetch in parallel)
        print(f"  Pre-loading data with {num_workers} workers...")
        with Pool(num_workers) as pool:
            examples = list(tqdm(
                pool.imap(dataset.__getitem__, range(len(dataset)), chunksize=100),
                total=len(dataset),
                desc="Loading"
            ))
    else:
        # Sequential loading
        examples = [dataset[i] for i in range(len(dataset))]

    print(f"  Packing {len(examples)} sequences...")

    if num_workers > 1:
        # Parallel packing: split examples into chunks
        chunk_size = len(examples) // num_workers
        chunks = []
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_workers - 1 else len(examples)
            chunk = examples[start_idx:end_idx]
            chunks.append((chunk, max_seq_length, pad_token_id, sep_token_id, num_sep_tokens))

        print(f"  Parallel packing with {num_workers} workers...")
        with Pool(num_workers) as pool:
            chunk_results = list(tqdm(
                pool.imap(_pack_chunk, chunks),
                total=len(chunks),
                desc="Packing chunks"
            ))

        # Combine results from all chunks
        packed_sequences = []
        for chunk_result in chunk_results:
            packed_sequences.extend(chunk_result)
    else:
        # Sequential packing
        packer = OfflinePacker(
            max_seq_length=max_seq_length,
            pad_token_id=pad_token_id,
            sep_token_id=sep_token_id,
            num_sep_tokens=num_sep_tokens,
        )

        for example in tqdm(examples, desc="Packing"):
            packer.add_hospitalization(example)

        packer.flush()
        packed_sequences = packer.get_packed_sequences()

    print(f"  ✓ Created {len(packed_sequences)} packed sequences")

    # Convert to polars DataFrame
    df = pl.DataFrame({
        "input_ids": [seq["input_ids"] for seq in packed_sequences],
        "attention_mask": [seq["attention_mask"] for seq in packed_sequences],
        "labels": [seq["labels"] for seq in packed_sequences],
        "document_ids": [seq["document_ids"] for seq in packed_sequences],  # 1D document IDs (memory efficient)
    })

    # Save to parquet
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"  ✓ Saved {len(packed_sequences)} sequences")

    # Print statistics
    total_tokens = len(packed_sequences) * max_seq_length
    original_tokens = sum(len(dataset[i]["input_ids"]) for i in range(len(dataset)))
    padding_pct = (1 - original_tokens / total_tokens) * 100

    print(f"\nPacking Statistics:")
    print(f"  Original tokens: {original_tokens:,}")
    print(f"  Packed tokens: {total_tokens:,}")
    print(f"  Padding: {padding_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Pre-pack hospitalization sequences")
    parser.add_argument("--clif-config", type=str, default="clif_config.json",
                        help="Path to CLIF config")
    parser.add_argument("--tokenizer-path", type=str,
                        default="AR/qwen2/tokenizer/clinical_tokenizer",
                        help="Path to tokenizer")
    parser.add_argument("--output-dir", type=str,
                        default="models/qwen2/preprocessed/packed_temporal_len8192",
                        help="Output directory for packed sequences")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Maximum sequence length")
    parser.add_argument("--num-sep-tokens", type=int, default=1,
                        help="Number of SEP tokens between documents (default: 1)")
    parser.add_argument("--num-workers", type=int, default=cpu_count(),
                        help=f"Number of parallel workers (default: {cpu_count()})")

    args = parser.parse_args()

    print("=" * 80)
    print("OFFLINE SEQUENCE PACKING")
    print("=" * 80)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ClinicalTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  ✓ Tokenizer loaded, vocab size: {len(tokenizer)}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pack train split
    print("Loading train dataset...")
    train_dataset = load_hospitalization_dataset(
        config_path=args.clif_config,
        split='train',
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(f"  ✓ Loaded {len(train_dataset)} hospitalizations")
    print()

    pack_and_save(
        dataset=train_dataset,
        output_path=output_dir / "train_packed.parquet",
        max_seq_length=args.max_length,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        num_sep_tokens=args.num_sep_tokens,
        num_workers=args.num_workers,
    )
    print()

    # Pack val split
    print("Loading val dataset...")
    val_dataset = load_hospitalization_dataset(
        config_path=args.clif_config,
        split='val',
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(f"  ✓ Loaded {len(val_dataset)} hospitalizations")
    print()

    pack_and_save(
        dataset=val_dataset,
        output_path=output_dir / "val_packed.parquet",
        max_seq_length=args.max_length,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        num_sep_tokens=args.num_sep_tokens,
        num_workers=args.num_workers,
    )
    print()

    # Pack test split
    print("Loading test dataset...")
    test_dataset = load_hospitalization_dataset(
        config_path=args.clif_config,
        split='test',
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(f"  ✓ Loaded {len(test_dataset)} hospitalizations")
    print()

    pack_and_save(
        dataset=test_dataset,
        output_path=output_dir / "test_packed.parquet",
        max_seq_length=args.max_length,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        num_sep_tokens=args.num_sep_tokens,
        num_workers=args.num_workers,
    )
    print()

    print("=" * 80)
    print("PACKING COMPLETE")
    print("=" * 80)
    print(f"Packed sequences saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
