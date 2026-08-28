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


PACKED_SCHEMA_VERSION = "2.0.0"


class OfflinePacker:
    """Packs pre-tokenized hospitalizations into fixed-length sequences with document isolation.

    v2 schema preserves episode keys and source-span metadata so continuation
    segments across packed rows remain traceable for TTE target construction.
    """

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

        self.buffer_input_ids = []
        self.buffer_attention_mask = []
        self.buffer_labels = []
        self.buffer_doc_boundaries = []
        self.buffer_episode_keys = []
        self.buffer_source_spans = []

        self.packed_sequences = []

    def add_hospitalization(self, example: Dict):
        """Add a hospitalization to the buffer and pack when ready."""
        doc_start = len(self.buffer_input_ids)

        self.buffer_input_ids.extend(example["input_ids"])
        self.buffer_attention_mask.extend(example["attention_mask"])
        self.buffer_labels.extend(example["labels"])

        doc_end = len(self.buffer_input_ids)
        self.buffer_doc_boundaries.append((doc_start, doc_end))
        episode_key = example.get("episode_key", f"doc-{len(self.buffer_doc_boundaries)}")
        source_start = example.get("source_start", 0)
        source_end = example.get("source_end", len(example["input_ids"]))
        self.buffer_episode_keys.append(episode_key)
        self.buffer_source_spans.append((source_start, source_end))

        separator_ids = [self.sep_token_id] * self.num_sep_tokens
        self.buffer_input_ids.extend(separator_ids)
        self.buffer_attention_mask.extend([1] * self.num_sep_tokens)
        self.buffer_labels.extend([-100] * self.num_sep_tokens)

        while len(self.buffer_input_ids) >= self.max_seq_length:
            self._pack_sequence()

    def _pack_sequence(self):
        """Extract a full sequence from buffer with 1D document IDs and segment metadata."""
        packed_input_ids = self.buffer_input_ids[:self.max_seq_length]
        packed_attention_mask = self.buffer_attention_mask[:self.max_seq_length]
        packed_labels = self.buffer_labels[:self.max_seq_length]

        current_boundaries = []
        for start, end in self.buffer_doc_boundaries:
            if end <= self.max_seq_length:
                current_boundaries.append((start, end))
            elif start < self.max_seq_length < end:
                current_boundaries.append((start, self.max_seq_length))

        document_ids = self._create_document_ids(
            seq_length=self.max_seq_length,
            doc_boundaries=current_boundaries
        )
        segments = self._segment_manifest(current_boundaries)

        self.buffer_input_ids = self.buffer_input_ids[self.max_seq_length:]
        self.buffer_attention_mask = self.buffer_attention_mask[self.max_seq_length:]
        self.buffer_labels = self.buffer_labels[self.max_seq_length:]

        new_boundaries = []
        new_episode_keys = []
        new_source_spans = []
        for i, (start, end) in enumerate(self.buffer_doc_boundaries):
            if end <= self.max_seq_length:
                continue
            elif start < self.max_seq_length < end:
                new_boundaries.append((0, end - self.max_seq_length))
                new_episode_keys.append(self.buffer_episode_keys[i])
                key = self.buffer_episode_keys[i]
                orig_start, orig_end = self.buffer_source_spans[i]
                new_source_spans.append((orig_start + (self.max_seq_length - start), orig_end))
            else:
                new_boundaries.append((start - self.max_seq_length, end - self.max_seq_length))
                new_episode_keys.append(self.buffer_episode_keys[i])
                new_source_spans.append(self.buffer_source_spans[i])
        self.buffer_doc_boundaries = new_boundaries
        self.buffer_episode_keys = new_episode_keys
        self.buffer_source_spans = new_source_spans

        self.packed_sequences.append({
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "labels": packed_labels,
            "document_ids": document_ids,
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "segments": segments,
        })

    def _segment_manifest(self, boundaries: list) -> dict[str, list]:
        episode_keys = []
        source_starts = []
        source_ends = []
        packed_starts = []
        packed_ends = []
        continuation_indices = []
        continues_from_prev = []
        continues_to_next = []
        episode_index: dict[str, int] = {}
        full_boundaries = self.buffer_doc_boundaries
        for i, (p_start, p_end) in enumerate(boundaries):
            key = self.buffer_episode_keys[i]
            orig_start, orig_end = self.buffer_source_spans[i]
            orig_full_start, orig_full_end = full_boundaries[i]
            episode_index.setdefault(key, 0)
            ci = episode_index[key]
            episode_index[key] = ci + 1
            continues_from = orig_start > orig_full_start
            continues_to = (orig_start + p_end - p_start) < orig_end
            episode_keys.append(key)
            source_starts.append(orig_start)
            source_ends.append(min(orig_end, orig_start + p_end - p_start))
            packed_starts.append(p_start)
            packed_ends.append(p_end)
            continuation_indices.append(ci)
            continues_from_prev.append(continues_from)
            continues_to_next.append(continues_to)
        return {
            "episode_keys": episode_keys,
            "segment_source_starts": source_starts,
            "segment_source_ends": source_ends,
            "segment_packed_starts": packed_starts,
            "segment_packed_ends": packed_ends,
            "segment_continuation_indices": continuation_indices,
            "segment_continues_from_previous": continues_from_prev,
            "segment_continues_to_next": continues_to_next,
        }

    def flush(self):
        """Pad and pack remaining buffer content with segment metadata."""
        if len(self.buffer_input_ids) > 0:
            pad_length = self.max_seq_length - len(self.buffer_input_ids)

            padded_input_ids = self.buffer_input_ids + [self.pad_token_id] * pad_length
            padded_attention_mask = self.buffer_attention_mask + [0] * pad_length
            padded_labels = self.buffer_labels + [-100] * pad_length

            document_ids = self._create_document_ids(
                seq_length=self.max_seq_length,
                doc_boundaries=self.buffer_doc_boundaries
            )
            segments = self._segment_manifest(self.buffer_doc_boundaries)

            self.packed_sequences.append({
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "labels": padded_labels,
                "document_ids": document_ids,
                "packed_schema_version": PACKED_SCHEMA_VERSION,
                "segments": segments,
            })

            self.buffer_input_ids = []
            self.buffer_attention_mask = []
            self.buffer_labels = []
            self.buffer_doc_boundaries = []
            self.buffer_episode_keys = []
            self.buffer_source_spans = []

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

    has_v2 = any("packed_schema_version" in seq for seq in packed_sequences)
    columns = {
        "input_ids": [seq["input_ids"] for seq in packed_sequences],
        "attention_mask": [seq["attention_mask"] for seq in packed_sequences],
        "labels": [seq["labels"] for seq in packed_sequences],
        "document_ids": [seq["document_ids"] for seq in packed_sequences],
    }
    if has_v2:
        for col in (
            "packed_schema_version",
            "episode_keys",
            "segment_source_starts",
            "segment_source_ends",
            "segment_packed_starts",
            "segment_packed_ends",
            "segment_continuation_indices",
            "segment_continues_from_previous",
            "segment_continues_to_next",
        ):
            segments = [seq.get("segments", {}) for seq in packed_sequences]
            columns[col] = [s.get(col, []) for s in segments]
    df = pl.DataFrame(columns)

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
