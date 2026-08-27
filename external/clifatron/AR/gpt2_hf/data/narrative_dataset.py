#!/usr/bin/env python3
"""
narrative_dataset.py - Dataset Loader for Clinical Narratives

Loads narrative sequences from parquet file with hospitalization isolation.

Key Features:
- Each hospitalization is kept separate (no cross-contamination)
- Hybrid chunking: splits at day boundaries when possible
- Falls back to sliding window if single day exceeds context limit
- Adds [BOS] and [EOS] tokens around each chunk
- Optional sequence packing to pack multiple hospitalizations into single sequences
"""

import os
import json
import warnings
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import polars as pl
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Import sequence packing utilities
from .sequence_packer import pack_sequences, create_document_attention_mask, PackedSequence


@dataclass
class NarrativeChunk:
    """Represents a chunk of a hospitalization narrative."""
    hospitalization_id: str
    tokens: List[str]
    chunk_index: int
    total_chunks: int
    is_complete_hospitalization: bool


class ClinicalNarrativeDataset(Dataset):
    """
    Dataset for clinical narratives with hospitalization isolation.

    Each example is a chunk of a single hospitalization, never mixing
    patients. Long hospitalizations are split using hybrid chunking:
    - Primary: Split at day boundaries (day_N markers)
    - Fallback: Sliding window with overlap if day exceeds max length

    Args:
        narrative_parquet_path: Path to narrative parquet file(s)
        tokenizer: ClinicalTokenizer instance
        max_length: Maximum chunk length (default: 8192)
        overlap_tokens: Number of overlapping tokens for sliding window (default: 819, ~10%)
        min_chunk_size: Minimum chunk size to keep (default: 50)
        split: Data split ('train', 'val', 'test')
        split_mode: Split strategy ('temporal' or 'random') (default: 'temporal')
        val_fraction: Fraction of data for validation (default: 0.1, only used for random mode)
        test_fraction: Fraction of data for test (default: 0.1, only used for random mode)
        train_val_fraction: For temporal mode, fraction of train_val data for training (default: 0.9)
        seed: Random seed for splitting (default: 42)

    Split Modes:
        - temporal: Uses pre-split parquet files from assemble_narratives.py
          - train/val: Loads from train_val_sequences.parquet (2018-2023 data)
          - test: Loads from test_sequences.parquet (2024 data)
        - random: Loads from narrative_sequences.parquet and splits randomly
    """

    def __init__(
        self,
        narrative_parquet_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 8192,
        overlap_tokens: int = 819,  # ~10% overlap
        min_chunk_size: int = 50,
        split: str = 'train',
        split_mode: str = 'temporal',
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        train_val_fraction: float = 0.9,
        seed: int = 42,
        pack_sequences: bool = True,  # Enable sequence packing
    ):
        self.narrative_parquet_path = narrative_parquet_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap_tokens = overlap_tokens
        self.min_chunk_size = min_chunk_size
        self.split = split
        self.split_mode = split_mode
        self.pack_sequences = pack_sequences

        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # Validate split_mode
        if split_mode not in ['temporal', 'random']:
            raise ValueError(f"Invalid split_mode: {split_mode}. Must be 'temporal' or 'random'")

        # Reserve tokens for special tokens
        # If packing: [BOS] + docs with [SEP] between + [EOS]
        # If not packing: [BOS] + doc + [EOS]
        self.effective_max_length = max_length - 2

        print(f"Loading narrative dataset ({split} split, {split_mode} mode)...")
        print(f"  Parquet: {narrative_parquet_path}")
        print(f"  Max length: {max_length} (effective: {self.effective_max_length})")
        print(f"  Overlap: {overlap_tokens} tokens")
        print(f"  Min chunk: {min_chunk_size} tokens")

        # Load and chunk narratives
        if split_mode == 'temporal':
            self.chunks = self._load_and_chunk_narratives_temporal(
                train_val_fraction=train_val_fraction,
                seed=seed
            )
        else:  # random
            self.chunks = self._load_and_chunk_narratives_random(
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed
            )

        print(f"  Loaded {len(self.chunks)} chunks for {split} split")

    def _load_and_chunk_narratives_temporal(
        self,
        train_val_fraction: float,
        seed: int
    ) -> List[NarrativeChunk]:
        """
        Load narratives from temporal split parquet files.

        For train/val: Loads from train_val_sequences.parquet (2018-2023)
        For test: Loads from test_sequences.parquet (2024)

        Args:
            train_val_fraction: Fraction of train_val data for training (rest for val)
            seed: Random seed for train/val split

        Returns:
            List of NarrativeChunk objects
        """
        # Determine which parquet file to load
        base_dir = os.path.dirname(self.narrative_parquet_path)

        if self.split in ['train', 'val']:
            # Load train_val_sequences.parquet (2018-2023 data)
            parquet_path = os.path.join(base_dir, 'train_val_sequences.parquet')
            print(f"  Loading temporal split: train_val_sequences.parquet (2018-2023)")
        else:  # test
            # Load test_sequences.parquet (2024 data)
            parquet_path = os.path.join(base_dir, 'test_sequences.parquet')
            print(f"  Loading temporal split: test_sequences.parquet (2024)")

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Temporal split file not found: {parquet_path}\n"
                f"Please run narrative assembly with temporal splits:\n"
                f"  uv run tokenETL/assemble_narratives.py"
            )

        # Load parquet file
        df = pl.read_parquet(parquet_path)

        # Get unique hospitalization IDs
        hosp_ids = df.select('hospitalization_id').unique().to_series().to_list()
        total_hosps = len(hosp_ids)

        print(f"  Total hospitalizations: {total_hosps:,}")

        # For train/val splits, further split the train_val data
        if self.split in ['train', 'val']:
            import numpy as np
            np.random.seed(seed)
            np.random.shuffle(hosp_ids)

            train_size = int(total_hosps * train_val_fraction)
            val_size = total_hosps - train_size

            if self.split == 'train':
                hosp_ids = hosp_ids[:train_size]
                print(f"  Train split: {len(hosp_ids):,} / {total_hosps:,} hospitalizations ({train_val_fraction*100:.1f}%)")
            else:  # val
                hosp_ids = hosp_ids[train_size:]
                print(f"  Val split: {len(hosp_ids):,} / {total_hosps:,} hospitalizations ({(1-train_val_fraction)*100:.1f}%)")
        else:  # test
            print(f"  Test split: {len(hosp_ids):,} hospitalizations (2024 data)")

        # Process each hospitalization
        if self.pack_sequences:
            # Collect all (hosp_id, tokens) pairs for packing
            all_hospitalizations = []
            for hosp_id in hosp_ids:
                # Get narrative for this hospitalization (preserve original parquet order)
                hosp_df = df.filter(pl.col('hospitalization_id') == hosp_id)

                # Extract tokens
                tokens = hosp_df.select('clif_sentence').to_series().to_list()
                all_hospitalizations.append((hosp_id, tokens))

            # Pack sequences using [SEP] token separator
            print(f"  Packing {len(all_hospitalizations)} hospitalizations into sequences...")
            packed_seqs = pack_sequences(
                all_hospitalizations,
                max_length=self.max_length,
                sep_token=self.tokenizer.sep_token
            )
            print(f"  Created {len(packed_seqs)} packed sequences")
            return packed_seqs
        else:
            # Original chunking logic (no packing)
            all_chunks = []
            for hosp_id in hosp_ids:
                # Get narrative for this hospitalization (preserve original parquet order)
                hosp_df = df.filter(pl.col('hospitalization_id') == hosp_id)

                # Extract tokens
                tokens = hosp_df.select('clif_sentence').to_series().to_list()

                # Chunk this hospitalization
                chunks = self._chunk_hospitalization(hosp_id, tokens)
                all_chunks.extend(chunks)

            return all_chunks

    def _load_and_chunk_narratives_random(
        self,
        val_fraction: float,
        test_fraction: float,
        seed: int
    ) -> List[NarrativeChunk]:
        """
        Load narratives from single parquet and split randomly.

        Args:
            val_fraction: Fraction for validation
            test_fraction: Fraction for test
            seed: Random seed

        Returns:
            List of NarrativeChunk objects
        """
        # Load parquet file using polars
        df = pl.read_parquet(self.narrative_parquet_path)

        # Get unique hospitalization IDs
        hosp_ids = df.select('hospitalization_id').unique().to_series().to_list()
        total_hosps = len(hosp_ids)

        print(f"  Total hospitalizations: {total_hosps:,}")

        # Split hospitalizations into train/val/test
        import numpy as np
        np.random.seed(seed)
        np.random.shuffle(hosp_ids)

        val_size = int(total_hosps * val_fraction)
        test_size = int(total_hosps * test_fraction)
        train_size = total_hosps - val_size - test_size

        if self.split == 'train':
            hosp_ids = hosp_ids[:train_size]
        elif self.split == 'val':
            hosp_ids = hosp_ids[train_size:train_size + val_size]
        else:  # test
            hosp_ids = hosp_ids[train_size + val_size:]

        print(f"  Split sizes - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")
        print(f"  Processing {len(hosp_ids):,} hospitalizations for {self.split}...")

        # Process each hospitalization
        if self.pack_sequences:
            # Collect all (hosp_id, tokens) pairs for packing
            all_hospitalizations = []
            for hosp_id in hosp_ids:
                # Get narrative for this hospitalization (preserve original parquet order)
                hosp_df = df.filter(pl.col('hospitalization_id') == hosp_id)

                # Extract tokens
                tokens = hosp_df.select('clif_sentence').to_series().to_list()
                all_hospitalizations.append((hosp_id, tokens))

            # Pack sequences using [SEP] token separator
            print(f"  Packing {len(all_hospitalizations)} hospitalizations into sequences...")
            packed_seqs = pack_sequences(
                all_hospitalizations,
                max_length=self.max_length,
                sep_token=self.tokenizer.sep_token
            )
            print(f"  Created {len(packed_seqs)} packed sequences")
            return packed_seqs
        else:
            # Original chunking logic (no packing)
            all_chunks = []
            for hosp_id in hosp_ids:
                # Get narrative for this hospitalization (preserve original parquet order)
                hosp_df = df.filter(pl.col('hospitalization_id') == hosp_id)

                # Extract tokens
                tokens = hosp_df.select('clif_sentence').to_series().to_list()

                # Chunk this hospitalization
                chunks = self._chunk_hospitalization(hosp_id, tokens)
                all_chunks.extend(chunks)

            return all_chunks

    def _chunk_hospitalization(
        self,
        hosp_id: str,
        tokens: List[str]
    ) -> List[NarrativeChunk]:
        """
        Chunk a single hospitalization using simple sequential chunking.

        Strategy:
        - Split at effective_max_length (8190 tokens)
        - No overlap, no waste
        - Truncated portions continue in next chunk
        - Each chunk gets [BOS] and [EOS] during tokenization

        Args:
            hosp_id: Hospitalization ID
            tokens: List of tokens for this hospitalization

        Returns:
            List of NarrativeChunk objects
        """
        # If entire hospitalization fits in max length, return as single chunk
        if len(tokens) <= self.effective_max_length:
            return [NarrativeChunk(
                hospitalization_id=hosp_id,
                tokens=tokens,
                chunk_index=0,
                total_chunks=1,
                is_complete_hospitalization=True
            )]

        # Otherwise, split into sequential chunks at effective_max_length boundary
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
                is_complete_hospitalization=False  # Split into multiple chunks
            ))

        return chunk_objects

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single chunk or packed sequence.

        Returns:
            Dictionary with:
                - input_ids: Token IDs with [BOS] and [EOS]
                - attention_mask: Attention mask (2D for packed sequences, 1D for unpacked)
                - labels: Same as input_ids (for causal LM)
                - hospitalization_id: Hospitalization ID(s) (for reference)
                - chunk_info: Chunk metadata
        """
        chunk = self.chunks[idx]

        # Check if this is a PackedSequence (multiple hospitalizations) or NarrativeChunk (single)
        if isinstance(chunk, PackedSequence):
            # Handle packed sequence with document-aware attention mask
            # Convert tokens to text (space-separated), filtering out None values
            text = " ".join(str(token) for token in chunk.tokens if token is not None)

            # Tokenize with special tokens
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,  # Adds [BOS] and [EOS]
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Extract tensors and squeeze batch dimension
            input_ids = encoding['input_ids'].squeeze(0)

            # Use standard 1D attention mask (1 for real tokens, 0 for padding)
            # SEP tokens separate hospitalizations, GPT2's causal attention handles document boundaries
            attention_mask = encoding['attention_mask'].squeeze(0)

            # For causal LM, labels are the same as input_ids
            labels = input_ids.clone()

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,  # 1D attention mask
                'labels': labels,
                'hospitalization_id': chunk.hospitalization_ids,  # List of IDs
                'chunk_info': {
                    'is_packed': True,
                    'num_documents': len(chunk.hospitalization_ids),
                    'num_tokens': len(chunk.tokens),
                    'document_boundaries': chunk.document_boundaries
                }
            }
        else:
            # Handle single hospitalization chunk (NarrativeChunk)
            # Convert tokens to text (space-separated), filtering out None values
            text = " ".join(str(token) for token in chunk.tokens if token is not None)

            # Tokenize with special tokens
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,  # Adds [BOS] and [EOS]
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Extract tensors and squeeze batch dimension
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            # For causal LM, labels are the same as input_ids
            labels = input_ids.clone()

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,  # 1D attention mask
                'labels': labels,
                'hospitalization_id': chunk.hospitalization_id,
                'chunk_info': {
                    'is_packed': False,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'is_complete': chunk.is_complete_hospitalization,
                    'num_tokens': len(chunk.tokens)
                }
            }

    @classmethod
    def from_cached_tensors(
        cls,
        cache_dir: str,
        split: str = 'train',
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> "CachedNarrativeDataset":
        """
        Create a dataset from pre-tokenized cached tensors.

        Args:
            cache_dir: Path to cache directory containing preprocessed data
            split: Dataset split ('train', 'val', or 'test')
            tokenizer: Optional tokenizer (kept for compatibility, not used)

        Returns:
            CachedNarrativeDataset instance wrapping pre-tokenized tensors
        """
        import torch
        from pathlib import Path

        cache_path = Path(cache_dir)
        dataset_path = cache_path / f"{split}_dataset.pt"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Cached {split} dataset not found at {dataset_path}.\n"
                f"Please run data_prep.py first to create cached datasets."
            )

        print(f"Loading {split} dataset from cache...")
        print(f"  Cache dir: {cache_dir}")

        # Load tensors
        tensors = torch.load(dataset_path)

        # Load metadata for validation
        metadata_path = cache_path / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"  Cache created: {metadata.get('created_at', 'unknown')}")
            print(f"  Model size: {metadata.get('model_size', 'unknown')}")
            print(f"  Split mode: {metadata.get('split_mode', 'unknown')}")

        print(f"  ✓ Loaded {len(tensors['input_ids'])} samples")

        return CachedNarrativeDataset(tensors, split)


class CachedNarrativeDataset(Dataset):
    """
    Lightweight dataset wrapper for pre-tokenized cached tensors.

    This class wraps pre-computed tensor dictionaries and provides
    efficient access without any on-the-fly processing.

    Args:
        tensors: Dictionary containing:
            - input_ids: List or tensor of token ID sequences
            - attention_mask: List or tensor of attention masks
            - labels: List or tensor of label sequences
            - hospitalization_ids: (optional) List of hospitalization IDs
            - chunk_info: (optional) List of chunk metadata dicts
        split: Dataset split name ('train', 'val', or 'test')
    """

    def __init__(
        self,
        tensors: Dict[str, any],
        split: str = 'train',
    ):
        import torch

        self.split = split

        # Store tensors
        self.input_ids = tensors['input_ids']
        self.attention_mask = tensors['attention_mask']
        self.labels = tensors['labels']

        # Optional metadata
        self.hospitalization_ids = tensors.get('hospitalization_ids', None)
        self.chunk_info = tensors.get('chunk_info', None)

        # Validate tensor lengths match
        num_samples = len(self.input_ids)
        if len(self.attention_mask) != num_samples or len(self.labels) != num_samples:
            raise ValueError(
                f"Tensor length mismatch: "
                f"input_ids={len(self.input_ids)}, "
                f"attention_mask={len(self.attention_mask)}, "
                f"labels={len(self.labels)}"
            )

        # Convert to list if they're stacked tensors (for efficient indexing)
        if isinstance(self.input_ids, torch.Tensor) and self.input_ids.dim() > 1:
            self.input_ids = [self.input_ids[i] for i in range(num_samples)]
            self.attention_mask = [self.attention_mask[i] for i in range(num_samples)]
            self.labels = [self.labels[i] for i in range(num_samples)]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample.

        Returns:
            Dictionary with pre-computed tensors:
                - input_ids: Token IDs with [BOS] and [EOS]
                - attention_mask: Attention mask
                - labels: Label sequence (same as input_ids for causal LM)
                - hospitalization_id: (optional) Hospitalization ID
                - chunk_info: (optional) Chunk metadata
        """
        result = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }

        # Add optional metadata if available
        if self.hospitalization_ids is not None:
            result['hospitalization_id'] = self.hospitalization_ids[idx]

        if self.chunk_info is not None:
            result['chunk_info'] = self.chunk_info[idx]

        return result


def load_narrative_dataset(
    config_path: str,
    tokenizer: PreTrainedTokenizer,
    split: str = 'train',
    split_mode: str = 'temporal',
    max_length: int = 8192,
    **kwargs
) -> ClinicalNarrativeDataset:
    """
    Convenience function to load dataset from clif_config.json.

    Args:
        config_path: Path to clif_config.json
        tokenizer: ClinicalTokenizer instance
        split: Data split ('train', 'val', 'test')
        split_mode: Split strategy ('temporal' or 'random')
        max_length: Maximum sequence length
        **kwargs: Additional arguments passed to ClinicalNarrativeDataset

    Returns:
        ClinicalNarrativeDataset instance
    """
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'output_dir' not in config:
        raise ValueError("Required key 'output_dir' missing from clif_config.json")

    # Get narrative parquet path (base path, actual files determined by split_mode)
    output_dir = config['output_dir']
    narratives_dir = os.path.join(output_dir, 'narratives')

    # For temporal mode, we need the narratives directory
    # The dataset will load train_val_sequences.parquet or test_sequences.parquet
    # For random mode, we use narrative_sequences.parquet
    if split_mode == 'temporal':
        narrative_parquet_path = os.path.join(narratives_dir, 'narrative_sequences.parquet')
    else:
        narrative_parquet_path = os.path.join(narratives_dir, 'narrative_sequences.parquet')
        if not os.path.exists(narrative_parquet_path):
            raise FileNotFoundError(
                f"Narrative parquet not found: {narrative_parquet_path}\n"
                f"Please run narrative assembly first:\n"
                f"  uv run tokenETL/assemble_narratives.py"
            )

    # Create dataset
    return ClinicalNarrativeDataset(
        narrative_parquet_path=narrative_parquet_path,
        tokenizer=tokenizer,
        split=split,
        split_mode=split_mode,
        max_length=max_length,
        **kwargs
    )
