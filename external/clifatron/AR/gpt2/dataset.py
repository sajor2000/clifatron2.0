#!/usr/bin/env python3

"""
Dataset loader for CLIF Llama training
Supports both padded and packed collation strategies
"""

import itertools
import pathlib
from typing import Literal

import datasets as ds
import numpy as np
import polars as pl
import torch as t

from vocabulary import Vocabulary


class ClifDataset:
    """Dataset handler for CLIF tokenized sequences"""

    def __init__(
        self,
        data_dir: pathlib.Path,
        vocab_path: pathlib.Path,
        collation: Literal["padded", "packed"] = "packed",
        max_seq_length: int = 4096,
        shuffle_buffer_size: int = 1024,
    ):
        """
        Initialize dataset

        Args:
            data_dir: Directory containing train/val/test splits
            vocab_path: Path to vocabulary file
            collation: "padded" or "packed" collation strategy
            max_seq_length: Maximum sequence length (default: 4096 tokens)
            shuffle_buffer_size: Buffer size for shuffling
        """
        self.data_dir = pathlib.Path(data_dir)
        self.collation = collation
        self.max_seq_length = max_seq_length
        self.shuffle_buffer_size = shuffle_buffer_size

        # Random generators
        self.t_rng = t.Generator().manual_seed(42)
        self.np_rng = np.random.default_rng(42)

        # Load vocabulary - handle both vocab_lock.json and vocab.gzip
        vocab_path = pathlib.Path(vocab_path)

        # If vocab_path is a directory, look for vocab_lock.json or vocab.gzip
        if vocab_path.is_dir():
            vocab_lock_path = vocab_path / "vocab_lock.json"
            vocab_gzip_path = vocab_path / "vocab.gzip"

            if vocab_lock_path.exists():
                self.vocab = Vocabulary.from_vocab_lock(vocab_lock_path)
            elif vocab_gzip_path.exists():
                self.vocab = Vocabulary().load(vocab_gzip_path)
            else:
                raise FileNotFoundError(
                    f"No vocabulary found in {vocab_path}. "
                    f"Expected vocab_lock.json or vocab.gzip"
                )
        # If vocab_path is a file, detect format by extension
        elif vocab_path.is_file():
            if vocab_path.suffix == ".json":
                self.vocab = Vocabulary.from_vocab_lock(vocab_path)
            elif vocab_path.suffix == ".gzip":
                self.vocab = Vocabulary().load(vocab_path)
            else:
                raise ValueError(
                    f"Unsupported vocabulary file format: {vocab_path}. "
                    f"Expected .json (vocab_lock) or .gzip"
                )
        else:
            raise FileNotFoundError(f"Vocabulary path not found: {vocab_path}")

        # Define splits
        self.splits = ("train", "val", "test")

        # Load datasets
        self._load_datasets()

    def _load_datasets(self):
        """Load train/val/test datasets from parquet files with memory optimization"""
        data_files = {}
        for split in self.splits:
            split_file = self.data_dir / split / "data.parquet"
            if split_file.exists():
                data_files[split] = str(split_file)

        if not data_files:
            raise FileNotFoundError(
                f"No dataset files found in {self.data_dir}\n"
                "Please run 03_create_splits.py first"
            )

        # OPTIMIZATION: Use streaming mode for memory efficiency with large datasets
        # Streaming mode prevents loading entire dataset into memory
        print(f"Loading datasets in streaming mode for memory efficiency...")
        self.dataset = ds.load_dataset("parquet", data_files=data_files, streaming=True)

        # Get dataset sizes from parquet metadata (without loading into memory)
        train_file = self.data_dir / "train" / "data.parquet"
        val_file = self.data_dir / "val" / "data.parquet"
        test_file = self.data_dir / "test" / "data.parquet"

        # Use polars to quickly get row counts from parquet metadata
        self.n_train = pl.scan_parquet(train_file).select(pl.len()).collect()[0, 0] if train_file.exists() else 0
        self.n_val = pl.scan_parquet(val_file).select(pl.len()).collect()[0, 0] if val_file.exists() else 0
        self.n_test = pl.scan_parquet(test_file).select(pl.len()).collect()[0, 0] if test_file.exists() else 0

        print(f"Dataset sizes - Train: {self.n_train:,}, Val: {self.n_val:,}, Test: {self.n_test:,}")

        # Note: With streaming=True, datasets are already IterableDatasets
        # Preparation is done in get_*_dataset methods to maintain streaming efficiency

    def generate_padding(self, poisson_rate: float = 7.0) -> t.Tensor:
        """
        Generate random padding for sequence packing

        Args:
            poisson_rate: Rate parameter for Poisson distribution

        Returns:
            Tensor of padding tokens
        """
        pad_token = self.vocab("PAD")
        size = t.poisson(t.tensor(poisson_rate), generator=self.t_rng).to(t.int64)
        return t.full(size=(size.item(),), fill_value=pad_token, dtype=t.int64)

    def chunk_iterable(self, it):
        """
        Pack sequences into fixed-length chunks with random padding

        Args:
            it: Iterable of examples

        Yields:
            Fixed-length packed sequences
        """
        ret = t.Tensor(size=(0,)).to(t.int64)

        for eg in it:
            # Convert input_ids to tensor if it's a list (from streaming dataset)
            input_ids = eg["input_ids"]
            if isinstance(input_ids, list):
                input_ids = t.tensor(input_ids, dtype=t.int64)
            elif not isinstance(input_ids, t.Tensor):
                input_ids = t.tensor(input_ids, dtype=t.int64)

            # Concatenate example with random padding
            x = t.cat((input_ids, self.generate_padding()))

            while x.size(dim=0) > 0:
                # Calculate how much to add to current chunk
                ndiff = min(self.max_seq_length - ret.size(dim=0), x.size(dim=0))

                # Add tokens to current chunk
                ret = t.cat((ret, x[:ndiff]))
                x = x[ndiff:]

                # Yield when chunk is full
                if ret.size(dim=0) == self.max_seq_length:
                    yield {"input_ids": ret.to(t.int64)}
                    ret = t.Tensor(size=(0,)).to(t.int64)

    def get_train_dataset(self, n_epochs: int = 1):
        """
        Get training dataset (streaming mode for memory efficiency)

        Args:
            n_epochs: Number of epochs (for packed collation)

        Returns:
            Training dataset
        """
        if "train" not in self.dataset:
            raise ValueError("Training dataset not found")

        # Extract just input_ids from streaming dataset
        def extract_input_ids(example):
            return {"input_ids": example["input_ids"]}

        if self.collation == "padded":
            # For padded collation with streaming, add padding on the fly
            def pad_example(example):
                input_ids = example["input_ids"]
                seq_len = len(input_ids)
                if seq_len < self.max_seq_length:
                    padding = [self.vocab("PAD")] * (self.max_seq_length - seq_len)
                    padded = input_ids + padding
                elif seq_len > self.max_seq_length:
                    padded = input_ids[: self.max_seq_length - 1] + [self.vocab("TRUNC")]
                else:
                    padded = input_ids
                return {"input_ids": padded}

            return (
                self.dataset["train"]
                .map(pad_example)
                .shuffle(buffer_size=self.shuffle_buffer_size, seed=42)
            )

        elif self.collation == "packed":
            # Create iterable dataset with packing from streaming dataset
            # First remove all columns except input_ids
            train_stream = self.dataset["train"].remove_columns(
                [col for col in self.dataset["train"].column_names if col != "input_ids"]
            )

            # Repeat for n_epochs and shuffle
            def epoch_repeater():
                for _ in range(n_epochs):
                    for example in train_stream:
                        yield example

            repeated_stream = ds.IterableDataset.from_generator(
                epoch_repeater,
                features=ds.Features({"input_ids": ds.Sequence(ds.Value("int64"))}),
            ).shuffle(buffer_size=self.shuffle_buffer_size, seed=42)

            # Pack sequences
            return ds.IterableDataset.from_generator(
                lambda: self.chunk_iterable(repeated_stream),
                features=ds.Features({"input_ids": ds.Sequence(ds.Value("int64"))}),
            )

    def get_val_dataset(self):
        """Get validation dataset (streaming mode for memory efficiency)"""
        if "val" not in self.dataset:
            raise ValueError("Validation dataset not found")

        # Extract just input_ids from streaming dataset
        def extract_input_ids(example):
            return {"input_ids": example["input_ids"]}

        if self.collation == "padded":
            # For padded collation with streaming, add padding on the fly
            def pad_example(example):
                input_ids = example["input_ids"]
                seq_len = len(input_ids)
                if seq_len < self.max_seq_length:
                    padding = [self.vocab("PAD")] * (self.max_seq_length - seq_len)
                    padded = input_ids + padding
                elif seq_len > self.max_seq_length:
                    padded = input_ids[: self.max_seq_length - 1] + [self.vocab("TRUNC")]
                else:
                    padded = input_ids
                return {"input_ids": padded}

            return self.dataset["val"].map(pad_example)

        elif self.collation == "packed":
            # Remove all columns except input_ids
            val_stream = self.dataset["val"].remove_columns(
                [col for col in self.dataset["val"].column_names if col != "input_ids"]
            )
            return ds.IterableDataset.from_generator(
                lambda: self.chunk_iterable(val_stream),
                features=ds.Features({"input_ids": ds.Sequence(ds.Value("int64"))}),
            )

    def get_test_dataset(self):
        """Get test dataset (streaming mode for memory efficiency)"""
        if "test" not in self.dataset:
            raise ValueError("Test dataset not found")

        # Extract just input_ids from streaming dataset
        def extract_input_ids(example):
            return {"input_ids": example["input_ids"]}

        return self.dataset["test"].map(extract_input_ids)

    def get_context_length(self) -> int:
        """Get the context length (max sequence length)"""
        return self.max_seq_length
