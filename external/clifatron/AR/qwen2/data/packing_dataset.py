#!/usr/bin/env python3
"""
packing_dataset.py - Efficient Sequence Packing with Document Isolation

Packs multiple hospitalizations into fixed-length sequences:
- No token waste: Overflow continues to next packed sequence
- Document isolation: Attention masks prevent cross-hospitalization leakage
- Label masking: PAD tokens masked with -100 (no loss contribution)
"""

import torch
from typing import Iterator, Dict, List
from datasets import IterableDataset, Features, Sequence, Value


class PackingIterator:
    """
    Iterator that packs pre-tokenized hospitalizations into fixed-length sequences.

    Strategy:
    1. Concatenate hospitalizations with PAD separator tokens
    2. Split into exact max_seq_length chunks (no waste - overflow continues)
    3. Track document boundaries for attention masking
    4. Mask PAD tokens in labels (-100)
    """

    def __init__(
        self,
        dataset: Iterator[Dict[str, List[int]]],
        max_seq_length: int,
        pad_token_id: int,
        num_pad_tokens: int = 8,
    ):
        """
        Args:
            dataset: Iterator yielding {"input_ids": [...], "attention_mask": [...], "labels": [...]}
            max_seq_length: Fixed sequence length for packing (e.g., 8192)
            pad_token_id: Token ID for padding between documents
            num_pad_tokens: Number of PAD tokens to insert between hospitalizations
        """
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.num_pad_tokens = num_pad_tokens

        # Buffer for accumulating tokens across documents
        self.buffer_input_ids = []
        self.buffer_attention_mask = []
        self.buffer_labels = []
        self.buffer_doc_boundaries = []  # Track where documents start/end

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, List[int]]:
        """Yield next packed sequence of exactly max_seq_length tokens."""

        # Keep adding documents until we have enough for a full sequence
        while len(self.buffer_input_ids) < self.max_seq_length:
            try:
                doc = next(self.dataset)
            except StopIteration:
                # No more documents - if buffer has content, pad and return it
                if len(self.buffer_input_ids) > 0:
                    return self._flush_buffer()
                raise StopIteration

            # Mark start of new document
            doc_start = len(self.buffer_input_ids)

            # Add document to buffer
            self.buffer_input_ids.extend(doc["input_ids"])
            self.buffer_attention_mask.extend(doc["attention_mask"])
            self.buffer_labels.extend(doc["labels"])

            # Mark end of document
            doc_end = len(self.buffer_input_ids)
            self.buffer_doc_boundaries.append((doc_start, doc_end))

            # Add separator PAD tokens between documents
            separator_ids = [self.pad_token_id] * self.num_pad_tokens
            self.buffer_input_ids.extend(separator_ids)
            self.buffer_attention_mask.extend([1] * self.num_pad_tokens)
            self.buffer_labels.extend([-100] * self.num_pad_tokens)  # Don't learn from PAD

        # Extract exactly max_seq_length tokens
        packed_input_ids = self.buffer_input_ids[:self.max_seq_length]
        packed_attention_mask = self.buffer_attention_mask[:self.max_seq_length]
        packed_labels = self.buffer_labels[:self.max_seq_length]

        # Keep overflow for next sequence
        self.buffer_input_ids = self.buffer_input_ids[self.max_seq_length:]
        self.buffer_attention_mask = self.buffer_attention_mask[self.max_seq_length:]
        self.buffer_labels = self.buffer_labels[self.max_seq_length:]

        # Update document boundaries for remaining buffer
        new_boundaries = []
        for start, end in self.buffer_doc_boundaries:
            if end <= self.max_seq_length:
                # Document fully consumed
                continue
            elif start < self.max_seq_length < end:
                # Document split across sequences
                new_boundaries.append((0, end - self.max_seq_length))
            else:
                # Document entirely in buffer
                new_boundaries.append((start - self.max_seq_length, end - self.max_seq_length))
        self.buffer_doc_boundaries = new_boundaries

        # Create 1D document IDs (much more memory efficient than 2D masks)
        document_ids = self._create_document_ids(
            seq_length=self.max_seq_length,
            doc_boundaries=[(s, e) for s, e in self.buffer_doc_boundaries if e <= self.max_seq_length]
        )

        return {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,  # Standard 1D mask
            "labels": packed_labels,
            "document_ids": document_ids,  # 1D array: document ID for each token
        }

    def _flush_buffer(self) -> Dict[str, List[int]]:
        """Pad and return remaining buffer content."""
        pad_length = self.max_seq_length - len(self.buffer_input_ids)

        # Pad to max_seq_length
        padded_input_ids = self.buffer_input_ids + [self.pad_token_id] * pad_length
        padded_attention_mask = self.buffer_attention_mask + [0] * pad_length
        padded_labels = self.buffer_labels + [-100] * pad_length

        document_ids = self._create_document_ids(
            seq_length=self.max_seq_length,
            doc_boundaries=self.buffer_doc_boundaries
        )

        # Clear buffer
        self.buffer_input_ids = []
        self.buffer_attention_mask = []
        self.buffer_labels = []
        self.buffer_doc_boundaries = []

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
            "document_ids": document_ids,
        }

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


def create_packed_dataset(
    dataset,
    max_seq_length: int,
    pad_token_id: int,
    num_pad_tokens: int = 8,
) -> IterableDataset:
    """
    Create an IterableDataset that yields packed sequences.

    Args:
        dataset: Source dataset (can be iterable or map-style)
        max_seq_length: Target sequence length for packing
        pad_token_id: PAD token ID
        num_pad_tokens: Number of PAD tokens between documents

    Returns:
        IterableDataset yielding packed sequences
    """
    def generator():
        dataset_iter = iter(dataset)
        packing_iter = PackingIterator(
            dataset=dataset_iter,
            max_seq_length=max_seq_length,
            pad_token_id=pad_token_id,
            num_pad_tokens=num_pad_tokens,
        )

        for packed_example in packing_iter:
            yield packed_example

    return IterableDataset.from_generator(
        generator,
        features=Features({
            "input_ids": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
            "document_ids": Sequence(Value("int64")),  # 1D document IDs (memory efficient)
        })
    )
