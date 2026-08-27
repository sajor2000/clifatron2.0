#!/usr/bin/env python3
"""
sequence_packer.py - Efficient Sequence Packing for Training

Packs multiple hospitalizations into single sequences to maximize GPU utilization.
Uses [SEP] tokens to separate documents and creates attention masks to prevent
cross-document attention.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch


@dataclass
class PackedSequence:
    """Represents a packed sequence with multiple documents."""
    tokens: List[str]
    hospitalization_ids: List[str]
    document_boundaries: List[Tuple[int, int]]  # (start, end) positions of each document


def pack_sequences(
    all_hospitalizations: List[Tuple[str, List[str]]],
    max_length: int,
    sep_token: str = "[SEP]"
) -> List[PackedSequence]:
    """
    Pack multiple hospitalizations into sequences of max_length.

    Strategy:
    - Greedily pack hospitalizations into sequences
    - Add [SEP] between different hospitalizations
    - Track document boundaries for attention masking

    Args:
        all_hospitalizations: List of (hosp_id, tokens) tuples
        max_length: Maximum sequence length (includes [BOS] and [EOS])
        sep_token: Separator token (default: [SEP])

    Returns:
        List of PackedSequence objects
    """
    packed_sequences = []

    # Reserve space for [BOS] and [EOS]
    available_length = max_length - 2

    current_tokens = []
    current_hosp_ids = []
    current_boundaries = []
    current_position = 0

    for hosp_id, hosp_tokens in all_hospitalizations:
        # Calculate tokens needed: hosp_tokens + [SEP] (if not first doc in sequence)
        sep_cost = 1 if current_tokens else 0
        tokens_needed = len(hosp_tokens) + sep_cost

        # If hospitalization too long to fit in a single sequence, split it
        if tokens_needed > available_length:
            # Save current packed sequence if not empty
            if current_tokens:
                packed_sequences.append(PackedSequence(
                    tokens=current_tokens.copy(),
                    hospitalization_ids=current_hosp_ids.copy(),
                    document_boundaries=current_boundaries.copy()
                ))
                current_tokens = []
                current_hosp_ids = []
                current_boundaries = []
                current_position = 0

            # Split long hospitalization into chunks
            for i in range(0, len(hosp_tokens), available_length):
                chunk = hosp_tokens[i:i + available_length]
                packed_sequences.append(PackedSequence(
                    tokens=chunk,
                    hospitalization_ids=[hosp_id],
                    document_boundaries=[(0, len(chunk))]
                ))
            continue

        # If adding this hospitalization would exceed max_length, start new sequence
        if current_tokens and (current_position + tokens_needed) > available_length:
            packed_sequences.append(PackedSequence(
                tokens=current_tokens.copy(),
                hospitalization_ids=current_hosp_ids.copy(),
                document_boundaries=current_boundaries.copy()
            ))
            current_tokens = []
            current_hosp_ids = []
            current_boundaries = []
            current_position = 0

        # Add [SEP] if not first document in sequence
        if current_tokens:
            current_tokens.append(sep_token)
            current_position += 1

        # Add hospitalization
        start_pos = current_position
        current_tokens.extend(hosp_tokens)
        current_position += len(hosp_tokens)
        end_pos = current_position

        current_hosp_ids.append(hosp_id)
        current_boundaries.append((start_pos, end_pos))

    # Add final sequence if not empty
    if current_tokens:
        packed_sequences.append(PackedSequence(
            tokens=current_tokens.copy(),
            hospitalization_ids=current_hosp_ids.copy(),
            document_boundaries=current_boundaries.copy()
        ))

    return packed_sequences


def create_document_attention_mask(
    seq_length: int,
    document_boundaries: List[Tuple[int, int]],
    add_special_tokens_length: int = 2  # [BOS] and [EOS]
) -> torch.Tensor:
    """
    Create attention mask that prevents cross-document attention.

    For a sequence with documents at positions:
    - Doc 1: [0, 100)
    - Doc 2: [101, 200)  (position 100 is [SEP])

    The attention mask allows:
    - Tokens in Doc 1 to attend to tokens in Doc 1
    - Tokens in Doc 2 to attend to tokens in Doc 2
    - But NOT cross-document attention

    Args:
        seq_length: Length of the packed sequence (without [BOS]/[EOS])
        document_boundaries: List of (start, end) positions for each document
        add_special_tokens_length: Number of special tokens added ([BOS] + [EOS])

    Returns:
        Attention mask of shape (total_length, total_length)
        where total_length = seq_length + add_special_tokens_length
    """
    total_length = seq_length + add_special_tokens_length

    # Start with causal mask (lower triangular)
    # Shape: (total_length, total_length)
    mask = torch.tril(torch.ones((total_length, total_length), dtype=torch.bool))

    # [BOS] (position 0) can attend to itself
    # All other tokens (1 to total_length-1) attend causally within their document

    # Convert document boundaries to account for [BOS] at position 0
    # Tokens are at positions 1 to seq_length, [EOS] at position seq_length+1
    for doc_start, doc_end in document_boundaries:
        # Adjust for [BOS] token at position 0
        adj_start = doc_start + 1
        adj_end = doc_end + 1

        # Block attention to tokens outside this document
        # Tokens in this document (adj_start:adj_end) should only attend to:
        # 1. [BOS] (position 0)
        # 2. Tokens in the same document (adj_start:adj_end)

        # For each token in this document
        for i in range(adj_start, adj_end):
            # Block attention to tokens before this document (except [BOS])
            if adj_start > 1:
                mask[i, 1:adj_start] = False

            # Block attention to tokens after this document
            if adj_end < total_length:
                mask[i, adj_end:] = False

    # [EOS] token (last position) can attend to all previous tokens in its document
    # If [EOS] is part of the last document, it follows the same rules

    return mask


def pack_and_create_batch(
    all_hospitalizations: List[Tuple[str, List[str]]],
    tokenizer,
    max_length: int = 8192
) -> Dict[str, torch.Tensor]:
    """
    Pack hospitalizations and create batched tensors with attention masks.

    Args:
        all_hospitalizations: List of (hosp_id, tokens) tuples
        tokenizer: Tokenizer with vocab
        max_length: Maximum sequence length

    Returns:
        Dictionary with:
            - input_ids: Tensor of shape (num_sequences, max_length)
            - attention_mask: Tensor of shape (num_sequences, max_length, max_length)
            - labels: Tensor of shape (num_sequences, max_length)
            - hospitalization_ids: List of lists of hosp IDs per sequence
    """
    packed_seqs = pack_sequences(
        all_hospitalizations,
        max_length=max_length,
        sep_token=tokenizer.sep_token
    )

    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []
    batch_hosp_ids = []

    for packed_seq in packed_seqs:
        # Convert tokens to text
        text = " ".join(packed_seq.tokens)

        # Tokenize with special tokens
        encoding = tokenizer(
            text,
            add_special_tokens=True,  # Adds [BOS] and [EOS]
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)  # Shape: (max_length,)

        # Create document-aware attention mask
        seq_len = len(packed_seq.tokens)
        doc_attention_mask = create_document_attention_mask(
            seq_length=seq_len,
            document_boundaries=packed_seq.document_boundaries,
            add_special_tokens_length=2
        )

        # Pad attention mask to max_length if needed
        if doc_attention_mask.shape[0] < max_length:
            padded_mask = torch.zeros((max_length, max_length), dtype=torch.bool)
            padded_mask[:doc_attention_mask.shape[0], :doc_attention_mask.shape[1]] = doc_attention_mask
            doc_attention_mask = padded_mask

        # Labels are same as input_ids (causal LM)
        labels = input_ids.clone()

        batch_input_ids.append(input_ids)
        batch_attention_masks.append(doc_attention_mask)
        batch_labels.append(labels)
        batch_hosp_ids.append(packed_seq.hospitalization_ids)

    return {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_masks),
        'labels': torch.stack(batch_labels),
        'hospitalization_ids': batch_hosp_ids
    }
