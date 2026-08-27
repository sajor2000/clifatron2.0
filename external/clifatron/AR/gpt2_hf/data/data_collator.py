#!/usr/bin/env python3
"""
data_collator.py - Data Collator for Clinical Narratives

Handles batching and packing of narrative sequences while maintaining
hospitalization isolation through attention masks.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForClinicalCausalLM:
    """
    Data collator for causal language modeling with clinical narratives.

    Supports two modes:
    1. Padding mode (enable_packing=False):
       - Each example is one hospitalization
       - Pad to max length in batch

    2. Packing mode (enable_packing=True):
       - Pack multiple hospitalizations per sequence
       - Structure: [BOS] hosp1 [EOS] [SEP] [BOS] hosp2 [EOS] [SEP] ...
       - Reduces padding waste from ~46% to <5%

    Args:
        tokenizer: ClinicalTokenizer instance
        mlm: Whether to use masked language modeling (default: False, use causal LM)
        pad_to_multiple_of: Pad lengths to multiple of this value (for efficiency)
        enable_packing: If True, pack multiple hospitalizations per sequence
        pack_to_max_length: Maximum length for packed sequences (default: 8192)
        repeat_short_sequences: If True, repeat sequences instead of padding remainder
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = False  # Causal LM by default
    pad_to_multiple_of: Optional[int] = None
    enable_packing: bool = False
    pack_to_max_length: int = 8192
    repeat_short_sequences: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of dictionaries from dataset __getitem__

        Returns:
            Dictionary with batched tensors:
                - input_ids: [batch_size, max_seq_len]
                - attention_mask: [batch_size, max_seq_len]
                - labels: [batch_size, max_seq_len] with -100 for padding
        """
        if self.enable_packing:
            return self._pack_sequences(features)
        else:
            # Original padding-only mode
            input_ids = [f['input_ids'] for f in features]
            attention_masks = [f['attention_mask'] for f in features]
            labels = [f['labels'] for f in features]

            batch = self._pad_sequences(
                input_ids=input_ids,
                attention_masks=attention_masks,
                labels=labels
            )

            return batch

    def _pack_sequences(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Pack multiple hospitalizations into fixed-length sequences.

        Structure: [BOS] hosp1_tokens [EOS] hosp2_tokens [EOS] hosp3_tokens [EOS] [PAD]...

        Note: Industry standard pattern (no SEP tokens, no redundant BOS).
        Each hospitalization has its own [BOS] and [EOS], packed hospitalizations
        after the first omit [BOS].

        Args:
            features: List of hospitalization features

        Returns:
            Dictionary with packed tensors
        """
        packed_input_ids = []
        packed_attention_masks = []
        packed_labels = []

        current_ids = []
        current_mask = []
        current_labels = []
        current_length = 0

        feature_idx = 0
        is_first_in_sequence = True

        while feature_idx < len(features):
            # Get next hospitalization
            hosp_ids = features[feature_idx]['input_ids']
            hosp_mask = features[feature_idx]['attention_mask']
            hosp_labels = features[feature_idx]['labels']

            # For packed hospitalizations after the first, remove the [BOS] token
            # Pattern: [BOS] hosp1 [EOS] hosp2 [EOS] (no [BOS] before hosp2)
            if not is_first_in_sequence and hosp_ids[0] == self.tokenizer.bos_token_id:
                hosp_ids = hosp_ids[1:]  # Remove [BOS]
                hosp_mask = hosp_mask[1:]
                hosp_labels = hosp_labels[1:]

            hosp_length = hosp_ids.size(0)

            # Check if we can fit this hospitalization
            if current_length + hosp_length <= self.pack_to_max_length:
                # Add hospitalization (no separator needed)
                current_ids.append(hosp_ids)
                current_mask.append(hosp_mask)
                current_labels.append(hosp_labels)
                current_length += hosp_ids.size(0)

                feature_idx += 1
                is_first_in_sequence = False

            else:
                # Current sequence is full, finalize it
                packed_seq = self._finalize_packed_sequence(
                    current_ids, current_mask, current_labels, current_length
                )
                packed_input_ids.append(packed_seq['input_ids'])
                packed_attention_masks.append(packed_seq['attention_mask'])
                packed_labels.append(packed_seq['labels'])

                # Start new sequence
                current_ids = []
                current_mask = []
                current_labels = []
                current_length = 0
                is_first_in_sequence = True  # Reset for new sequence

        # Finalize last sequence if not empty
        if current_length > 0:
            packed_seq = self._finalize_packed_sequence(
                current_ids, current_mask, current_labels, current_length
            )
            packed_input_ids.append(packed_seq['input_ids'])
            packed_attention_masks.append(packed_seq['attention_mask'])
            packed_labels.append(packed_seq['labels'])

        # Stack into batch
        batch_input_ids = torch.stack(packed_input_ids)
        batch_attention_masks = torch.stack(packed_attention_masks)
        batch_labels = torch.stack(packed_labels)

        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'labels': batch_labels
        }

    def _finalize_packed_sequence(
        self,
        ids_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        current_length: int
    ) -> Dict[str, torch.Tensor]:
        """
        Finalize a packed sequence by concatenating and padding/repeating.

        Args:
            ids_list: List of token ID tensors to concatenate
            mask_list: List of attention mask tensors
            labels_list: List of label tensors
            current_length: Current length of packed sequence

        Returns:
            Dictionary with final tensors of length pack_to_max_length
        """
        # Concatenate all parts
        packed_ids = torch.cat(ids_list)
        packed_mask = torch.cat(mask_list)
        packed_labels = torch.cat(labels_list)

        remaining = self.pack_to_max_length - current_length

        if remaining > 0:
            if self.repeat_short_sequences and len(ids_list) > 0:
                # Repeat sequences to fill remainder
                repeated_ids = []
                repeated_mask = []
                repeated_labels = []

                idx = 0
                while remaining > 0:
                    hosp_idx = idx % len(ids_list)
                    hosp_ids = ids_list[hosp_idx]
                    hosp_mask = mask_list[hosp_idx]
                    hosp_labels = labels_list[hosp_idx]

                    # No separator needed - just concatenate
                    # Pattern: [BOS] hosp1 [EOS] hosp2 [EOS] hosp3 [EOS] ...

                    # Add hospitalization (or part of it if not enough space)
                    take_length = min(hosp_ids.size(0), remaining)
                    repeated_ids.append(hosp_ids[:take_length])
                    repeated_mask.append(hosp_mask[:take_length])
                    repeated_labels.append(hosp_labels[:take_length])
                    remaining -= take_length

                    idx += 1

                if repeated_ids:
                    packed_ids = torch.cat([packed_ids] + repeated_ids)
                    packed_mask = torch.cat([packed_mask] + repeated_mask)
                    packed_labels = torch.cat([packed_labels] + repeated_labels)

                # If still remaining (rare), pad
                remaining = self.pack_to_max_length - packed_ids.size(0)
                if remaining > 0:
                    pad_ids = torch.full((remaining,), self.tokenizer.pad_token_id, dtype=torch.long)
                    pad_mask = torch.zeros(remaining, dtype=torch.long)
                    pad_labels = torch.full((remaining,), -100, dtype=torch.long)

                    packed_ids = torch.cat([packed_ids, pad_ids])
                    packed_mask = torch.cat([packed_mask, pad_mask])
                    packed_labels = torch.cat([packed_labels, pad_labels])
            else:
                # Pad with [PAD] tokens
                pad_ids = torch.full((remaining,), self.tokenizer.pad_token_id, dtype=torch.long)
                pad_mask = torch.zeros(remaining, dtype=torch.long)
                pad_labels = torch.full((remaining,), -100, dtype=torch.long)

                packed_ids = torch.cat([packed_ids, pad_ids])
                packed_mask = torch.cat([packed_mask, pad_mask])
                packed_labels = torch.cat([packed_labels, pad_labels])

        return {
            'input_ids': packed_ids,
            'attention_mask': packed_mask,
            'labels': packed_labels
        }

    def _pad_sequences(
        self,
        input_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor],
        labels: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Pad sequences to uniform length within batch (original padding mode).

        Args:
            input_ids: List of input_ids tensors
            attention_masks: List of attention_mask tensors (1D or 2D)
            labels: List of label tensors

        Returns:
            Dictionary with padded tensors
        """
        # Find max length in batch
        max_length = max(ids.size(0) for ids in input_ids)

        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch_size = len(input_ids)

        # Check if we have 2D attention masks (from packed sequences with document boundaries)
        is_2d_attention = len(attention_masks[0].shape) == 2

        # Initialize padded tensors
        padded_input_ids = torch.full(
            (batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )

        if is_2d_attention:
            # 2D attention masks for packed sequences with document boundaries
            padded_attention_masks = torch.zeros(
                (batch_size, max_length, max_length),
                dtype=torch.bool
            )
        else:
            # 1D attention masks for regular sequences
            padded_attention_masks = torch.zeros(
                (batch_size, max_length),
                dtype=torch.long
            )

        padded_labels = torch.full(
            (batch_size, max_length),
            -100,  # Ignore index for loss calculation
            dtype=torch.long
        )

        # Fill in actual sequences
        for i, (ids, mask, label) in enumerate(zip(input_ids, attention_masks, labels)):
            seq_len = ids.size(0)

            # Copy sequences (left-aligned, right-padded)
            padded_input_ids[i, :seq_len] = ids

            if is_2d_attention:
                # Copy 2D attention mask
                mask_len = mask.size(0)
                padded_attention_masks[i, :mask_len, :mask_len] = mask
            else:
                # Copy 1D attention mask
                padded_attention_masks[i, :seq_len] = mask

            padded_labels[i, :seq_len] = label

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': padded_labels
        }


@dataclass
class DataCollatorForClinicalMLM:
    """
    Data collator for masked language modeling with clinical narratives.

    This is an alternative to causal LM if you want to use MLM objective.

    Args:
        tokenizer: ClinicalTokenizer instance
        mlm_probability: Probability of masking tokens (default: 0.15)
        pad_to_multiple_of: Pad lengths to multiple of this value
    """

    tokenizer: PreTrainedTokenizer
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features with masking.

        Args:
            features: List of dictionaries from dataset __getitem__

        Returns:
            Dictionary with batched tensors including masked inputs
        """
        # Extract input_ids and attention_mask
        input_ids = [f['input_ids'] for f in features]
        attention_masks = [f['attention_mask'] for f in features]

        # Pad sequences
        batch = self._pad_sequences(input_ids, attention_masks)

        # Apply masking
        batch['input_ids'], batch['labels'] = self._mask_tokens(batch['input_ids'])

        return batch

    def _pad_sequences(
        self,
        input_ids: List[torch.Tensor],
        attention_masks: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Pad sequences to uniform length."""
        max_length = max(ids.size(0) for ids in input_ids)

        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch_size = len(input_ids)

        padded_input_ids = torch.full(
            (batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        padded_attention_masks = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long
        )

        for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
            seq_len = ids.size(0)
            padded_input_ids[i, :seq_len] = ids
            padded_attention_masks[i, :seq_len] = mask

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks
        }

    def _mask_tokens(
        self,
        input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for MLM.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Tuple of (masked_input_ids, labels)
        """
        labels = input_ids.clone()

        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        for special_token_id in [
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.sep_token_id
        ]:
            if special_token_id is not None:
                special_tokens_mask |= (labels == special_token_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of the time, replace with [MASK] token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else self.tokenizer.unk_token_id

        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long
        )
        input_ids[indices_random] = random_tokens[indices_random]

        # 10% of the time, keep original token

        return input_ids, labels


def create_data_collator(
    tokenizer: PreTrainedTokenizer,
    mlm: bool = False,
    mlm_probability: float = 0.15,
    pad_to_multiple_of: Optional[int] = None,
    enable_packing: bool = False,
    pack_to_max_length: int = 8192,
    repeat_short_sequences: bool = True
):
    """
    Factory function to create appropriate data collator.

    Args:
        tokenizer: ClinicalTokenizer instance
        mlm: If True, use MLM collator; if False, use causal LM collator
        mlm_probability: Masking probability for MLM
        pad_to_multiple_of: Pad lengths to multiple of this value
        enable_packing: If True, pack multiple hospitalizations per sequence
        pack_to_max_length: Maximum length for packed sequences
        repeat_short_sequences: If True, repeat sequences instead of padding

    Returns:
        DataCollator instance
    """
    if mlm:
        return DataCollatorForClinicalMLM(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of
        )
    else:
        return DataCollatorForClinicalCausalLM(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
            enable_packing=enable_packing,
            pack_to_max_length=pack_to_max_length,
            repeat_short_sequences=repeat_short_sequences
        )
