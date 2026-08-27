#!/usr/bin/env python3
"""
clinical_tokenizer.py - Custom Tokenizer for Clinical Narratives

Creates a simple whitespace tokenizer using the token registry from tokenETL.
The clinical tokens are already pre-tokenized as 'clif_sentence' in the narrative data.

Vocabulary:
  - 5 special tokens ([PAD], [UNK], [BOS], [EOS], [SEP])
  - 1,368 clinical tokens from token_registry.json
  - 55 time markers (day_1...day_30, day_30+, hour_1...hour_24)
  = 1,428 total tokens
"""

import json
import hashlib
from typing import List, Dict, Optional, Union
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class ClinicalTokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer for pre-tokenized clinical narratives.

    The input text is already tokenized as space-separated clinical tokens.
    This tokenizer simply maps tokens to IDs based on the vocabulary.

    Special tokens:
        [PAD] (0): Padding token
        [UNK] (1): Unknown token (for tokens not in vocab)
        [BOS] (2): Beginning of sequence (start of hospitalization)
        [EOS] (3): End of sequence (end of hospitalization)
        [SEP] (4): Separator token (reserved for future use)
    """

    # Define special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    SEP_TOKEN = "[SEP]"

    def __init__(
        self,
        vocab: Dict[str, int],
        model_max_length: int = 8192,
        **kwargs
    ):
        """
        Initialize the clinical tokenizer.

        Args:
            vocab: Dictionary mapping token strings to token IDs
            model_max_length: Maximum sequence length (default: 8192)
            **kwargs: Additional arguments passed to PreTrainedTokenizer
        """
        # Set vocab BEFORE calling super().__init__() since parent class may access it
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

        # Set default special tokens if not provided
        kwargs.setdefault("pad_token", self.PAD_TOKEN)
        kwargs.setdefault("unk_token", self.UNK_TOKEN)
        kwargs.setdefault("bos_token", self.BOS_TOKEN)
        kwargs.setdefault("eos_token", self.EOS_TOKEN)
        kwargs.setdefault("sep_token", self.SEP_TOKEN)

        super().__init__(
            model_max_length=model_max_length,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary."""
        return self.vocab.copy()

    def get_vocab_hash(self) -> str:
        """
        Compute SHA256 hash of the vocabulary for consistency validation.

        Returns:
            Hexadecimal hash string of the vocabulary
        """
        # Create deterministic JSON string (sorted keys)
        vocab_json = json.dumps(self.vocab, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(vocab_json.encode('utf-8')).hexdigest()

    def validate_vocab_size(self, expected_size: int = 1428) -> bool:
        """
        Validate that vocabulary size matches expected size.

        Args:
            expected_size: Expected vocabulary size (default: 1428)

        Returns:
            True if vocab size matches, False otherwise

        Raises:
            ValueError: If vocabulary size doesn't match expected size
        """
        actual_size = len(self.vocab)
        if actual_size != expected_size:
            raise ValueError(
                f"Vocabulary size mismatch! Expected {expected_size}, got {actual_size}. "
                f"This tokenizer is incompatible with models trained on the standard vocabulary."
            )
        return True

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text by splitting on whitespace.

        Args:
            text: Space-separated clinical tokens

        Returns:
            List of token strings
        """
        # Split on whitespace - tokens are already pre-tokenized
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        """
        Convert a token string to an ID.

        Args:
            token: Token string

        Returns:
            Token ID (or UNK token ID if not in vocab)
        """
        return self.vocab.get(token, self.vocab.get(self.UNK_TOKEN, 1))

    def _convert_id_to_token(self, index: int) -> str:
        """
        Convert a token ID to a token string.

        Args:
            index: Token ID

        Returns:
            Token string (or UNK token if ID not in vocab)
        """
        return self.ids_to_tokens.get(index, self.UNK_TOKEN)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens to a single string.

        Args:
            tokens: List of token strings

        Returns:
            Space-separated token string
        """
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs by adding special tokens.

        Adds [BOS] at the beginning and [EOS] at the end.

        Args:
            token_ids_0: First sequence token IDs
            token_ids_1: Optional second sequence token IDs

        Returns:
            List of token IDs with special tokens added
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return bos + token_ids_0 + eos

        # For two sequences, separate with SEP token
        sep = [self.sep_token_id]
        return bos + token_ids_0 + sep + token_ids_1 + eos

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Create mask for special tokens.

        Args:
            token_ids_0: First sequence token IDs
            token_ids_1: Optional second sequence token IDs
            already_has_special_tokens: Whether special tokens are already added

        Returns:
            List of 0s and 1s (1 for special tokens)
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]

        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs for two sequences.

        Args:
            token_ids_0: First sequence token IDs
            token_ids_1: Optional second sequence token IDs

        Returns:
            List of token type IDs (0 for first sequence, 1 for second)
        """
        bos = [0]
        eos = [0]

        if token_ids_1 is None:
            return bos + ([0] * len(token_ids_0)) + eos

        sep = [0]
        return bos + ([0] * len(token_ids_0)) + sep + ([1] * len(token_ids_1)) + eos

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> tuple:
        """
        Save the vocabulary to a file.

        Args:
            save_directory: Directory to save vocabulary
            filename_prefix: Optional prefix for filename

        Returns:
            Tuple of vocabulary file path
        """
        import os

        if not os.path.isdir(save_directory):
            raise ValueError(f"Directory {save_directory} does not exist")

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    @classmethod
    def from_token_registry(
        cls,
        token_registry_path: str,
        model_max_length: int = 8192,
        **kwargs
    ) -> "ClinicalTokenizer":
        """
        Create tokenizer from token_registry.json file.

        Args:
            token_registry_path: Path to token_registry.json
            model_max_length: Maximum sequence length
            **kwargs: Additional arguments

        Returns:
            ClinicalTokenizer instance
        """
        # Load token registry
        with open(token_registry_path, "r") as f:
            token_registry = json.load(f)

        # Build vocabulary starting with special tokens
        vocab = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
            cls.BOS_TOKEN: 2,
            cls.EOS_TOKEN: 3,
            cls.SEP_TOKEN: 4,
        }

        # Add all tokens from registry (organized by source)
        token_id = len(vocab)  # Start after special tokens

        # Collect all tokens from all sources (preserve token_registry order)
        all_tokens = []
        for source, tokens_dict in token_registry.items():
            for token_name in tokens_dict.keys():
                all_tokens.append(token_name)

        # Add to vocabulary (no sorting - preserve registry order)
        for token in all_tokens:
            if token not in vocab:  # Avoid duplicates
                vocab[token] = token_id
                token_id += 1

        # Add time marker tokens dynamically (not in registry)
        # Day tokens: day_1 through day_30, plus day_30+
        for i in range(1, 31):
            token = f"day_{i}"
            if token not in vocab:
                vocab[token] = token_id
                token_id += 1

        if "day_30+" not in vocab:
            vocab["day_30+"] = token_id
            token_id += 1

        # Hour tokens: hour_1 through hour_24
        for i in range(1, 25):
            token = f"hour_{i}"
            if token not in vocab:
                vocab[token] = token_id
                token_id += 1

        print(f"Built vocabulary with {len(vocab)} tokens:")
        print(f"  - Special tokens: 5")
        print(f"  - Clinical tokens: {len(all_tokens)}")
        print(f"  - Time markers: 55 (30 days + 1 day_30+ + 24 hours)")
        print(f"  - Total: {len(vocab)}")

        return cls(vocab=vocab, model_max_length=model_max_length, **kwargs)

    @classmethod
    def from_vocab_lock(
        cls,
        vocab_lock_path: str,
        model_max_length: int = 8192,
        **kwargs
    ) -> "ClinicalTokenizer":
        """
        Create tokenizer from vocab_lock.json file (data-driven approach).

        This is the preferred method for loading vocabulary built from actual
        parquet data. The vocab_lock.json file contains vocabulary extracted
        directly from narratives with order preservation and explicit time markers.

        Args:
            vocab_lock_path: Path to vocab_lock.json
            model_max_length: Maximum sequence length
            **kwargs: Additional arguments

        Returns:
            ClinicalTokenizer instance
        """
        from pathlib import Path

        vocab_lock_path = Path(vocab_lock_path)

        if not vocab_lock_path.exists():
            raise FileNotFoundError(f"Vocabulary lock file not found: {vocab_lock_path}")

        # Load vocabulary lock file
        with open(vocab_lock_path, "r") as f:
            vocab_data = json.load(f)

        vocab = vocab_data["vocab"]

        print(f"Loaded vocabulary from {vocab_lock_path}:")
        print(f"  - Total tokens: {len(vocab)}")
        print(f"  - Vocabulary size: {vocab_data.get('vocab_size', len(vocab))}")

        return cls(vocab=vocab, model_max_length=model_max_length, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a previously saved ClinicalTokenizer.

        Checks for vocab_lock.json first (preferred), falls back to vocab.json.

        Args:
            pretrained_model_name_or_path: Path to directory containing saved tokenizer files
            **kwargs: Additional arguments

        Returns:
            ClinicalTokenizer instance
        """
        import json
        from pathlib import Path

        tokenizer_dir = Path(pretrained_model_name_or_path)

        # Try vocab_lock.json first (preferred - data-driven vocabulary)
        vocab_lock_path = tokenizer_dir / "vocab_lock.json"
        if vocab_lock_path.exists():
            return cls.from_vocab_lock(str(vocab_lock_path), **kwargs)

        # Fall back to vocab.json (legacy)
        vocab_path = tokenizer_dir / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found. Looked for:\n"
                f"  - {vocab_lock_path} (preferred)\n"
                f"  - {vocab_path} (legacy)"
            )

        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        # Load tokenizer config if exists
        config_path = tokenizer_dir / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                kwargs.setdefault('model_max_length', config.get('model_max_length', 8192))

        return cls(vocab=vocab, **kwargs)


def load_clinical_tokenizer(
    token_registry_path: str,
    model_max_length: int = 8192
) -> ClinicalTokenizer:
    """
    Convenience function to load clinical tokenizer from token registry.

    Args:
        token_registry_path: Path to token_registry.json
        model_max_length: Maximum sequence length

    Returns:
        ClinicalTokenizer instance
    """
    return ClinicalTokenizer.from_token_registry(
        token_registry_path=token_registry_path,
        model_max_length=model_max_length
    )
