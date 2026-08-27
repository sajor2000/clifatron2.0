"""
Clinical Tokenizer for Qwen2 - No Token Registry Dependency

This tokenizer loads vocabulary directly from vocab_lock.json,
which is built from the parquet data files.

Key features:
- Whitespace-based tokenization (one token = one clinical concept)
- No subword tokenization (unlike BPE/WordPiece)
- Preserves token order from data (no sorting)
- No dependency on token_registry
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import hashlib


class ClinicalTokenizer:
    """
    Whitespace-based tokenizer for clinical narratives.

    Each token represents a complete clinical concept (e.g., "vitals_hr_80_100").
    Tokens are separated by whitespace in the input text.

    This tokenizer is designed for pre-tokenized clinical data where each
    token has already been extracted and binned during the ETL process.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        special_tokens: Optional[Dict[str, str]] = None,
        max_length: int = 8192
    ):
        """
        Initialize tokenizer with vocabulary.

        Args:
            vocab: Dictionary mapping token -> token_id
            special_tokens: Dictionary of special token names and values
            max_length: Maximum sequence length
        """
        self.vocab = vocab
        self.max_length = max_length

        # Reverse vocab for decoding
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

        # Special tokens
        if special_tokens is None:
            special_tokens = {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "sep_token": "[SEP]"
            }

        self.special_tokens = special_tokens

        # Special token IDs
        self.pad_token = special_tokens["pad_token"]
        self.unk_token = special_tokens["unk_token"]
        self.bos_token = special_tokens["bos_token"]
        self.eos_token = special_tokens["eos_token"]
        self.sep_token = special_tokens["sep_token"]

        self.pad_token_id = vocab[self.pad_token]
        self.unk_token_id = vocab[self.unk_token]
        self.bos_token_id = vocab[self.bos_token]
        self.eos_token_id = vocab[self.eos_token]
        self.sep_token_id = vocab[self.sep_token]

        self.vocab_size = len(vocab)

        # Validate vocabulary
        self._validate_vocab()

    def _validate_vocab(self):
        """Validate vocabulary has required structure."""
        # Check special tokens are present
        for token_name, token in self.special_tokens.items():
            if token not in self.vocab:
                raise ValueError(f"Special token '{token_name}' ({token}) not in vocabulary")

        # Check vocab size matches expected
        if self.vocab_size < 1000:
            raise ValueError(f"Vocabulary too small: {self.vocab_size} tokens")

        print(f"Clinical Tokenizer initialized:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Special tokens: {list(self.special_tokens.values())}")
        print(f"  Max length: {self.max_length:,}")

    @classmethod
    def from_vocab_file(cls, vocab_path: Union[str, Path], max_length: int = 8192):
        """
        Load tokenizer from vocab_lock.json file.

        Args:
            vocab_path: Path to vocab_lock.json
            max_length: Maximum sequence length

        Returns:
            ClinicalTokenizer instance
        """
        vocab_path = Path(vocab_path)

        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        vocab = vocab_data["vocab"]
        special_tokens = vocab_data.get("special_tokens", None)

        print(f"Loaded vocabulary from {vocab_path}")
        print(f"  Vocabulary size: {len(vocab):,}")

        return cls(vocab=vocab, special_tokens=special_tokens, max_length=max_length)

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, Path], max_length: int = 8192):
        """
        Load tokenizer from pretrained directory (HuggingFace compatible).

        Args:
            pretrained_path: Path to directory containing vocab.json
            max_length: Maximum sequence length

        Returns:
            ClinicalTokenizer instance
        """
        pretrained_path = Path(pretrained_path)

        # Look for vocab.json in the directory
        vocab_file = pretrained_path / "vocab.json"
        if not vocab_file.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_file}\n"
                f"Expected vocab.json in {pretrained_path}"
            )

        return cls.from_vocab_file(vocab_file, max_length=max_length)

    def __len__(self) -> int:
        """Return vocabulary size (HuggingFace compatible)."""
        return self.vocab_size

    def get_vocab_hash(self) -> str:
        """
        Compute SHA256 hash of vocabulary for consistency checking.

        This hash can be used to verify that all sites have the same
        vocabulary lock file.

        Returns:
            Hex string of SHA256 hash
        """
        # Sort vocab by token_id to ensure deterministic ordering
        sorted_items = sorted(self.vocab.items(), key=lambda x: x[1])
        vocab_str = json.dumps(sorted_items)
        return hashlib.sha256(vocab_str.encode()).hexdigest()

    def compute_vocab_hash(self) -> str:
        """Alias for get_vocab_hash() for compatibility."""
        return self.get_vocab_hash()

    def validate_vocab_size(self, expected_size: int):
        """
        Validate that vocabulary size matches expected size.

        Args:
            expected_size: Expected vocabulary size

        Raises:
            ValueError: If vocab size doesn't match expected
        """
        if self.vocab_size != expected_size:
            raise ValueError(
                f"Vocabulary size mismatch!\n"
                f"  Expected: {expected_size}\n"
                f"  Actual: {self.vocab_size}\n"
                f"This usually means the vocabulary was rebuilt with different data."
            )
        print(f"  ✓ Vocabulary size validated: {self.vocab_size}")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text by splitting on whitespace.

        Args:
            text: Input text (already tokenized, space-separated)

        Returns:
            List of tokens
        """
        return text.strip().split()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to IDs using vocabulary.

        Args:
            tokens: List of token strings

        Returns:
            List of token IDs (unknown tokens mapped to [UNK])
        """
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs back to tokens.

        Args:
            ids: List of token IDs

        Returns:
            List of token strings
        """
        return [self.ids_to_tokens.get(id, self.unk_token) for id in ids]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add [BOS] and [EOS]
            max_length: Maximum length (default: self.max_length)
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            List of token IDs
        """
        if max_length is None:
            max_length = self.max_length

        # Tokenize
        tokens = self.tokenize(text)

        # Convert to IDs
        token_ids = self.convert_tokens_to_ids(tokens)

        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        # Truncate if needed
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Pad if needed
        if padding and len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        tokens = self.convert_ids_to_tokens(token_ids)

        # Filter special tokens if requested
        if skip_special_tokens:
            special_token_values = set(self.special_tokens.values())
            tokens = [t for t in tokens if t not in special_token_values]

        return " ".join(tokens)

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Dict:
        """
        Tokenize text (compatible with HuggingFace interface).

        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add [BOS] and [EOS]
            max_length: Maximum length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: "pt" for PyTorch tensors, None for lists

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Handle single text or batch
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        if max_length is None:
            max_length = self.max_length

        # Encode each text
        all_input_ids = []
        all_attention_masks = []

        for t in texts:
            input_ids = self.encode(
                t,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1 if id != self.pad_token_id else 0 for id in input_ids]

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # Create labels (copy of input_ids for causal LM)
        all_labels = [ids.copy() for ids in all_input_ids]

        # Prepare output
        output = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels
        }

        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            output = {
                k: torch.tensor(v) for k, v in output.items()
            }

        # If single text, unwrap the batch dimension
        if isinstance(text, str) and return_tensors is None:
            output = {k: v[0] for k, v in output.items()}

        return output

    def save_vocabulary(self, save_directory: Union[str, Path]) -> None:
        """
        Save vocabulary to directory.

        Args:
            save_directory: Directory to save vocab files
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_path = save_directory / "vocab.json"
        vocab_data = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }

        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

        print(f"Vocabulary saved to {vocab_path}")

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Save tokenizer (alias for save_vocabulary for HuggingFace compatibility).

        Args:
            save_directory: Directory to save tokenizer files
        """
        self.save_vocabulary(save_directory)

        # Also save tokenizer config for HuggingFace compatibility
        config_path = Path(save_directory) / "tokenizer_config.json"
        config = {
            "tokenizer_class": "ClinicalTokenizer",
            "model_max_length": self.max_length,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Tokenizer config saved to {config_path}")


def load_tokenizer(tokenizer_path: Union[str, Path], max_length: int = 8192) -> ClinicalTokenizer:
    """
    Convenience function to load tokenizer from directory.

    Args:
        tokenizer_path: Path to directory containing vocab.json or vocab_lock.json
        max_length: Maximum sequence length

    Returns:
        ClinicalTokenizer instance
    """
    tokenizer_path = Path(tokenizer_path)

    # Look for vocab files
    vocab_file = None
    for filename in ["vocab.json", "vocab_lock.json"]:
        candidate = tokenizer_path / filename if tokenizer_path.is_dir() else tokenizer_path
        if candidate.exists() and candidate.is_file():
            vocab_file = candidate
            break

    if vocab_file is None:
        # If path is a file, use it directly
        if tokenizer_path.is_file():
            vocab_file = tokenizer_path
        else:
            raise FileNotFoundError(
                f"No vocab file found in {tokenizer_path}. "
                "Expected vocab.json or vocab_lock.json"
            )

    return ClinicalTokenizer.from_vocab_file(vocab_file, max_length=max_length)
