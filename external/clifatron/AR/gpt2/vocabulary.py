#!/usr/bin/env python3

"""
Vocabulary class for managing token mappings for CLIF domain-specific tokens
Adapted from FMs-EHRs-Rep-Dynamics-and-Transfer/vocabulary.py
"""

import collections
import functools
import gzip
import hashlib
import json
import pathlib
import pickle
import typing
import warnings

import polars as pl

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Hashable: typing.TypeAlias = collections.abc.Hashable
Pathlike: typing.TypeAlias = pathlib.PurePath | str


class Vocabulary:
    """
    maintains a dictionary `lookup` mapping words -> tokens,
    a dictionary `reverse` inverting the lookup, and a dictionary
    `aux` mapping words -> auxiliary info
    """

    def __init__(self, words: tuple = (), *, is_training: bool = True):
        assert len(set(words)) == len(words)
        self.lookup = {v: i for i, v in enumerate(words)}
        self.reverse = dict(enumerate(words))
        self.aux = {}
        self._is_training = is_training

    def __call__(self, word: Hashable) -> int | None:
        try:
            return self.lookup[word]
        except KeyError:
            if self._is_training:
                self.lookup[word], self.reverse[n] = (n := len(self.lookup)), word
                return n
            else:
                warnings.warn(
                    "Encountered previously unseen token: {} {}".format(
                        word, type(word)
                    )
                )
                return self.lookup[None] if None in self.lookup else None

    def set_aux(self, word: Hashable, aux_data):
        if self._is_training:
            self.aux[word] = aux_data
        else:
            raise Exception("Tokenizer is frozen after training.")
        return self

    def has_aux(self, word: Hashable) -> bool:
        return word in self.aux

    def in_lookup(self, word: Hashable) -> bool:
        return word in self.lookup

    def get_aux(self, word: Hashable):
        return self.aux[word]

    def save(self, filepath: Pathlike) -> typing.Self:
        with gzip.open(pathlib.Path(filepath).expanduser(), "w+") as f:
            pickle.dump(
                {
                    "lookup": self.lookup,
                    "reverse": self.reverse,
                    "aux": {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v for k, v in self.aux.items()},
                },
                f,
            )
        return self

    def load(self, filepath: Pathlike) -> typing.Self:
        with gzip.open(pathlib.Path(filepath).expanduser(), mode="r+") as f:
            for k, v in pickle.load(f).items():
                setattr(self, k, v)
        return self

    def get_frame(self) -> Frame:
        return pl.from_records(
            list(self.lookup.items()), schema=("word", "token"), orient="row"
        )

    def __len__(self) -> int:
        return len(self.lookup)

    @property
    def is_training(self) -> bool:
        return self._is_training

    @is_training.setter
    def is_training(self, value: bool):
        self._is_training = value

    def get_vocab_hash(self) -> str:
        """
        Generate SHA256 hash of vocabulary for consistency validation.

        Uses sorted JSON format for compatibility with vocab_lock.json.

        Returns:
            Hex string of vocabulary hash
        """
        # Create deterministic JSON string (sorted keys)
        vocab_json = json.dumps(self.lookup, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(vocab_json.encode('utf-8')).hexdigest()

    def validate_vocab_size(self, expected_size: int = 1373):
        """
        Validate vocabulary has expected number of tokens.

        Args:
            expected_size: Expected vocabulary size (default 1373)

        Raises:
            ValueError if size mismatch
        """
        actual_size = len(self.lookup)
        if actual_size != expected_size:
            raise ValueError(
                f"Vocabulary size mismatch! Expected {expected_size}, got {actual_size}. "
                f"This indicates vocabulary was built with different token registry."
            )

    def save_metadata(self, output_path: Pathlike):
        """Save vocabulary metadata including hash for vocabulary lock system."""
        metadata = {
            'vocab_size': len(self.lookup),
            'vocab_hash': self.get_vocab_hash(),
            'special_tokens': {
                '[PAD]': 0,
                '[BOS]': 1,
                '[EOS]': 2,
                '[UNK]': 3,
                '[SEP]': 4
            }
        }
        with open(pathlib.Path(output_path).expanduser(), 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def from_vocab_lock(cls, vocab_lock_path: Pathlike) -> typing.Self:
        """
        Load vocabulary from vocab_lock.json file.

        Args:
            vocab_lock_path: Path to vocab_lock.json file

        Returns:
            Vocabulary instance
        """
        with open(pathlib.Path(vocab_lock_path).expanduser(), 'r') as f:
            vocab_data = json.load(f)

        vocab_dict = vocab_data['vocab']

        # Create tuple of words ordered by token ID
        max_id = max(vocab_dict.values())
        words = [''] * (max_id + 1)
        for word, token_id in vocab_dict.items():
            words[token_id] = word

        # Create vocabulary instance (frozen mode)
        instance = cls(tuple(words), is_training=False)

        print(f"Loaded vocabulary from vocab_lock:")
        print(f"  - Size: {len(instance)}")
        print(f"  - Hash: {instance.get_vocab_hash()[:16]}...")

        return instance

    def save_to_vocab_lock(self, output_path: Pathlike):
        """
        Save vocabulary in vocab_lock.json format.

        Args:
            output_path: Path to save vocab_lock.json
        """
        vocab_data = {
            "vocab": self.lookup,
            "vocab_size": len(self.lookup),
            "special_tokens": {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "sep_token": "[SEP]"
            }
        }

        with open(pathlib.Path(output_path).expanduser(), 'w') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        print(f"Saved vocab_lock.json to: {output_path}")
        print(f"  - Size: {len(self.lookup)}")
        print(f"  - Hash: {self.get_vocab_hash()[:16]}...")

    def print_aux(self):
        for k, v in self.aux.items():
            if hasattr(v, '__iter__') and not isinstance(v, str):
                print(
                    "{k}: {v}".format(
                        k=k, v=list(map(functools.partial(round, ndigits=2), v))
                    )
                )
            else:
                print("{k}: {v}".format(k=k, v=v))
