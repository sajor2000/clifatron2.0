"""
Clinical Tokenizer for Qwen2

Provides whitespace-based tokenization for pre-tokenized clinical narratives.
No dependency on token_registry - loads vocabulary from vocab_lock.json.
"""

from .clinical_tokenizer import ClinicalTokenizer, load_tokenizer

__all__ = ["ClinicalTokenizer", "load_tokenizer"]
