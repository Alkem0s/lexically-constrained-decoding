"""
model_loader.py — Load and cache Helsinki-NLP MarianMT models for EN↔TR.

Mirrors the caching pattern from the reference MachineTranslator class:
  - Download from HuggingFace Hub on first run, save locally.
  - Subsequent runs load from local path (local_files_only=True).
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple

import config


# Fix #14: Remind the user to install sacremoses so MarianMT tokenisation
# quality is not degraded.  We do this once at import time rather than
# drowning every tokenizer load in the warning.
try:
    import sacremoses  # noqa: F401
except ImportError:
    import warnings
    warnings.warn(
        "sacremoses is not installed — MarianMT tokenisation may be lower quality.\n"
        "  pip install sacremoses",
        RuntimeWarning,
        stacklevel=1,
    )


class MTModel:
    """
    Wrapper around a single MarianMT seq2seq model + tokenizer.
    Handles local caching, device placement, and token-ID lookup.
    """

    def __init__(self, model_name: str, model_path: str, device: str = None):
        self.model_name  = model_name
        self.model_path  = model_path
        self.device      = device or config.DEVICE

        self._load_or_download()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_or_download(self):
        if os.path.exists(self.model_path):
            print(f"[{self.model_name}] Loading from local cache: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path, local_files_only=True
            )
        else:
            print(f"[{self.model_name}] Downloading from HuggingFace Hub...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            os.makedirs(self.model_path, exist_ok=True)
            self.tokenizer.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)
            print(f"[{self.model_name}] Saved to {self.model_path}")

        self.model.to(self.device)
        self.model.eval()
        print(f"[{self.model_name}] Ready on {self.device}")

    # ── Token-ID utilities ────────────────────────────────────────────────────

    def words_to_token_ids(self, words: List[str]) -> List[List[int]]:
        """
        Convert a list of surface-form words to their token ID(s).

        Fix #2: MarianMT / SentencePiece prepends a word-boundary marker
        (▁, U+2581) when a word appears mid-sentence but NOT when encoded
        in isolation.  Encoding a word in isolation therefore misses the
        token ID that the model actually emits during generation, causing
        hard-exclusion masks and inclusion boosts to silently fail.

        We collect IDs for BOTH variants:
          - bare encoding  (word in isolation, no leading space)
          - spaced encoding (word with a leading space, which SPM maps to ▁word)

        This ensures the processor targets the token the model will actually
        produce, regardless of position in the sequence.

        Returns a list-of-lists: one inner list per input word (deduplicated).
        """
        result = []
        for word in words:
            id_set = set()

            # Variant 1: encode the bare word
            bare_ids = self.tokenizer.encode(word, add_special_tokens=False)
            id_set.update(bare_ids)

            # Variant 2: encode with a leading space so SPM sees it as
            # a continuation token (▁word), which is what the decoder emits
            # for most non-first-position words.
            spaced_ids = self.tokenizer.encode(" " + word, add_special_tokens=False)
            id_set.update(spaced_ids)

            ids = list(id_set)
            if ids:
                result.append(ids)
            else:
                print(f"  Warning: '{word}' produced no token IDs — skipping.")
        return result

    def flat_token_ids(self, words: List[str]) -> List[int]:
        """
        Flatten words_to_token_ids into a single list of unique token IDs.
        Useful for logit masking where we just need a set of IDs to suppress.
        """
        nested = self.words_to_token_ids(words)
        flat   = [tid for sublist in nested for tid in sublist]
        return list(set(flat))

    # ── Encode / decode helpers ───────────────────────────────────────────────

    def encode(self, text: str, max_length: int = config.MAX_LENGTH):
        """Tokenize a single source sentence and return tensors on device."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(self.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode a 1-D or 2-D token tensor to a string."""
        if token_ids.dim() == 2:
            token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


# ── Factory helpers ───────────────────────────────────────────────────────────

def load_en_tr() -> MTModel:
    return MTModel(config.EN_TR_MODEL_NAME, config.EN_TR_MODEL_PATH)


def load_tr_en() -> MTModel:
    return MTModel(config.TR_EN_MODEL_NAME, config.TR_EN_MODEL_PATH)