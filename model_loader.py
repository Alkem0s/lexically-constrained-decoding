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

    def _build_vocab_surface_map(self):
        """
        Build and cache a mapping: token_id → decoded surface string (lowercased,
        stripped of the SentencePiece boundary marker ▁).

        This is computed once per model load and reused for all constraint lookups.
        Scanning the full vocabulary (~60k tokens) takes ~50ms and is far more
        reliable than trying to guess which encoding variant SPM will choose.
        """
        if hasattr(self, "_vocab_surface"):
            return self._vocab_surface

        vocab_size = self.tokenizer.vocab_size
        surface_map = {}
        for tid in range(vocab_size):
            decoded = self.tokenizer.decode([tid], skip_special_tokens=True)
            # Strip leading space / SPM boundary marker and lowercase
            surface_map[tid] = decoded.strip().lower()
        self._vocab_surface = surface_map
        return surface_map

    def words_to_token_ids(self, words: List[str]) -> List[List[int]]:
        """
        Convert a list of surface-form words to ALL token IDs that could
        contribute to generating that word.

        Root-cause fix for hard/soft exclusion failures and inclusion garbling:

        SentencePiece has multiple valid segmentations of any word.  Simply
        encoding a word in isolation (or with a leading space) only captures
        ONE segmentation path.  The model can still emit the forbidden word via
        a different subword split that was never masked.

        Strategy — scan the entire vocabulary and collect every token whose
        decoded surface:
          a) IS a case-insensitive substring of the target word  → it could be
             a subword piece that assembles into the word during generation.
          b) CONTAINS the target word as a substring → it is the whole word or
             a longer form (e.g. morphological suffix attached).

        For EXCLUSION: we want set (a) ∪ (b) — mask any token that could
        participate in spelling the forbidden word.

        For INCLUSION: we want only set (b) — boost tokens whose surface already
        contains the complete required word, so we never boost ambiguous subword
        fragments that also appear in unrelated words (which caused "The hound",
        "The dossier", "The cadets" instead of the intended words).

        Both sets are returned together here; the caller (flat_token_ids vs
        words_to_token_ids_strict) decides which subset to use.

        Returns list-of-lists: one inner list per word, using the STRICT (b-only)
        subset — safe for inclusion boosting.
        """
        surface_map = self._build_vocab_surface_map()
        result = []
        for word in words:
            word_lower = word.strip().lower()
            # Strict set: token surface contains the full target word.
            # Used for inclusion so we only boost tokens that ARE the word,
            # not ambiguous fragments shared with other words.
            strict_ids = [
                tid for tid, surface in surface_map.items()
                if word_lower in surface and surface  # surface must be non-empty
            ]
            if strict_ids:
                result.append(strict_ids)
            else:
                # Fallback: any token that is a substring piece of the word
                fragment_ids = [
                    tid for tid, surface in surface_map.items()
                    if surface and surface in word_lower
                ]
                if fragment_ids:
                    print(f"  [{word}] No whole-word tokens found; using {len(fragment_ids)} fragments.")
                    result.append(fragment_ids)
                else:
                    print(f"  Warning: '{word}' produced no token IDs — skipping.")
        return result

    def flat_token_ids(self, words: List[str]) -> List[int]:
        """
        Return ALL token IDs that could spell any of the given words —
        including subword fragments (substring pieces).

        Used for EXCLUSION: we need to mask every token that could participate
        in assembling the forbidden word under any segmentation, so we use the
        broader (a) ∪ (b) set rather than the strict (b)-only set used for
        inclusion.
        """
        surface_map = self._build_vocab_surface_map()
        id_set = set()
        for word in words:
            word_lower = word.strip().lower()
            for tid, surface in surface_map.items():
                if not surface:
                    continue
                # Include if: token IS a part of the word, OR token CONTAINS word
                if surface in word_lower or word_lower in surface:
                    id_set.add(tid)
        return list(id_set)

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