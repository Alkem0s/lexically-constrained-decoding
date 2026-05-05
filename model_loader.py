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

    @staticmethod
    def _turkish_lower(s: str) -> str:
        """Python's str.lower() mishandles Turkish İ/I. Apply correct mapping first."""
        return s.replace('İ', 'i').replace('I', 'ı').lower()

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
            surface_map[tid] = self._turkish_lower(decoded.lstrip(" ").strip())
        self._vocab_surface = surface_map
        return surface_map
    
    def get_boundary_mask(self) -> torch.Tensor:
        """
        Create a boolean mask of all tokens that represent a word boundary.
        In SentencePiece, these are tokens starting with ' ' (U+2581), 
        punctuation, or special tokens like <eos>.
        """
        if hasattr(self, "_boundary_mask"):
            return self._boundary_mask
            
        vocab_size = self.tokenizer.vocab_size
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        for tid in range(vocab_size):
            tok = self.tokenizer.convert_ids_to_tokens(tid)
            # Handle special tokens
            if tid in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.tokenizer.unk_token_id]:
                mask[tid] = True
                continue
                
            # SentencePiece boundary marker is ' ' (U+2581)
            if tok.startswith('\u2581') or tok.startswith(' '):
                mask[tid] = True
            # Punctuation / non-alphanumeric (e.g. '.', ',', '!', '-', etc.)
            elif not any(c.isalnum() for c in tok):
                mask[tid] = True
                
        self._boundary_mask = mask
        return mask
    
    @staticmethod
    def expand_turkish_word(word: str) -> List[str]:
        """Generates valid Turkish surface forms using 2-Way and 4-Way Vowel Harmony."""
        word = word.lower()
        vowels = "aeıioöuü"
        last_vowel = next((c for c in reversed(word) if c in vowels), 'a')
        
        is_front = last_vowel in "eiöü"
        is_rounded = last_vowel in "ouöü"
        
        a_vowel = "e" if is_front else "a"
        if is_front and not is_rounded: i_vowel = "i"
        elif is_front and is_rounded: i_vowel = "ü"
        elif not is_front and not is_rounded: i_vowel = "ı"
        else: i_vowel = "u"
        
        ends_with_vowel = word[-1] in vowels
        ends_with_voiceless = word[-1] in "fstkçşhp"
        
        # USE A LIST so the uninflected root word is ALWAYS index 0
        variants_list = [word]
        suffixes = []
        
        lar = "ler" if is_front else "lar"
        suffixes.append(lar)
        
        y_buffer = "y" if ends_with_vowel else ""
        suffixes.append(f"{y_buffer}{a_vowel}")  
        suffixes.append(f"{y_buffer}{i_vowel}")  
        
        d_cons = "t" if ends_with_voiceless else "d"
        suffixes.append(f"{d_cons}{a_vowel}")    
        suffixes.append(f"{d_cons}{a_vowel}n")   
        
        n_buffer = "n" if ends_with_vowel else ""
        suffixes.append(f"{n_buffer}{i_vowel}n") 
        
        if ends_with_vowel:
            suffixes.extend(["m", "n", f"s{i_vowel}"])
        else:
            suffixes.extend([f"{i_vowel}m", f"{i_vowel}n", f"{i_vowel}"])
            
        for suf in suffixes:
            variants_list.append(word + suf)
            
        # Deduplicate while preserving order
        seen = set()
        ordered_variants = []
        for v in variants_list:
            if v not in seen:
                ordered_variants.append(v)
                seen.add(v)
                
        return ordered_variants

    def words_to_token_ids(self, words: List[str], expand_tr: bool = False) -> List[List[int]]:
        """Maps words to token IDs, used by Soft Constraints."""
        surface_map = self._build_vocab_surface_map()
        word_groups = []
        
        for word in words:
            # Expand forms if translating to Turkish
            surface_forms = self.expand_turkish_word(word) if expand_tr else [word]
            
            id_set = set()
            for form in surface_forms:
                word_lower = form.strip().lower()
                
                # Strict matching against the pre-built surface_map
                strict_ids = [tid for tid, surface in surface_map.items()
                              if surface and surface.startswith(word_lower)]
                strict_ids = self.filter_ambiguous_tokens(strict_ids, word_lower, surface_map)
                
                if strict_ids:
                    id_set.update(strict_ids)
                else:
                    # Fragment fallback
                    fragment_ids = [tid for tid, surface in surface_map.items()
                                    if surface and surface in word_lower]
                    if fragment_ids:
                        id_set.update(fragment_ids)
            
            if id_set:
                word_groups.append(list(id_set))
            else:
                print(f"  Warning: '{word}' produced no token IDs — skipping.")
                
        return word_groups

    def words_to_sequences(self, words: List[str], expand_tr: bool = False) -> List[List[List[int]]]:
        """Maps words to continuous token sequences, used by Hard Constraints."""
        word_groups = []
        for word in words:
            # Expand forms if translating to Turkish
            surface_forms = self.expand_turkish_word(word) if expand_tr else [word]
            
            group = []
            for form in surface_forms:
                variants = [
                    " " + form.lower(),
                    " " + form.capitalize(),
                    form.capitalize(),
                    form.lower()
                ]
                for variant in variants:
                    seq = self.tokenizer(variant, add_special_tokens=False).input_ids
                    if seq and seq not in group:
                        group.append(seq)
            if group:
                word_groups.append(group)
        return word_groups
    
    def words_to_stem_sequences(self, words: List[str]) -> List[List[int]]:
        """
        Like words_to_sequences but tokenises only the consonant stem,
        allowing the model to attach Turkish case/possessive suffixes freely.
        """
        sequences = []
        for word in words:
            # Simple stem: strip final vowel if word ends in one
            # (covers halı→hal, hava→hav, etc.)
            stem = word
            if word and word[-1] in 'aeıioöuü' and len(word) > 2:
                stem = word[:-1]
            prefix_stem = " " + stem
            seq = self.tokenizer(prefix_stem, add_special_tokens=False).input_ids
            if seq:
                sequences.append(seq)
            else:
                print(f"  Warning: stem of '{word}' produced no token IDs — skipping.")
        return sequences
    
    @staticmethod
    def filter_ambiguous_tokens(strict_ids, word_lower, surface_map):
        """Remove tokens whose surface is a prefix of words OTHER than target."""
        safe_ids = []
        for tid in strict_ids:
            surface = surface_map[tid]
            other_words = [
                s for s in surface_map.values() 
                if s.startswith(surface) 
                and not s.startswith(word_lower)
                and len(s) <= len(word_lower) + 2  
            ]
            if not other_words:
                safe_ids.append(tid)
        return safe_ids if safe_ids else strict_ids

    def flat_token_ids(self, words: List[str]) -> List[int]:
        """
        Return ALL token IDs that could spell any of the given words.
        """
        MAX_EXCLUSION_IDS = 200
        surface_map = self._build_vocab_surface_map()
        id_set = set()
        for word in words:
            word_lower = word.strip().lower()
            
            # FIX: English "e-drop" heuristic (strike -> strik)
            # This allows us to catch "striking", "strikes", etc.
            stem = word_lower[:-1] if word_lower.endswith('e') else word_lower
            
            broad_ids = [
                tid for tid, surface in surface_map.items()
                if surface and (
                    # THE FIX: Only ban fragments if they are at least 3 characters long.
                    # This prevents banning single letters (like 's' or 'e') that the 
                    # inclusion processor desperately needs to build other words.
                    (len(surface) >= 3 and surface in word_lower)
                    or surface.startswith(stem)    # token starts with the stem
                    or word_lower in surface       # word is embedded in token (e.g. "plane" in "airplane")
                )
            ]
            
            if len(broad_ids) > MAX_EXCLUSION_IDS:
                strict_ids = [
                    tid for tid, surface in surface_map.items()
                    if surface and surface.startswith(word_lower)
                ]
                import warnings
                warnings.warn(
                    f"  [flat_token_ids] '{word}' matched {len(broad_ids)} tokens "
                    f"(>{MAX_EXCLUSION_IDS}); falling back to {len(strict_ids)} "
                    f"whole-word tokens to avoid over-masking.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                id_set.update(strict_ids)
            else:
                id_set.update(broad_ids)
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