"""
decoding.py — Constrained generation functions.

Each function accepts an MTModel, a source string, constraint word lists,
and returns:
  - translation (str)
  - interpretability log (List[Dict])

Four modes:
  1. unconstrained()       — plain beam search, no constraints (baseline)
  2. hard_exclusion()      — forbid specific words via logit masking
  3. hard_inclusion()      — require specific words via logit boosting
  4. soft_constrained()    — reward / penalise words with a scalar delta
"""

import torch
from typing import List, Dict, Tuple, Optional

import config
from model_loader import MTModel
from constraints import (
    HardExclusionProcessor,
    HardInclusionProcessor,
    SoftConstraintProcessor,
)


# ── Shared generate wrapper ────────────────────────────────────────────────────

def _generate(
    model_wrapper : MTModel,
    source_text   : str,
    processors    : list,
    num_beams     : int = config.NUM_BEAMS,
    max_length    : int = config.MAX_LENGTH,
) -> torch.Tensor:
    """Run model.generate() with the supplied logits processors."""
    inputs = model_wrapper.encode(source_text, max_length=max_length)

    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            **inputs,
            max_length             = max_length,
            num_beams              = num_beams,
            no_repeat_ngram_size   = config.NO_REPEAT_NGRAM,
            logits_processor       = processors,
            early_stopping         = True,
        )
    return outputs


# ── 1. Unconstrained ──────────────────────────────────────────────────────────

def unconstrained(
    model_wrapper : MTModel,
    source_text   : str,
) -> Tuple[str, List[Dict]]:
    """
    Plain beam search — no logits manipulation.
    Returns (translation, empty_log).
    """
    outputs     = _generate(model_wrapper, source_text, processors=[])
    translation = model_wrapper.decode(outputs)
    return translation, []


# ── 2. Hard Exclusion ─────────────────────────────────────────────────────────

def hard_exclusion(
    model_wrapper  : MTModel,
    source_text    : str,
    forbidden_words: List[str],
) -> Tuple[str, List[Dict]]:
    """
    Forbid a list of words by setting their token logits to -inf.

    Args:
        forbidden_words: surface-form words in the TARGET language.
    Returns:
        (translation, log)  where log contains per-step original logit info.
    """
    forbidden_ids = model_wrapper.flat_token_ids(forbidden_words)
    if not forbidden_ids:
        print("  [hard_exclusion] No valid token IDs found for forbidden words.")
        return unconstrained(model_wrapper, source_text)

    log       = []
    processor = HardExclusionProcessor(forbidden_ids, log_store=log)
    outputs   = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)
    return translation, log


# ── 3. Hard Inclusion ─────────────────────────────────────────────────────────

def hard_inclusion(
    model_wrapper  : MTModel,
    source_text    : str,
    required_words : List[str],
) -> Tuple[str, List[Dict]]:
    """
    Require specific words to appear in the output by boosting their logits
    every step until they are generated.

    Args:
        required_words: surface-form words in the TARGET language.
    Returns:
        (translation, log)
    """
    token_groups = model_wrapper.words_to_token_ids(required_words)
    if not token_groups:
        print("  [hard_inclusion] No valid token IDs found for required words.")
        return unconstrained(model_wrapper, source_text)

    log       = []
    processor = HardInclusionProcessor(token_groups, log_store=log)
    outputs   = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)
    return translation, log


# ── 4. Soft Constraints ───────────────────────────────────────────────────────

def soft_constrained(
    model_wrapper : MTModel,
    source_text   : str,
    reward_words  : Optional[List[str]] = None,
    penalty_words : Optional[List[str]] = None,
    reward_val    : float = config.SOFT_REWARD_STRENGTH,
    penalty_val   : float = config.SOFT_PENALTY_STRENGTH,
) -> Tuple[str, List[Dict]]:
    """
    Nudge generation by adding a reward or penalty scalar to token logits.
    Unlike hard constraints, the model may still override the nudge if the
    constraint strongly conflicts with fluency.

    Args:
        reward_words  : words (target language) to encourage.
        penalty_words : words (target language) to discourage.
    Returns:
        (translation, log)
    """
    reward_ids  = model_wrapper.flat_token_ids(reward_words  or [])
    penalty_ids = model_wrapper.flat_token_ids(penalty_words or [])

    if not reward_ids and not penalty_ids:
        print("  [soft_constrained] No valid token IDs — running unconstrained.")
        return unconstrained(model_wrapper, source_text)

    log       = []
    processor = SoftConstraintProcessor(
        reward_ids  = reward_ids,
        penalty_ids = penalty_ids,
        reward_val  = reward_val,
        penalty_val = penalty_val,
        log_store   = log,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)
    return translation, log
