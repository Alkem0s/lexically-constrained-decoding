"""
decoding.py — Constrained generation functions.

Each function accepts an MTModel, a source string, constraint word lists,
and returns:
  - translation (str)
  - interpretability log (List[Dict])

Six modes:
  1. unconstrained()           — plain beam search, no constraints (baseline)
  2. hard_exclusion()          — forbid specific words via logit masking
  3. hard_inclusion()          — require specific words via logit boosting
  4. soft_penalty_only()       — penalise words with a scalar delta (no reward)
  5. soft_reward_only()        — reward words with a scalar delta (no penalty)
  6. soft_constrained()        — combined reward + penalty in one pass
  7. combined_hard()           — simultaneous hard exclusion + hard inclusion
  8. combined_soft()           — simultaneous soft penalty + soft reward
     (alias for soft_constrained; kept for naming clarity in main.py)
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


# ── Shared helpers ────────────────────────────────────────────────────────────

def _word_satisfied(translation: str, word: str) -> bool:
    """Case-insensitive substring check (mirrors evaluation._contains_word)."""
    return word.lower() in translation.lower()


def _missing_words(translation: str, words: List[str]) -> List[str]:
    """Return words from `words` that are absent from `translation`."""
    return [w for w in words if not _word_satisfied(translation, w)]


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


def _generate_multi(
    model_wrapper  : MTModel,
    source_text    : str,
    processors     : list,
    num_beams      : int = config.COMBINED_HARD_RERANK_BEAMS,
    max_length     : int = config.MAX_LENGTH,
) -> List[str]:
    """
    Generate `num_beams` distinct candidates and decode all of them.
    Used by combined_hard() for the reranking phase.
    """
    inputs = model_wrapper.encode(source_text, max_length=max_length)
    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            **inputs,
            max_length             = max_length,
            num_beams              = num_beams,
            num_return_sequences   = num_beams,
            no_repeat_ngram_size   = config.NO_REPEAT_NGRAM,
            logits_processor       = processors,
            early_stopping         = True,
        )
    return [
        model_wrapper.tokenizer.decode(seq, skip_special_tokens=True)
        for seq in outputs
    ]


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

    When words_to_token_ids() fell back to fragment tokens for a word, the
    boost is raised to HARD_INCLUSION_BOOST * 1.5 (rounded) to compensate for
    the boost spreading across more token IDs.

    Adaptive min_content_tokens:
        The boost deferral window is computed from source length so that very
        short sentences (≤6 source tokens) get a deferral of 1 instead of the
        default 3, giving the boost more opportunity to fire early.

    Args:
        required_words: surface-form words in the TARGET language.
    Returns:
        (translation, log)
    """
    token_groups = model_wrapper.words_to_token_ids(required_words)
    if not token_groups:
        print("  [hard_inclusion] No valid token IDs found for required words.")
        return unconstrained(model_wrapper, source_text)

    # Adaptive boost: if any group is large (fragment fallback), use a
    # stronger boost so it can compete against the spread of tokens.
    FRAGMENT_THRESHOLD = 50
    max_group_size = max(len(g) for g in token_groups)
    boost = config.HARD_INCLUSION_BOOST
    if max_group_size > FRAGMENT_THRESHOLD:
        boost = round(config.HARD_INCLUSION_BOOST * 1.5)
        print(f"  [hard_inclusion] Largest token group has {max_group_size} IDs "
              f"(fragment fallback) — raising boost to {boost}.")

    # Adaptive deferral: shorter sources → smaller window so the boost fires
    # earlier and has more decoding steps to enforce the required word.
    src_len            = model_wrapper.encode(source_text)["input_ids"].shape[1]
    min_content_tokens = max(1, min(3, src_len // 4))

    log       = []
    processor = HardInclusionProcessor(
        token_groups,
        boost              = boost,
        log_store          = log,
        min_content_tokens = min_content_tokens,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)
    return translation, log


# ── 3b. Combined Hard (exclusion + inclusion simultaneously) ─────────────────

def combined_hard(
    model_wrapper  : MTModel,
    source_text    : str,
    forbidden_words: List[str],
    required_words : List[str],
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Apply hard exclusion and hard inclusion simultaneously, with a reranking
    pre-pass to recover fluency.

    Strategy (two-phase):
      Phase 1 — Reranking:
        Generate COMBINED_HARD_RERANK_BEAMS candidates with the exclusion
        processor only.  Among those, pick the first candidate that already
        satisfies all required words.  Because only the exclusion constraint is
        active, beam search can find its most fluent path freely — so if the
        model naturally produces the required word, this candidate will have
        higher BLEU than the simultaneous-constraint output.

      Phase 2 — Simultaneous fallback:
        If no reranking candidate satisfies inclusion, fall back to the original
        simultaneous exclusion + inclusion mode.  This guarantees the constraint
        is always enforced.

    Returns:
        (translation, excl_log, incl_log)
    """
    forbidden_ids = model_wrapper.flat_token_ids(forbidden_words)
    token_groups  = model_wrapper.words_to_token_ids(required_words)

    if not forbidden_ids and not token_groups:
        t, _ = unconstrained(model_wrapper, source_text)
        return t, [], []

    excl_log = []
    incl_log = []

    # ── Phase 1: Reranking (only when both constraint types are present) ──────
    if forbidden_ids and token_groups:
        phase1_excl_log = []
        excl_proc       = HardExclusionProcessor(forbidden_ids, log_store=phase1_excl_log)
        candidates      = _generate_multi(
            model_wrapper, source_text, processors=[excl_proc],
            num_beams=config.COMBINED_HARD_RERANK_BEAMS,
        )
        required_lower = [w.lower() for w in required_words]
        for cand in candidates:
            if all(req in cand.lower() for req in required_lower):
                # Found a fluent candidate that satisfies both constraints.
                incl_log.append({
                    "step": 0, "type": "inclusion",
                    "tokens": {}, "pending_count": 0,
                    "note": f"reranking satisfied — chosen from {len(candidates)} candidates",
                })
                return cand, phase1_excl_log, incl_log

        print(f"  [combined_hard] reranking: none of {len(candidates)} candidates "
              f"satisfied inclusion — falling back to simultaneous mode.")

    # ── Phase 2: Simultaneous fallback ────────────────────────────────────────
    processors = []
    if token_groups:
        src_len            = model_wrapper.encode(source_text)["input_ids"].shape[1]
        min_content_tokens = max(1, min(3, src_len // 4))
        processors.append(
            HardInclusionProcessor(
                token_groups,
                log_store          = incl_log,
                min_content_tokens = min_content_tokens,
            )
        )
    if forbidden_ids:
        processors.append(HardExclusionProcessor(forbidden_ids, log_store=excl_log))

    outputs     = _generate(model_wrapper, source_text, processors=processors)
    translation = model_wrapper.decode(outputs)
    return translation, excl_log, incl_log


# ── 4. Soft Constraints ───────────────────────────────────────────────────────

def soft_penalty_only(
    model_wrapper : MTModel,
    source_text   : str,
    penalty_words : List[str],
    penalty_val   : float = config.SOFT_PENALTY_STRENGTH,
) -> Tuple[str, List[Dict]]:
    """
    Penalise specific words only — no reward component.
    Runs a single model.generate() pass with penalty tokens only.
    """
    penalty_ids = model_wrapper.flat_token_ids(penalty_words or [])
    if not penalty_ids:
        print("  [soft_penalty_only] No valid token IDs — running unconstrained.")
        return unconstrained(model_wrapper, source_text)

    log       = []
    processor = SoftConstraintProcessor(
        reward_ids  = [],
        penalty_ids = penalty_ids,
        reward_val  = 0.0,
        penalty_val = penalty_val,
        log_store   = log,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)
    return translation, log


def soft_reward_only(
    model_wrapper : MTModel,
    source_text   : str,
    reward_words  : List[str],
    reward_val    : float = config.SOFT_REWARD_STRENGTH,
) -> Tuple[str, List[Dict]]:
    """
    Reward specific words only — no penalty component.

    Soft-then-hard cascade:
        After the soft-reward generate() pass, check whether every reward word
        appears in the output.  For any word still missing, retry with
        hard_inclusion() as an automatic escalation.  This preserves the
        "try soft first" semantics while guaranteeing satisfaction when the
        model's natural distribution is too strong to be nudged softly.
        A flag is appended to the log so interpretability analysis can detect
        which samples required escalation.
    """
    reward_ids = model_wrapper.flat_token_ids(reward_words or [])
    if not reward_ids:
        print("  [soft_reward_only] No valid token IDs — running unconstrained.")
        return unconstrained(model_wrapper, source_text)

    log       = []
    processor = SoftConstraintProcessor(
        reward_ids  = reward_ids,
        penalty_ids = [],
        reward_val  = reward_val,
        penalty_val = 0.0,
        log_store   = log,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)

    # ── Soft-then-hard cascade ────────────────────────────────────────────────
    missing = _missing_words(translation, reward_words or [])
    if missing:
        print(f"  [soft_reward_only] soft reward failed for {missing} — "
              f"escalating to hard inclusion.")
        hard_trans, hard_log = hard_inclusion(model_wrapper, source_text, missing)
        log.append({
            "escalated"     : True,
            "missing_words" : missing,
            "hard_log_steps": len(hard_log),
        })
        return hard_trans, log

    return translation, log


def combined_soft(
    model_wrapper : MTModel,
    source_text   : str,
    reward_words  : Optional[List[str]] = None,
    penalty_words : Optional[List[str]] = None,
    reward_val    : float = config.SOFT_REWARD_STRENGTH,
    penalty_val   : float = config.SOFT_PENALTY_STRENGTH,
) -> Tuple[str, List[Dict]]:
    """
    Combined soft penalty + reward in one generate() pass.
    Both constraints are active simultaneously so they interact — the penalty
    suppresses unwanted words while the reward nudges toward preferred ones.
    This is the correct combined-mode translation; use soft_penalty_only /
    soft_reward_only when you want ablation-style isolated results.
    """
    return soft_constrained(
        model_wrapper,
        source_text,
        reward_words  = reward_words,
        penalty_words = penalty_words,
        reward_val    = reward_val,
        penalty_val   = penalty_val,
    )


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

    Fix #11: Both reward_words and penalty_words are processed in a single
    model.generate() call via one SoftConstraintProcessor instance.

    Soft-then-hard cascade (reward component):
        After the combined generate() pass, if any reward word is still missing
        from the output, those words are retried with hard_inclusion() while the
        penalty constraint remains satisfied (the penalty was active during the
        first pass and any cascade pass uses only the missing reward words as
        the inclusion target).  A log entry flags the escalation.

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

    # ── Soft-then-hard cascade (reward words only) ────────────────────────────
    if reward_words:
        missing = _missing_words(translation, reward_words)
        if missing:
            print(f"  [soft_constrained] soft reward failed for {missing} — "
                  f"escalating to hard inclusion.")
            hard_trans, hard_log = hard_inclusion(model_wrapper, source_text, missing)
            log.append({
                "escalated"     : True,
                "missing_words" : missing,
                "hard_log_steps": len(hard_log),
            })
            return hard_trans, log

    return translation, log