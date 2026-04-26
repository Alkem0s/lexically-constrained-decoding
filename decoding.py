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

import re

def _turkish_lower(s: str) -> str:
    return s.replace('İ', 'i').replace('I', 'ı').lower()

def _word_satisfied(translation: str, word: str) -> bool:
    """
    Turkish-aware word presence check — exactly mirrors evaluation._contains_word
    so that escalation triggers fire on the same violations the evaluator catches.
    """
    word_lower = _turkish_lower(word)
    stem = word_lower[:-1] if word_lower.endswith('e') else word_lower
    pattern = r'(^|\W)' + re.escape(stem)
    return bool(re.search(pattern, _turkish_lower(translation)))


def _missing_words(translation: str, words: List[str]) -> List[str]:
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
    _force_hard    : bool = False,      # ← new: skip soft gate when True
) -> Tuple[str, List[Dict]]:
    """
    Require specific words via logit boosting.
    Soft-first gate: attempt soft reward first; escalate to hard boosting only
    for words the soft reward cannot place.
    Set _force_hard=True when calling from soft_reward_only to break the
    mutual recursion cycle.
    """
    # ── Soft-first gate (skipped when called from soft_reward_only) ───────────
    if not _force_hard:
        soft_trans, soft_log = soft_reward_only(model_wrapper, source_text, required_words)
        still_missing = _missing_words(soft_trans, required_words)
        if not still_missing:
            print(f"  [hard_inclusion] soft-first gate satisfied all words — skipping hard boost.")
            return soft_trans, soft_log
        print(f"  [hard_inclusion] soft-first gate failed for {still_missing} — applying hard boost.")
        required_words = still_missing   # only hard-boost the stubborn words

    # ── Hard boosting ─────────────────────────────────────────────────────────
    token_sequences = model_wrapper.words_to_sequences(required_words)
    if not token_sequences:
        print("  [hard_inclusion] No valid token sequences found — returning unconstrained.")
        return unconstrained(model_wrapper, source_text)

    src_len            = model_wrapper.encode(source_text)["input_ids"].shape[1]
    min_content_tokens = max(1, min(3, src_len // 4))

    log       = []
    processor = HardInclusionProcessor(
        token_sequences,
        boost              = config.HARD_INCLUSION_BOOST,
        log_store          = log,
        min_content_tokens = min_content_tokens,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)
    return translation, log

def combined_hard(
    model_wrapper  : MTModel,
    source_text    : str,
    forbidden_words: List[str],
    required_words : List[str],
) -> Tuple[str, List[Dict], List[Dict]]:
    
    forbidden_ids   = model_wrapper.flat_token_ids(forbidden_words)
    token_sequences = model_wrapper.words_to_sequences(required_words)

    if not forbidden_ids and not token_sequences:
        t, _ = unconstrained(model_wrapper, source_text)
        return t, [], []

    excl_log = []
    incl_log = []

    # ── Phase 1: Reranking ────────────────────────────────────────────────────
    if forbidden_ids and token_sequences:
        phase1_excl_log = []
        excl_proc       = HardExclusionProcessor(forbidden_ids, log_store=phase1_excl_log)
        candidates      = _generate_multi(
            model_wrapper, source_text, processors=[excl_proc],
            num_beams=config.COMBINED_HARD_RERANK_BEAMS,
        )
        required_lower = [w.lower() for w in required_words]
        for cand in candidates:
            if all(req in cand.lower() for req in required_lower):
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
    if token_sequences:
        src_len            = model_wrapper.encode(source_text)["input_ids"].shape[1]
        min_content_tokens = max(1, min(3, src_len // 4))
        processors.append(
            HardInclusionProcessor(
                token_sequences, 
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
        reward_token_groups = [],  # Updated argument name to match the new processor
        penalty_ids         = penalty_ids,
        reward_val          = 0.0,
        penalty_val         = penalty_val,
        log_store           = log,
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
    Escalation ladder: soft reward → boosted soft reward → hard inclusion.
    _force_hard=True is passed to hard_inclusion to prevent mutual recursion
    with the soft-first gate in hard_inclusion().
    """
    reward_groups = model_wrapper.words_to_token_ids(reward_words or [])
    if not reward_groups:
        print("  [soft_reward_only] No valid token IDs — running unconstrained.")
        return unconstrained(model_wrapper, source_text)

    # ── Tier 1: standard soft reward ─────────────────────────────────────────
    log       = []
    processor = SoftConstraintProcessor(
        reward_token_groups = reward_groups,
        penalty_ids         = [],
        reward_val          = reward_val,
        penalty_val         = 0.0,
        log_store           = log,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)

    # ── Tier 2: boosted soft reward ───────────────────────────────────────────
    missing = _missing_words(translation, reward_words or [])
    if missing:
        print(f"  [soft_reward_only] soft reward failed for {missing} — "
              f"retrying with boosted reward (tier 2).")
        boost_groups = model_wrapper.words_to_token_ids(missing)
        boost_log    = []
        boost_proc   = SoftConstraintProcessor(
            reward_token_groups = boost_groups,
            penalty_ids         = [],
            reward_val          = config.SOFT_REWARD_MAX,
            penalty_val         = 0.0,
            log_store           = boost_log,
        )
        boost_outputs = _generate(model_wrapper, source_text, processors=[boost_proc])
        boost_trans   = model_wrapper.decode(boost_outputs)
        still_missing = _missing_words(boost_trans, missing)

        if not still_missing:
            log.append({"escalated": "soft_boost", "missing_words": missing})
            return boost_trans, log

        # ── Tier 3: hard inclusion — _force_hard=True breaks mutual recursion ─
        print(f"  [soft_reward_only] boosted reward still failed for {still_missing} — "
              f"escalating to hard inclusion.")
        hard_trans, hard_log = hard_inclusion(
            model_wrapper, source_text, still_missing, _force_hard=True
        )
        log.append({
            "escalated"      : "hard_inclusion",
            "missing_words"  : still_missing,
            "hard_log_steps" : len(hard_log),
        })
        return hard_trans, log

    return translation, log


def soft_constrained(
    model_wrapper : MTModel,
    source_text   : str,
    reward_words  : Optional[List[str]] = None,
    penalty_words : Optional[List[str]] = None,
    reward_val    : float = config.SOFT_REWARD_STRENGTH,
    penalty_val   : float = config.SOFT_PENALTY_STRENGTH,
) -> Tuple[str, List[Dict]]:
    reward_groups = model_wrapper.words_to_token_ids(reward_words or [])
    penalty_ids   = model_wrapper.flat_token_ids(penalty_words or [])
    print(f"  [soft_constrained] reward_groups={[len(g) for g in reward_groups]}, penalty_ids={len(penalty_ids)}")

    if not reward_groups and not penalty_ids:
        print("  [soft_constrained] No valid token IDs — running unconstrained.")
        return unconstrained(model_wrapper, source_text)

    log       = []
    processor = SoftConstraintProcessor(
        reward_token_groups = reward_groups,
        penalty_ids         = penalty_ids,
        reward_val          = reward_val,
        penalty_val         = penalty_val,
        log_store           = log,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)

    # ── Check penalty satisfaction — escalate to hard exclusion if soft failed ─
    # soft_penalty can fail silently: if the forbidden word still appears after
    # the -12 nudge, there is no existing fallback, causing soft_combined to
    # report 0.95 satisfaction.  We add one here.
    if penalty_words:
        still_forbidden = [w for w in penalty_words if _word_satisfied(translation, w)]
        if still_forbidden:
            print(f"  [soft_constrained] soft penalty failed for {still_forbidden} — "
                  f"retrying with hard exclusion.")
            hard_excl_ids = model_wrapper.flat_token_ids(still_forbidden)
            retry_log     = []
            retry_processors = [
                HardExclusionProcessor(hard_excl_ids, log_store=retry_log),
            ]
            # Preserve any reward processor if reward words are still needed
            if reward_groups:
                still_missing = _missing_words(translation, reward_words or [])
                if still_missing:
                    retry_reward_groups = model_wrapper.words_to_token_ids(still_missing)
                    retry_processors.append(SoftConstraintProcessor(
                        reward_token_groups = retry_reward_groups,
                        penalty_ids         = [],
                        reward_val          = config.SOFT_REWARD_MAX,
                        penalty_val         = 0.0,
                        log_store           = retry_log,
                    ))
            retry_outputs = _generate(model_wrapper, source_text, processors=retry_processors)
            translation   = model_wrapper.decode(retry_outputs)
            log.append({
                "escalated"      : "hard_exclusion",
                "still_forbidden": still_forbidden,
                "retry_log_steps": len(retry_log),
            })

    # ── Check reward satisfaction — cascade as before ─────────────────────────
    if reward_words:
        missing = _missing_words(translation, reward_words)
        if missing:
            print(f"  [soft_constrained] soft reward failed for {missing} — "
                  f"retrying with boosted reward before hard fallback.")
            retry_groups = model_wrapper.words_to_token_ids(missing)
            retry_log    = []
            retry_proc   = SoftConstraintProcessor(
                reward_token_groups = retry_groups,
                penalty_ids         = model_wrapper.flat_token_ids(penalty_words or []),
                reward_val          = config.SOFT_REWARD_MAX,
                penalty_val         = penalty_val,
                log_store           = retry_log,
            )
            retry_outputs = _generate(model_wrapper, source_text, processors=[retry_proc])
            retry_trans   = model_wrapper.decode(retry_outputs)
            still_missing = _missing_words(retry_trans, missing)
            if not still_missing:
                log.append({"escalated": "soft_boost", "missing_words": missing})
                return retry_trans, log
            hard_trans, hard_log = hard_inclusion(model_wrapper, source_text, still_missing)
            log.append({"escalated": "hard_inclusion", "missing_words": still_missing,
                        "hard_log_steps": len(hard_log)})
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