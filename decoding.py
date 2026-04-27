import torch
import re
from typing import List, Dict, Tuple, Optional

import config
from model_loader import MTModel
from constraints import (
    HardExclusionProcessor,
    HardInclusionProcessor,
    SoftConstraintProcessor,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _turkish_lower(s: str) -> str:
    return s.replace('İ', 'i').replace('I', 'ı').lower()

def _missing_words(translation: str, words: List[str]) -> List[str]:
    """Exact-word check, consistent with evaluation.satisfaction_inclusion."""
    def _contains(text, word):
        w = _turkish_lower(word)
        pattern = r'(^|\W)' + re.escape(w) + r'(?=\W|$)'
        return bool(re.search(pattern, _turkish_lower(text)))
    return [w for w in words if not _contains(translation, w)]


# ── Shared generate wrapper ────────────────────────────────────────────────────

def _generate(
    model_wrapper : MTModel,
    source_text   : str,
    processors    : list,
    num_beams     : int = config.NUM_BEAMS,
    max_length    : int = config.MAX_LENGTH,
) -> torch.Tensor:
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
) -> List[Tuple[str, float]]:
    """
    Generate `num_beams` distinct candidates and decode all of them.
    Returns a list of tuples: (decoded_string, sequence_score).
    Used by combined_hard() for the probabilistic reranking phase.
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
            return_dict_in_generate= True,  # Required to get scores
            output_scores          = True,  # Required to get scores
        )
        
    # Calculate the actual log-probability sequence scores
    transition_scores = model_wrapper.model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    
    # Sum the log-probabilities across the generated tokens (dim=1)
    sequence_scores = transition_scores.sum(dim=1).tolist()
    
    decoded = [
        model_wrapper.tokenizer.decode(seq, skip_special_tokens=True)
        for seq in outputs.sequences
    ]
    
    return list(zip(decoded, sequence_scores))


# ── 1. Unconstrained ──────────────────────────────────────────────────────────

def unconstrained(
    model_wrapper : MTModel,
    source_text   : str,
) -> Tuple[str, List[Dict]]:
    outputs     = _generate(model_wrapper, source_text, processors=[])
    translation = model_wrapper.decode(outputs)
    return translation, []


# ── 2. Hard Exclusion ─────────────────────────────────────────────────────────

def hard_exclusion(
    model_wrapper  : MTModel,
    source_text    : str,
    forbidden_words: List[str],
) -> Tuple[str, List[Dict]]:
    forbidden_ids = model_wrapper.flat_token_ids(forbidden_words)
    if not forbidden_ids:
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
    _force_hard    : bool = False,
) -> Tuple[str, List[Dict]]:
    
    if not _force_hard:
        soft_trans, soft_log = soft_reward_only(
            model_wrapper, source_text, required_words,
            reward_val=config.SOFT_REWARD_STRENGTH * 1.5,   # more aggressive on first pass
        )
        still_missing = _missing_words(soft_trans, required_words)
        if not still_missing:
            return soft_trans, soft_log
        required_words = still_missing

    token_sequences = model_wrapper.words_to_sequences(required_words)
    if not token_sequences:
        return unconstrained(model_wrapper, source_text)

    src_len            = model_wrapper.encode(source_text)["input_ids"].shape[1]
    min_content_tokens = max(3, min(5, src_len // 3))
    eos_id             = model_wrapper.tokenizer.eos_token_id
    boundary_mask      = model_wrapper.get_boundary_mask().to(config.DEVICE)

    log       = []
    processor = HardInclusionProcessor(
        required_token_sequences = token_sequences,
        src_len                  = src_len,
        eos_token_id             = eos_id,
        boundary_mask            = boundary_mask,
        boost                    = config.HARD_INCLUSION_BOOST,
        log_store                = log,
        min_content_tokens       = min_content_tokens,
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

    unconstrained_trans, _ = unconstrained(model_wrapper, source_text)
    baseline_tokens = set(unconstrained_trans.lower().split())

    excl_log = []
    incl_log = []

    # ── THE RERANKING PHASE ──
    if forbidden_ids and token_sequences:
        phase1_excl_log = []
        excl_proc       = HardExclusionProcessor(forbidden_ids, log_store=phase1_excl_log)
        
        # Now returns a list of (candidate_string, score) tuples
        candidates_with_scores = _generate_multi(
            model_wrapper, source_text, processors=[excl_proc],
            num_beams=config.COMBINED_HARD_RERANK_BEAMS,
        )
        
        # Filter for candidates that satisfy the required words
        passing = [
            cand for cand in candidates_with_scores 
            if not _missing_words(cand[0], required_words)
        ]
        
        if passing:
            best_candidate = max(
                passing,
                key=lambda c: (
                    len(set(c[0].lower().split()) & baseline_tokens) / max(len(baseline_tokens), 1)
                    + c[1] / max(len(c[0].split()), 1) * 0.3   # weighted score term
                )
            )[0]
            
            incl_log.append({
                "step": 0, "type": "inclusion", "tokens": {},
                "pending_count": 0,
                "note": f"reranking: {len(passing)}/{len(candidates_with_scores)} passed, best selected by sequence score"
            })
            return best_candidate, phase1_excl_log, incl_log

    processors = []
    if token_sequences:
        src_len            = model_wrapper.encode(source_text)["input_ids"].shape[1]
        min_content_tokens = max(1, min(3, src_len // 4))
        eos_id             = model_wrapper.tokenizer.eos_token_id
        boundary_mask      = model_wrapper.get_boundary_mask().to(config.DEVICE)

        processors.append(
            HardInclusionProcessor(
                required_token_sequences = token_sequences,
                src_len                  = src_len,
                eos_token_id             = eos_id,
                boundary_mask            = boundary_mask,
                log_store                = incl_log,
                min_content_tokens       = min_content_tokens,
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
    penalty_ids = model_wrapper.flat_token_ids(penalty_words or [])
    if not penalty_ids:
        return unconstrained(model_wrapper, source_text)

    log       = []
    processor = SoftConstraintProcessor(
        reward_token_groups = [],
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
    reward_groups = model_wrapper.words_to_token_ids(reward_words or [])
    if not reward_groups:
        return unconstrained(model_wrapper, source_text)

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

    missing = _missing_words(translation, reward_words or [])
    if missing:
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


# ── 5. Pure soft combined (used by combined_soft) ─────────────────────────────

def _soft_combined_pure(
    model_wrapper : MTModel,
    source_text   : str,
    reward_words  : Optional[List[str]] = None,
    penalty_words : Optional[List[str]] = None,
    reward_val    : float = config.SOFT_REWARD_STRENGTH,
    penalty_val   : float = config.SOFT_PENALTY_STRENGTH,
) -> Tuple[str, List[Dict]]:
    
    reward_groups = model_wrapper.words_to_token_ids(reward_words or [])
    penalty_ids   = model_wrapper.flat_token_ids(penalty_words or [])

    if not reward_groups and not penalty_ids:
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
    return translation, log


def combined_soft(
    model_wrapper : MTModel,
    source_text   : str,
    reward_words  : Optional[List[str]] = None,
    penalty_words : Optional[List[str]] = None,
    reward_val    : float = config.SOFT_REWARD_STRENGTH,
    penalty_val   : float = config.SOFT_PENALTY_STRENGTH,
) -> Tuple[str, List[Dict]]:
    
    trans, log = _soft_combined_pure(
        model_wrapper, source_text,
        reward_words=reward_words, penalty_words=penalty_words,
        reward_val=reward_val, penalty_val=penalty_val,
    )

    # ── Post-generation retry (mirrors soft_reward_only escalation) ───────────
    missing = _missing_words(trans, reward_words or [])
    if missing:
        retry_trans, _ = _soft_combined_pure(
            model_wrapper, source_text,
            reward_words=missing, penalty_words=penalty_words,
            reward_val=config.SOFT_REWARD_MAX,   # full strength on retry
            penalty_val=penalty_val,
        )
        log.append({"escalated": "soft_boost", "missing_words": missing})
        still_missing = _missing_words(retry_trans, missing)
        if not still_missing:
            return retry_trans, log
        # If soft retry still fails, log it but return best effort
        log.append({"escalated": "soft_boost_failed", "missing_words": still_missing})
        return trans, log

    return trans, log


# ── 6. Soft constrained with hard fallbacks (production mode) ─────────────────

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

    if not reward_groups and not penalty_ids:
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

    if penalty_words:
        still_forbidden = [w for w in penalty_words if _word_satisfied(translation, w)]
        if still_forbidden:
            hard_excl_ids    = model_wrapper.flat_token_ids(still_forbidden)
            retry_log        = []
            retry_processors = [
                HardExclusionProcessor(hard_excl_ids, log_store=retry_log),
            ]
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

    if reward_words:
        missing = _missing_words(translation, reward_words)
        if missing:
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