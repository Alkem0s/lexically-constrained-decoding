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

def _missing_words(translation: str, words: List[str], is_tr_target: bool = False) -> List[str]:
    """Exact-word check, or stem-check for Turkish."""
    def _contains(text, word):
        w = _turkish_lower(word)
        if is_tr_target:
            # Match the word boundary before, but allow valid Turkish letters after!
            pattern = r'(^|\W)' + re.escape(w) + r'[a-zçğıöşü]*'
        else:
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
    
    # Check if we are translating into Turkish
    is_en_tr = "en-tr" in model_wrapper.model_name
    
    # The soft-first gate has been removed. HardInclusionProcessor now natively
    # handles early token leniency via HARD_INCL_EARLY_TOKENS.
    # Expand Turkish constraints, leave English alone
    token_sequences = model_wrapper.words_to_sequences(required_words, expand_tr=is_en_tr)
    if not token_sequences:
        return unconstrained(model_wrapper, source_text)

    src_len            = model_wrapper.encode(source_text)["input_ids"].shape[1]
    min_content_tokens = max(3, min(5, src_len // 3))
    eos_id             = model_wrapper.tokenizer.eos_token_id
    boundary_mask      = model_wrapper.get_boundary_mask().to(config.DEVICE)

    # Select the correct parameters based on language direction
    if is_en_tr:
        active_boost        = config.HARD_INCLUSION_BOOST_TR
        active_penalty      = config.SUFFIX_PENALTY_TR
        active_early_tokens = config.HARD_INCL_EARLY_TOKENS_TR
        active_sweet_rank   = config.HARD_INCL_SWEET_RANK_TR
        active_sweet_buffer = config.HARD_INCL_SWEET_BUFFER_TR
        active_anchor_start = config.HARD_INCL_ANCHOR_START_TR
        active_anchor_range = config.HARD_INCL_ANCHOR_RANGE_TR
    else:
        active_boost        = config.HARD_INCLUSION_BOOST_EN
        active_penalty      = config.SUFFIX_PENALTY_EN
        active_early_tokens = config.HARD_INCL_EARLY_TOKENS_EN
        active_sweet_rank   = config.HARD_INCL_SWEET_RANK_EN
        active_sweet_buffer = config.HARD_INCL_SWEET_BUFFER_EN
        active_anchor_start = config.HARD_INCL_ANCHOR_START_EN
        active_anchor_range = config.HARD_INCL_ANCHOR_RANGE_EN

    log       = []
    processor = HardInclusionProcessor(
        required_token_sequences = token_sequences,
        src_len                  = src_len,
        eos_token_id             = eos_id,
        boundary_mask            = boundary_mask,
        boost                    = active_boost,
        log_store                = log,
        min_content_tokens       = min_content_tokens,
        suffix_penalty           = active_penalty,
        early_tokens             = active_early_tokens,
        sweet_rank               = active_sweet_rank,
        sweet_buffer             = active_sweet_buffer,
        anchor_start             = active_anchor_start,
        anchor_range             = active_anchor_range
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

    is_en_tr = "en-tr" in model_wrapper.model_name

    forbidden_ids   = model_wrapper.flat_token_ids(forbidden_words)
    token_sequences = model_wrapper.words_to_sequences(required_words, expand_tr=is_en_tr)

    if not forbidden_ids and not token_sequences:
        t, _ = unconstrained(model_wrapper, source_text)
        return t, [], []

    unconstrained_trans, _ = unconstrained(model_wrapper, source_text)
    baseline_tokens = set(unconstrained_trans.lower().split())

    excl_log = []
    incl_log = []

    processors = []
    if forbidden_ids:
        processors.append(HardExclusionProcessor(forbidden_ids, log_store=excl_log))

    if token_sequences:
        src_len            = model_wrapper.encode(source_text)["input_ids"].shape[1]
        min_content_tokens = max(1, min(3, src_len // 4))
        eos_id             = model_wrapper.tokenizer.eos_token_id
        boundary_mask      = model_wrapper.get_boundary_mask().to(config.DEVICE)
        if is_en_tr:
            active_boost        = config.HARD_COMBINED_BOOST_TR
            active_penalty      = config.HARD_COMBINED_SUFFIX_PENALTY_TR
            active_early_tokens = config.HARD_COMBINED_EARLY_TOKENS_TR
            active_sweet_rank   = config.HARD_COMBINED_SWEET_RANK_TR
            active_sweet_buffer = config.HARD_COMBINED_SWEET_BUFFER_TR
            active_anchor_start = config.HARD_COMBINED_ANCHOR_START_TR
            active_anchor_range = config.HARD_COMBINED_ANCHOR_RANGE_TR
        else:
            active_boost        = config.HARD_COMBINED_BOOST_EN
            active_penalty      = config.HARD_COMBINED_SUFFIX_PENALTY_EN
            active_early_tokens = config.HARD_COMBINED_EARLY_TOKENS_EN
            active_sweet_rank   = config.HARD_COMBINED_SWEET_RANK_EN
            active_sweet_buffer = config.HARD_COMBINED_SWEET_BUFFER_EN
            active_anchor_start = config.HARD_COMBINED_ANCHOR_START_EN
            active_anchor_range = config.HARD_COMBINED_ANCHOR_RANGE_EN

        processors.append(
            HardInclusionProcessor(
                required_token_sequences = token_sequences,
                src_len                  = src_len,
                eos_token_id             = eos_id,
                boundary_mask            = boundary_mask,
                boost                    = active_boost,
                log_store                = incl_log,
                min_content_tokens       = min_content_tokens,
                suffix_penalty           = active_penalty,
                early_tokens             = active_early_tokens,
                sweet_rank               = active_sweet_rank,
                sweet_buffer             = active_sweet_buffer,
                anchor_start             = active_anchor_start,
                anchor_range             = active_anchor_range
            )
        )

    outputs     = _generate(model_wrapper, source_text, processors=processors)
    translation = model_wrapper.decode(outputs)

    return translation, excl_log, incl_log


# ── 4. Soft Constraints ───────────────────────────────────────────────────────

def soft_penalty_only(
    model_wrapper : MTModel,
    source_text   : str,
    penalty_words : List[str],
    penalty_val   : Optional[float] = None,
) -> Tuple[str, List[Dict]]:
    is_en_tr = "en-tr" in model_wrapper.model_name
    if penalty_val is None:
        penalty_val = config.SOFT_PENALTY_STRENGTH_TR if is_en_tr else config.SOFT_PENALTY_STRENGTH_EN

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
    reward_val    : Optional[float] = None,
) -> Tuple[str, List[Dict]]:
    
    is_en_tr = "en-tr" in model_wrapper.model_name
    if reward_val is None:
        reward_val = config.SOFT_REWARD_STRENGTH_TR if is_en_tr else config.SOFT_REWARD_STRENGTH_EN

    reward_groups = model_wrapper.words_to_token_ids(reward_words or [], expand_tr=is_en_tr)
    if not reward_groups:
        return unconstrained(model_wrapper, source_text)

    if is_en_tr:
        active_curriculum_rate = config.SOFT_REWARD_CURRICULUM_RATE_TR
        active_reward_max = config.SOFT_REWARD_MAX_TR
        active_anchor_offset = config.ANCHOR_OFFSET_TR
        active_contextual_nudge = config.CONTEXTUAL_NUDGE_TR
    else:
        active_curriculum_rate = config.SOFT_REWARD_CURRICULUM_RATE_EN
        active_reward_max = config.SOFT_REWARD_MAX_EN
        active_anchor_offset = config.ANCHOR_OFFSET_EN
        active_contextual_nudge = config.CONTEXTUAL_NUDGE_EN

    log       = []
    processor = SoftConstraintProcessor(
        reward_token_groups = reward_groups,
        penalty_ids         = [],
        reward_val          = reward_val,
        penalty_val         = 0.0,
        curriculum_rate     = active_curriculum_rate,
        max_reward          = active_reward_max,
        anchor_offset       = active_anchor_offset,
        contextual_nudge    = active_contextual_nudge,
        log_store           = log,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)

    # PROPERLY PASS is_tr_target
    missing = _missing_words(translation, reward_words or [], is_tr_target=is_en_tr)
    if missing:
        boost_groups = model_wrapper.words_to_token_ids(missing, expand_tr=is_en_tr)
        boost_log    = []
        boost_proc   = SoftConstraintProcessor(
            reward_token_groups = boost_groups,
            penalty_ids         = [],
            reward_val          = active_reward_max,
            penalty_val         = 0.0,
            curriculum_rate     = active_curriculum_rate,
            max_reward          = active_reward_max,
            anchor_offset       = active_anchor_offset,
            contextual_nudge    = active_contextual_nudge,
            log_store           = boost_log,
        )
        boost_outputs = _generate(model_wrapper, source_text, processors=[boost_proc])
        boost_trans   = model_wrapper.decode(boost_outputs)
        still_missing = _missing_words(boost_trans, missing, is_tr_target=is_en_tr)

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
    reward_val    : Optional[float] = None,
    penalty_val   : Optional[float] = None,
    curriculum_rate: Optional[float] = None,
    reward_max    : Optional[float] = None,
    anchor_offset : Optional[float] = None,
) -> Tuple[str, List[Dict]]:
    
    is_en_tr = "en-tr" in model_wrapper.model_name
    if reward_val is None:
        reward_val = config.SOFT_REWARD_STRENGTH_TR if is_en_tr else config.SOFT_REWARD_STRENGTH_EN
    if penalty_val is None:
        penalty_val = config.SOFT_PENALTY_STRENGTH_TR if is_en_tr else config.SOFT_PENALTY_STRENGTH_EN

    reward_groups = model_wrapper.words_to_token_ids(reward_words or [], expand_tr=is_en_tr)
    penalty_ids   = model_wrapper.flat_token_ids(penalty_words or [])

    if not reward_groups and not penalty_ids:
        return unconstrained(model_wrapper, source_text)

    if curriculum_rate is None:
        curriculum_rate = config.SOFT_REWARD_CURRICULUM_RATE_TR if is_en_tr else config.SOFT_REWARD_CURRICULUM_RATE_EN
    if reward_max is None:
        reward_max = config.SOFT_REWARD_MAX_TR if is_en_tr else config.SOFT_REWARD_MAX_EN
    if anchor_offset is None:
        anchor_offset = config.ANCHOR_OFFSET_TR if is_en_tr else config.ANCHOR_OFFSET_EN
    active_contextual_nudge = config.CONTEXTUAL_NUDGE_TR if is_en_tr else config.CONTEXTUAL_NUDGE_EN

    log       = []
    processor = SoftConstraintProcessor(
        reward_token_groups = reward_groups,
        penalty_ids         = penalty_ids,
        reward_val          = reward_val,
        penalty_val         = penalty_val,
        curriculum_rate     = curriculum_rate,
        max_reward          = reward_max,
        anchor_offset       = anchor_offset,
        contextual_nudge    = active_contextual_nudge,
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
    reward_val    : Optional[float] = None,
    penalty_val   : Optional[float] = None,
) -> Tuple[str, List[Dict]]:
    
    is_en_tr = "en-tr" in model_wrapper.model_name
    if reward_val is None:
        reward_val = config.SOFT_COMBINED_REWARD_STRENGTH_TR if is_en_tr else config.SOFT_COMBINED_REWARD_STRENGTH_EN
    if penalty_val is None:
        penalty_val = config.SOFT_COMBINED_PENALTY_STRENGTH_TR if is_en_tr else config.SOFT_COMBINED_PENALTY_STRENGTH_EN
    
    active_curriculum_rate = config.SOFT_COMBINED_REWARD_CURRICULUM_RATE_TR if is_en_tr else config.SOFT_COMBINED_REWARD_CURRICULUM_RATE_EN
    active_reward_max = config.SOFT_COMBINED_REWARD_MAX_TR if is_en_tr else config.SOFT_COMBINED_REWARD_MAX_EN
    active_anchor_offset = config.SOFT_COMBINED_ANCHOR_OFFSET_TR if is_en_tr else config.SOFT_COMBINED_ANCHOR_OFFSET_EN

    trans, log = _soft_combined_pure(
        model_wrapper, source_text,
        reward_words=reward_words, penalty_words=penalty_words,
        reward_val=reward_val, penalty_val=penalty_val,
        curriculum_rate=active_curriculum_rate,
        reward_max=active_reward_max,
        anchor_offset=active_anchor_offset,
    )

    # ── Post-generation retry (mirrors soft_reward_only escalation) ───────────
    missing = _missing_words(trans, reward_words or [], is_tr_target=is_en_tr)
    if missing:
        retry_trans, _ = _soft_combined_pure(
            model_wrapper, source_text,
            reward_words=missing, penalty_words=penalty_words,
            reward_val=active_reward_max,   # full strength on retry
            penalty_val=penalty_val,
            curriculum_rate=active_curriculum_rate,
            reward_max=active_reward_max,
            anchor_offset=active_anchor_offset,
        )
        log.append({"escalated": "soft_boost", "missing_words": missing})
        still_missing = _missing_words(retry_trans, missing, is_tr_target=is_en_tr)
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
    reward_val    : Optional[float] = None,
    penalty_val   : Optional[float] = None,
) -> Tuple[str, List[Dict]]:

    is_en_tr = "en-tr" in model_wrapper.model_name
    if reward_val is None:
        reward_val = config.SOFT_REWARD_STRENGTH_TR if is_en_tr else config.SOFT_REWARD_STRENGTH_EN
    if penalty_val is None:
        penalty_val = config.SOFT_PENALTY_STRENGTH_TR if is_en_tr else config.SOFT_PENALTY_STRENGTH_EN

    reward_groups = model_wrapper.words_to_token_ids(reward_words or [], expand_tr=is_en_tr)
    penalty_ids   = model_wrapper.flat_token_ids(penalty_words or [])

    if not reward_groups and not penalty_ids:
        return unconstrained(model_wrapper, source_text)

    if is_en_tr:
        active_curriculum_rate = config.SOFT_REWARD_CURRICULUM_RATE_TR
        active_reward_max = config.SOFT_REWARD_MAX_TR
        active_anchor_offset = config.ANCHOR_OFFSET_TR
        active_contextual_nudge = config.CONTEXTUAL_NUDGE_TR
    else:
        active_curriculum_rate = config.SOFT_REWARD_CURRICULUM_RATE_EN
        active_reward_max = config.SOFT_REWARD_MAX_EN
        active_anchor_offset = config.ANCHOR_OFFSET_EN
        active_contextual_nudge = config.CONTEXTUAL_NUDGE_EN

    log       = []
    processor = SoftConstraintProcessor(
        reward_token_groups = reward_groups,
        penalty_ids         = penalty_ids,
        reward_val          = reward_val,
        penalty_val         = penalty_val,
        curriculum_rate     = active_curriculum_rate,
        max_reward          = active_reward_max,
        anchor_offset       = active_anchor_offset,
        contextual_nudge    = active_contextual_nudge,
        log_store           = log,
    )
    outputs     = _generate(model_wrapper, source_text, processors=[processor])
    translation = model_wrapper.decode(outputs)

    if penalty_words:
        # Replaced _word_satisfied with _missing_words check to avoid crashing
        still_forbidden = [w for w in penalty_words if not _missing_words(translation, [w])]
        if still_forbidden:
            hard_excl_ids    = model_wrapper.flat_token_ids(still_forbidden)
            retry_log        = []
            retry_processors = [
                HardExclusionProcessor(hard_excl_ids, log_store=retry_log),
            ]
            if reward_groups:
                still_missing = _missing_words(translation, reward_words or [], is_tr_target=is_en_tr)
                if still_missing:
                    retry_reward_groups = model_wrapper.words_to_token_ids(still_missing, expand_tr=is_en_tr)
                    retry_processors.append(SoftConstraintProcessor(
                        reward_token_groups = retry_reward_groups,
                        penalty_ids         = [],
                        reward_val          = active_reward_max,
                        penalty_val         = 0.0,
                        curriculum_rate     = active_curriculum_rate,
                        max_reward          = active_reward_max,
                        anchor_offset       = active_anchor_offset,
                        contextual_nudge    = active_contextual_nudge,
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
        missing = _missing_words(translation, reward_words, is_tr_target=is_en_tr)
        if missing:
            retry_groups = model_wrapper.words_to_token_ids(missing, expand_tr=is_en_tr)
            retry_log    = []
            retry_proc   = SoftConstraintProcessor(
                reward_token_groups = retry_groups,
                penalty_ids         = model_wrapper.flat_token_ids(penalty_words or []),
                reward_val          = active_reward_max,
                penalty_val         = penalty_val,
                curriculum_rate     = active_curriculum_rate,
                max_reward          = active_reward_max,
                anchor_offset       = active_anchor_offset,
                contextual_nudge    = active_contextual_nudge,
                log_store           = retry_log,
            )
            retry_outputs = _generate(model_wrapper, source_text, processors=[retry_proc])
            retry_trans   = model_wrapper.decode(retry_outputs)
            still_missing = _missing_words(retry_trans, missing, is_tr_target=is_en_tr)
            if not still_missing:
                log.append({"escalated": "soft_boost", "missing_words": missing})
                return retry_trans, log
            hard_trans, hard_log = hard_inclusion(model_wrapper, source_text, still_missing)
            log.append({"escalated": "hard_inclusion", "missing_words": still_missing,
                        "hard_log_steps": len(hard_log)})
            return hard_trans, log

    return translation, log


def _filter_subsets(sequences: List[List[int]]) -> List[List[int]]:
    """
    Remove duplicate sequences and sequences that contain another sequence as a prefix.
    This is required by HuggingFace's DisjunctiveConstraint, which uses a DisjunctiveTrie
    that raises a ValueError if any sequence is a complete prefix of another.
    """
    unique_seqs = sorted(list(set(tuple(s) for s in sequences)), key=len)
    filtered = []
    for seq in unique_seqs:
        is_prefix_found = False
        for kept in filtered:
            if len(kept) <= len(seq) and seq[:len(kept)] == kept:
                is_prefix_found = True
                break
        if not is_prefix_found:
            filtered.append(seq)
    return [list(s) for s in filtered]


def huggingface_dba(
    model_wrapper  : MTModel,
    source_text    : str,
    required_words : List[str],
    forbidden_words: List[str],
) -> Tuple[str, List[Dict]]:
    """
    Standard HuggingFace Dynamic Beam Allocation (DBA) baseline.
    Enforces inclusion constraints via DisjunctiveConstraint (morphology-aware) and exclusion constraints via bad_words_ids.
    """
    from transformers import DisjunctiveConstraint

    is_en_tr = "en-tr" in model_wrapper.model_name
    token_sequences = model_wrapper.words_to_sequences(required_words, expand_tr=is_en_tr)

    constraints = []
    for seqs in token_sequences:
        if seqs:
            filtered_seqs = _filter_subsets(seqs)
            if filtered_seqs:
                constraints.append(DisjunctiveConstraint(filtered_seqs))

    bad_words_ids = []
    if forbidden_words:
        flat_ids = model_wrapper.flat_token_ids(forbidden_words)
        for tid in flat_ids:
            bad_words_ids.append([tid])

    active_num_beams = config.DBA_NUM_BEAMS_TR if is_en_tr else config.DBA_NUM_BEAMS_EN
    active_length_penalty = config.DBA_LENGTH_PENALTY_TR if is_en_tr else config.DBA_LENGTH_PENALTY_EN
    active_repetition_penalty = config.DBA_REPETITION_PENALTY_TR if is_en_tr else config.DBA_REPETITION_PENALTY_EN

    inputs = model_wrapper.encode(source_text, max_length=config.MAX_LENGTH)

    gen_kwargs = {
        "max_length": config.MAX_LENGTH,
        "num_beams": active_num_beams,
        "no_repeat_ngram_size": config.NO_REPEAT_NGRAM,
        "length_penalty": active_length_penalty,
        "repetition_penalty": active_repetition_penalty,
        "constraints": constraints if constraints else None,
        "bad_words_ids": bad_words_ids if bad_words_ids else None,
        "early_stopping": True,
    }
    
    if constraints:
        # Avoid AttributeError: 'MarianMTModel' has no attribute 'transformers-community/constrained-beam-search'
        gen_kwargs["custom_generate"] = "transformers-community/constrained-beam-search"
        gen_kwargs["trust_remote_code"] = True

    with torch.no_grad():
        outputs = model_wrapper.model.generate(
            **inputs,
            **gen_kwargs
        )

    translation = model_wrapper.decode(outputs)
    return translation, []