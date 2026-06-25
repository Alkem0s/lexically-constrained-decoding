"""
evaluation.py — Measure constraint satisfaction and translation quality.

Metrics:
  - Constraint satisfaction rate : did forbidden words stay out / required
    words appear in the output?
  - BLEU score vs unconstrained baseline (via sacrebleu).
  - Simple fluency proxy: output length ratio relative to unconstrained output
    (heavy truncation or bloat suggests constraint pressure hurt fluency).
  - Escalation count : how many samples needed tier-2 or tier-3 fallbacks to
    satisfy the constraint.  A mode with 100% satisfaction but high escalation
    count is relying on hard fallbacks, not its own soft nudges.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ── sacrebleu import (optional — graceful fallback if not installed) ──────────
_bleu_scorer = None
_chrf_scorer = None
try:
    from sacrebleu.metrics import BLEU, CHRF
    _BLEU_AVAILABLE = True
    _bleu_scorer = BLEU(effective_order=True)
    _chrf_scorer = CHRF()
except ImportError:
    _BLEU_AVAILABLE = False
    print("[evaluation] sacrebleu not found — BLEU and ChrF scores will be skipped. "
          "Install with: pip install sacrebleu")


def compute_chrf(hypothesis: str, reference: str) -> Optional[float]:
    """
    Compute sentence-level ChrF of `hypothesis` against `reference`.
    """
    if not _BLEU_AVAILABLE or not hypothesis or not reference:
        return None
    score = _chrf_scorer.sentence_score(hypothesis, [reference])
    return round(score.score, 2)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    """Holds all outputs for a single source sentence across all constraint modes."""
    source          : str
    direction       : str                   # e.g. "EN→TR"
    unconstrained   : str = ""
    reference       : str = ""

    # Per-mode translations
    hard_exclusion          : str = ""
    hard_inclusion          : str = ""      # soft-first gate, hard fallback
    hard_inclusion_ablation : str = ""      # hard-only (bypasses soft gate)
    hard_combined           : str = ""      # simultaneous exclusion + inclusion

    soft_penalty    : str = ""             # penalty only (isolated, no escalation)
    soft_reward     : str = ""             # reward only (isolated, with escalation)
    soft_combined   : str = ""             # simultaneous penalty + reward, pure soft
    huggingface_dba : str = ""             # HF native DBA baseline

    # Constraint specs
    forbidden_words : List[str] = field(default_factory=list)
    required_words  : List[str] = field(default_factory=list)
    penalty_words   : List[str] = field(default_factory=list)
    reward_words    : List[str] = field(default_factory=list)

    # Computed metrics (filled in by evaluate_sample)
    metrics         : Dict = field(default_factory=dict)

    # Escalation events per mode (filled in by run_sample before evaluate_sample)
    # Keys match the metrics mode names; value = count of escalation events in that log.
    escalation_counts : Dict = field(default_factory=dict)
    
    # Timing and generation pass metrics
    latencies         : Dict = field(default_factory=dict)
    pass_counts       : Dict = field(default_factory=dict)


# ── Constraint satisfaction ────────────────────────────────────────────────────

def _turkish_lower(s: str) -> str:
    """Python's str.lower() mishandles Turkish İ/I. Apply correct mapping first."""
    return s.replace('İ', 'i').replace('I', 'ı').lower()

def _contains_word(text: str, word: str, is_tr_target: bool = False) -> bool:
    word_lower = _turkish_lower(word)
    if is_tr_target:
        # Match the word boundary before, but allow valid Turkish letters after!
        pattern = r'(^|\W)' + re.escape(word_lower) + r'[a-zçğıöşü]*'
    else:
        pattern = r'(^|\W)' + re.escape(word_lower) + r'(?=\W|$)'
    return bool(re.search(pattern, _turkish_lower(text)))

def satisfaction_exclusion(translation: str, forbidden_words: List[str], is_tr_target: bool = False) -> Dict:
    """Return per-word and overall exclusion satisfaction."""
    details = {}
    for w in forbidden_words:
        present = _contains_word(translation, w, is_tr_target)
        details[w] = {"present": present, "satisfied": not present}
    overall = all(not v["present"] for v in details.values())
    return {"overall": overall, "details": details}

def satisfaction_inclusion(translation: str, required_words: List[str], is_tr_target: bool = False) -> Dict:
    """Return per-word and overall inclusion satisfaction."""
    details = {}
    for w in required_words:
        present = _contains_word(translation, w, is_tr_target)
        details[w] = {"present": present, "satisfied": present}
    overall = all(v["present"] for v in details.values())
    return {"overall": overall, "details": details}


# ── BLEU ──────────────────────────────────────────────────────────────────────

def compute_bleu(hypothesis: str, reference: str) -> Optional[float]:
    """
    Compute sentence-level BLEU of `hypothesis` against `reference`.
    Returns None if sacrebleu is unavailable or inputs are empty.
    """
    if not _BLEU_AVAILABLE or not hypothesis or not reference:
        return None
    score = _bleu_scorer.sentence_score(hypothesis, [reference])
    return round(score.score, 2)


# ── Fluency proxy ─────────────────────────────────────────────────────────────

def length_ratio(constrained: str, unconstrained: str) -> Optional[float]:
    """
    Token-count ratio: constrained / unconstrained.
    Values far from 1.0 suggest the constraint hurt fluency or completeness.
    """
    u_len = len(unconstrained.split())
    c_len = len(constrained.split())
    if u_len == 0:
        return None
    return round(c_len / u_len, 3)


# ── Escalation counting ───────────────────────────────────────────────────────

def count_escalations(log: List[Dict]) -> int:
    """
    Count escalation events in a constraint processor log.

    Escalation entries are appended by soft_reward_only() / soft_constrained()
    when they fall back to a harder strategy, e.g.:
        {"escalated": "soft_boost", "missing_words": [...]}
        {"escalated": "hard_inclusion", "missing_words": [...]}
        {"escalated": "hard_exclusion", "still_forbidden": [...]}

    Returns the count of such entries.  Zero means the mode satisfied the
    constraint on the first pass without any fallback.
    """
    return sum(1 for entry in log if isinstance(entry, dict) and "escalated" in entry)


# ── Full sample evaluation ────────────────────────────────────────────────────

def evaluate_sample(result: SampleResult) -> Dict:
    """
    Compute all metrics for a single SampleResult and store in result.metrics.
    Reads result.escalation_counts (pre-populated by run_sample) and records
    per-mode escalation counts in the metrics dict.
    Returns the metrics dict for convenience.
    """
    baseline = result.unconstrained
    reference = result.reference
    metrics  = {}
    
    # Check if the target language is Turkish based on the direction string (e.g., "EN→TR")
    is_tr_target = "→TR" in result.direction or "en-tr" in result.direction.lower()

    # Helper to populate metrics dict
    def populate_mode(mode_key, translation, satisfaction, extra_sat_dict=None):
        if not translation:
            return
        
        # Determine satisfaction
        sat_val = satisfaction
            
        bleu_base = compute_bleu(translation, baseline)
        bleu_ref = compute_bleu(translation, reference)
        chrf_ref = compute_chrf(translation, reference)
        
        m = {
            "bleu_vs_baseline" : bleu_base,
            "bleu_ref"         : bleu_ref,
            "chrf_ref"         : chrf_ref,
            "length_ratio"     : length_ratio(translation, baseline),
            "n_escalated"      : result.escalation_counts.get(mode_key, 0),
            "latency_ms"       : result.latencies.get(mode_key, 0.0),
            "pass_count"       : result.pass_counts.get(mode_key, 1),
        }
        
        if extra_sat_dict is not None:
            # Combined modes
            m.update(extra_sat_dict)
            m["overall_satisfaction"] = sat_val
        else:
            # Simple modes
            m["satisfaction"] = {"overall": sat_val}
            
        metrics[mode_key] = m

    # 0. Unconstrained baseline
    metrics["unconstrained"] = {
        "bleu_vs_baseline" : 100.0,
        "bleu_ref"         : compute_bleu(baseline, reference),
        "chrf_ref"         : compute_chrf(baseline, reference),
        "length_ratio"     : 1.0,
        "n_escalated"      : 0,
        "latency_ms"       : result.latencies.get("unconstrained", 0.0),
        "pass_count"       : 1,
        "satisfaction"     : {"overall": True}
    }

    # 1. Hard exclusion
    if result.hard_exclusion and result.forbidden_words:
        sat  = satisfaction_exclusion(result.hard_exclusion, result.forbidden_words, is_tr_target)
        populate_mode("hard_exclusion", result.hard_exclusion, sat["overall"])
        metrics["hard_exclusion"]["constraint_violated_at_baseline"] = (not sat["overall"]) and (metrics["hard_exclusion"]["bleu_vs_baseline"] is not None and metrics["hard_exclusion"]["bleu_vs_baseline"] >= 99.9)

    # 2. Hard inclusion
    if result.hard_inclusion and result.required_words:
        sat = satisfaction_inclusion(result.hard_inclusion, result.required_words, is_tr_target)
        populate_mode("hard_inclusion", result.hard_inclusion, sat["overall"])

    # 3. Hard inclusion ablation
    if result.hard_inclusion_ablation and result.required_words:
        sat = satisfaction_inclusion(result.hard_inclusion_ablation, result.required_words, is_tr_target)
        populate_mode("hard_inclusion_ablation", result.hard_inclusion_ablation, sat["overall"])

    # 4. Hard combined
    if result.hard_combined and (result.forbidden_words or result.required_words):
        excl_sat = (
            satisfaction_exclusion(result.hard_combined, result.forbidden_words, is_tr_target)
            if result.forbidden_words else {"overall": True, "details": {}}
        )
        incl_sat = (
            satisfaction_inclusion(result.hard_combined, result.required_words, is_tr_target)
            if result.required_words else {"overall": True, "details": {}}
        )
        overall_sat = excl_sat["overall"] and incl_sat["overall"]
        populate_mode("hard_combined", result.hard_combined, overall_sat, {
            "exclusion_satisfaction": excl_sat,
            "inclusion_satisfaction": incl_sat,
        })

    # 5. Soft penalty
    if result.soft_penalty and result.penalty_words:
        sat  = satisfaction_exclusion(result.soft_penalty, result.penalty_words, is_tr_target)
        populate_mode("soft_penalty", result.soft_penalty, sat["overall"])
        metrics["soft_penalty"]["constraint_violated_at_baseline"] = (not sat["overall"]) and (metrics["soft_penalty"]["bleu_vs_baseline"] is not None and metrics["soft_penalty"]["bleu_vs_baseline"] >= 99.9)

    # 6. Soft reward
    if result.soft_reward and result.reward_words:
        sat = satisfaction_inclusion(result.soft_reward, result.reward_words, is_tr_target)
        populate_mode("soft_reward", result.soft_reward, sat["overall"])

    # 7. Soft combined
    if result.soft_combined and (result.penalty_words or result.reward_words):
        pen_sat = (
            satisfaction_exclusion(result.soft_combined, result.penalty_words, is_tr_target)
            if result.penalty_words else {"overall": True, "details": {}}
        )
        rew_sat = (
            satisfaction_inclusion(result.soft_combined, result.reward_words, is_tr_target)
            if result.reward_words else {"overall": True, "details": {}}
        )
        overall_sat = pen_sat["overall"] and rew_sat["overall"]
        populate_mode("soft_combined", result.soft_combined, overall_sat, {
            "penalty_satisfaction": pen_sat,
            "reward_satisfaction": rew_sat,
        })
        metrics["soft_combined"]["constraint_violated_at_baseline"] = (not pen_sat["overall"]) and (metrics["soft_combined"]["bleu_vs_baseline"] is not None and metrics["soft_combined"]["bleu_vs_baseline"] >= 99.9)

    # 8. HuggingFace DBA baseline
    if result.huggingface_dba:
        excl_sat = (
            satisfaction_exclusion(result.huggingface_dba, result.forbidden_words, is_tr_target)
            if result.forbidden_words else {"overall": True, "details": {}}
        )
        incl_sat = (
            satisfaction_inclusion(result.huggingface_dba, result.required_words, is_tr_target)
            if result.required_words else {"overall": True, "details": {}}
        )
        overall_sat = excl_sat["overall"] and incl_sat["overall"]
        populate_mode("huggingface_dba", result.huggingface_dba, overall_sat, {
            "exclusion_satisfaction": excl_sat,
            "inclusion_satisfaction": incl_sat,
        })

    result.metrics = metrics
    return metrics


# ── Aggregate across samples ──────────────────────────────────────────────────

def aggregate_results(results: List[SampleResult]) -> Dict:
    """
    Compute aggregate metrics across all samples.
    """
    agg = {}

    # Unconstrained baseline first
    unconstrained_bleus, unconstrained_chrfs, unconstrained_latencies = [], [], []
    for r in results:
        m = r.metrics.get("unconstrained")
        if m is not None:
            if m["bleu_ref"] is not None:
                unconstrained_bleus.append(m["bleu_ref"])
            if m["chrf_ref"] is not None:
                unconstrained_chrfs.append(m["chrf_ref"])
            if m.get("latency_ms") is not None:
                unconstrained_latencies.append(m["latency_ms"])
    agg["unconstrained"] = {
        "n_samples"              : len(results),
        "avg_satisfaction"       : None,
        "avg_bleu_vs_base"       : 100.00,
        "avg_bleu_vs_ref"        : round(sum(unconstrained_bleus) / len(unconstrained_bleus), 2) if unconstrained_bleus else None,
        "avg_chrf_vs_ref"        : round(sum(unconstrained_chrfs) / len(unconstrained_chrfs), 2) if unconstrained_chrfs else None,
        "avg_length_ratio"       : 1.000,
        "n_violated_at_baseline" : 0,
        "n_escalated"            : 0,
        "avg_latency_ms"         : round(sum(unconstrained_latencies) / len(unconstrained_latencies), 1) if unconstrained_latencies else None,
        "avg_passes"             : 1.0,
    }

    # Modes that use a single "satisfaction.overall" key
    simple_modes = [
        "hard_exclusion",
        "hard_inclusion",
        "hard_inclusion_ablation",
        "soft_penalty",
        "soft_reward",
    ]
    # Combined modes that use "overall_satisfaction" key
    combined_modes = ["hard_combined", "soft_combined", "huggingface_dba"]

    for mode in simple_modes:
        sats, bleus, ratios, ref_bleus, ref_chrfs, latencies, passes = [], [], [], [], [], [], []
        n_violated_at_baseline = 0
        n_escalated            = 0
        for r in results:
            m = r.metrics.get(mode)
            if m is None:
                continue
            sats.append(float(m["satisfaction"]["overall"]))
            if m["bleu_vs_baseline"] is not None:
                bleus.append(m["bleu_vs_baseline"])
            if m["length_ratio"] is not None:
                ratios.append(m["length_ratio"])
            if m.get("constraint_violated_at_baseline"):
                n_violated_at_baseline += 1
            if m.get("n_escalated", 0) > 0:
                n_escalated += 1
            if m.get("bleu_ref") is not None:
                ref_bleus.append(m["bleu_ref"])
            if m.get("chrf_ref") is not None:
                ref_chrfs.append(m["chrf_ref"])
            if m.get("latency_ms") is not None:
                latencies.append(m["latency_ms"])
            if m.get("pass_count") is not None:
                passes.append(m["pass_count"])

        if not sats:
            continue

        agg[mode] = {
            "n_samples"              : len(sats),
            "avg_satisfaction"       : round(sum(sats) / len(sats), 3),
            "avg_bleu_vs_base"       : round(sum(bleus) / len(bleus), 2) if bleus else None,
            "avg_bleu_vs_ref"        : round(sum(ref_bleus) / len(ref_bleus), 2) if ref_bleus else None,
            "avg_chrf_vs_ref"        : round(sum(ref_chrfs) / len(ref_chrfs), 2) if ref_chrfs else None,
            "avg_length_ratio"       : round(sum(ratios) / len(ratios), 3) if ratios else None,
            "n_violated_at_baseline" : n_violated_at_baseline,
            "n_escalated"            : n_escalated,
            "avg_latency_ms"         : round(sum(latencies) / len(latencies), 1) if latencies else None,
            "avg_passes"             : round(sum(passes) / len(passes), 2) if passes else 1.0,
        }

    for mode in combined_modes:
        sats, bleus, ratios, ref_bleus, ref_chrfs, latencies, passes = [], [], [], [], [], [], []
        n_violated_at_baseline = 0
        n_escalated            = 0
        for r in results:
            m = r.metrics.get(mode)
            if m is None:
                continue
            sats.append(float(m["overall_satisfaction"]))
            if m["bleu_vs_baseline"] is not None:
                bleus.append(m["bleu_vs_baseline"])
            if m["length_ratio"] is not None:
                ratios.append(m["length_ratio"])
            if m.get("constraint_violated_at_baseline"):
                n_violated_at_baseline += 1
            if m.get("n_escalated", 0) > 0:
                n_escalated += 1
            if m.get("bleu_ref") is not None:
                ref_bleus.append(m["bleu_ref"])
            if m.get("chrf_ref") is not None:
                ref_chrfs.append(m["chrf_ref"])
            if m.get("latency_ms") is not None:
                latencies.append(m["latency_ms"])
            if m.get("pass_count") is not None:
                passes.append(m["pass_count"])

        if not sats:
            continue

        agg[mode] = {
            "n_samples"              : len(sats),
            "avg_satisfaction"       : round(sum(sats) / len(sats), 3),
            "avg_bleu_vs_base"       : round(sum(bleus) / len(bleus), 2) if bleus else None,
            "avg_bleu_vs_ref"        : round(sum(ref_bleus) / len(ref_bleus), 2) if ref_bleus else None,
            "avg_chrf_vs_ref"        : round(sum(ref_chrfs) / len(ref_chrfs), 2) if ref_chrfs else None,
            "avg_length_ratio"       : round(sum(ratios) / len(ratios), 3) if ratios else None,
            "n_violated_at_baseline" : n_violated_at_baseline,
            "n_escalated"            : n_escalated,
            "avg_latency_ms"         : round(sum(latencies) / len(latencies), 1) if latencies else None,
            "avg_passes"             : round(sum(passes) / len(passes), 2) if passes else 1.0,
        }

    return agg


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_sample_result(result: SampleResult) -> None:
    """Print a formatted summary for a single sample."""
    print(f"\n  Source ({result.direction}): {result.source}")
    print(f"  Unconstrained : {result.unconstrained}")

    rows = [
        ("Hard Exclusion"     , result.hard_exclusion          , "hard_exclusion"),
        ("Hard Inclusion"     , result.hard_inclusion          , "hard_inclusion"),
        ("Hard Incl Ablation" , result.hard_inclusion_ablation , "hard_inclusion_ablation"),
        ("Hard Combined"      , result.hard_combined           , "hard_combined"),
        ("Soft Penalty"       , result.soft_penalty            , "soft_penalty"),
        ("Soft Reward"        , result.soft_reward             , "soft_reward"),
        ("Soft Combined"      , result.soft_combined           , "soft_combined"),
        ("HuggingFace DBA"    , result.huggingface_dba         , "huggingface_dba"),
    ]

    for label, translation, mode_key in rows:
        if not translation:
            continue
        m = result.metrics.get(mode_key, {})
        esc_flag = f"  ↑escalated×{m['n_escalated']}" if m.get("n_escalated", 0) > 0 else ""

        if mode_key in ("hard_combined", "soft_combined", "huggingface_dba"):
            overall   = m.get("overall_satisfaction", "—")
            bleu_val  = m.get("bleu_vs_baseline", "—")
            bleu_ref  = m.get("bleu_ref", "—")
            chrf_ref  = m.get("chrf_ref", "—")
            lr_val    = m.get("length_ratio", "—")
            lat_val   = f"{m.get('latency_ms', 0.0):.1f}ms"
            viol_flag = "  ⚠ violated at baseline" if m.get("constraint_violated_at_baseline") else ""
            print(
                f"  {label:<22}: {translation}\n"
                f"    ↳ overall_satisfied={overall}  bleu_vs_base={bleu_val}  bleu_ref={bleu_ref}  chrf_ref={chrf_ref}  "
                f"len_ratio={lr_val}  latency={lat_val}{viol_flag}{esc_flag}"
            )
        else:
            sat_val   = m.get("satisfaction", {}).get("overall", "—")
            bleu_val  = m.get("bleu_vs_baseline", "—")
            bleu_ref  = m.get("bleu_ref", "—")
            chrf_ref  = m.get("chrf_ref", "—")
            lr_val    = m.get("length_ratio", "—")
            lat_val   = f"{m.get('latency_ms', 0.0):.1f}ms"
            viol_flag = "  ⚠ violated at baseline" if m.get("constraint_violated_at_baseline") else ""
            print(
                f"  {label:<22}: {translation}\n"
                f"    ↳ satisfied={sat_val}  bleu_vs_base={bleu_val}  bleu_ref={bleu_ref}  chrf_ref={chrf_ref}  "
                f"len_ratio={lr_val}  latency={lat_val}{viol_flag}{esc_flag}"
            )


def print_aggregate(agg: Dict) -> None:
    """Print a formatted aggregate results table."""
    print("\n" + "="*112)
    print("AGGREGATE RESULTS")
    print("="*112)
    header = (
        f"  {'Mode':<24}  {'N':>4}  {'Sat%':>6}  {'BLEU_Base':>9}  {'BLEU_Ref':>8}  "
        f"{'ChrF_Ref':>8}  {'LenRatio':>9}  {'Latency_ms':>10}  {'Passes':>6}"
    )
    print(header)
    print("  " + "-" * 106)
    for mode, vals in agg.items():
        sat  = f"{vals['avg_satisfaction']*100:.1f}%" if vals['avg_satisfaction'] is not None else "  N/A"
        bleu_base = f"{vals['avg_bleu_vs_base']:.2f}" if vals['avg_bleu_vs_base']  is not None else "  N/A"
        bleu_ref = f"{vals['avg_bleu_vs_ref']:.2f}"   if vals['avg_bleu_vs_ref']   is not None else "  N/A"
        chrf_ref = f"{vals['avg_chrf_vs_ref']:.2f}"   if vals['avg_chrf_vs_ref']   is not None else "  N/A"
        lr   = f"{vals['avg_length_ratio']:.3f}"      if vals['avg_length_ratio']  is not None else "  N/A"
        lat  = f"{vals['avg_latency_ms']:.1f}"        if vals['avg_latency_ms']    is not None else "  N/A"
        passes = f"{vals['avg_passes']:.2f}"          if vals['avg_passes']        is not None else " 1.00"
        print(
            f"  {mode:<24}  {vals['n_samples']:>4}  {sat:>6}  {bleu_base:>9}  {bleu_ref:>8}  "
            f"{chrf_ref:>8}  {lr:>9}  {lat:>10}  {passes:>6}"
        )
    print("="*112)