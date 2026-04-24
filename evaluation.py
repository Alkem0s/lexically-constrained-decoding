"""
evaluation.py — Measure constraint satisfaction and translation quality.

Metrics:
  - Constraint satisfaction rate : did forbidden words stay out / required
    words appear in the output?
  - BLEU score vs unconstrained baseline (via sacrebleu).
  - Simple fluency proxy: output length ratio relative to unconstrained output
    (heavy truncation or bloat suggests constraint pressure hurt fluency).

Fix #7: When hard exclusion fails (forbidden word still present) and the
output is identical to baseline (BLEU = 100), the metrics dict now also
carries a 'constraint_violated_at_baseline' flag so callers can distinguish
"constraint respected and output unchanged" from "constraint failed silently".
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ── sacrebleu import (optional — graceful fallback if not installed) ──────────
try:
    from sacrebleu.metrics import BLEU
    _BLEU_AVAILABLE = True
except ImportError:
    _BLEU_AVAILABLE = False
    print("[evaluation] sacrebleu not found — BLEU scores will be skipped. "
          "Install with: pip install sacrebleu")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    """Holds all outputs for a single source sentence across all constraint modes."""
    source          : str
    direction       : str                   # e.g. "EN→TR"
    unconstrained   : str = ""

    # Per-mode translations
    hard_exclusion  : str = ""
    hard_inclusion  : str = ""
    hard_combined   : str = ""             # simultaneous exclusion + inclusion

    soft_penalty    : str = ""             # penalty only (isolated)
    soft_reward     : str = ""             # reward only (isolated)
    soft_combined   : str = ""            # simultaneous penalty + reward

    # Constraint specs
    forbidden_words : List[str] = field(default_factory=list)
    required_words  : List[str] = field(default_factory=list)
    penalty_words   : List[str] = field(default_factory=list)
    reward_words    : List[str] = field(default_factory=list)

    # Computed metrics (filled in by evaluate_sample)
    metrics         : Dict = field(default_factory=dict)


# ── Constraint satisfaction ────────────────────────────────────────────────────

def _contains_word(text: str, word: str) -> bool:
    """
    Check if `word` appears in `text` (case-insensitive, substring match).
    For Turkish we use a simple case-fold rather than regex word boundaries
    because Turkish morphology can attach suffixes.
    """
    return word.lower() in text.lower()


def satisfaction_exclusion(translation: str, forbidden_words: List[str]) -> Dict:
    """Return per-word and overall exclusion satisfaction."""
    details = {}
    for w in forbidden_words:
        present = _contains_word(translation, w)
        details[w] = {"present": present, "satisfied": not present}
    overall = all(not v["present"] for v in details.values())
    return {"overall": overall, "details": details}


def satisfaction_inclusion(translation: str, required_words: List[str]) -> Dict:
    """Return per-word and overall inclusion satisfaction."""
    details = {}
    for w in required_words:
        present = _contains_word(translation, w)
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
    bleu = BLEU(effective_order=True)
    score = bleu.sentence_score(hypothesis, [reference])
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


# ── Full sample evaluation ────────────────────────────────────────────────────

def evaluate_sample(result: SampleResult) -> Dict:
    """
    Compute all metrics for a single SampleResult and store in result.metrics.
    Returns the metrics dict for convenience.
    """
    baseline = result.unconstrained
    metrics  = {}

    # ── Hard exclusion ────────────────────────────────────────────────────────
    if result.hard_exclusion and result.forbidden_words:
        sat  = satisfaction_exclusion(result.hard_exclusion, result.forbidden_words)
        bleu = compute_bleu(result.hard_exclusion, baseline)
        violated_at_baseline = (not sat["overall"]) and (bleu is not None and bleu >= 99.9)
        metrics["hard_exclusion"] = {
            "satisfaction"              : sat,
            "bleu_vs_baseline"          : bleu,
            "length_ratio"              : length_ratio(result.hard_exclusion, baseline),
            "constraint_violated_at_baseline": violated_at_baseline,
        }

    # ── Hard inclusion ────────────────────────────────────────────────────────
    if result.hard_inclusion and result.required_words:
        sat = satisfaction_inclusion(result.hard_inclusion, result.required_words)
        metrics["hard_inclusion"] = {
            "satisfaction"     : sat,
            "bleu_vs_baseline" : compute_bleu(result.hard_inclusion, baseline),
            "length_ratio"     : length_ratio(result.hard_inclusion, baseline),
        }

    # ── Hard combined (exclusion + inclusion simultaneously) ──────────────────
    if result.hard_combined and (result.forbidden_words or result.required_words):
        excl_sat = (
            satisfaction_exclusion(result.hard_combined, result.forbidden_words)
            if result.forbidden_words else {"overall": True, "details": {}}
        )
        incl_sat = (
            satisfaction_inclusion(result.hard_combined, result.required_words)
            if result.required_words else {"overall": True, "details": {}}
        )
        bleu = compute_bleu(result.hard_combined, baseline)
        metrics["hard_combined"] = {
            "exclusion_satisfaction"    : excl_sat,
            "inclusion_satisfaction"    : incl_sat,
            "overall_satisfaction"      : excl_sat["overall"] and incl_sat["overall"],
            "bleu_vs_baseline"          : bleu,
            "length_ratio"              : length_ratio(result.hard_combined, baseline),
        }

    # ── Soft penalty (isolated — no reward active) ────────────────────────────
    if result.soft_penalty and result.penalty_words:
        sat  = satisfaction_exclusion(result.soft_penalty, result.penalty_words)
        bleu = compute_bleu(result.soft_penalty, baseline)
        violated_at_baseline = (not sat["overall"]) and (bleu is not None and bleu >= 99.9)
        metrics["soft_penalty"] = {
            "satisfaction"              : sat,
            "bleu_vs_baseline"          : bleu,
            "length_ratio"              : length_ratio(result.soft_penalty, baseline),
            "constraint_violated_at_baseline": violated_at_baseline,
        }

    # ── Soft reward (isolated — no penalty active) ────────────────────────────
    if result.soft_reward and result.reward_words:
        sat = satisfaction_inclusion(result.soft_reward, result.reward_words)
        metrics["soft_reward"] = {
            "satisfaction"     : sat,
            "bleu_vs_baseline" : compute_bleu(result.soft_reward, baseline),
            "length_ratio"     : length_ratio(result.soft_reward, baseline),
        }

    # ── Soft combined (penalty + reward simultaneously) ───────────────────────
    if result.soft_combined and (result.penalty_words or result.reward_words):
        pen_sat = (
            satisfaction_exclusion(result.soft_combined, result.penalty_words)
            if result.penalty_words else {"overall": True, "details": {}}
        )
        rew_sat = (
            satisfaction_inclusion(result.soft_combined, result.reward_words)
            if result.reward_words else {"overall": True, "details": {}}
        )
        bleu = compute_bleu(result.soft_combined, baseline)
        violated_at_baseline = (not pen_sat["overall"]) and (bleu is not None and bleu >= 99.9)
        metrics["soft_combined"] = {
            "penalty_satisfaction"      : pen_sat,
            "reward_satisfaction"       : rew_sat,
            "overall_satisfaction"      : pen_sat["overall"] and rew_sat["overall"],
            "bleu_vs_baseline"          : bleu,
            "length_ratio"              : length_ratio(result.soft_combined, baseline),
            "constraint_violated_at_baseline": violated_at_baseline,
        }

    result.metrics = metrics
    return metrics


# ── Aggregate across samples ──────────────────────────────────────────────────

def aggregate_results(results: List[SampleResult]) -> Dict:
    """
    Compute aggregate metrics across all samples.
    Returns a dict: mode → {avg_satisfaction, avg_bleu, avg_length_ratio,
                             n_violated_at_baseline}.

    For combined modes (hard_combined, soft_combined) overall_satisfaction is
    used (both sub-constraints must pass).
    """
    agg = {}

    # Modes that use a single "satisfaction.overall" key
    simple_modes = ["hard_exclusion", "hard_inclusion", "soft_penalty", "soft_reward"]
    # Combined modes that use "overall_satisfaction" key
    combined_modes = ["hard_combined", "soft_combined"]

    for mode in simple_modes:
        sats, bleus, ratios = [], [], []
        n_violated_at_baseline = 0
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

        if not sats:
            continue

        agg[mode] = {
            "n_samples"               : len(sats),
            "avg_satisfaction"        : round(sum(sats) / len(sats), 3),
            "avg_bleu_vs_base"        : round(sum(bleus) / len(bleus), 2) if bleus else None,
            "avg_length_ratio"        : round(sum(ratios) / len(ratios), 3) if ratios else None,
            "n_violated_at_baseline"  : n_violated_at_baseline,
        }

    for mode in combined_modes:
        sats, bleus, ratios = [], [], []
        n_violated_at_baseline = 0
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

        if not sats:
            continue

        agg[mode] = {
            "n_samples"               : len(sats),
            "avg_satisfaction"        : round(sum(sats) / len(sats), 3),
            "avg_bleu_vs_base"        : round(sum(bleus) / len(bleus), 2) if bleus else None,
            "avg_length_ratio"        : round(sum(ratios) / len(ratios), 3) if ratios else None,
            "n_violated_at_baseline"  : n_violated_at_baseline,
        }

    return agg


# ── Pretty print ──────────────────────────────────────────────────────────────

def print_sample_result(result: SampleResult) -> None:
    """Print a formatted summary for a single sample."""
    print(f"\n  Source ({result.direction}): {result.source}")
    print(f"  Unconstrained : {result.unconstrained}")

    rows = [
        ("Hard Exclusion" , result.hard_exclusion , "hard_exclusion"),
        ("Hard Inclusion" , result.hard_inclusion , "hard_inclusion"),
        ("Hard Combined"  , result.hard_combined  , "hard_combined"),
        ("Soft Penalty"   , result.soft_penalty   , "soft_penalty"),
        ("Soft Reward"    , result.soft_reward    , "soft_reward"),
        ("Soft Combined"  , result.soft_combined  , "soft_combined"),
    ]

    for label, translation, mode_key in rows:
        if not translation:
            continue
        m = result.metrics.get(mode_key, {})

        # Combined modes expose overall_satisfaction + sub-satisfaction dicts
        if mode_key in ("hard_combined", "soft_combined"):
            overall   = m.get("overall_satisfaction", "—")
            bleu_val  = m.get("bleu_vs_baseline", "—")
            lr_val    = m.get("length_ratio", "—")
            viol_flag = "  ⚠ violated at baseline" if m.get("constraint_violated_at_baseline") else ""
            print(
                f"  {label:<18}: {translation}\n"
                f"    ↳ overall_satisfied={overall}  bleu_vs_base={bleu_val}  len_ratio={lr_val}{viol_flag}"
            )
        else:
            sat_val   = m.get("satisfaction", {}).get("overall", "—")
            bleu_val  = m.get("bleu_vs_baseline", "—")
            lr_val    = m.get("length_ratio", "—")
            viol_flag = "  ⚠ violated at baseline" if m.get("constraint_violated_at_baseline") else ""
            print(
                f"  {label:<18}: {translation}\n"
                f"    ↳ satisfied={sat_val}  bleu_vs_base={bleu_val}  len_ratio={lr_val}{viol_flag}"
            )


def print_aggregate(agg: Dict) -> None:
    """Print a formatted aggregate results table."""
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    header = f"  {'Mode':<18}  {'N':>4}  {'Sat%':>6}  {'BLEU':>6}  {'LenRatio':>9}  {'ViolAtBase':>10}"
    print(header)
    print("  " + "-" * 65)
    for mode, vals in agg.items():
        sat  = f"{vals['avg_satisfaction']*100:.1f}%" if vals['avg_satisfaction'] is not None else "  N/A"
        bleu = f"{vals['avg_bleu_vs_base']:.2f}"      if vals['avg_bleu_vs_base']  is not None else "  N/A"
        lr   = f"{vals['avg_length_ratio']:.3f}"      if vals['avg_length_ratio']  is not None else "  N/A"
        vab  = str(vals.get('n_violated_at_baseline', 'N/A'))
        print(f"  {mode:<18}  {vals['n_samples']:>4}  {sat:>6}  {bleu:>6}  {lr:>9}  {vab:>10}")
    print("="*70)