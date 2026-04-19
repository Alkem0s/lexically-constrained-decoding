"""
evaluation.py — Measure constraint satisfaction and translation quality.

Metrics:
  - Constraint satisfaction rate : did forbidden words stay out / required
    words appear in the output?
  - BLEU score vs unconstrained baseline (via sacrebleu).
  - Simple fluency proxy: output length ratio relative to unconstrained output
    (heavy truncation or bloat suggests constraint pressure hurt fluency).
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
    soft_penalty    : str = ""
    soft_reward     : str = ""

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
    Check if `word` appears in `text` (case-insensitive, word-boundary aware).
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
    baseline  = result.unconstrained
    metrics   = {}

    # ── Hard exclusion ────────────────────────────────────────────────────────
    if result.hard_exclusion and result.forbidden_words:
        sat = satisfaction_exclusion(result.hard_exclusion, result.forbidden_words)
        metrics["hard_exclusion"] = {
            "satisfaction"  : sat,
            "bleu_vs_baseline" : compute_bleu(result.hard_exclusion, baseline),
            "length_ratio"  : length_ratio(result.hard_exclusion, baseline),
        }

    # ── Hard inclusion ────────────────────────────────────────────────────────
    if result.hard_inclusion and result.required_words:
        sat = satisfaction_inclusion(result.hard_inclusion, result.required_words)
        metrics["hard_inclusion"] = {
            "satisfaction"     : sat,
            "bleu_vs_baseline" : compute_bleu(result.hard_inclusion, baseline),
            "length_ratio"     : length_ratio(result.hard_inclusion, baseline),
        }

    # ── Soft penalty ──────────────────────────────────────────────────────────
    if result.soft_penalty and result.penalty_words:
        sat = satisfaction_exclusion(result.soft_penalty, result.penalty_words)
        metrics["soft_penalty"] = {
            "satisfaction"     : sat,        # soft may not fully exclude
            "bleu_vs_baseline" : compute_bleu(result.soft_penalty, baseline),
            "length_ratio"     : length_ratio(result.soft_penalty, baseline),
        }

    # ── Soft reward ───────────────────────────────────────────────────────────
    if result.soft_reward and result.reward_words:
        sat = satisfaction_inclusion(result.soft_reward, result.reward_words)
        metrics["soft_reward"] = {
            "satisfaction"     : sat,
            "bleu_vs_baseline" : compute_bleu(result.soft_reward, baseline),
            "length_ratio"     : length_ratio(result.soft_reward, baseline),
        }

    result.metrics = metrics
    return metrics


# ── Aggregate across samples ──────────────────────────────────────────────────

def aggregate_results(results: List[SampleResult]) -> Dict:
    """
    Compute aggregate metrics across all samples.
    Returns a dict: mode → {avg_satisfaction, avg_bleu, avg_length_ratio}.
    """
    agg = {}

    for mode in ["hard_exclusion", "hard_inclusion", "soft_penalty", "soft_reward"]:
        sats, bleus, ratios = [], [], []
        for r in results:
            m = r.metrics.get(mode)
            if m is None:
                continue
            sats.append(float(m["satisfaction"]["overall"]))
            if m["bleu_vs_baseline"] is not None:
                bleus.append(m["bleu_vs_baseline"])
            if m["length_ratio"] is not None:
                ratios.append(m["length_ratio"])

        if not sats:
            continue

        agg[mode] = {
            "n_samples"          : len(sats),
            "avg_satisfaction"   : round(sum(sats) / len(sats), 3),
            "avg_bleu_vs_base"   : round(sum(bleus) / len(bleus), 2) if bleus else None,
            "avg_length_ratio"   : round(sum(ratios) / len(ratios), 3) if ratios else None,
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
        ("Soft Penalty"   , result.soft_penalty   , "soft_penalty"),
        ("Soft Reward"    , result.soft_reward    , "soft_reward"),
    ]

    for label, translation, mode_key in rows:
        if not translation:
            continue
        m = result.metrics.get(mode_key, {})
        sat_val  = m.get("satisfaction", {}).get("overall", "—")
        bleu_val = m.get("bleu_vs_baseline", "—")
        lr_val   = m.get("length_ratio", "—")
        print(
            f"  {label:<18}: {translation}\n"
            f"    ↳ satisfied={sat_val}  bleu_vs_base={bleu_val}  len_ratio={lr_val}"
        )


def print_aggregate(agg: Dict) -> None:
    """Print a formatted aggregate results table."""
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    header = f"  {'Mode':<18}  {'N':>4}  {'Sat%':>6}  {'BLEU':>6}  {'LenRatio':>9}"
    print(header)
    print("  " + "-" * 60)
    for mode, vals in agg.items():
        sat  = f"{vals['avg_satisfaction']*100:.1f}%" if vals['avg_satisfaction'] is not None else "  N/A"
        bleu = f"{vals['avg_bleu_vs_base']:.2f}"      if vals['avg_bleu_vs_base']  is not None else "  N/A"
        lr   = f"{vals['avg_length_ratio']:.3f}"      if vals['avg_length_ratio']  is not None else "  N/A"
        print(f"  {mode:<18}  {vals['n_samples']:>4}  {sat:>6}  {bleu:>6}  {lr:>9}")
    print("="*70)
