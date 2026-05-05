"""
interpretability.py — Analyse the per-step logit logs produced by the
constraint processors and surface human-readable insights.

Key metrics extracted (per the project spec):
  - Original rank & probability of constrained tokens BEFORE intervention.
  - Magnitude of probability manipulation required to enforce constraints.
  - Beam trajectory changes summarised from step-level pending counts
    (for hard inclusion) or from logit deltas (for exclusion / soft).

Fix #13: token_level_report() was dead code — defined but never called.
It is now integrated into compare_analyses() as an optional detail block
that fires automatically when a tokenizer is provided.
"""

from typing import List, Dict, Optional
import numpy as np


# ── Single-sample analysis ────────────────────────────────────────────────────

def analyse_log(log: List[Dict], constraint_type: str) -> Dict:
    """
    Summarise a log produced by one of the three constraint processors.

    Returns a dict with:
      - constraint_type
      - num_steps_active : how many decoding steps had active constraints
      - avg_original_rank : mean rank of constrained tokens before intervention
      - avg_original_prob : mean pre-intervention probability
      - avg_delta         : mean absolute logit shift applied
      - max_delta         : maximum logit shift applied (inf → 999 for display)
      - beam_pressure     : (hard inclusion only) avg number of pending groups
                            across steps — higher means more pressure on beam
    """
    if not log:
        return {
            "constraint_type"   : constraint_type,
            "num_steps_active"  : 0,
            "avg_original_rank" : None,
            "avg_original_prob" : None,
            "avg_delta"         : None,
            "max_delta"         : None,
            "beam_pressure"     : None,
        }

    ranks     = []
    probs     = []
    deltas    = []
    pressures = []

    for step in log:
        token_data = step.get("tokens", {})
        for tid, info in token_data.items():
            ranks.append(info.get("rank", None))
            probs.append(info.get("prob", None))
            raw_delta = info.get("delta", 0)
            # Convert inf (hard exclusion) to a large finite number for stats
            if raw_delta == float("inf") or raw_delta == float("-inf"):
                deltas.append(999.0)
            else:
                deltas.append(abs(raw_delta))

        # Beam pressure: relevant for hard inclusion
        if "pending_count" in step:
            pressures.append(step["pending_count"])

    def safe_mean(lst):
        lst = [x for x in lst if x is not None]
        return float(np.mean(lst)) if lst else None

    return {
        "constraint_type"   : constraint_type,
        "num_steps_active"  : len(log),
        "avg_original_rank" : safe_mean(ranks),
        "avg_original_prob" : safe_mean(probs),
        "avg_delta"         : safe_mean(deltas),
        "max_delta"         : max(deltas) if deltas else None,
        "beam_pressure"     : safe_mean(pressures) if pressures else None,
    }


# ── Comparison across constraint modes ────────────────────────────────────────

def compare_analyses(
    analyses  : Dict[str, Dict],
    logs      : Optional[Dict[str, List[Dict]]] = None,
    tokenizer = None,
    top_n_steps: int = 3,
    log_file: str = "results/deep_dive.log"
) -> None:
    """
    Pretty-print a side-by-side comparison of interpretability metrics
    across constraint modes for a single translation sample.

    Fix #13: If `logs` and `tokenizer` are provided, the previously dead
    token_level_report() is called here for each mode, surfacing per-step
    token detail by saving it to a log file instead of cluttering the console.

    Args:
        analyses   : dict mapping mode_name → analyse_log() output
        logs       : dict mapping mode_name → raw log list (optional)
        tokenizer  : model tokenizer for decoding token IDs (optional)
        top_n_steps: how many steps to show in the token-level detail
        log_file   : filepath to write the deep dive results to
    """
    print("\n  ┌─ Interpretability Summary " + "─" * 52)

    fields = [
        ("num_steps_active",  "Steps with active constraint"),
        ("avg_original_rank", "Avg rank of constrained token (pre-intervention)"),
        ("avg_original_prob", "Avg prob of constrained token (pre-intervention)"),
        ("avg_delta",         "Avg |logit delta| applied"),
        ("max_delta",         "Max |logit delta| applied"),
        ("beam_pressure",     "Avg pending inclusion groups (beam pressure)"),
    ]

    for field, label in fields:
        row = f"  │ {label:<52}"
        for mode, data in analyses.items():
            val = data.get(field)
            if val is None:
                cell = "   N/A  "
            elif isinstance(val, float):
                cell = f" {val:>7.3f}"
            else:
                cell = f" {val:>7}"
            row += f"  [{mode[:10]:<10}: {cell}]"
        print(row)

    print("  └" + "─" * 80)

    # Output token-level detail to a file when the caller provides raw logs + tokenizer.
    if logs is not None and tokenizer is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            for mode, log in logs.items():
                if log:
                    print(f"\n  ── Token-level detail: {mode} ──", file=f)
                    token_level_report(log, tokenizer, top_n_steps=top_n_steps, file_obj=f)
        
        print(f"\n  [!] Detailed token-level deep dive saved to: {log_file}")


# ── Token-level deep dive ─────────────────────────────────────────────────────

def token_level_report(log: List[Dict], tokenizer, top_n_steps: int = 3, file_obj=None) -> None:
    """
    Print a detailed step-by-step view of what happened to constrained tokens,
    limited to top_n_steps for readability, directly to a file object.

    Args:
        log         : raw log list from a constraint processor.
        tokenizer   : the model's tokenizer (for decoding token IDs to strings).
        top_n_steps : how many steps to display in detail.
        file_obj    : The file object to write to (defaults to sys.stdout if None)
    """
    import sys
    if file_obj is None:
        file_obj = sys.stdout

    if not log:
        print("    (no constraint log to display)", file=file_obj)
        return

    steps_to_show = log[:top_n_steps]
    print(f"\n    Token-level detail (first {len(steps_to_show)} steps):", file=file_obj)

    for entry in steps_to_show:
        step    = entry["step"]
        ctype   = entry["type"]
        tokens  = entry.get("tokens", {})
        pending = entry.get("pending_count", "—")
        note    = entry.get("note", "")

        print(f"\n    Step {step} [{ctype}]  pending_groups={pending}"
              + (f"  [{note}]" if note else ""), file=file_obj)

        for tid, info in tokens.items():
            word      = tokenizer.decode([int(tid)], skip_special_tokens=True)
            raw_delta = info.get("delta", 0)
            delta_str = (
                "±inf" if abs(raw_delta) == 999 or raw_delta == float("inf")
                else f"{raw_delta:+.1f}"
            )
            print(
                f"      token_id={tid:<6}  surface='{word:<12}'  "
                f"rank={info.get('rank', '?'):<6}  "
                f"prob={info.get('prob', 0):.4f}  "
                f"logit={info.get('logit', 0):+.3f}  "
                f"delta={delta_str}",
                file=file_obj
            )