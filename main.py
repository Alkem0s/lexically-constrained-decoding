"""
main.py — Pipeline entry point for Lexically Constrained MT project.

Runs the full experiment for both EN→TR and TR→EN directions:
  1. Load models (with local caching).
  2. For each test sentence + constraint pair:
       a. Unconstrained baseline.
       b. Hard exclusion.
       c. Hard inclusion       (soft-first gate, hard fallback if needed).
       d. Hard inclusion abl.  (ablation: pure HardInclusionProcessor, no soft gate).
       e. Hard combined        (simultaneous exclusion + inclusion).
       f. Soft penalty only    (pure soft, no escalation).
       g. Soft reward only     (soft with escalation ladder).
       h. Soft combined        (pure soft: single pass, no hard fallback).
  3. Compute per-sample escalation counts from the returned logs.
  4. Evaluate constraint satisfaction and BLEU vs baseline.
  5. Run interpretability analysis on each constraint log.
  6. Save all results to ./results/.

Usage:
    python main.py
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import pathlib

import config
import model_loader
import decoding
import interpretability
import evaluation
from evaluation import SampleResult, count_escalations


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Test case loading ──────────────────────────────────────────────────────────

def load_test_cases(path: str = "test_cases.json") -> tuple:
    """
    Load EN_TR and TR_EN test cases from a JSON file.

    The JSON schema is:
      {
        "EN_TR": [ { source, direction, forbidden_words, required_words,
                     penalty_words, reward_words, comment? }, ... ],
        "TR_EN": [ ... ]
      }

    Falls back to empty lists with a clear error message if the file is missing
    or malformed so the rest of the pipeline can still run.
    """
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[WARNING] test_cases.json not found at '{path}'. "
              "Using empty case lists.")
        return [], []

    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Could not parse '{path}': {exc}")
        return [], []

    en_tr = data.get("EN_TR", [])
    tr_en = data.get("TR_EN", [])
    print(f"  Loaded {len(en_tr)} EN→TR cases and {len(tr_en)} TR→EN cases "
          f"from '{path}'")
    return en_tr, tr_en


EN_TR_CASES, TR_EN_CASES = load_test_cases("test_cases_eval.json")


# ── Per-sample runner ─────────────────────────────────────────────────────────

def run_sample(model_wrapper, case: Dict, interp_logs: List) -> tuple:
    """
    Run all decoding modes for one test case and return (SampleResult, raw_logs).

    Modes run:
      - unconstrained baseline
      - hard_exclusion      (isolated)
      - hard_inclusion      (soft-first gate, hard fallback for stubborn words)
      - hard_inclusion_abl  (ablation: _force_hard=True, bypasses soft gate)
      - hard_combined       (exclusion + inclusion simultaneously)
      - soft_penalty        (isolated, pure soft — no escalation)
      - soft_reward         (isolated, with escalation ladder)
      - soft_combined       (pure soft combined — no hard fallback)

    After all decoding, escalation counts are computed from the returned logs
    and stored on result.escalation_counts so evaluate_sample can surface them.
    """
    src  = case["source"]
    dir_ = case["direction"]

    result = SampleResult(
        source          = src,
        direction       = dir_,
        forbidden_words = case.get("forbidden_words", []),
        required_words  = case.get("required_words",  []),
        penalty_words   = case.get("penalty_words",   []),
        reward_words    = case.get("reward_words",    []),
    )

    # 1. Unconstrained baseline
    result.unconstrained, _ = decoding.unconstrained(model_wrapper, src)

    # 2. Hard exclusion (isolated)
    excl_trans, excl_log = decoding.hard_exclusion(
        model_wrapper, src, case.get("forbidden_words", [])
    )
    result.hard_exclusion = excl_trans

    # 3. Hard inclusion — production mode (soft-first gate)
    incl_trans, incl_log = decoding.hard_inclusion(
        model_wrapper, src, case.get("required_words", [])
    )
    result.hard_inclusion = incl_trans

    # 4. Hard inclusion ablation — bypasses soft gate entirely for fair comparison
    hincl_abl_trans, hincl_abl_log = decoding.hard_inclusion(
        model_wrapper, src, case.get("required_words", []), _force_hard=True
    )
    result.hard_inclusion_ablation = hincl_abl_trans

    # 5. Hard combined (exclusion + inclusion simultaneously)
    hcomb_trans, hcomb_excl_log, hcomb_incl_log = decoding.combined_hard(
        model_wrapper, src,
        forbidden_words = case.get("forbidden_words", []),
        required_words  = case.get("required_words",  []),
    )
    result.hard_combined = hcomb_trans

    # 6. Soft penalty only (isolated, pure soft — no escalation)
    spen_trans, spen_log = decoding.soft_penalty_only(
        model_wrapper, src,
        penalty_words = case.get("penalty_words", []),
    )
    result.soft_penalty = spen_trans

    # 7. Soft reward only (isolated, with escalation ladder)
    srew_trans, srew_log = decoding.soft_reward_only(
        model_wrapper, src,
        reward_words = case.get("reward_words", []),
    )
    result.soft_reward = srew_trans

    # 8. Soft combined — pure soft, no hard fallback (see combined_soft docstring)
    scomb_trans, scomb_log = decoding.combined_soft(
        model_wrapper, src,
        penalty_words = case.get("penalty_words", []),
        reward_words  = case.get("reward_words",  []),
    )
    result.soft_combined = scomb_trans

    # ── Escalation counting ───────────────────────────────────────────────────
    # Count how many tier-2/tier-3 fallback events appear in each mode's log.
    # Stored on result so evaluate_sample can surface them in per-mode metrics.
    esc: Dict[str, int] = {}
    simple_mode_logs = [
        ("hard_exclusion",          excl_log),
        ("hard_inclusion",          incl_log),
        ("hard_inclusion_ablation", hincl_abl_log),
        ("soft_penalty",            spen_log),
        ("soft_reward",             srew_log),
        ("soft_combined",           scomb_log),
    ]
    for mode_name, log in simple_mode_logs:
        n = count_escalations(log)
        if n > 0:
            esc[mode_name] = n
    # combined_hard draws from two logs
    hcomb_esc = count_escalations(hcomb_excl_log) + count_escalations(hcomb_incl_log)
    if hcomb_esc > 0:
        esc["hard_combined"] = hcomb_esc
    result.escalation_counts = esc

    # ── Interpretability ──────────────────────────────────────────────────────
    raw_logs = {
        "hard_excl"       : excl_log,
        "hard_incl"       : incl_log,
        "hard_incl_abl"   : hincl_abl_log,
        "hard_comb_excl"  : hcomb_excl_log,
        "hard_comb_incl"  : hcomb_incl_log,
        "soft_pen"        : spen_log,
        "soft_rew"        : srew_log,
        "soft_comb"       : scomb_log,
    }
    analyses = {
        "hard_excl"      : interpretability.analyse_log(excl_log,       "hard_exclusion"),
        "hard_incl"      : interpretability.analyse_log(incl_log,       "hard_inclusion"),
        "hard_incl_abl"  : interpretability.analyse_log(hincl_abl_log,  "hard_inclusion_ablation"),
        "hard_comb_excl" : interpretability.analyse_log(hcomb_excl_log, "hard_combined_excl"),
        "hard_comb_incl" : interpretability.analyse_log(hcomb_incl_log, "hard_combined_incl"),
        "soft_pen"       : interpretability.analyse_log(spen_log,       "soft_penalty"),
        "soft_rew"       : interpretability.analyse_log(srew_log,       "soft_reward"),
        "soft_comb"      : interpretability.analyse_log(scomb_log,      "soft_combined"),
    }
    interp_logs.append({
        "source"   : src,
        "direction": dir_,
        "analyses" : analyses,
        "raw_logs" : raw_logs,
    })

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluation.evaluate_sample(result)

    return result, raw_logs


# ── Direction runner ──────────────────────────────────────────────────────────

def run_direction(model_wrapper, cases: List[Dict], direction_label: str):
    print(f"\n{'='*70}")
    print(f"  Direction: {direction_label}")
    print(f"{'='*70}")

    results     = []
    interp_logs = []

    for case in tqdm(cases, desc=f"  {direction_label}"):
        result, raw_logs = run_sample(model_wrapper, case, interp_logs)
        evaluation.print_sample_result(result)

        # Fix #13: pass raw logs and the model's tokenizer so
        # compare_analyses() can call token_level_report() instead of
        # silently skipping it.
        interpretability.compare_analyses(
            analyses    = {k: v for k, v in interp_logs[-1]["analyses"].items()},
            logs        = raw_logs,
            tokenizer   = model_wrapper.tokenizer,
            top_n_steps = 2,
        )
        results.append(result)

    return results, interp_logs


# ── Save helpers ──────────────────────────────────────────────────────────────

def _serialise_result(r: SampleResult) -> Dict:
    """Convert SampleResult to a JSON-serialisable dict."""
    return {
        "source"                  : r.source,
        "direction"               : r.direction,
        "unconstrained"           : r.unconstrained,
        "hard_exclusion"          : r.hard_exclusion,
        "hard_inclusion"          : r.hard_inclusion,
        "hard_inclusion_ablation" : r.hard_inclusion_ablation,
        "hard_combined"           : r.hard_combined,
        "soft_penalty"            : r.soft_penalty,
        "soft_reward"             : r.soft_reward,
        "soft_combined"           : r.soft_combined,
        "forbidden_words"         : r.forbidden_words,
        "required_words"          : r.required_words,
        "penalty_words"           : r.penalty_words,
        "reward_words"            : r.reward_words,
        "escalation_counts"       : r.escalation_counts,
        "metrics"                 : r.metrics,
    }


def _serialise_debug_result(r: SampleResult) -> Dict:
    """Convert SampleResult to a shorter JSON dict without metrics for debugging."""
    return {
        "source"                  : r.source,
        "direction"               : r.direction,
        "unconstrained"           : r.unconstrained,
        "hard_exclusion"          : r.hard_exclusion,
        "hard_inclusion"          : r.hard_inclusion,
        "hard_inclusion_ablation" : r.hard_inclusion_ablation,
        "hard_combined"           : r.hard_combined,
        "soft_penalty"            : r.soft_penalty,
        "soft_reward"             : r.soft_reward,
        "soft_combined"           : r.soft_combined,
        "forbidden_words"         : r.forbidden_words,
        "required_words"          : r.required_words,
        "penalty_words"           : r.penalty_words,
        "reward_words"            : r.reward_words,
        "escalation_counts"       : r.escalation_counts,
    }


def _serialise_interp(entry: Dict) -> Dict:
    """Strip the raw_logs tensor data before JSON serialisation."""
    return {
        "source"   : entry["source"],
        "direction": entry["direction"],
        "analyses" : entry["analyses"],
        # raw_logs are kept in memory but not written to JSON (too verbose)
    }


def save_results(all_results: List[SampleResult], all_interp: List[Dict], run_id: str):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Translation + evaluation results
    trans_path = os.path.join(config.RESULTS_DIR, f"results_{run_id}.json")
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump([_serialise_result(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"\n  Saved translation results → {trans_path}")

    # Shorter debug results
    debug_path = os.path.join(config.RESULTS_DIR, f"debug_results_{run_id}.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump([_serialise_debug_result(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"  Saved debug results → {debug_path}")

    # Interpretability logs
    interp_path = os.path.join(config.RESULTS_DIR, f"interpretability_{run_id}.json")
    with open(interp_path, "w", encoding="utf-8") as f:
        json.dump([_serialise_interp(e) for e in all_interp], f, ensure_ascii=False, indent=2)
    print(f"  Saved interpretability logs → {interp_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  Lexically Constrained & Interpretable Decoding for NMT")
    print("="*70)
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Device   : {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠  No CUDA device detected — see config.py for reinstall instructions.")
    print("="*70)

    set_seeds()
    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    all_interp  = []

    # ── EN → TR ───────────────────────────────────────────────────────────────
    print("\n[Step 1] Loading EN→TR model...")
    en_tr_model = model_loader.load_en_tr()

    if EN_TR_CASES:
        en_tr_results, en_tr_interp = run_direction(en_tr_model, EN_TR_CASES, "EN→TR")
        all_results.extend(en_tr_results)
        all_interp.extend(en_tr_interp)
    else:
        print("  [SKIP] No EN→TR cases loaded.")

    del en_tr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── TR → EN ───────────────────────────────────────────────────────────────
    print("\n[Step 2] Loading TR→EN model...")
    tr_en_model = model_loader.load_tr_en()

    if TR_EN_CASES:
        tr_en_results, tr_en_interp = run_direction(tr_en_model, TR_EN_CASES, "TR→EN")
        all_results.extend(tr_en_results)
        all_interp.extend(tr_en_interp)
    else:
        print("  [SKIP] No TR→EN cases loaded.")

    del tr_en_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Aggregate evaluation ──────────────────────────────────────────────────
    print("\n[Step 3] Aggregating results...")
    agg = evaluation.aggregate_results(all_results)
    evaluation.print_aggregate(agg)

    # Save aggregate
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    agg_path = os.path.join(config.RESULTS_DIR, f"aggregate_{run_id}.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"  Saved aggregate → {agg_path}")

    # ── Save full results ─────────────────────────────────────────────────────
    save_results(all_results, all_interp, run_id)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()