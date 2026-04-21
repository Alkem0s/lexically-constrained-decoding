"""
main.py — Pipeline entry point for Lexically Constrained MT project.

Runs the full experiment for both EN→TR and TR→EN directions:
  1. Load models (with local caching).
  2. For each test sentence + constraint pair:
       a. Unconstrained baseline.
       b. Hard exclusion.
       c. Hard inclusion.
       d. Soft penalty + reward (single combined call — fix #11).
  3. Evaluate constraint satisfaction and BLEU vs baseline.
  4. Run interpretability analysis on each constraint log.
  5. Save all results to ./results/.

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
from evaluation import SampleResult


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Test cases ─────────────────────────────────────────────────────────────────
#
# Each entry is a dict:
#   source          : source sentence (in source language)
#   direction       : "EN→TR" or "TR→EN"
#   forbidden_words : target-language words to EXCLUDE (hard exclusion & soft penalty)
#   required_words  : target-language words to INCLUDE (hard inclusion & soft reward)
#
# Words are in the TARGET language so they can be directly mapped to token IDs.

# ── Per-sample runner ─────────────────────────────────────────────────────────

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


EN_TR_CASES, TR_EN_CASES = load_test_cases()


def run_sample(model_wrapper, case: Dict, interp_logs: List) -> SampleResult:
    """
    Run all decoding modes for one test case and return a SampleResult.

    Fix #11: soft penalty and soft reward are now handled in ONE combined
    soft_constrained() call that runs a single model.generate() pass, instead
    of two separate calls that each ran a full generate().  The combined
    translation is stored in both soft_penalty and soft_reward fields of the
    result (they share the same output when both constraints are active).
    If you need separate penalty-only and reward-only outputs for ablation,
    call soft_constrained twice explicitly — but for the main pipeline one
    combined run is correct and twice as fast.
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

    # 2. Hard exclusion
    excl_trans, excl_log = decoding.hard_exclusion(
        model_wrapper, src, case.get("forbidden_words", [])
    )
    result.hard_exclusion = excl_trans

    # 3. Hard inclusion
    incl_trans, incl_log = decoding.hard_inclusion(
        model_wrapper, src, case.get("required_words", [])
    )
    result.hard_inclusion = incl_trans

    # 4. Soft — single combined call (fix #11)
    soft_trans, soft_log = decoding.soft_constrained(
        model_wrapper, src,
        penalty_words = case.get("penalty_words", []),
        reward_words  = case.get("reward_words",  []),
    )
    result.soft_penalty = soft_trans
    result.soft_reward  = soft_trans   # same run; evaluated against both word lists

    # ── Interpretability ──────────────────────────────────────────────────────
    raw_logs = {
        "hard_excl" : excl_log,
        "hard_incl" : incl_log,
        "soft"      : soft_log,
    }
    analyses = {
        "hard_excl" : interpretability.analyse_log(excl_log, "hard_exclusion"),
        "hard_incl" : interpretability.analyse_log(incl_log, "hard_inclusion"),
        "soft_pen"  : interpretability.analyse_log(soft_log, "soft_penalty"),
        "soft_rew"  : interpretability.analyse_log(soft_log, "soft_reward"),
    }
    interp_logs.append({
        "source"   : src,
        "direction": dir_,
        "analyses" : analyses,
        "raw_logs" : raw_logs,   # retained for offline token_level_report
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
            analyses  = {k: v for k, v in interp_logs[-1]["analyses"].items()},
            logs      = raw_logs,
            tokenizer = model_wrapper.tokenizer,
            top_n_steps = 2,
        )
        results.append(result)

    return results, interp_logs


# ── Save helpers ──────────────────────────────────────────────────────────────

def _serialise_result(r: SampleResult) -> Dict:
    """Convert SampleResult to a JSON-serialisable dict."""
    return {
        "source"         : r.source,
        "direction"      : r.direction,
        "unconstrained"  : r.unconstrained,
        "hard_exclusion" : r.hard_exclusion,
        "hard_inclusion" : r.hard_inclusion,
        "soft_penalty"   : r.soft_penalty,
        "soft_reward"    : r.soft_reward,
        "forbidden_words": r.forbidden_words,
        "required_words" : r.required_words,
        "metrics"        : r.metrics,
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