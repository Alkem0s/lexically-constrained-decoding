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
import time
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
    
    # Inject references from the paired opposite direction
    for en, tr in zip(en_tr, tr_en):
        en["reference"] = tr["source"]
        tr["reference"] = en["source"]
        
    print(f"  Loaded {len(en_tr)} EN→TR cases and {len(tr_en)} TR→EN cases "
          f"from '{path}'")
    return en_tr, tr_en


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
        reference       = case.get("reference",       ""),
    )

    # 1. Unconstrained baseline
    t0 = time.perf_counter()
    result.unconstrained, _ = decoding.unconstrained(model_wrapper, src)
    result.latencies["unconstrained"] = (time.perf_counter() - t0) * 1000.0
    result.pass_counts["unconstrained"] = 1

    # 2. Hard exclusion (isolated)
    t0 = time.perf_counter()
    excl_trans, excl_log = decoding.hard_exclusion(
        model_wrapper, src, case.get("forbidden_words", [])
    )
    result.latencies["hard_exclusion"] = (time.perf_counter() - t0) * 1000.0
    result.hard_exclusion = excl_trans
    result.pass_counts["hard_exclusion"] = 1

    # 3. Hard inclusion — production mode (soft-first gate)
    t0 = time.perf_counter()
    incl_trans, incl_log = decoding.hard_inclusion(
        model_wrapper, src, case.get("required_words", [])
    )
    result.latencies["hard_inclusion"] = (time.perf_counter() - t0) * 1000.0
    result.hard_inclusion = incl_trans
    result.pass_counts["hard_inclusion"] = 1

    # 4. Hard inclusion ablation — bypasses soft gate entirely for fair comparison
    t0 = time.perf_counter()
    hincl_abl_trans, hincl_abl_log = decoding.hard_inclusion(
        model_wrapper, src, case.get("required_words", []), _force_hard=True
    )
    result.latencies["hard_inclusion_ablation"] = (time.perf_counter() - t0) * 1000.0
    result.hard_inclusion_ablation = hincl_abl_trans
    result.pass_counts["hard_inclusion_ablation"] = 1

    # 5. Hard combined (exclusion + inclusion simultaneously)
    t0 = time.perf_counter()
    hcomb_trans, hcomb_excl_log, hcomb_incl_log = decoding.combined_hard(
        model_wrapper, src,
        forbidden_words = case.get("forbidden_words", []),
        required_words  = case.get("required_words",  []),
    )
    result.latencies["hard_combined"] = (time.perf_counter() - t0) * 1000.0
    result.hard_combined = hcomb_trans
    result.pass_counts["hard_combined"] = 1

    # 6. Soft penalty only (isolated, pure soft — no escalation)
    t0 = time.perf_counter()
    spen_trans, spen_log = decoding.soft_penalty_only(
        model_wrapper, src,
        penalty_words = case.get("penalty_words", []),
    )
    result.latencies["soft_penalty"] = (time.perf_counter() - t0) * 1000.0
    result.soft_penalty = spen_trans
    result.pass_counts["soft_penalty"] = 1

    # 7. Soft reward only (isolated, with escalation ladder)
    t0 = time.perf_counter()
    srew_trans, srew_log = decoding.soft_reward_only(
        model_wrapper, src,
        reward_words = case.get("reward_words", []),
    )
    result.latencies["soft_reward"] = (time.perf_counter() - t0) * 1000.0
    result.soft_reward = srew_trans
    
    # Calculate passes for soft reward
    srew_passes = 1
    for entry in srew_log:
        if isinstance(entry, dict):
            if entry.get("escalated") == "hard_inclusion":
                srew_passes = 3
                break
            elif entry.get("escalated") == "soft_boost":
                srew_passes = 2
    result.pass_counts["soft_reward"] = srew_passes

    # 8. Soft combined — pure soft, no hard fallback (see combined_soft docstring)
    t0 = time.perf_counter()
    scomb_trans, scomb_log = decoding.combined_soft(
        model_wrapper, src,
        penalty_words = case.get("penalty_words", []),
        reward_words  = case.get("reward_words",  []),
    )
    result.latencies["soft_combined"] = (time.perf_counter() - t0) * 1000.0
    result.soft_combined = scomb_trans
    
    # Calculate passes for soft combined
    scomb_passes = 1
    for entry in scomb_log:
        if isinstance(entry, dict) and entry.get("escalated") in ("soft_boost", "soft_boost_failed"):
            scomb_passes = 2
            break
    result.pass_counts["soft_combined"] = scomb_passes

    # 9. HuggingFace DBA baseline
    t0 = time.perf_counter()
    dba_trans, dba_log = decoding.huggingface_dba(
        model_wrapper, src,
        required_words  = case.get("required_words",  []),
        forbidden_words = case.get("forbidden_words", []),
    )
    result.latencies["huggingface_dba"] = (time.perf_counter() - t0) * 1000.0
    result.huggingface_dba = dba_trans
    result.pass_counts["huggingface_dba"] = 1

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
        "huggingface_dba" : dba_log,
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
        "huggingface_dba": interpretability.analyse_log(dba_log,        "huggingface_dba"),
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
        "huggingface_dba"         : r.huggingface_dba,
        "forbidden_words"         : r.forbidden_words,
        "required_words"          : r.required_words,
        "penalty_words"           : r.penalty_words,
        "reward_words"            : r.reward_words,
        "escalation_counts"       : r.escalation_counts,
        "latencies"               : r.latencies,
        "pass_counts"             : r.pass_counts,
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
        "huggingface_dba"         : r.huggingface_dba,
        "forbidden_words"         : r.forbidden_words,
        "required_words"          : r.required_words,
        "penalty_words"           : r.penalty_words,
        "reward_words"            : r.reward_words,
        "escalation_counts"       : r.escalation_counts,
        "latencies"               : r.latencies,
        "pass_counts"             : r.pass_counts,
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


def _average_aggregates(agg_list: List[Dict]) -> Dict:
    """
    Given a list of aggregate dicts (one per seed), return a new dict where
    every numeric leaf is replaced by the mean across all seeds.
    Non-numeric values (e.g. None) are passed through from the first dict.
    """
    if len(agg_list) == 1:
        return agg_list[0]

    merged: Dict = {}
    for mode in agg_list[0]:
        merged[mode] = {}
        for key in agg_list[0][mode]:
            vals = [a[mode][key] for a in agg_list if mode in a and a[mode][key] is not None]
            if vals and isinstance(vals[0], (int, float)):
                merged[mode][key] = sum(vals) / len(vals)
            else:
                merged[mode][key] = agg_list[0][mode][key]
    return merged


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lexically Constrained MT Evaluation Pipeline")
    parser.add_argument("--data", type=str, default="test_cases_eval.json",
                        help="Path to evaluation test cases JSON file")
    args = parser.parse_args()

    print("="*70)
    print("  Lexically Constrained & Interpretable Decoding for NMT")
    print("="*70)
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Device   : {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠  No CUDA device detected — see config.py for reinstall instructions.")
    seeds = config.EVAL_SEEDS
    print(f"  Seeds    : {seeds} ({len(seeds)} run(s) will be averaged)")
    print("="*70)

    # Load test cases dynamically
    en_tr_cases, tr_en_cases = load_test_cases(args.data)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_seeds = len(seeds)

    seed_aggregates_en_tr: List[Dict] = []
    seed_aggregates_tr_en: List[Dict] = []
    last_en_tr_results: List[SampleResult] = []
    last_tr_en_results: List[SampleResult] = []
    last_en_tr_interp:  List[Dict] = []
    last_tr_en_interp:  List[Dict] = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"  Seed run {seed_idx + 1}/{n_seeds}  (seed={seed})")
        print(f"{'='*70}")
        set_seeds(seed)

        # ── EN → TR ───────────────────────────────────────────────────────────
        if seed_idx == 0:
            print("\n[Step 1] Loading EN→TR model...")
        en_tr_model = model_loader.load_en_tr()

        seed_en_tr_results: List[SampleResult] = []
        seed_en_tr_interp:  List[Dict] = []
        if en_tr_cases:
            seed_en_tr_results, seed_en_tr_interp = run_direction(en_tr_model, en_tr_cases, "EN→TR")
        else:
            print("  [SKIP] No EN→TR cases loaded.")

        del en_tr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── TR → EN ───────────────────────────────────────────────────────────
        if seed_idx == 0:
            print("\n[Step 2] Loading TR→EN model...")
        tr_en_model = model_loader.load_tr_en()

        seed_tr_en_results: List[SampleResult] = []
        seed_tr_en_interp:  List[Dict] = []
        if tr_en_cases:
            seed_tr_en_results, seed_tr_en_interp = run_direction(tr_en_model, tr_en_cases, "TR→EN")
        else:
            print("  [SKIP] No TR→EN cases loaded.")

        del tr_en_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Aggregate this seed's results — directions kept strictly separate
        if seed_en_tr_results:
            print(f"\n[Seed {seed_idx + 1}] Aggregating EN→TR...")
            agg_en_tr = evaluation.aggregate_results(seed_en_tr_results)
            evaluation.print_aggregate(agg_en_tr)
            seed_aggregates_en_tr.append(agg_en_tr)

        if seed_tr_en_results:
            print(f"\n[Seed {seed_idx + 1}] Aggregating TR→EN...")
            agg_tr_en = evaluation.aggregate_results(seed_tr_en_results)
            evaluation.print_aggregate(agg_tr_en)
            seed_aggregates_tr_en.append(agg_tr_en)

        last_en_tr_results = seed_en_tr_results
        last_tr_en_results = seed_tr_en_results
        last_en_tr_interp  = seed_en_tr_interp
        last_tr_en_interp  = seed_tr_en_interp

    # ── Average across seeds — directions kept strictly separate ──────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if seed_aggregates_en_tr:
        print(f"\n{'='*70}")
        print(f"  EN→TR  —  final averaged aggregate over {len(seed_aggregates_en_tr)} seed(s): {seeds}")
        print(f"{'='*70}")
        agg_en_tr_final = _average_aggregates(seed_aggregates_en_tr)
        evaluation.print_aggregate(agg_en_tr_final)
        agg_en_tr_path = os.path.join(config.RESULTS_DIR, f"aggregate_en_tr_{run_id}.json")
        with open(agg_en_tr_path, "w") as f:
            json.dump(agg_en_tr_final, f, indent=2)
        print(f"  Saved EN→TR averaged aggregate → {agg_en_tr_path}")

    if seed_aggregates_tr_en:
        print(f"\n{'='*70}")
        print(f"  TR→EN  —  final averaged aggregate over {len(seed_aggregates_tr_en)} seed(s): {seeds}")
        print(f"{'='*70}")
        agg_tr_en_final = _average_aggregates(seed_aggregates_tr_en)
        evaluation.print_aggregate(agg_tr_en_final)
        agg_tr_en_path = os.path.join(config.RESULTS_DIR, f"aggregate_tr_en_{run_id}.json")
        with open(agg_tr_en_path, "w") as f:
            json.dump(agg_tr_en_final, f, indent=2)
        print(f"  Saved TR→EN averaged aggregate → {agg_tr_en_path}")

    # Save last seed's full translations for qualitative review — per direction
    if last_en_tr_results:
        save_results(last_en_tr_results, last_en_tr_interp, f"en_tr_{run_id}")
    if last_tr_en_results:
        save_results(last_tr_en_results, last_tr_en_interp, f"tr_en_{run_id}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()