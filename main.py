"""
main.py — Pipeline entry point for Lexically Constrained MT project.

Runs the full experiment for both EN→TR and TR→EN directions:
  1. Load models (with local caching).
  2. For each test sentence + constraint pair:
       a. Unconstrained baseline.
       b. Hard exclusion.
       c. Hard inclusion.
       d. Soft penalty / reward.
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
#   forbidden_words : target-language words to EXCLUDE (hard & soft penalty)
#   required_words  : target-language words to INCLUDE (hard & soft reward)
#
# Words are in the TARGET language so they can be directly mapped to token IDs.

EN_TR_CASES = [
    {
        "source"         : "The cat sat on the mat and looked out the window.",
        "direction"      : "EN→TR",
        "forbidden_words": ["pencere"],          # "window" in Turkish
        "required_words" : ["kedi"],             # "cat" in Turkish
        "penalty_words"  : ["pencere"],
        "reward_words"   : ["kedi"],
    },
    {
        "source"         : "I want to eat food at the restaurant tonight.",
        "direction"      : "EN→TR",
        "forbidden_words": ["restoran"],         # "restaurant"
        "required_words" : ["yemek"],            # "food / meal"
        "penalty_words"  : ["restoran"],
        "reward_words"   : ["yemek"],
    },
    {
        "source"         : "The student studied hard and passed the exam.",
        "direction"      : "EN→TR",
        "forbidden_words": ["sınav"],            # "exam"
        "required_words" : ["öğrenci"],          # "student"
        "penalty_words"  : ["sınav"],
        "reward_words"   : ["öğrenci"],
    },
    {
        "source"         : "Water is essential for life and health.",
        "direction"      : "EN→TR",
        "forbidden_words": ["hayat"],            # "life"
        "required_words" : ["su"],               # "water"
        "penalty_words"  : ["hayat"],
        "reward_words"   : ["su"],
    },
    {
        "source"         : "She read a book in the library all afternoon.",
        "direction"      : "EN→TR",
        "forbidden_words": ["kütüphane"],        # "library"
        "required_words" : ["kitap"],            # "book"
        "penalty_words"  : ["kütüphane"],
        "reward_words"   : ["kitap"],
    },
]

TR_EN_CASES = [
    {
        "source"         : "Köpek parkta koştu ve çok yoruldu.",
        "direction"      : "TR→EN",
        "forbidden_words": ["park"],             # same in English
        "required_words" : ["dog"],
        "penalty_words"  : ["park"],
        "reward_words"   : ["dog"],
    },
    {
        "source"         : "Annem her sabah kahve içer ve gazete okur.",
        "direction"      : "TR→EN",
        "forbidden_words": ["coffee"],
        "required_words" : ["mother"],
        "penalty_words"  : ["coffee"],
        "reward_words"   : ["mother"],
    },
    {
        "source"         : "Hava bugün çok sıcak, denize gidebiliriz.",
        "direction"      : "TR→EN",
        "forbidden_words": ["hot"],
        "required_words" : ["sea"],
        "penalty_words"  : ["hot"],
        "reward_words"   : ["sea"],
    },
    {
        "source"         : "Çocuklar okuldan sonra futbol oynadı.",
        "direction"      : "TR→EN",
        "forbidden_words": ["football"],
        "required_words" : ["children"],
        "penalty_words"  : ["football"],
        "reward_words"   : ["children"],
    },
    {
        "source"         : "Doktor hastaya ilaç yazdı ve dinlenmesini söyledi.",
        "direction"      : "TR→EN",
        "forbidden_words": ["medicine"],
        "required_words" : ["doctor"],
        "penalty_words"  : ["medicine"],
        "reward_words"   : ["doctor"],
    },
]


# ── Per-sample runner ─────────────────────────────────────────────────────────

def run_sample(model_wrapper, case: Dict, interp_logs: List) -> SampleResult:
    """
    Run all four decoding modes for one test case and return a SampleResult.
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

    # 4a. Soft penalty
    sp_trans, sp_log = decoding.soft_constrained(
        model_wrapper, src,
        penalty_words = case.get("penalty_words", []),
    )
    result.soft_penalty = sp_trans

    # 4b. Soft reward
    sr_trans, sr_log = decoding.soft_constrained(
        model_wrapper, src,
        reward_words  = case.get("reward_words", []),
    )
    result.soft_reward = sr_trans

    # ── Interpretability ──────────────────────────────────────────────────────
    analyses = {
        "hard_excl" : interpretability.analyse_log(excl_log, "hard_exclusion"),
        "hard_incl" : interpretability.analyse_log(incl_log, "hard_inclusion"),
        "soft_pen"  : interpretability.analyse_log(sp_log,   "soft_penalty"),
        "soft_rew"  : interpretability.analyse_log(sr_log,   "soft_reward"),
    }
    interp_logs.append({"source": src, "direction": dir_, "analyses": analyses})

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluation.evaluate_sample(result)

    return result


# ── Direction runner ──────────────────────────────────────────────────────────

def run_direction(model_wrapper, cases: List[Dict], direction_label: str) -> List[SampleResult]:
    print(f"\n{'='*70}")
    print(f"  Direction: {direction_label}")
    print(f"{'='*70}")

    results     = []
    interp_logs = []

    for case in tqdm(cases, desc=f"  {direction_label}"):
        result = run_sample(model_wrapper, case, interp_logs)
        evaluation.print_sample_result(result)
        interpretability.compare_analyses(
            {k: v for k, v in interp_logs[-1]["analyses"].items()}
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
        json.dump(all_interp, f, ensure_ascii=False, indent=2)
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
    print("="*70)

    set_seeds()
    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    all_interp  = []

    # ── EN → TR ───────────────────────────────────────────────────────────────
    print("\n[Step 1] Loading EN→TR model...")
    en_tr_model = model_loader.load_en_tr()

    en_tr_results, en_tr_interp = run_direction(en_tr_model, EN_TR_CASES, "EN→TR")
    all_results.extend(en_tr_results)
    all_interp.extend(en_tr_interp)

    # Free VRAM before loading second model
    del en_tr_model
    torch.cuda.empty_cache()

    # ── TR → EN ───────────────────────────────────────────────────────────────
    print("\n[Step 2] Loading TR→EN model...")
    tr_en_model = model_loader.load_tr_en()

    tr_en_results, tr_en_interp = run_direction(tr_en_model, TR_EN_CASES, "TR→EN")
    all_results.extend(tr_en_results)
    all_interp.extend(tr_en_interp)

    del tr_en_model
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
