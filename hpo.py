import optuna
import json
import os
import torch
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

import config
import model_loader
import decoding
import evaluation
from main import load_test_cases, set_seeds

# ── HPO Settings ─────────────────────────────────────────────────────────────
N_TRIALS = 50
SEED = 42

# ── Objective Function ───────────────────────────────────────────────────────

def objective(trial, en_tr_model, tr_en_model, en_tr_cases, tr_en_cases):
    # 1. Suggest parameters
    config.HARD_INCL_EARLY_TOKENS = trial.suggest_int("HARD_INCL_EARLY_TOKENS", 1, 6)
    config.HARD_INCL_SWEET_RANK   = trial.suggest_int("HARD_INCL_SWEET_RANK", 50, 1000)
    config.HARD_INCL_SWEET_BUFFER = trial.suggest_float("HARD_INCL_SWEET_BUFFER", 1.0, 15.0)
    config.HARD_INCL_ANCHOR_START = trial.suggest_float("HARD_INCL_ANCHOR_START", -15.0, -2.0)
    config.HARD_INCL_ANCHOR_RANGE = trial.suggest_float("HARD_INCL_ANCHOR_RANGE", 5.0, 25.0)
    
    config.SOFT_REWARD_STRENGTH  = trial.suggest_float("SOFT_REWARD_STRENGTH", 1.0, 10.0)
    config.SOFT_PENALTY_STRENGTH = trial.suggest_float("SOFT_PENALTY_STRENGTH", -20.0, -5.0)
    config.ANCHOR_OFFSET         = trial.suggest_float("ANCHOR_OFFSET", -10.0, 0.0)
    config.HARD_INCLUSION_BOOST  = trial.suggest_float("HARD_INCLUSION_BOOST", 15.0, 40.0)
    config.SUFFIX_PENALTY_TR     = trial.suggest_float("SUFFIX_PENALTY_TR", -15.0, 0.0)

    # 2. Run evaluation
    all_results = []
    
    # EN -> TR
    for case in en_tr_cases:
        trans, log = decoding.hard_inclusion(en_tr_model, case["source"], case["required_words"])
        
        res = evaluation.SampleResult(
            source=case["source"],
            direction=case["direction"],
            required_words=case["required_words"],
            unconstrained=case["baseline"] # Use pre-calculated baseline
        )
        res.hard_inclusion = trans
        evaluation.evaluate_sample(res)
        all_results.append(res)

    # TR -> EN
    for case in tr_en_cases:
        trans, log = decoding.hard_inclusion(tr_en_model, case["source"], case["required_words"])
        res = evaluation.SampleResult(
            source=case["source"],
            direction=case["direction"],
            required_words=case["required_words"],
            unconstrained=case["baseline"] # Use pre-calculated baseline
        )
        res.hard_inclusion = trans
        evaluation.evaluate_sample(res)
        all_results.append(res)

    # 3. Aggregate metrics
    agg = evaluation.aggregate_results(all_results)
    mode = "hard_inclusion"
    
    if mode not in agg:
        return 0.0
    
    sat_rate = agg[mode]["avg_satisfaction"]  # 0.0 to 1.0
    avg_bleu = agg[mode]["avg_bleu_vs_base"] or 0.0 # 0 to 100
    
    # Store attributes for visibility
    trial.set_user_attr("satisfaction", sat_rate)
    trial.set_user_attr("bleu", avg_bleu)
    
    # Composite score: Lexicographical. 100% Satisfaction strongly beats any lower satisfaction.
    score = (sat_rate * 1000) + avg_bleu
    
    # Print progress with breakdown
    print(f"  Trial {trial.number:2d} | Sat: {sat_rate*100:5.1f}% | BLEU: {avg_bleu:5.2f} | Score: {score:7.3f}")
    
    return score

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  Lexically Constrained MT — Hyperparameter Optimization")
    print("="*70)
    
    set_seeds(SEED)
    
    # 1. Load models once
    print("\n[Step 1] Loading models...")
    en_tr_model = model_loader.load_en_tr()
    tr_en_model = model_loader.load_tr_en()
    
    # 2. Load test cases and pre-calculate baselines
    en_tr_all, tr_en_all = load_test_cases()
    
    print("\n[Step 2] Pre-calculating unconstrained baselines...")
    for case in tqdm(en_tr_all, desc="  EN-TR Baselines"):
        case["baseline"], _ = decoding.unconstrained(en_tr_model, case["source"])
    
    for case in tqdm(tr_en_all, desc="  TR-EN Baselines"):
        case["baseline"], _ = decoding.unconstrained(tr_en_model, case["source"])

    # 3. Start Study
    print(f"\n[Step 3] Starting Optuna study ({N_TRIALS} trials on {len(en_tr_all) + len(tr_en_all)} samples)...")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda t: objective(t, en_tr_model, tr_en_model, en_tr_all, tr_en_all),
        n_trials=N_TRIALS
    )

    # 4. Results
    print("\n" + "="*70)
    print("  HPO COMPLETE")
    print("="*70)
    print(f"  Best score: {study.best_value:.4f}")
    print("  Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {k:<25}: {v}")
    
    # Save to file
    out_path = os.path.join(config.RESULTS_DIR, "best_params.json")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\n  Best parameters saved to → {out_path}")
    print("  Update config.py with these values for final evaluation.")

if __name__ == "__main__":
    main()
