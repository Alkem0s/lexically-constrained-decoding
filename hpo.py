import optuna
import json
import os
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import config
import model_loader
import decoding
import evaluation
from main import load_test_cases, set_seeds


def objective_generic(trial, mode, direction, en_tr_model, tr_en_model, en_tr_cases, tr_en_cases):
    # 1. Suggest parameters based on mode and direction
    if mode == "hard_inclusion":
        if direction in ["en-tr", "both"]:
            config.HARD_INCLUSION_BOOST_TR  = trial.suggest_float("HARD_INCLUSION_BOOST_TR", 2.0, 25.0)
            config.SUFFIX_PENALTY_TR        = trial.suggest_float("SUFFIX_PENALTY_TR", -25.0, -1.0)
            config.HARD_INCL_EARLY_TOKENS_TR = trial.suggest_int("HARD_INCL_EARLY_TOKENS_TR", 0, 10)
            config.HARD_INCL_SWEET_RANK_TR   = trial.suggest_int("HARD_INCL_SWEET_RANK_TR", 50, 600)
            config.HARD_INCL_SWEET_BUFFER_TR = trial.suggest_float("HARD_INCL_SWEET_BUFFER_TR", 1.0, 15.0)
            config.HARD_INCL_ANCHOR_START_TR = trial.suggest_float("HARD_INCL_ANCHOR_START_TR", -40.0, 5.0)
            config.HARD_INCL_ANCHOR_RANGE_TR = trial.suggest_float("HARD_INCL_ANCHOR_RANGE_TR", 5.0, 35.0)
            
        if direction in ["tr-en", "both"]:
            config.HARD_INCLUSION_BOOST_EN  = trial.suggest_float("HARD_INCLUSION_BOOST_EN", 2.0, 25.0)
            config.SUFFIX_PENALTY_EN        = trial.suggest_float("SUFFIX_PENALTY_EN", -25.0, -1.0)
            config.HARD_INCL_EARLY_TOKENS_EN = trial.suggest_int("HARD_INCL_EARLY_TOKENS_EN", 0, 10)
            config.HARD_INCL_SWEET_RANK_EN   = trial.suggest_int("HARD_INCL_SWEET_RANK_EN", 50, 600)
            config.HARD_INCL_SWEET_BUFFER_EN = trial.suggest_float("HARD_INCL_SWEET_BUFFER_EN", 1.0, 15.0)
            config.HARD_INCL_ANCHOR_START_EN = trial.suggest_float("HARD_INCL_ANCHOR_START_EN", -40.0, 5.0)
            config.HARD_INCL_ANCHOR_RANGE_EN = trial.suggest_float("HARD_INCL_ANCHOR_RANGE_EN", 5.0, 35.0)

    elif mode == "hard_combined":
        if direction in ["en-tr", "both"]:
            config.HARD_COMBINED_BOOST_TR  = trial.suggest_float("HARD_COMBINED_BOOST_TR", 2.0, 25.0)
            config.HARD_COMBINED_SUFFIX_PENALTY_TR = trial.suggest_float("HARD_COMBINED_SUFFIX_PENALTY_TR", -25.0, -1.0)
            config.HARD_COMBINED_EARLY_TOKENS_TR = trial.suggest_int("HARD_COMBINED_EARLY_TOKENS_TR", 0, 10)
            config.HARD_COMBINED_SWEET_RANK_TR   = trial.suggest_int("HARD_COMBINED_SWEET_RANK_TR", 50, 600)
            config.HARD_COMBINED_SWEET_BUFFER_TR = trial.suggest_float("HARD_COMBINED_SWEET_BUFFER_TR", 1.0, 15.0)
            config.HARD_COMBINED_ANCHOR_START_TR = trial.suggest_float("HARD_COMBINED_ANCHOR_START_TR", -40.0, 5.0)
            config.HARD_COMBINED_ANCHOR_RANGE_TR = trial.suggest_float("HARD_COMBINED_ANCHOR_RANGE_TR", 5.0, 35.0)
            
        if direction in ["tr-en", "both"]:
            config.HARD_COMBINED_BOOST_EN  = trial.suggest_float("HARD_COMBINED_BOOST_EN", 2.0, 25.0)
            config.HARD_COMBINED_SUFFIX_PENALTY_EN = trial.suggest_float("HARD_COMBINED_SUFFIX_PENALTY_EN", -25.0, -1.0)
            config.HARD_COMBINED_EARLY_TOKENS_EN = trial.suggest_int("HARD_COMBINED_EARLY_TOKENS_EN", 0, 10)
            config.HARD_COMBINED_SWEET_RANK_EN   = trial.suggest_int("HARD_COMBINED_SWEET_RANK_EN", 50, 600)
            config.HARD_COMBINED_SWEET_BUFFER_EN = trial.suggest_float("HARD_COMBINED_SWEET_BUFFER_EN", 1.0, 15.0)
            config.HARD_COMBINED_ANCHOR_START_EN = trial.suggest_float("HARD_COMBINED_ANCHOR_START_EN", -40.0, 5.0)
            config.HARD_COMBINED_ANCHOR_RANGE_EN = trial.suggest_float("HARD_COMBINED_ANCHOR_RANGE_EN", 5.0, 35.0)
            
    elif mode == "soft_reward":
        if direction in ["en-tr", "both"]:
            config.SOFT_REWARD_STRENGTH_TR  = trial.suggest_float("SOFT_REWARD_STRENGTH_TR", 2.0, 12.0)
            config.SOFT_REWARD_MAX_TR       = trial.suggest_float("SOFT_REWARD_MAX_TR", 5.0, 25.0)
            config.SOFT_REWARD_CURRICULUM_RATE_TR = trial.suggest_float("SOFT_REWARD_CURRICULUM_RATE_TR", 0.05, 2.0)
            config.ANCHOR_OFFSET_TR         = trial.suggest_float("ANCHOR_OFFSET_TR", -20.0, 0.0)
            
        if direction in ["tr-en", "both"]:
            config.SOFT_REWARD_STRENGTH_EN  = trial.suggest_float("SOFT_REWARD_STRENGTH_EN", 2.0, 12.0)
            config.SOFT_REWARD_MAX_EN       = trial.suggest_float("SOFT_REWARD_MAX_EN", 5.0, 25.0)
            config.SOFT_REWARD_CURRICULUM_RATE_EN = trial.suggest_float("SOFT_REWARD_CURRICULUM_RATE_EN", 0.05, 2.0)
            config.ANCHOR_OFFSET_EN         = trial.suggest_float("ANCHOR_OFFSET_EN", -20.0, 0.0)
            
    elif mode == "soft_penalty":
        if direction in ["en-tr", "both"]:
            config.SOFT_PENALTY_STRENGTH_TR = trial.suggest_float("SOFT_PENALTY_STRENGTH_TR", -100.0, -10.0)
        if direction in ["tr-en", "both"]:
            config.SOFT_PENALTY_STRENGTH_EN = trial.suggest_float("SOFT_PENALTY_STRENGTH_EN", -100.0, -10.0)
            
    elif mode == "soft_combined":
        if direction in ["en-tr", "both"]:
            config.SOFT_COMBINED_REWARD_STRENGTH_TR = trial.suggest_float("SOFT_COMBINED_REWARD_STRENGTH_TR", 2.0, 12.0)
            config.SOFT_COMBINED_REWARD_MAX_TR      = trial.suggest_float("SOFT_COMBINED_REWARD_MAX_TR", 5.0, 40.0)
            config.SOFT_COMBINED_REWARD_CURRICULUM_RATE_TR = trial.suggest_float("SOFT_COMBINED_REWARD_CURRICULUM_RATE_TR", 0.05, 2.0)
            config.SOFT_COMBINED_ANCHOR_OFFSET_TR   = trial.suggest_float("SOFT_COMBINED_ANCHOR_OFFSET_TR", -20.0, 0.0)
            config.SOFT_COMBINED_PENALTY_STRENGTH_TR = trial.suggest_float("SOFT_COMBINED_PENALTY_STRENGTH_TR", -120.0, 0.0)
            
        if direction in ["tr-en", "both"]:
            config.SOFT_COMBINED_REWARD_STRENGTH_EN = trial.suggest_float("SOFT_COMBINED_REWARD_STRENGTH_EN", 2.0, 12.0)
            config.SOFT_COMBINED_REWARD_MAX_EN      = trial.suggest_float("SOFT_COMBINED_REWARD_MAX_EN", 5.0, 40.0)
            config.SOFT_COMBINED_REWARD_CURRICULUM_RATE_EN = trial.suggest_float("SOFT_COMBINED_REWARD_CURRICULUM_RATE_EN", 0.05, 2.0)
            config.SOFT_COMBINED_ANCHOR_OFFSET_EN   = trial.suggest_float("SOFT_COMBINED_ANCHOR_OFFSET_EN", -20.0, 0.0)
            config.SOFT_COMBINED_PENALTY_STRENGTH_EN = trial.suggest_float("SOFT_COMBINED_PENALTY_STRENGTH_EN", -120.0, 0.0)

    # 2. Run evaluation
    all_results = []
    
    eval_tasks = []
    if direction in ["en-tr", "both"] and en_tr_model is not None:
        eval_tasks.append((en_tr_cases, en_tr_model))
    if direction in ["tr-en", "both"] and tr_en_model is not None:
        eval_tasks.append((tr_en_cases, tr_en_model))
        
    for cases, model in eval_tasks:
        for case in cases:
            if mode == "hard_inclusion":
                trans, _ = decoding.hard_inclusion(model, case["source"], case["required_words"])
            elif mode == "hard_combined":
                trans, _, _ = decoding.combined_hard(model, case["source"], case.get("forbidden_words", []), case["required_words"])
            elif mode == "soft_reward":
                trans, _ = decoding.soft_reward_only(model, case["source"], case.get("reward_words", []))
            elif mode == "soft_penalty":
                trans, _ = decoding.soft_penalty_only(model, case["source"], case.get("penalty_words", []))
            elif mode == "soft_combined":
                trans, _ = decoding.combined_soft(
                    model, case["source"],
                    reward_words=case.get("reward_words", []),
                    penalty_words=case.get("penalty_words", [])
                )
                
            res = evaluation.SampleResult(
                source=case["source"],
                direction=case["direction"],
                required_words=case.get("required_words", []),
                forbidden_words=case.get("forbidden_words", []),
                penalty_words=case.get("penalty_words", []),
                reward_words=case.get("reward_words", []),
                reference=case.get("reference", ""),
                unconstrained=case["baseline"]
            )
            setattr(res, mode, trans)
            evaluation.evaluate_sample(res)
            all_results.append(res)

    # 3. Aggregate metrics
    agg = evaluation.aggregate_results(all_results)
    
    if mode not in agg:
        return 0.0
    
    sat_rate = agg[mode]["avg_satisfaction"]  # 0.0 to 1.0
    avg_bleu = agg[mode]["avg_bleu_vs_base"] or 0.0 # 0 to 100
    avg_len_ratio = agg[mode]["avg_length_ratio"] or 1.0
    
    # Store attributes for visibility
    trial.set_user_attr("satisfaction", sat_rate)
    trial.set_user_attr("bleu", avg_bleu)
    trial.set_user_attr("length_ratio", avg_len_ratio)
    
    # Continuous length ratio penalty (encourage terminating output near 1.0)
    length_penalty = max(0.0, abs(avg_len_ratio - 1.0) - 0.05) * 1000.0
    
    # Composite score
    score = (sat_rate * 1000) + avg_bleu - length_penalty
    
    # Print progress with breakdown
    print(f"  Trial {trial.number:2d} | Sat: {sat_rate*100:5.1f}% | BLEU: {avg_bleu:5.2f} | LenRatio: {avg_len_ratio:5.3f} | Score: {score:7.3f}")
    
    return score


def objective_dba(trial, direction, en_tr_model, tr_en_model, en_tr_cases, tr_en_cases):
    # 1. Suggest parameters for DBA
    if direction in ["en-tr", "both"]:
        config.DBA_NUM_BEAMS_TR = trial.suggest_int("DBA_NUM_BEAMS_TR", 4, 16)
        config.DBA_LENGTH_PENALTY_TR = trial.suggest_float("DBA_LENGTH_PENALTY_TR", -5.0, 2.0)
        config.DBA_REPETITION_PENALTY_TR = trial.suggest_float("DBA_REPETITION_PENALTY_TR", 1.0, 1.5)
        
    if direction in ["tr-en", "both"]:
        config.DBA_NUM_BEAMS_EN = trial.suggest_int("DBA_NUM_BEAMS_EN", 4, 16)
        config.DBA_LENGTH_PENALTY_EN = trial.suggest_float("DBA_LENGTH_PENALTY_EN", -5.0, 2.0)
        config.DBA_REPETITION_PENALTY_EN = trial.suggest_float("DBA_REPETITION_PENALTY_EN", 1.0, 1.5)

    # 2. Run evaluation
    all_results = []
    
    eval_tasks = []
    if direction in ["en-tr", "both"] and en_tr_model is not None:
        eval_tasks.append((en_tr_cases, en_tr_model))
    if direction in ["tr-en", "both"] and tr_en_model is not None:
        eval_tasks.append((tr_en_cases, tr_en_model))
        
    for cases, model in eval_tasks:
        for case in cases:
            trans, log = decoding.huggingface_dba(
                model, case["source"], 
                required_words=case["required_words"],
                forbidden_words=case.get("forbidden_words", [])
            )
            
            res = evaluation.SampleResult(
                source=case["source"],
                direction=case["direction"],
                required_words=case["required_words"],
                forbidden_words=case.get("forbidden_words", []),
                unconstrained=case["baseline"]
            )
            res.huggingface_dba = trans
            evaluation.evaluate_sample(res)
            all_results.append(res)

    # 3. Aggregate metrics
    agg = evaluation.aggregate_results(all_results)
    mode = "huggingface_dba"
    
    if mode not in agg:
        return 0.0
    
    sat_rate = agg[mode]["avg_satisfaction"]  # 0.0 to 1.0
    avg_bleu = agg[mode]["avg_bleu_vs_base"] or 0.0 # 0 to 100
    avg_len_ratio = agg[mode]["avg_length_ratio"] or 1.0
    
    # Store attributes for visibility
    trial.set_user_attr("satisfaction", sat_rate)
    trial.set_user_attr("bleu", avg_bleu)
    trial.set_user_attr("length_ratio", avg_len_ratio)
    
    # Continuous length ratio penalty (encourage terminating output near 1.0)
    length_penalty = max(0.0, abs(avg_len_ratio - 1.0) - 0.05) * 1000.0
        
    score = (sat_rate * 1000) + avg_bleu - length_penalty
    
    # Print progress with breakdown
    print(f"  Trial {trial.number:2d} | Sat: {sat_rate*100:5.1f}% | BLEU: {avg_bleu:5.2f} | LenRatio: {avg_len_ratio:5.3f} | Score: {score:7.3f}")
    
    return score


def run_study(mode, direction, en_tr_model, tr_en_model, en_tr_all, tr_en_all, n_trials):
    print("\n" + "="*70)
    print(f"  Starting HPO Study for: {mode.upper()} [{direction.upper()}] ({n_trials} trials)")
    print("="*70)
    
    study = optuna.create_study(direction="maximize")
    
    if mode == "dba":
        obj_func = lambda t: objective_dba(t, direction, en_tr_model, tr_en_model, en_tr_all, tr_en_all)
    elif mode in ["hard_inclusion", "hard_combined", "soft_reward", "soft_penalty", "soft_combined"]:
        obj_func = lambda t: objective_generic(t, mode, direction, en_tr_model, tr_en_model, en_tr_all, tr_en_all)
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    study.optimize(obj_func, n_trials=n_trials)
    
    print(f"\n  Study for {mode.upper()} [{direction.upper()}] complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print("  Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {k:<25}: {v}")
        
    # Save to file
    suffix = f"_{direction}" if direction != "both" else ""
    out_path = os.path.join(config.RESULTS_DIR, f"best_params_{mode}{suffix}.json")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"  Saved parameters to → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization Sweep")
    parser.add_argument("--mode", type=str, choices=["hard_inclusion", "hard_combined", "soft_reward", "soft_penalty", "soft_combined", "dba", "all"], default="hard_inclusion",
                        help="Which mode to optimize (hard_inclusion, hard_combined, soft_reward, soft_penalty, soft_combined, dba, or all)")
    parser.add_argument("--direction", type=str, choices=["en-tr", "tr-en", "both", "all"], default="both",
                        help="Which direction to optimize (en-tr, tr-en, both, or all)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of HPO trials per sweep")
    args = parser.parse_args()

    print("="*70)
    print("  Lexically Constrained MT — Hyperparameter Optimization")
    print("="*70)
    
    set_seeds(config.SEED)
    
    # Resolve directions to run
    if args.direction == "all":
        directions_to_run = ["en-tr", "tr-en"]
    else:
        directions_to_run = [args.direction]
        
    for direction in directions_to_run:
        print(f"\n{'#'*80}")
        print(f"  Running HPO optimization for target direction: {direction.upper()}")
        print(f"{'#'*80}")
        
        # 1. Load models based on selected direction
        print("\n[Step 1] Loading models...")
        en_tr_model = None
        tr_en_model = None
        if direction in ["en-tr", "both"]:
            en_tr_model = model_loader.load_en_tr()
        if direction in ["tr-en", "both"]:
            tr_en_model = model_loader.load_tr_en()
        
        # 2. Load test cases and pre-calculate baselines
        en_tr_all, tr_en_all = load_test_cases("test_cases_eval.json")
        
        # Filter test cases by selected direction to speed up baselines pre-calculation
        if direction == "en-tr":
            tr_en_all = []
        elif direction == "tr-en":
            en_tr_all = []
    
        print("\n[Step 2] Pre-calculating unconstrained baselines...")
        if en_tr_model and en_tr_all:
            for case in tqdm(en_tr_all, desc="  EN-TR Baselines"):
                case["baseline"], _ = decoding.unconstrained(en_tr_model, case["source"])
        
        if tr_en_model and tr_en_all:
            for case in tqdm(tr_en_all, desc="  TR-EN Baselines"):
                case["baseline"], _ = decoding.unconstrained(tr_en_model, case["source"])
    
        # 3. Start optimization
        if args.mode == "all":
            modes_to_run = ["hard_inclusion", "hard_combined", "soft_reward", "soft_penalty", "soft_combined", "dba"]
            for m in modes_to_run:
                run_study(m, direction, en_tr_model, tr_en_model, en_tr_all, tr_en_all, args.trials)
        else:
            run_study(args.mode, direction, en_tr_model, tr_en_model, en_tr_all, tr_en_all, args.trials)
            
        # Clean up models and clear GPU cache
        del en_tr_model
        del tr_en_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
