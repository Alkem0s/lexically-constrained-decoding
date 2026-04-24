"""
config.py — Central configuration for Lexically Constrained MT project.
All hyperparameters, model names, and paths live here.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    # Fix #1: Emit a clear warning so the user knows they are on CPU.
    # If you have an RTX 4070, your PyTorch install is likely the CPU-only
    # wheel.  Reinstall with CUDA 12.x support:
    #   pip install torch --index-url https://download.pytorch.org/whl/cu121
    import warnings
    warnings.warn(
        "CUDA is NOT available — running on CPU.  If you have an NVIDIA GPU, "
        "reinstall PyTorch with CUDA support:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121",
        RuntimeWarning,
        stacklevel=1,
    )
    DEVICE = "cpu"

# ── Models ────────────────────────────────────────────────────────────────────
# Helsinki-NLP MarianMT models for EN↔TR
EN_TR_MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-tr"
TR_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-tr-en"

EN_TR_MODEL_PATH = "models/opus-mt-en-tr"   # local cache dir
TR_EN_MODEL_PATH = "models/opus-mt-tr-en"

# ── Generation hyperparameters ────────────────────────────────────────────────
MAX_LENGTH      = 128
NUM_BEAMS       = 4      # beam width for constrained beam search
NO_REPEAT_NGRAM = 3      # standard repetition penalty

# ── Soft constraint strength ──────────────────────────────────────────────────
# With full-vocabulary token coverage (all segmentations caught), a penalty of
# -12 reliably suppresses forbidden words.  Reward starts at 5.0 and escalates
# via a curriculum schedule (see SOFT_REWARD_CURRICULUM_RATE) so that if the
# model ignores the nudge in early steps, the boost grows until it matches
# HARD_INCLUSION_BOOST by ~step 7.
SOFT_REWARD_STRENGTH  =  5.0
SOFT_PENALTY_STRENGTH = -12.0

# ── Curriculum reward escalation ─────────────────────────────────────────────
# Effective reward at step n (while word is still pending):
#   effective = min(SOFT_REWARD_MAX, SOFT_REWARD_STRENGTH * (1 + RATE * n))
# With RATE=0.3 and STRENGTH=5.0, the cap of 15.0 is reached at step 7,
# matching HARD_INCLUSION_BOOST so the soft reward gracefully becomes hard-like
# for words the model persistently ignores.
SOFT_REWARD_CURRICULUM_RATE = 0.3
SOFT_REWARD_MAX             = 15.0   # matches HARD_INCLUSION_BOOST

# ── Hard inclusion: logit boost applied each step until word appears ──────────
# Lowered from 20 → 15 because we now only boost whole-word tokens (high base
# rank), so a smaller nudge is sufficient and causes less fluency disruption.
HARD_INCLUSION_BOOST  = 15.0

# ── Combined-hard reranking ───────────────────────────────────────────────────
# Number of beam candidates to generate (exclusion-only pass) for the reranking
# phase of combined_hard().  A larger pool gives more chances to find a
# candidate that naturally satisfies inclusion; 8 = 2× the default beam width.
COMBINED_HARD_RERANK_BEAMS = 8

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42