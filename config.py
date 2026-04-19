"""
config.py — Central configuration for Lexically Constrained MT project.
All hyperparameters, model names, and paths live here.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# Positive = reward (inclusion), Negative = penalty (exclusion)
SOFT_REWARD_STRENGTH  =  8.0
SOFT_PENALTY_STRENGTH = -8.0

# ── Hard inclusion: logit boost applied each step until word appears ──────────
HARD_INCLUSION_BOOST  = 20.0

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42
