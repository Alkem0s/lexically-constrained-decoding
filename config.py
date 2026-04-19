"""
config.py — Central configuration for Lexically Constrained MT project.
All hyperparameters, model names, and paths live here.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
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
# Fix #12: original ±8.0 was too weak for penalty (0% exclusion satisfaction)
# and too strong / unguarded for reward (catastrophic repetition on some cases).
# Penalty raised to -15.0 so it meaningfully suppresses high-confidence tokens.
# Reward kept at 5.0; the "stop once satisfied" guard in SoftConstraintProcessor
# prevents runaway repetition, so a moderate value is safe.
SOFT_REWARD_STRENGTH  =  5.0
SOFT_PENALTY_STRENGTH = -15.0

# ── Hard inclusion: logit boost applied each step until word appears ──────────
HARD_INCLUSION_BOOST  = 20.0

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42