"""
config.py — Central configuration for Lexically Constrained MT project.
All hyperparameters, model names, and paths live here.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
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
EN_TR_MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-tr"
TR_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-tr-en"

EN_TR_MODEL_PATH = "models/opus-mt-en-tr"
TR_EN_MODEL_PATH = "models/opus-mt-tr-en"

# ── Generation hyperparameters ────────────────────────────────────────────────
MAX_LENGTH      = 128
NUM_BEAMS       = 16
NO_REPEAT_NGRAM = 3

# ── Soft constraint strength ──────────────────────────────────────────────────
SOFT_REWARD_STRENGTH  =  4.0
SOFT_PENALTY_STRENGTH = -12.0
ANCHOR_OFFSET = -5.0            
CONTEXTUAL_NUDGE = 2.0 

# ── Curriculum reward escalation ─────────────────────────────────────────────
SOFT_REWARD_CURRICULUM_RATE = 0.25
SOFT_REWARD_MAX             = 12.0

# ── Hard inclusion: logit boost applied each step until word appears ──────────
HARD_INCLUSION_BOOST  = 15.0
SUFFIX_PENALTY_EN = -15.0
SUFFIX_PENALTY_TR =   0.0
READINESS_THRESHOLD = 200

# ── Hard Inclusion Dynamic Anchoring (HPO Search Space) ──────────────────────
# Tokens to skip before applying any pressure (prevents front-loading)
HARD_INCL_EARLY_TOKENS = 3         # Suggested HPO range: [1, 5]

# The rank threshold where the word is considered a "natural fit"
HARD_INCL_SWEET_RANK   = 100       # Suggested HPO range: [10, 500]

# The logit buffer granted when the word falls in the sweet spot
HARD_INCL_SWEET_BUFFER = 5.0       # Suggested HPO range: [1.0, 15.0]

# Starting offset from max_logit at 0% sentence completion
HARD_INCL_ANCHOR_START = -4.0      # Suggested HPO range: [-10.0, -1.0]

# Total logit climb from 0% to 100% completion
HARD_INCL_ANCHOR_RANGE = 5.0       # Suggested HPO range: [2.0, 10.0]

# ── Combined-hard reranking ───────────────────────────────────────────────────
COMBINED_HARD_RERANK_BEAMS = 16

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED = 42