"""
config.py — Central configuration
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    import warnings
    warnings.warn("CUDA is NOT available — running on CPU.", RuntimeWarning, stacklevel=1)
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

# ── HuggingFace DBA baseline parameters (Turkish Target) ──────────────────────
DBA_NUM_BEAMS_TR = 6
DBA_LENGTH_PENALTY_TR = 0.4939031949478693
DBA_REPETITION_PENALTY_TR = 1.0008664344404636

# ── HuggingFace DBA baseline parameters (English Target) ──────────────────────
DBA_NUM_BEAMS_EN = 13
DBA_LENGTH_PENALTY_EN = -2.6996447685168357
DBA_REPETITION_PENALTY_EN = 1.4650169030085232

# ── Soft reward only parameters ──────────────────────────────────────────────
SOFT_REWARD_STRENGTH_TR = 8.92938497828889
SOFT_REWARD_STRENGTH_EN = 4.197470333952294
SOFT_REWARD_MAX_TR = 20.575216644588068
SOFT_REWARD_MAX_EN = 7.664562418114533
SOFT_REWARD_CURRICULUM_RATE_TR = 0.9648681462242688
SOFT_REWARD_CURRICULUM_RATE_EN = 0.47147565271592456
ANCHOR_OFFSET_TR = -3.900007827831196
ANCHOR_OFFSET_EN = -11.980686734072117
CONTEXTUAL_NUDGE_TR = 2.0
CONTEXTUAL_NUDGE_EN = 2.0

# ── Soft penalty only parameters ──────────────────────────────────────────────
SOFT_PENALTY_STRENGTH_TR = -38.4043670426713
SOFT_PENALTY_STRENGTH_EN = -38.75794574460309

# ── Soft combined parameters ──────────────────────────────────────────────────
SOFT_COMBINED_REWARD_STRENGTH_TR = 5.657883476468935
SOFT_COMBINED_REWARD_STRENGTH_EN = 2.8620058796353556
SOFT_COMBINED_REWARD_MAX_TR = 24.55307958305177
SOFT_COMBINED_REWARD_MAX_EN = 12.810036452510525
SOFT_COMBINED_REWARD_CURRICULUM_RATE_TR = 0.899819010888688
SOFT_COMBINED_REWARD_CURRICULUM_RATE_EN = 0.32947659805771523
SOFT_COMBINED_ANCHOR_OFFSET_TR = -12.890141941229228
SOFT_COMBINED_ANCHOR_OFFSET_EN = -8.487808291905306
SOFT_COMBINED_PENALTY_STRENGTH_TR = -14.377393147094622
SOFT_COMBINED_PENALTY_STRENGTH_EN = -26.838487835531552

# ── Hard inclusion parameters ─────────────────────────────────────────────────
HARD_INCLUSION_BOOST_TR  = 9.402467737770602
HARD_INCLUSION_BOOST_EN  = 11.904520450360314
SUFFIX_PENALTY_TR = -11.909600025499575
SUFFIX_PENALTY_EN = -7.733009238117177
READINESS_THRESHOLD = 200

# ── Hard Inclusion Dynamic Anchoring (Turkish Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_TR = 3
HARD_INCL_SWEET_RANK_TR   = 100
HARD_INCL_SWEET_BUFFER_TR = 7.822984298768535
HARD_INCL_ANCHOR_START_TR = -6.929641484824387
HARD_INCL_ANCHOR_RANGE_TR = 15.656740187342306

# ── Hard Inclusion Dynamic Anchoring (English Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_EN = 5
HARD_INCL_SWEET_RANK_EN   = 307
HARD_INCL_SWEET_BUFFER_EN = 5.818685657503794
HARD_INCL_ANCHOR_START_EN = -16.065062964311274
HARD_INCL_ANCHOR_RANGE_EN = 26.067537650273763

# ── Hard combined parameters (Turkish Target) ─────────────────────────────────
HARD_COMBINED_BOOST_TR = 14.27887621243904
HARD_COMBINED_SUFFIX_PENALTY_TR = -12.506913387335155
HARD_COMBINED_EARLY_TOKENS_TR = 2
HARD_COMBINED_SWEET_RANK_TR = 145
HARD_COMBINED_SWEET_BUFFER_TR = 6.991524943855499
HARD_COMBINED_ANCHOR_START_TR = -5.9374962197243
HARD_COMBINED_ANCHOR_RANGE_TR = 20.11886794343148

# ── Hard combined parameters (English Target) ─────────────────────────────────
HARD_COMBINED_BOOST_EN = 11.922712626116601
HARD_COMBINED_SUFFIX_PENALTY_EN = -16.338948145301
HARD_COMBINED_EARLY_TOKENS_EN = 3
HARD_COMBINED_SWEET_RANK_EN = 92
HARD_COMBINED_SWEET_BUFFER_EN = 3.52005140337291
HARD_COMBINED_ANCHOR_START_EN = -20.52153450272938
HARD_COMBINED_ANCHOR_RANGE_EN = 19.582196532724176

# ── Combined-hard reranking ───────────────────────────────────────────────────
COMBINED_HARD_RERANK_BEAMS = 16

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"

# ── Seeds ─────────────────────────────────────────────────────────────────────
# Single seed used by HPO and for reproducibility of individual runs.
SEED = 42

# Seeds used by the evaluation pipeline to average results over multiple runs.
# Set to a list of one seed to reproduce single-seed behaviour.
# Using multiple seeds reduces variance from beam-search stochasticity and
# any GPU-level non-determinism.
EVAL_SEEDS = [42, 123, 7]