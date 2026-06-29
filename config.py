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
DBA_NUM_BEAMS_TR = 7
DBA_LENGTH_PENALTY_TR = -1.170514862290621
DBA_REPETITION_PENALTY_TR = 1.0805311101450759

# ── HuggingFace DBA baseline parameters (English Target) ──────────────────────
DBA_NUM_BEAMS_EN = 5
DBA_LENGTH_PENALTY_EN = -2.0262898465237904
DBA_REPETITION_PENALTY_EN = 1.340074435458198

# ── Soft reward only parameters ──────────────────────────────────────────────
SOFT_REWARD_STRENGTH_TR = 7.034694149130213
SOFT_REWARD_STRENGTH_EN = 9.036324571487722
SOFT_REWARD_MAX_TR = 23.870430014236067
SOFT_REWARD_MAX_EN = 20.30791307146735
SOFT_REWARD_CURRICULUM_RATE_TR = 4.977654793811748
SOFT_REWARD_CURRICULUM_RATE_EN = 0.8310823069670247
ANCHOR_OFFSET_TR = -28.604573397084756
ANCHOR_OFFSET_EN = -32.55402610620901
CONTEXTUAL_NUDGE_TR = 2.0
CONTEXTUAL_NUDGE_EN = 2.0

# ── Soft penalty only parameters ──────────────────────────────────────────────
SOFT_PENALTY_STRENGTH_TR = -118.06345349434449
SOFT_PENALTY_STRENGTH_EN = -77.16972072479382

# ── Soft combined parameters ──────────────────────────────────────────────────
SOFT_COMBINED_REWARD_STRENGTH_TR = 8.374259438207808
SOFT_COMBINED_REWARD_STRENGTH_EN = 3.7772171623082307
SOFT_COMBINED_REWARD_MAX_TR = 30.721425137350636
SOFT_COMBINED_REWARD_MAX_EN = 39.65203421580849
SOFT_COMBINED_REWARD_CURRICULUM_RATE_TR = 4.1042840681080985
SOFT_COMBINED_REWARD_CURRICULUM_RATE_EN = 2.649896167847922
SOFT_COMBINED_ANCHOR_OFFSET_TR = -27.68711911809957
SOFT_COMBINED_ANCHOR_OFFSET_EN = -25.34179504389559
SOFT_COMBINED_PENALTY_STRENGTH_TR = -61.91802804055028
SOFT_COMBINED_PENALTY_STRENGTH_EN = -40.68113564575316

# ── Hard inclusion parameters ─────────────────────────────────────────────────
HARD_INCLUSION_BOOST_TR  = 24.77116512030992
HARD_INCLUSION_BOOST_EN  = 7.470006259556959
SUFFIX_PENALTY_TR = -19.945350134067315
SUFFIX_PENALTY_EN = -17.19091984623191
READINESS_THRESHOLD = 200

# ── Hard Inclusion Dynamic Anchoring (Turkish Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_TR = 3
HARD_INCL_SWEET_RANK_TR   = 300
HARD_INCL_SWEET_BUFFER_TR = 13.332386000847434
HARD_INCL_ANCHOR_START_TR = -17.003751982111723
HARD_INCL_ANCHOR_RANGE_TR = 36.503547152050736

# ── Hard Inclusion Dynamic Anchoring (English Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_EN = 0
HARD_INCL_SWEET_RANK_EN   = 66
HARD_INCL_SWEET_BUFFER_EN = 14.85497715043644
HARD_INCL_ANCHOR_START_EN = -28.247558337638488
HARD_INCL_ANCHOR_RANGE_EN = 30.685751307196327

# ── Hard combined parameters (Turkish Target) ─────────────────────────────────
HARD_COMBINED_BOOST_TR = 23.63279125322714
HARD_COMBINED_SUFFIX_PENALTY_TR = -8.828726743701234
HARD_COMBINED_EARLY_TOKENS_TR = 0
HARD_COMBINED_SWEET_RANK_TR = 405
HARD_COMBINED_SWEET_BUFFER_TR = 11.372726056245263
HARD_COMBINED_ANCHOR_START_TR = -1.8821929951768368
HARD_COMBINED_ANCHOR_RANGE_TR = 20.566704246393414

# ── Hard combined parameters (English Target) ─────────────────────────────────
HARD_COMBINED_BOOST_EN = 22.263749672365247
HARD_COMBINED_SUFFIX_PENALTY_EN = -14.944588001249056
HARD_COMBINED_EARLY_TOKENS_EN = 2
HARD_COMBINED_SWEET_RANK_EN = 156
HARD_COMBINED_SWEET_BUFFER_EN = 4.078158140223465
HARD_COMBINED_ANCHOR_START_EN = -14.803142607118655
HARD_COMBINED_ANCHOR_RANGE_EN = 16.702214183163846

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