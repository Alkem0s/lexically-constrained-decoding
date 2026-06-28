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
DBA_NUM_BEAMS_TR = 4
DBA_LENGTH_PENALTY_TR = -3.258403729811374
DBA_REPETITION_PENALTY_TR = 1.3154924246870694

# ── HuggingFace DBA baseline parameters (English Target) ──────────────────────
DBA_NUM_BEAMS_EN = 5
DBA_LENGTH_PENALTY_EN = -3.320994476336735
DBA_REPETITION_PENALTY_EN = 1.2937930689690016

# ── Soft reward only parameters ──────────────────────────────────────────────
SOFT_REWARD_STRENGTH_TR = 12.598497142008368
SOFT_REWARD_STRENGTH_EN = 3.608473391291833
SOFT_REWARD_MAX_TR = 10.893778789456125
SOFT_REWARD_MAX_EN = 32.72704741584594
SOFT_REWARD_CURRICULUM_RATE_TR = 0.5373403752586311
SOFT_REWARD_CURRICULUM_RATE_EN = 0.4521715217869347
ANCHOR_OFFSET_TR = -16.074940017106385
ANCHOR_OFFSET_EN = -39.514862073903245
CONTEXTUAL_NUDGE_TR = 2.0
CONTEXTUAL_NUDGE_EN = 2.0

# ── Soft penalty only parameters ──────────────────────────────────────────────
SOFT_PENALTY_STRENGTH_TR = -125.32424921619241
SOFT_PENALTY_STRENGTH_EN = -71.30466344585409

# ── Soft combined parameters ──────────────────────────────────────────────────
SOFT_COMBINED_REWARD_STRENGTH_TR = 15.899063183294366
SOFT_COMBINED_REWARD_STRENGTH_EN = 3.855283720068468
SOFT_COMBINED_REWARD_MAX_TR = 9.910690142962896
SOFT_COMBINED_REWARD_MAX_EN = 51.73649665337388
SOFT_COMBINED_REWARD_CURRICULUM_RATE_TR = 3.8860590706475597
SOFT_COMBINED_REWARD_CURRICULUM_RATE_EN = 2.5424594465328667
SOFT_COMBINED_ANCHOR_OFFSET_TR = -47.9807160002656
SOFT_COMBINED_ANCHOR_OFFSET_EN = -18.81384985207404
SOFT_COMBINED_PENALTY_STRENGTH_TR = -180.69946948285713
SOFT_COMBINED_PENALTY_STRENGTH_EN = -40.02639180894677

# ── Hard inclusion parameters ─────────────────────────────────────────────────
HARD_INCLUSION_BOOST_TR  = 31.10109191554598
HARD_INCLUSION_BOOST_EN  = 36.91603048341559
SUFFIX_PENALTY_TR = -1.3709425542252711
SUFFIX_PENALTY_EN = -27.324973841405434
READINESS_THRESHOLD = 200

# ── Hard Inclusion Dynamic Anchoring (Turkish Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_TR = 2
HARD_INCL_SWEET_RANK_TR   = 853
HARD_INCL_SWEET_BUFFER_TR = 10.497253262979196
HARD_INCL_ANCHOR_START_TR = 1.3458785255293089
HARD_INCL_ANCHOR_RANGE_TR = 29.344673386291273

# ── Hard Inclusion Dynamic Anchoring (English Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_EN = 5
HARD_INCL_SWEET_RANK_EN   = 477
HARD_INCL_SWEET_BUFFER_EN = 4.280437466299246
HARD_INCL_ANCHOR_START_EN = -26.165961519907086
HARD_INCL_ANCHOR_RANGE_EN = 31.737092740783922

# ── Hard combined parameters (Turkish Target) ─────────────────────────────────
HARD_COMBINED_BOOST_TR = 25.820221910536155
HARD_COMBINED_SUFFIX_PENALTY_TR = -24.499761823267644
HARD_COMBINED_EARLY_TOKENS_TR = 0
HARD_COMBINED_SWEET_RANK_TR = 678
HARD_COMBINED_SWEET_BUFFER_TR = 9.291667500492247
HARD_COMBINED_ANCHOR_START_TR = -23.57463662184104
HARD_COMBINED_ANCHOR_RANGE_TR = 14.025782082102342

# ── Hard combined parameters (English Target) ─────────────────────────────────
HARD_COMBINED_BOOST_EN = 24.30772827893468
HARD_COMBINED_SUFFIX_PENALTY_EN = -18.355690475080145
HARD_COMBINED_EARLY_TOKENS_EN = 4
HARD_COMBINED_SWEET_RANK_EN = 683
HARD_COMBINED_SWEET_BUFFER_EN = 13.791370621658581
HARD_COMBINED_ANCHOR_START_EN = -19.99976453149047
HARD_COMBINED_ANCHOR_RANGE_EN = 30.561823184268093

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