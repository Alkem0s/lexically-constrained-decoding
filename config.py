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
DBA_LENGTH_PENALTY_TR = -2.3071076962383796
DBA_REPETITION_PENALTY_TR = 1.083265332727951

# ── HuggingFace DBA baseline parameters (English Target) ──────────────────────
DBA_NUM_BEAMS_EN = 5
DBA_LENGTH_PENALTY_EN = -3.5809536246293034
DBA_REPETITION_PENALTY_EN = 1.4383135728669465

# ── Soft reward only parameters ──────────────────────────────────────────────
SOFT_REWARD_STRENGTH_TR = 10.147480743046776
SOFT_REWARD_STRENGTH_EN = 3.797036270238195
SOFT_REWARD_MAX_TR = 23.407415697341516
SOFT_REWARD_MAX_EN = 11.91683897706493
SOFT_REWARD_CURRICULUM_RATE_TR = 3.481693853129853
SOFT_REWARD_CURRICULUM_RATE_EN = 0.10284883168053965
ANCHOR_OFFSET_TR = -39.5558719340102
ANCHOR_OFFSET_EN = -12.367943969374618
CONTEXTUAL_NUDGE_TR = 2.0
CONTEXTUAL_NUDGE_EN = 2.0

# ── Soft penalty only parameters ──────────────────────────────────────────────
SOFT_PENALTY_STRENGTH_TR = -94.78325214372592
SOFT_PENALTY_STRENGTH_EN = -125.7447294351319

# ── Soft combined parameters ──────────────────────────────────────────────────
SOFT_COMBINED_REWARD_STRENGTH_TR = 7.552779048157623
SOFT_COMBINED_REWARD_STRENGTH_EN = 3.127689036221139
SOFT_COMBINED_REWARD_MAX_TR = 29.843347036673496
SOFT_COMBINED_REWARD_MAX_EN = 56.43591773796329
SOFT_COMBINED_REWARD_CURRICULUM_RATE_TR = 1.0076144513689183
SOFT_COMBINED_REWARD_CURRICULUM_RATE_EN = 2.6712889172821708
SOFT_COMBINED_ANCHOR_OFFSET_TR = -21.934359045424667
SOFT_COMBINED_ANCHOR_OFFSET_EN = -15.66925214511932
SOFT_COMBINED_PENALTY_STRENGTH_TR = -97.89749235309901
SOFT_COMBINED_PENALTY_STRENGTH_EN = -61.96899493600391

# ── Hard inclusion parameters ─────────────────────────────────────────────────
HARD_INCLUSION_BOOST_TR  = 13.13321571428587
HARD_INCLUSION_BOOST_EN  = 26.660430527906378
SUFFIX_PENALTY_TR = -2.451639332300523
SUFFIX_PENALTY_EN = -16.13668042108037
READINESS_THRESHOLD = 200

# ── Hard Inclusion Dynamic Anchoring (Turkish Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_TR = 1
HARD_INCL_SWEET_RANK_TR   = 402
HARD_INCL_SWEET_BUFFER_TR = 9.493643272109697
HARD_INCL_ANCHOR_START_TR = 1.8079856516903092
HARD_INCL_ANCHOR_RANGE_TR = 5.0635903829627775

# ── Hard Inclusion Dynamic Anchoring (English Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_EN = 1
HARD_INCL_SWEET_RANK_EN   = 566
HARD_INCL_SWEET_BUFFER_EN = 5.361011125170711
HARD_INCL_ANCHOR_START_EN = -33.08636780963822
HARD_INCL_ANCHOR_RANGE_EN = 37.00284363248174

# ── Hard combined parameters (Turkish Target) ─────────────────────────────────
HARD_COMBINED_BOOST_TR = 14.798808025785567
HARD_COMBINED_SUFFIX_PENALTY_TR = -22.539127082881294
HARD_COMBINED_EARLY_TOKENS_TR = 0
HARD_COMBINED_SWEET_RANK_TR = 310
HARD_COMBINED_SWEET_BUFFER_TR = 8.754530392505961
HARD_COMBINED_ANCHOR_START_TR = -17.766858667619427
HARD_COMBINED_ANCHOR_RANGE_TR = 45.90004463962985

# ── Hard combined parameters (English Target) ─────────────────────────────────
HARD_COMBINED_BOOST_EN = 30.4889877572603
HARD_COMBINED_SUFFIX_PENALTY_EN = -11.676638262043435
HARD_COMBINED_EARLY_TOKENS_EN = 4
HARD_COMBINED_SWEET_RANK_EN = 316
HARD_COMBINED_SWEET_BUFFER_EN = 1.9204447521049763
HARD_COMBINED_ANCHOR_START_EN = -16.88763074830799
HARD_COMBINED_ANCHOR_RANGE_EN = 19.637724608990904

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