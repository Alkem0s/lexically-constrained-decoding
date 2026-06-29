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
DBA_LENGTH_PENALTY_TR = 0.006326283364614049
DBA_REPETITION_PENALTY_TR = 1.0304988620379727

# ── HuggingFace DBA baseline parameters (English Target) ──────────────────────
DBA_NUM_BEAMS_EN = 5
DBA_LENGTH_PENALTY_EN = -4.6086734734747745
DBA_REPETITION_PENALTY_EN = 1.2046863355531654

# ── Soft reward only parameters ──────────────────────────────────────────────
SOFT_REWARD_STRENGTH_TR = 2.034751506018973
SOFT_REWARD_STRENGTH_EN = 2.0074439838805587
SOFT_REWARD_MAX_TR = 15.369451667662998
SOFT_REWARD_MAX_EN = 19.037476262523693
SOFT_REWARD_CURRICULUM_RATE_TR = 2.958466455711092
SOFT_REWARD_CURRICULUM_RATE_EN = 0.17255495616360675
ANCHOR_OFFSET_TR = -23.885560696477985
ANCHOR_OFFSET_EN = -28.739574461457774
CONTEXTUAL_NUDGE_TR = 2.0
CONTEXTUAL_NUDGE_EN = 2.0

# ── Soft penalty only parameters ──────────────────────────────────────────────
SOFT_PENALTY_STRENGTH_TR = -98.37477074412291
SOFT_PENALTY_STRENGTH_EN = -118.52447192603353

# ── Soft combined parameters ──────────────────────────────────────────────────
SOFT_COMBINED_REWARD_STRENGTH_TR = 12.754968439065916
SOFT_COMBINED_REWARD_STRENGTH_EN = 4.717973336945308
SOFT_COMBINED_REWARD_MAX_TR = 25.852042853143573
SOFT_COMBINED_REWARD_MAX_EN = 25.969242966481676
SOFT_COMBINED_REWARD_CURRICULUM_RATE_TR = 0.9199123482556277
SOFT_COMBINED_REWARD_CURRICULUM_RATE_EN = 2.430185279310913
SOFT_COMBINED_ANCHOR_OFFSET_TR = -22.000166482897555
SOFT_COMBINED_ANCHOR_OFFSET_EN = -18.856820920604143
SOFT_COMBINED_PENALTY_STRENGTH_TR = -41.22168686753129
SOFT_COMBINED_PENALTY_STRENGTH_EN = -54.722656156763406

# ── Hard inclusion parameters ─────────────────────────────────────────────────
HARD_INCLUSION_BOOST_TR  = 5.743414108444233
HARD_INCLUSION_BOOST_EN  = 7.182293872522841
SUFFIX_PENALTY_TR = -21.918526843561935
SUFFIX_PENALTY_EN = -9.344246208789052
READINESS_THRESHOLD = 200

# ── Hard Inclusion Dynamic Anchoring (Turkish Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_TR = 0
HARD_INCL_SWEET_RANK_TR   = 557
HARD_INCL_SWEET_BUFFER_TR = 12.270148382090829
HARD_INCL_ANCHOR_START_TR = -6.552844930239333
HARD_INCL_ANCHOR_RANGE_TR = 10.314292138926305

# ── Hard Inclusion Dynamic Anchoring (English Target) ─────────────────────────
HARD_INCL_EARLY_TOKENS_EN = 5
HARD_INCL_SWEET_RANK_EN   = 293
HARD_INCL_SWEET_BUFFER_EN = 5.855872539437234
HARD_INCL_ANCHOR_START_EN = -7.498777335021458
HARD_INCL_ANCHOR_RANGE_EN = 9.831046441283089

# ── Hard combined parameters (Turkish Target) ─────────────────────────────────
HARD_COMBINED_BOOST_TR = 16.191355430287526
HARD_COMBINED_SUFFIX_PENALTY_TR = -8.019039135757986
HARD_COMBINED_EARLY_TOKENS_TR = 1
HARD_COMBINED_SWEET_RANK_TR = 476
HARD_COMBINED_SWEET_BUFFER_TR = 14.968198606137086
HARD_COMBINED_ANCHOR_START_TR = -35.06821465771045
HARD_COMBINED_ANCHOR_RANGE_TR = 10.2385959653275

# ── Hard combined parameters (English Target) ─────────────────────────────────
HARD_COMBINED_BOOST_EN = 22.794340695682752
HARD_COMBINED_SUFFIX_PENALTY_EN = -9.33004427464971
HARD_COMBINED_EARLY_TOKENS_EN = 1
HARD_COMBINED_SWEET_RANK_EN = 294
HARD_COMBINED_SWEET_BUFFER_EN = 2.5459156103398226
HARD_COMBINED_ANCHOR_START_EN = -22.84952287564377
HARD_COMBINED_ANCHOR_RANGE_EN = 25.0054242009445

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