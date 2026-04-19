"""
constraints.py — HuggingFace LogitsProcessor implementations for all three
constraint types described in the project spec.

  1. HardExclusionProcessor  — logit masking (-inf) for forbidden tokens
  2. SoftConstraintProcessor — additive reward/penalty for specified tokens
  3. HardInclusionProcessor  — logit boosting until required words appear

All processors log the *original* logit values before intervention so
interpretability.py can reconstruct what the model originally preferred.
"""

import torch
from transformers import LogitsProcessor
from typing import List, Dict, Tuple
import config


# ── 1. Hard Exclusion ─────────────────────────────────────────────────────────

class HardExclusionProcessor(LogitsProcessor):
    """
    Set the logit of every forbidden token to -inf before sampling.
    This guarantees those tokens can never be selected.

    Args:
        forbidden_ids : flat list of token IDs to suppress.
        log_store     : mutable list; each step appends a dict with
                        the original logit of each forbidden token.
    """

    def __init__(self, forbidden_ids: List[int], log_store: List[Dict]):
        self.forbidden_ids = torch.tensor(forbidden_ids, dtype=torch.long)
        self.log_store     = log_store
        self._step         = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        # ── Log original values before masking ───────────────────────────────
        probs    = torch.softmax(scores[0], dim=-1)   # batch dim 0
        logits   = scores[0]
        original = {}
        for tid in self.forbidden_ids.tolist():
            original[tid] = {
                "logit"     : logits[tid].item(),
                "prob"      : probs[tid].item(),
                "rank"      : int((logits > logits[tid]).sum().item()) + 1,
                "delta"     : float("inf"),            # magnitude = ∞ (hard mask)
            }
        self.log_store.append({"step": self._step, "type": "exclusion", "tokens": original})
        self._step += 1

        # ── Apply mask ───────────────────────────────────────────────────────
        ids = self.forbidden_ids.to(scores.device)
        scores[:, ids] = float("-inf")
        return scores


# ── 2. Soft Constraint ────────────────────────────────────────────────────────

class SoftConstraintProcessor(LogitsProcessor):
    """
    Add a fixed scalar reward (positive) or penalty (negative) to specified
    token logits at every decoding step.

    Args:
        reward_ids    : token IDs to reward (nudge toward inclusion).
        penalty_ids   : token IDs to penalise (nudge toward exclusion).
        reward_val    : scalar to add to reward token logits.
        penalty_val   : scalar to add to penalty token logits (typically negative).
        log_store     : mutable list for per-step interpretability logs.
    """

    def __init__(
        self,
        reward_ids  : List[int],
        penalty_ids : List[int],
        reward_val  : float = config.SOFT_REWARD_STRENGTH,
        penalty_val : float = config.SOFT_PENALTY_STRENGTH,
        log_store   : List[Dict] = None,
    ):
        self.reward_ids  = torch.tensor(reward_ids,  dtype=torch.long) if reward_ids  else None
        self.penalty_ids = torch.tensor(penalty_ids, dtype=torch.long) if penalty_ids else None
        self.reward_val  = reward_val
        self.penalty_val = penalty_val
        self.log_store   = log_store if log_store is not None else []
        self._step       = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        logits = scores[0]
        probs  = torch.softmax(logits, dim=-1)
        step_log = {"step": self._step, "type": "soft", "tokens": {}}

        # ── Reward ────────────────────────────────────────────────────────────
        if self.reward_ids is not None:
            ids = self.reward_ids.to(scores.device)
            for tid in ids.tolist():
                step_log["tokens"][tid] = {
                    "logit" : logits[tid].item(),
                    "prob"  : probs[tid].item(),
                    "rank"  : int((logits > logits[tid]).sum().item()) + 1,
                    "delta" : self.reward_val,
                    "mode"  : "reward",
                }
            scores[:, ids] += self.reward_val

        # ── Penalty ───────────────────────────────────────────────────────────
        if self.penalty_ids is not None:
            ids = self.penalty_ids.to(scores.device)
            for tid in ids.tolist():
                step_log["tokens"][tid] = {
                    "logit" : logits[tid].item(),
                    "prob"  : probs[tid].item(),
                    "rank"  : int((logits > logits[tid]).sum().item()) + 1,
                    "delta" : self.penalty_val,
                    "mode"  : "penalty",
                }
            scores[:, ids] += self.penalty_val

        self.log_store.append(step_log)
        self._step += 1
        return scores


# ── 3. Hard Inclusion ─────────────────────────────────────────────────────────

class HardInclusionProcessor(LogitsProcessor):
    """
    Guarantee that each required word appears at least once by boosting its
    token IDs every step until the word has been generated.

    Strategy:
      - Maintain a set of *pending* required token groups.
      - Each group is a list of subword IDs for one required word
        (we consider the word 'satisfied' if ANY of its subword IDs appear).
      - While pending, add HARD_INCLUSION_BOOST to those token logits.
      - Once generated, remove from pending.

    Args:
        required_token_groups : list-of-lists from MTModel.words_to_token_ids()
        boost                 : logit boost to apply.
        log_store             : mutable interpretability log.
    """

    def __init__(
        self,
        required_token_groups : List[List[int]],
        boost                 : float = config.HARD_INCLUSION_BOOST,
        log_store             : List[Dict] = None,
    ):
        # Store as list of (word_idx, token_ids) tuples
        self.pending   = {i: ids for i, ids in enumerate(required_token_groups)}
        self.boost     = boost
        self.log_store = log_store if log_store is not None else []
        self._step     = 0

    def _update_pending(self, generated_ids: torch.LongTensor):
        """Remove groups whose token has appeared in generated output."""
        generated_set = set(generated_ids.tolist())
        satisfied = [
            idx for idx, ids in self.pending.items()
            if generated_set.intersection(ids)
        ]
        for idx in satisfied:
            del self.pending[idx]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        # Update which required words have already been generated
        # input_ids shape: (batch * num_beams, seq_len)
        self._update_pending(input_ids[0])

        logits    = scores[0]
        probs     = torch.softmax(logits, dim=-1)
        step_log  = {"step": self._step, "type": "inclusion", "tokens": {}, "pending_count": len(self.pending)}

        for word_idx, ids in self.pending.items():
            id_tensor = torch.tensor(ids, dtype=torch.long, device=scores.device)
            for tid in ids:
                step_log["tokens"][tid] = {
                    "logit"      : logits[tid].item(),
                    "prob"       : probs[tid].item(),
                    "rank"       : int((logits > logits[tid]).sum().item()) + 1,
                    "delta"      : self.boost,
                    "word_group" : word_idx,
                }
            scores[:, id_tensor] += self.boost

        self.log_store.append(step_log)
        self._step += 1
        return scores
