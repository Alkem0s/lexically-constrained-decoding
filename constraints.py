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
from typing import List, Dict, Set
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

        # Fix #10: log beam-0 pre-intervention values (beam 0 is representative;
        # all beams get the same mask so the delta is identical across beams).
        logits   = scores[0]
        probs    = torch.softmax(logits, dim=-1)
        original = {}
        for tid in self.forbidden_ids.tolist():
            original[tid] = {
                "logit" : logits[tid].item(),
                "prob"  : probs[tid].item(),
                "rank"  : int((logits > logits[tid]).sum().item()) + 1,
                "delta" : float("inf"),   # magnitude = ∞ (hard mask)
            }
        self.log_store.append({"step": self._step, "type": "exclusion", "tokens": original})
        self._step += 1

        # ── Apply mask across ALL beams ───────────────────────────────────────
        ids = self.forbidden_ids.to(scores.device)
        scores[:, ids] = float("-inf")
        return scores


# ── 2. Soft Constraint ────────────────────────────────────────────────────────

class SoftConstraintProcessor(LogitsProcessor):
    """
    Add a reward (positive) or penalty (negative) to specified token logits at
    every decoding step.

    Fix #4 / #8: reward tokens are only boosted while still *pending* (not yet
    appeared in any beam's output).  Once a reward token is generated the boost
    stops, preventing runaway repetition.

    Penalty tokens are suppressed every step — consistent with the semantics of
    "do not use this word at all".

    NEW — Curriculum reward escalation:
        The reward is not a fixed scalar but grows each step the word remains
        unsatisfied:
            effective_reward = min(max_reward,
                                   reward_val * (1 + curriculum_rate * n))
        where n = number of steps during which pending rewards were active.
        With defaults (reward_val=5, rate=0.3, max=15) the reward hits its cap
        at step 7, matching HARD_INCLUSION_BOOST so the soft reward becomes
        hard-like for words the model persistently ignores.

    NEW — Sequential penalty -> re-normalise -> reward:
        When both penalty and reward are active in the same step, applying them
        additively can cause the penalty to suppress probability mass that the
        reward is trying to build — the two constraints partially cancel.
        Instead we now:
          1. Apply the penalty to push forbidden tokens down.
          2. Re-normalise the distribution (log-softmax over vocab dim) so the
             remaining mass is redistributed across non-penalised tokens.
          3. Apply the escalated reward on top of the re-normalised scores.
        This ensures the reward operates on a clean post-penalty distribution
        with no cancellation artefacts.

    Args:
        reward_ids      : token IDs to reward (nudge toward inclusion).
        penalty_ids     : token IDs to penalise (nudge toward exclusion).
        reward_val      : base scalar for reward (grows via curriculum).
        penalty_val     : scalar for penalty (typically negative, fixed).
        curriculum_rate : per-step growth factor for the reward.
        max_reward      : absolute cap on the escalated reward.
        log_store       : mutable list for per-step interpretability logs.
    """

    def __init__(
        self,
        reward_ids      : List[int],
        penalty_ids     : List[int],
        reward_val      : float = config.SOFT_REWARD_STRENGTH,
        penalty_val     : float = config.SOFT_PENALTY_STRENGTH,
        curriculum_rate : float = config.SOFT_REWARD_CURRICULUM_RATE,
        max_reward      : float = config.SOFT_REWARD_MAX,
        log_store       : List[Dict] = None,
    ):
        self.reward_ids      = torch.tensor(reward_ids,  dtype=torch.long) if reward_ids  else None
        self.penalty_ids     = torch.tensor(penalty_ids, dtype=torch.long) if penalty_ids else None
        self.reward_val      = reward_val
        self.penalty_val     = penalty_val
        self.curriculum_rate = curriculum_rate
        self.max_reward      = max_reward
        self.log_store       = log_store if log_store is not None else []
        self._step           = 0

        # Fix #4/#8: track which reward IDs are still unsatisfied.
        self._pending_reward_ids: Set[int] = set(reward_ids) if reward_ids else set()
        # Curriculum counter: increments only while any reward is pending.
        self._reward_steps_active: int = 0

    def _update_pending_rewards(self, input_ids: torch.LongTensor):
        """
        Remove reward IDs that have already appeared in ANY beam's output.
        input_ids shape: (num_beams, seq_len)
        """
        # Fix #9 (for soft reward): check all beams, not just beam 0.
        generated = set(input_ids.reshape(-1).tolist())
        self._pending_reward_ids -= generated

    def _effective_reward(self) -> float:
        """Curriculum-scaled reward for the current active step."""
        return min(
            self.max_reward,
            self.reward_val * (1.0 + self.curriculum_rate * self._reward_steps_active),
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        # Update which reward tokens have already been generated.
        self._update_pending_rewards(input_ids)

        logits   = scores[0]
        probs    = torch.softmax(logits, dim=-1)
        step_log = {
            "step": self._step, "type": "soft", "tokens": {},
            "reward_steps_active": self._reward_steps_active,
        }

        # ── Step 1: Penalty (fixed, applied before re-normalisation) ──────────
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

        # ── Step 2: Re-normalise (only when both constraints are active) ───────
        # After penalty shifts the distribution, re-normalise so the reward
        # operates on a clean post-penalty baseline with no cancellation.
        if self.penalty_ids is not None and self._pending_reward_ids:
            scores = scores - torch.logsumexp(scores, dim=-1, keepdim=True)

        # ── Step 3: Curriculum reward (only for tokens not yet generated) ──────
        if self.reward_ids is not None and self._pending_reward_ids:
            eff_reward     = self._effective_reward()
            pending_tensor = torch.tensor(
                list(self._pending_reward_ids), dtype=torch.long, device=scores.device
            )
            for tid in pending_tensor.tolist():
                step_log["tokens"][tid] = {
                    "logit"               : logits[tid].item(),
                    "prob"                : probs[tid].item(),
                    "rank"                : int((logits > logits[tid]).sum().item()) + 1,
                    "delta"               : eff_reward,
                    "delta_base"          : self.reward_val,
                    "reward_steps_active" : self._reward_steps_active,
                    "mode"                : "reward",
                }
            scores[:, pending_tensor] += eff_reward
            # Curriculum counter only advances while there are pending rewards.
            self._reward_steps_active += 1

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

    Fix #3: Do NOT boost at step 0 (decoder start token position).  The
    decoder's first real decision is step 1 after the forced BOS/language tag.
    Boosting at step 0 injects a required subword before the sentence begins,
    producing garbled prefixes like "di.Kedi …".

    Fix #9: Satisfied-check inspects ALL beams (not just beam 0) so a word
    generated on any beam correctly stops being boosted.

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
        min_content_tokens    : int = 3,
    ):
        self.pending              = {i: ids for i, ids in enumerate(required_token_groups)}
        self.boost                = boost
        self.log_store            = log_store if log_store is not None else []
        self._step                = 0
        self.min_content_tokens   = min_content_tokens

    def _update_pending(self, input_ids: torch.LongTensor):
        """
        Remove groups whose token has appeared in ANY beam's generated output.

        Fix #9: input_ids shape is (num_beams, seq_len).  We flatten across
        all beams so a word generated on beam k (even if not beam 0) is marked
        satisfied and no longer boosted.
        """
        generated_set = set(input_ids.reshape(-1).tolist())
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

        # Fix #3 (extended): skip boosting until at least min_content_tokens
        # real tokens have been generated.  The threshold is computed adaptively
        # from the source length by the caller (decoding.hard_inclusion) so that
        # short sentences do not defer the boost too long.
        current_len = input_ids.shape[1]   # includes the forced BOS token
        if current_len <= self.min_content_tokens:
            self.log_store.append({
                "step": self._step, "type": "inclusion",
                "tokens": {}, "pending_count": len(self.pending),
                "note": f"boost deferred (only {current_len} tokens generated so far)",
            })
            self._step += 1
            return scores

        # Update which required words have already been generated.
        self._update_pending(input_ids)

        logits   = scores[0]
        probs    = torch.softmax(logits, dim=-1)
        step_log = {
            "step": self._step, "type": "inclusion",
            "tokens": {}, "pending_count": len(self.pending),
        }

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