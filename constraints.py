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

    Updated to group reward tokens: instead of a flat list of fragments, it 
    accepts list-of-lists (groups). Once ANY token from a group appears in 
    the output, the entire group is marked satisfied and the reward stops.
    """

    def __init__(
        self,
        reward_token_groups : List[List[int]],
        penalty_ids         : List[int],
        reward_val          : float = config.SOFT_REWARD_STRENGTH,
        penalty_val         : float = config.SOFT_PENALTY_STRENGTH,
        curriculum_rate     : float = config.SOFT_REWARD_CURRICULUM_RATE,
        max_reward          : float = config.SOFT_REWARD_MAX,
        log_store           : List[Dict] = None,
    ):
        self.pending_rewards = {i: ids for i, ids in enumerate(reward_token_groups)} if reward_token_groups else {}
        self.penalty_ids     = torch.tensor(penalty_ids, dtype=torch.long) if penalty_ids else None
        self.reward_val      = reward_val
        self.penalty_val     = penalty_val
        self.curriculum_rate = curriculum_rate
        self.max_reward      = max_reward
        self.log_store       = log_store if log_store is not None else []
        self._step           = 0
        self._reward_steps_active = 0

    def _update_pending_rewards(self, input_ids: torch.LongTensor):
        """
        Remove reward groups whose token has appeared in ANY beam's generated output.
        """
        generated_set = set(input_ids.reshape(-1).tolist())
        satisfied = [
            idx for idx, ids in self.pending_rewards.items()
            if generated_set.intersection(ids)
        ]
        for idx in satisfied:
            del self.pending_rewards[idx]

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

        # Gather currently pending reward IDs
        pending_ids = set()
        for ids in self.pending_rewards.values():
            pending_ids.update(ids)

        # ── Step 2: Re-normalise (only when both constraints are active) ───────
        if self.penalty_ids is not None and pending_ids:
            scores = scores - torch.logsumexp(scores, dim=-1, keepdim=True)

        # ── Step 3: Curriculum reward (only for tokens not yet generated) ──────
        if pending_ids:
            eff_reward     = self._effective_reward()
            pending_tensor = torch.tensor(
                list(pending_ids), dtype=torch.long, device=scores.device
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
            self._reward_steps_active += 1

        self.log_store.append(step_log)
        self._step += 1
        return scores


# ── 3. Hard Inclusion ─────────────────────────────────────────────────────────

class HardInclusionProcessor(LogitsProcessor):
    def __init__(
        self,
        required_token_sequences : List[List[int]],
        boost                    : float = config.HARD_INCLUSION_BOOST,
        log_store                : List[Dict] = None,
        min_content_tokens       : int = 3,
    ):
        self.pending              = {i: seq for i, seq in enumerate(required_token_sequences)}
        self.boost                = boost
        self.log_store            = log_store if log_store is not None else []
        self._step                = 0
        self.min_content_tokens   = min_content_tokens

    def _update_pending(self, input_ids: torch.LongTensor):
        num_beams = input_ids.shape[0]
        satisfied = []

        for word_idx, seq in self.pending.items():
            seq_len = len(seq)
            for b in range(num_beams):
                beam_list = input_ids[b].tolist()
                found = False
                for i in range(len(beam_list) - seq_len + 1):
                    if beam_list[i:i+seq_len] == seq:
                        found = True
                        break
                
                if found:
                    satisfied.append(word_idx)
                    break 

        for idx in satisfied:
            del self.pending[idx]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        current_len = input_ids.shape[1]
        if current_len <= self.min_content_tokens:
            self.log_store.append({
                "step": self._step, "type": "inclusion",
                "tokens": {}, "pending_count": len(self.pending),
                "note": f"boost deferred (only {current_len} tokens generated so far)",
            })
            self._step += 1
            return scores

        self._update_pending(input_ids)

        logits   = scores[0].clone() 
        probs    = torch.softmax(logits, dim=-1)
        step_log = {
            "step": self._step, "type": "inclusion",
            "tokens": {}, "pending_count": len(self.pending),
        }

        num_beams = input_ids.shape[0]

        for word_idx, seq in self.pending.items():
            seq_len = len(seq)
            
            for b in range(num_beams):
                beam_list = input_ids[b].tolist()
                target_token = seq[0] 
                is_continuation = False
                
                max_prefix_len = min(seq_len - 1, current_len)
                for prefix_len in range(max_prefix_len, 0, -1):
                    if beam_list[-prefix_len:] == seq[:prefix_len]:
                        target_token = seq[prefix_len] 
                        is_continuation = True
                        break
                
                escalated_boost = min(
                    self.boost * 2,
                    self.boost + (max(0, self._step - 5) * 1.5)
                )
                applied_boost = 1000.0 if is_continuation else escalated_boost
                scores[b, target_token] += applied_boost

                if b == 0:
                    step_log["tokens"][target_token] = {
                        "logit"           : logits[target_token].item(),
                        "prob"            : probs[target_token].item(),
                        "rank"            : int((logits > logits[target_token]).sum().item()) + 1,
                        "delta"           : applied_boost,
                        "word_group"      : word_idx,
                        "is_continuation" : is_continuation
                    }

        self.log_store.append(step_log)
        self._step += 1
        return scores