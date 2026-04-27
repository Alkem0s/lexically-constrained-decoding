import torch
from transformers import LogitsProcessor
from typing import List, Dict
import config


# ── 1. Hard Exclusion ─────────────────────────────────────────────────────────

class HardExclusionProcessor(LogitsProcessor):
    def __init__(self, forbidden_ids: List[int], log_store: List[Dict]):
        self.forbidden_ids = torch.tensor(forbidden_ids, dtype=torch.long)
        self.log_store     = log_store
        self._step         = 0

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        logits   = scores[0]
        probs    = torch.softmax(logits, dim=-1)
        original = {}
        for tid in self.forbidden_ids.tolist():
            original[tid] = {
                "logit" : logits[tid].item(),
                "prob"  : probs[tid].item(),
                "rank"  : int((logits > logits[tid]).sum().item()) + 1,
                "delta" : float("inf"), 
            }
        self.log_store.append({"step": self._step, "type": "exclusion", "tokens": original})
        self._step += 1

        ids = self.forbidden_ids.to(scores.device)
        scores[:, ids] = float("-inf")
        return scores


# ── 2. Soft Constraint ────────────────────────────────────────────────────────

class SoftConstraintProcessor(LogitsProcessor):
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
        generated_set = set(input_ids.reshape(-1).tolist())
        satisfied = [
            idx for idx, ids in self.pending_rewards.items()
            if generated_set.intersection(ids)
        ]
        for idx in satisfied:
            del self.pending_rewards[idx]

    def _effective_reward(self) -> float:
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

        # Step 1: Penalty
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

        # Step 2: Curriculum reward
        pending_ids = set()
        for ids in self.pending_rewards.values():
            pending_ids.update(ids)

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
        src_len                  : int,
        eos_token_id             : int,
        boundary_mask            : torch.Tensor,
        boost                    : float = config.HARD_INCLUSION_BOOST,
        log_store                : List[Dict] = None,
        min_content_tokens       : int = 3,
    ):
        self.master_sequences     = {i: seq for i, seq in enumerate(required_token_sequences)}
        self.src_len              = src_len
        self.eos_token_id         = eos_token_id
        self.boundary_mask        = boundary_mask
        self.boost                = boost
        self.log_store            = log_store if log_store is not None else []
        self._step                = 0
        self.min_content_tokens   = min_content_tokens

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        current_len = input_ids.shape[1]
        num_beams   = input_ids.shape[0]

        if current_len <= self.min_content_tokens:
            self.log_store.append({
                "step": self._step, "type": "inclusion",
                "tokens": {}, "pending_count": len(self.master_sequences),
                "note": f"boost deferred (only {current_len} tokens generated so far)",
            })
            self._step += 1
            return scores

        logits   = scores[0].clone() 
        probs    = torch.softmax(logits, dim=-1)
        step_log = {
            "step": self._step, "type": "inclusion",
            "tokens": {}, "pending_count": 0,
        }

        target_len = max(1, self.src_len // 2)
        progress_ratio = min(1.0, current_len / target_len)
        dynamic_boost = 5.0 + (progress_ratio * (self.boost - 5.0))

        min_pending = 999

        for b in range(num_beams):
            beam_list = input_ids[b].tolist()
            pending_for_beam = {}
            just_completed_any = False
            
            # Determine what THIS specific beam is missing
            for word_idx, seq in self.master_sequences.items():
                seq_len = len(seq)
                found = False
                
                # Check if it was JUST completed at the very end of the beam
                if len(beam_list) >= seq_len and beam_list[-seq_len:] == seq:
                    found = True
                    just_completed_any = True
                else:
                    # Check if it was completed earlier
                    for i in range(len(beam_list) - seq_len):
                        if beam_list[i:i+seq_len] == seq:
                            found = True
                            break
                            
                if not found:
                    pending_for_beam[word_idx] = seq

            min_pending = min(min_pending, len(pending_for_beam))

            # ── NEW: Morphological Escape Prevention ──────────────────────
            # If a word was just completed, violently reject any suffix token.
            # The model MUST choose a boundary (space, punctuation, eos).
            if just_completed_any and self.boundary_mask is not None:
                scores[b, ~self.boundary_mask] = float("-inf")
            # ──────────────────────────────────────────────────────────────
                
            # Apply EOS mask ONLY if this specific beam is still missing words
            if pending_for_beam and self.eos_token_id is not None:
                scores[b, self.eos_token_id] = float("-inf")

            # Apply logit boost ONLY for this specific beam's missing words
            for word_idx, seq in pending_for_beam.items():
                seq_len = len(seq)
                target_token = seq[0] 
                is_continuation = False
                
                max_prefix_len = min(seq_len - 1, current_len)
                for prefix_len in range(max_prefix_len, 0, -1):
                    if beam_list[-prefix_len:] == seq[:prefix_len]:
                        target_token = seq[prefix_len] 
                        is_continuation = True
                        break
                
                applied_boost = 1000.0 if is_continuation else dynamic_boost
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

        step_log["pending_count"] = min_pending if min_pending != 999 else 0
        self.log_store.append(step_log)
        self._step += 1
        return scores