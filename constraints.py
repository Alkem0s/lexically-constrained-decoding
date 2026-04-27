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
        max_logits = scores.max(dim=-1, keepdim=True).values

        step_log = {
            "step": self._step, "type": "soft", "tokens": {},
            "reward_steps_active": self._reward_steps_active,
        }

        # Step 1: Penalty (as now)
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

        # Step 2: Reward via Curriculum Anchoring
        pending_ids = set()
        for ids in self.pending_rewards.values():
            pending_ids.update(ids)

        if pending_ids:
            is_early = self._step <= 3
            eff_reward = 0.0 if is_early else self._effective_reward()
            
            pending_tensor = torch.tensor(
                list(pending_ids), dtype=torch.long, device=scores.device
            )

            for b in range(scores.shape[0]):
                for tid in pending_tensor.tolist():
                    current_logit = scores[b, tid].item()
                    
                    # Baseline incorporates the silenced or active curriculum reward
                    target_baseline = max_logits[b, 0].item() + config.ANCHOR_OFFSET + eff_reward
                    
                    if current_logit < target_baseline:
                        # The token is buried: Pull it exactly to the curriculum baseline
                        scores[b, tid] = target_baseline
                        delta_applied = target_baseline - current_logit
                    else:
                        # The token is already highly probable organically: Just nudge it to win
                        scores[b, tid] += config.CONTEXTUAL_NUDGE
                        delta_applied = config.CONTEXTUAL_NUDGE
                    
                    step_log["tokens"][tid] = {
                        "logit"               : scores[b, tid].item(),
                        "prob"                : probs[tid].item(),
                        "rank"                : int((logits > logits[tid]).sum().item()) + 1,
                        "delta"               : delta_applied,
                        "delta_base"          : self.reward_val,
                        "reward_steps_active" : self._reward_steps_active,
                        "mode"                : "reward (curriculum)",
                    }

            self._reward_steps_active += 1

        self.log_store.append(step_log)
        self._step += 1
        return scores


# ── 3. Hard Inclusion ─────────────────────────────────────────────────────────

class HardInclusionProcessor(LogitsProcessor):
    def __init__(
        self,
        required_token_sequences : List[List[List[int]]],
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

        # FIX: Keep a pristine copy of all beams' scores to accurately check 
        # the model's true natural intent before we apply masks.
        original_scores = scores.clone() 
        probs = torch.softmax(original_scores, dim=-1)

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
            for word_idx, variant_group in self.master_sequences.items():
                found = False
                
                # Check if ANY of the casing variants exist in the generated text
                for seq in variant_group:
                    seq_len = len(seq)
                    if len(beam_list) >= seq_len and beam_list[-seq_len:] == seq:
                        found = True
                        just_completed_any = True
                        break
                    else:
                        for i in range(len(beam_list) - seq_len):
                            if beam_list[i:i+seq_len] == seq:
                                found = True
                                break
                    if found:
                        break
                            
                if not found:
                    # If missing, target the first variant (usually mid-sentence lowercase)
                    pending_for_beam[word_idx] = variant_group[0]

            min_pending = min(min_pending, len(pending_for_beam))

            # --- Babbling Escape Hatch ---
            is_babbling = (current_len > self.src_len * 1.5)
            
            if pending_for_beam and self.eos_token_id is not None:
                if not is_babbling:
                    # Only block EOS if it hasn't completely lost its mind yet
                    scores[b, self.eos_token_id] -= 50.0

            # Morphological Escape Prevention 
            if just_completed_any and self.boundary_mask is not None:
                non_boundary = ~self.boundary_mask.to(scores.device)
                scores[b, non_boundary] += config.SUFFIX_PENALTY
                
            if pending_for_beam and self.eos_token_id is not None:
                scores[b, self.eos_token_id] -= 50.0

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
                
                # ── Robust Context-Aware Logic ──
                beam_logits  = original_scores[b]
                target_logit = beam_logits[target_token].item()
                target_rank  = int((beam_logits > target_logit).sum().item()) + 1
                
                if is_continuation:
                    applied_boost = 50.0 
                else:
                    is_early = current_len <= config.HARD_INCL_EARLY_TOKENS
                    
                    if is_early:
                        applied_boost = 0.0
                    elif target_rank <= config.HARD_INCL_SWEET_RANK:
                        applied_boost = config.HARD_INCL_SWEET_BUFFER
                    else:
                        progress_multiplier = min(1.0, current_len / max(1, self.src_len * 0.8))
                        max_logit = beam_logits.max().item()
                        
                        target_anchor = max_logit + config.HARD_INCL_ANCHOR_START + (config.HARD_INCL_ANCHOR_RANGE * progress_multiplier)
                        
                        if target_logit < target_anchor:
                            applied_boost = target_anchor - target_logit
                        else:
                            applied_boost = 0.0
                
                scores[b, target_token] += applied_boost

                if b == 0:
                    step_log["tokens"][target_token] = {
                        "logit"           : target_logit,
                        "prob"            : probs[b, target_token].item(),
                        "rank"            : target_rank,
                        "delta"           : applied_boost,
                        "word_group"      : word_idx,
                        "is_continuation" : is_continuation,
                        "desperation"     : is_babbling
                    }

        step_log["pending_count"] = min_pending if min_pending != 999 else 0
        self.log_store.append(step_log)
        self._step += 1
        return scores