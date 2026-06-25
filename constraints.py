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
        reward_val          : float = 3.75,
        penalty_val         : float = -34.93,
        curriculum_rate     : float = 0.25,
        max_reward          : float = 12.0,
        anchor_offset       : float = -5.94,
        contextual_nudge    : float = 2.0,
        log_store           : List[Dict] = None,
    ):
        self.pending_rewards = {i: ids for i, ids in enumerate(reward_token_groups)} if reward_token_groups else {}
        self.penalty_ids     = torch.tensor(penalty_ids, dtype=torch.long) if penalty_ids else None
        self.reward_val      = reward_val
        self.penalty_val     = penalty_val
        self.curriculum_rate = curriculum_rate
        self.max_reward      = max_reward
        self.anchor_offset   = anchor_offset
        self.contextual_nudge = contextual_nudge
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

        if pending_ids and self._step > 3:
            eff_reward = self._effective_reward()
            
            pending_tensor = torch.tensor(
                list(pending_ids), dtype=torch.long, device=scores.device
            )

            for b in range(scores.shape[0]):
                for tid in pending_tensor.tolist():
                    current_logit = scores[b, tid].item()
                    
                    # Baseline incorporates the silenced or active curriculum reward
                    target_baseline = max_logits[b, 0].item() + self.anchor_offset + eff_reward
                    
                    if current_logit < target_baseline:
                        # The token is buried: Pull it exactly to the curriculum baseline
                        scores[b, tid] = target_baseline
                        delta_applied = target_baseline - current_logit
                    else:
                        # The token is already highly probable organically: Just nudge it to win
                        scores[b, tid] += self.contextual_nudge
                        delta_applied = self.contextual_nudge
                    
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
        boost                    : float = 7.0,
        log_store                : List[Dict] = None,
        min_content_tokens       : int = 3,
        suffix_penalty           : float = 0.0,
        early_tokens             : int = 1,
        sweet_rank               : int = 394,
        sweet_buffer             : float = 4.66,
        anchor_start             : float = -16.54,
        anchor_range             : float = 14.05,
    ):
        self.master_sequences     = {i: seq for i, seq in enumerate(required_token_sequences)}
        self.src_len              = src_len
        self.eos_token_id         = eos_token_id
        self.boundary_mask        = boundary_mask
        self.boost                = boost
        self.log_store            = log_store if log_store is not None else []
        self._step                = 0
        self.min_content_tokens   = min_content_tokens
        self.suffix_penalty       = suffix_penalty
        self.early_tokens         = early_tokens
        self.sweet_rank           = sweet_rank
        self.sweet_buffer         = sweet_buffer
        self.anchor_start         = anchor_start
        self.anchor_range         = anchor_range

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

        min_pending = 999

        for b in range(num_beams):
            beam_list = input_ids[b].tolist()
            pending_for_beam = {}
            just_completed_any = False
            
            # Determine what THIS specific beam is missing
            for word_idx, variant_group in self.master_sequences.items():
                found = False
                
                # Check ALL possible starting positions for any variant
                for seq in variant_group:
                    seq_len = len(seq)
                    if len(beam_list) < seq_len:
                        continue
                        
                    # Search entire history including the most recent tokens
                    for i in range(len(beam_list) - seq_len + 1):
                        if beam_list[i:i+seq_len] == seq:
                            found = True
                            # If it was just finished this step, mark for morphology check
                            if i == len(beam_list) - seq_len:
                                just_completed_any = True
                            break
                    if found:
                        break
                            
                if not found:
                    pending_for_beam[word_idx] = variant_group[0]

            min_pending = min(min_pending, len(pending_for_beam))

            # --- Babbling Escape Hatch ---
            # If the sentence is getting too long, stop forcing and allow EOS.
            is_babbling = (current_len > self.src_len * 1.5)
            
            if pending_for_beam and self.eos_token_id is not None:
                if not is_babbling:
                    # Gradually reduce the EOS penalty as we approach max_length
                    # to allow the model to finish naturally if it's stuck.
                    eos_penalty = -20.0 if current_len > self.src_len else -50.0
                    scores[b, self.eos_token_id] += eos_penalty

            # Morphological Escape Prevention 
            if just_completed_any and self.boundary_mask is not None and self.suffix_penalty < 0.0:
                non_boundary = ~self.boundary_mask.to(scores.device)
                scores[b, non_boundary] += self.suffix_penalty

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
                max_logit    = beam_logits.max().item()
                target_logit = beam_logits[target_token].item()
                target_rank  = int((beam_logits > target_logit).sum().item()) + 1
                
                if is_continuation:
                    # Just enough to guarantee it's the top choice
                    # ONLY apply if not already blocked by another processor
                    if target_logit > float("-inf"):
                        applied_boost = (max_logit + self.boost) - target_logit
                    else:
                        applied_boost = 0.0
                else:
                    is_early = current_len <= self.early_tokens
                    
                    if is_early:
                        applied_boost = 0.0
                    elif target_rank <= self.sweet_rank:
                        applied_boost = self.sweet_buffer
                    else:
                        # Dynamic Anchoring: Pull the token up from the depths
                        progress_multiplier = min(1.0, current_len / max(1, self.src_len * 0.8))
                        
                        target_anchor = max_logit + self.anchor_start + (self.anchor_range * progress_multiplier)
                        
                        if target_logit > float("-inf") and target_logit < target_anchor:
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