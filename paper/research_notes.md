# Research Notes — HPO Parameter Plateaus and Optimization Landscape

During our Hyperparameter Optimization (HPO) sweeps, we observed that many trials for decoding modes like `soft_penalty` and `soft_reward` yield identical metric scores. This document analyzes these findings and details the underlying mechanics of this parameter plateau phenomenon.

---

## 1. The Threshold Effect in Soft Penalty
For `soft_penalty`, we observed that any penalty strength (e.g., `-13.1` to `-188.4`) produces identical translation results.

### Mechanics
```python
scores[:, ids] += self.penalty_val
```
*   **Beam Search Suppression**: Beam search only considers the top candidate sequences. Once `penalty_val` is negative enough to push a forbidden token out of the top `NUM_BEAMS` window, the probability of selecting that token falls to $0$.
*   **Flat Landscape**: Increasing the penalty beyond this threshold (e.g., making it even more negative) does not change the selected beams. Thus, any penalty below the threshold yields the exact same translation.
*   **Heuristic**: You do not need to fine-tune the penalty value. Any value below a safe threshold (like `-30.0`) guarantees optimal suppression without affecting other vocabulary logits.

---

## 2. The Anchor Offset Plateau in Soft Reward
For `soft_reward`, many trials with diverse parameter configurations yield the exact same score of `1064.400`, while others drop to `1049.680`.

### Mechanics
$$\text{target\_baseline} = \text{max\_logits} + \text{anchor\_offset} + \text{effective\_reward}$$

```
                          ANCHOR OFFSET LANDSCAPE
                          
    Aggressive Nudging (Bad BLEU)      Gentle Safety Net (Optimal BLEU)
  ◄───────────────────────────────────┬───────────────────────────────────►
 -5.0                               -15.0                               -60.0
 (Forces early constraint injection)   (Allows natural organic generation)
```

*   **Aggressive Nudging (Low BLEU)**: When `anchor_offset` is too close to `0.0` (e.g., `-3.0`), the target baseline is extremely high. The decoder aggressively forces constraints into early or unnatural sentence positions, disrupting translation quality and dropping BLEU.
*   **Gentle Safety Net (Plateau)**: When `anchor_offset` is negative enough (e.g., less than `-15.0`), it acts as a passive safety net. The model generates constraint words organically. The curriculum reward baseline only intervenes when a token is buried.
*   **Flat Landscape**: Once the offset is in this "gentle" zone, the exact values of reward strengths and offsets do not change the generation paths, creating a large, flat, optimal plateau.

---

## 3. Key Takeaways for Future HPO and Deployment

> [!TIP]
> **Heuristic for Soft Penalties**: Do not perform micro-tuning sweeps on penalty variables. A simple default value of `-50.0` is robust and behaves identically to any larger penalty.

> [!IMPORTANT]
> **Offset Guidelines**: Keep `anchor_offset` ranges bounded away from `0` (e.g., search within `[-60.0, -15.0]`) to avoid wasting HPO trial budget on aggressive, low-BLEU search regions.

### Recommendations on Trial Budgets:
*   **`soft_penalty`**: Requires **5–10 trials** maximum. It has a flat optimization landscape (suppressing forbidden tokens requires any sufficiently negative logit value), so it converges almost instantly.
*   **`hard_inclusion`**: Requires **50+ trials**. It has multiple continuous parameters (boost, rank, range, starts, buffers) with complex interactions. A larger trial budget is needed to find the optimal trade-off between constraint satisfaction and translation BLEU.
*   **`dba`**: Requires **15–20 trials**. It has only 3 discrete/continuous parameters (beams, length penalty, repetition penalty), allowing Optuna to cover the search space quickly.
*   **`soft_reward`**: Requires **30–40 trials**. It has 4 parameters. While it exhibits a large, flat optimal plateau when the anchor offset is below `-15.0`, it still requires some search budget to avoid the aggressive nudging zone (offsets close to `0.0`).
*   **`soft_combined`**: Requires **50+ trials**. It contains 5 parameters, combining reward and penalty forces which introduce complex trade-offs in search path exploration.
*   **`hard_combined`**: Requires **50+ trials**. This mode has 7 parameters and optimizes simultaneously for exclusion and inclusion logic, necessitating a larger search budget for convergence.