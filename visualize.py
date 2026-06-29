"""
visualize.py
Run this script from the project root to generate all figures for the paper.
It reads the latest results files from the results/ directory.

Usage:
    python visualize.py

Output:  paper/figures/  (created automatically)
"""

import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.patheffects as pe

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# ── colour palette ─────────────────────────────────────────────────────────────
HARD_COLOR  = "#E07B54"
SOFT_COLOR  = "#5B9BD5"
DBA_COLOR   = "#8E44AD"
BASE_COLOR  = "#AAB7B8"
REF_ALPHA   = 0.55       # transparency for Reference-BLEU overlay bars

# ── helpers ───────────────────────────────────────────────────────────────────

def _latest(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {RESULTS_DIR}")
    return files[-1]

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f'  Saved → "{path}"')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 0: System Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    def draw_box(ax, x, y, w, h, text, color="#FFFFFF", edge="#2E4057", lw=1.5, fs=10, zorder=3):
        box = mpatches.Rectangle((x, y), w, h, fill=True, facecolor=color,
                                 edgecolor=edge, linewidth=lw, zorder=zorder)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color="#2E4057", fontweight="bold", zorder=zorder+1)
        return x + w, y + h/2

    def draw_arrow(ax, x1, y1, x2, y2, text=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2E4057", lw=1.5), zorder=2)
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha="center", va="bottom",
                    fontsize=8.5, color="#2E4057", zorder=4,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    draw_box(ax, 0.0, 2.0, 1.2, 0.8, "Source Text", color="#F2F4F4")
    draw_box(ax, 0.0, 0.5, 1.2, 0.8, "Constraints\n(Lexical)", color="#F2F4F4")

    draw_box(ax, 2.0, 2.0, 1.5, 0.8, "Tokenizer", color="#E8F8F5", edge="#1ABC9C")
    draw_box(ax, 2.0, 0.5, 1.5, 0.8, "Token\nExpansion", color="#E8F8F5", edge="#1ABC9C")

    draw_arrow(ax, 1.2, 2.4, 2.0, 2.4)
    draw_arrow(ax, 1.2, 0.9, 2.0, 0.9)

    loop_box = mpatches.Rectangle((4.3, 0.2), 4.2, 3.8, fill=True, facecolor="#FDF2E9",
                                  edgecolor="#E67E22", linewidth=2, linestyle="--", zorder=1)
    ax.add_patch(loop_box)
    ax.text(4.4, 3.8, "Beam Search Decoding Loop (Step t)", fontsize=9, color="#E67E22",
            fontweight="bold", va="top")

    draw_arrow(ax, 3.5, 2.4, 4.3, 2.4, "input_ids")

    draw_box(ax, 4.6, 2.0, 1.6, 0.8, "OPUS-MT\nTransformer", color="#D6EAF8", edge="#3498DB")

    proc_y = 0.5
    draw_box(ax, 6.7, proc_y + 1.2, 1.6, 0.6, "Hard Exclusion\n(-∞ mask)", color="#FADBD8", edge="#E74C3C")
    draw_box(ax, 6.7, proc_y + 0.5, 1.6, 0.6, "Hard Inclusion\n(Anchor Boost)", color="#FCF3CF", edge="#F1C40F")
    draw_box(ax, 6.7, proc_y, 1.6, 0.6, "Soft Constraint\n(Curriculum)", color="#D5F5E3", edge="#2ECC71")

    draw_arrow(ax, 3.5, 0.9, 4.3, 0.9, "target_sequences")
    draw_arrow(ax, 6.2, 2.3, 6.7, 2.3, "Raw Logits")

    ax.plot([7.5, 7.5, 5.4], [0.5, 0.3, 0.3], color="#E74C3C", lw=1.5, zorder=1)
    ax.annotate("", xy=(5.4, 2.0), xytext=(5.4, 0.3),
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5), zorder=2)
    ax.text(5.4, 0.45, "Modified Logits", ha="center", va="bottom",
            fontsize=8.5, color="#E74C3C", fontweight="bold")

    draw_arrow(ax, 8.5, 2.0, 9.3, 2.0, "Beam end")
    draw_box(ax, 9.3, 1.6, 1.2, 0.8, "Target\nTranslation", color="#F2F4F4")

    ax.set_xlim(-0.2, 10.7)
    ax.set_ylim(0, 4.2)
    fig.tight_layout()
    save(fig, "fig0_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Consolidated Quality Overview
# Baseline BLEU + Reference BLEU + ChrF grouped bars, satisfaction overlay
# ─────────────────────────────────────────────────────────────────────────────

def fig_quality_overview(direction):
    agg_path = _latest(f"aggregate_{direction}_*.json")
    with open(agg_path) as f:
        agg = json.load(f)

    mode_labels = {
        "unconstrained":   "Unconstrained",
        "hard_exclusion":  "Hard\nExclusion",
        "hard_inclusion":  "Hard\nInclusion",
        "hard_combined":   "Hard\nCombined",
        "soft_penalty":    "Soft\nPenalty",
        "soft_reward":     "Soft\nReward\n(Escal.)",
        "soft_combined":   "Soft\nCombined",
        "huggingface_dba": "HF DBA\n(Baseline)",
    }

    modes    = [m for m in mode_labels if m in agg]
    labels   = [mode_labels[m] for m in modes]
    bleu_b   = [agg[m]["avg_bleu_vs_base"] for m in modes]
    bleu_r   = [agg[m]["avg_bleu_vs_ref"]  for m in modes]
    chrf_r   = [agg[m]["avg_chrf_vs_ref"]  for m in modes]
    sat_pct  = [(agg[m].get("avg_satisfaction") if agg[m].get("avg_satisfaction") is not None else 1.0) * 100 for m in modes]

    x = np.arange(len(modes))
    w = 0.24   # width of each bar group member

    def _color(m):
        if m == "unconstrained":   return BASE_COLOR
        if m == "huggingface_dba": return DBA_COLOR
        return HARD_COLOR if "hard" in m else SOFT_COLOR

    bar_colors = [_color(m) for m in modes]

    fig, ax1 = plt.subplots(figsize=(13, 5.2))

    # Three grouped bars per mode
    bars_bb = ax1.bar(x - w,   bleu_b, w, color=bar_colors, alpha=0.95,
                      edgecolor="white", linewidth=0.6, zorder=3)
    bars_br = ax1.bar(x,       bleu_r, w, color=bar_colors, alpha=0.65,
                      edgecolor="white", linewidth=0.6, zorder=3, hatch="//")
    bars_cf = ax1.bar(x + w,   chrf_r, w, color=bar_colors, alpha=0.40,
                      edgecolor="white", linewidth=0.6, zorder=3, hatch="xx")

    # Logit-squeeze annotation on Hard Combined
    if "hard_combined" in modes and "hard_exclusion" in modes and "hard_inclusion" in modes:
        mean_hard_b = (agg["hard_exclusion"]["avg_bleu_vs_base"] +
                       agg["hard_inclusion"]["avg_bleu_vs_base"]) / 2
        ci = modes.index("hard_combined")
        ax1.hlines(mean_hard_b, ci - w*1.6, ci - w*0.4,
                   colors="crimson", linewidths=1.8, linestyles="--", zorder=5)
        y_offset = 40 if direction == "en_tr" else 28
        ax1.annotate(f"Exp. Base\n({mean_hard_b:.1f})",
                     xy=(ci - w, mean_hard_b),
                     xytext=(ci - w - 0.6, mean_hard_b + y_offset),
                     ha="center", va="bottom", fontsize=7, color="crimson", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="crimson", lw=0.9,
                                     connectionstyle="arc3,rad=-0.25"))

    # Value labels — only on top of each bar, tiny font to avoid clutter
    def _annotate(bars, vals, suffix):
        for bar, val in zip(bars, vals):
            if val is None: continue
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                     f"{val:.1f}\n{suffix}", ha="center", va="bottom",
                     fontsize=5.8, fontweight="bold", linespacing=1.1)
    _annotate(bars_bb, bleu_b, "Base")
    _annotate(bars_br, bleu_r, "Ref")
    _annotate(bars_cf, chrf_r, "ChrF")

    # Satisfaction as secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, sat_pct, "D--", color="#2E4057", markersize=6,
             linewidth=1.4, zorder=6, label="Satisfaction %")
    ax2.set_ylabel("Constraint Satisfaction (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.yaxis.label.set_color("#2E4057")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    dir_title = "EN→TR" if direction == "en_tr" else "TR→EN"
    ax1.set_title(f"Translation Quality & Constraint Satisfaction ({dir_title})\n"
                  "Baseline BLEU / Reference BLEU / ChrF per Decoding Mode", fontsize=11, y=1.16)

    # Legend
    hard_p  = mpatches.Patch(color=HARD_COLOR, label="Hard modes")
    soft_p  = mpatches.Patch(color=SOFT_COLOR, label="Soft modes")
    dba_p   = mpatches.Patch(color=DBA_COLOR,  label="HuggingFace DBA")
    base_p  = mpatches.Patch(color=BASE_COLOR,  label="Unconstrained")
    bb_p    = mpatches.Patch(facecolor="grey", alpha=0.95, label="Baseline BLEU (solid)")
    br_p    = mpatches.Patch(facecolor="grey", alpha=0.65, hatch="//", label="Ref BLEU (hatched)")
    cf_p    = mpatches.Patch(facecolor="grey", alpha=0.40, hatch="xx", label="ChrF (cross-hatched)")
    sat_l   = plt.Line2D([0], [0], color="#2E4057", marker="D",
                         linestyle="--", linewidth=1.4, label="Satisfaction %")
    ax1.legend(handles=[hard_p, soft_p, dba_p, base_p, bb_p, br_p, cf_p, sat_l],
               fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=4)

    fig.tight_layout()
    save(fig, f"fig1_quality_overview_{direction}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Dynamic Anchor Schedule Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_anchor_heatmap(direction):
    """
    Simulates the mathematical dynamic anchoring schedule based on the
    optimized hyperparameters. (Raw token logs are not saved to disk).
    """
    import config
    n_steps = 25
    src_len = 20
    
    suffix = "TR" if direction == "en_tr" else "EN"
    grace_period = getattr(config, f"HARD_INCL_EARLY_TOKENS_{suffix}", 1)
    buffer       = getattr(config, f"HARD_INCL_SWEET_BUFFER_{suffix}", 4.66)
    anchor_start = getattr(config, f"HARD_INCL_ANCHOR_START_{suffix}", -16.54)
    anchor_range = getattr(config, f"HARD_INCL_ANCHOR_RANGE_{suffix}", 14.05)

    n_tokens = 4
    matrix = np.zeros((n_tokens, n_steps))

    for step in range(n_steps):
        t = step + 1
        if t <= grace_period:
            continue

        progress = min(1.0, t / max(1, src_len * 0.8))
        target_anchor = anchor_start + (anchor_range * progress)
        base_boost = max(0, 10.0 + target_anchor)

        matrix[0, step] = base_boost
        matrix[1, step] = buffer if t == 8  else base_boost
        matrix[2, step] = buffer if t == 15 else base_boost
        matrix[3, step] = buffer if t in [10, 11] else base_boost

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(range(1, n_steps + 1), fontsize=8)
    ax.set_yticks(range(n_tokens))
    ax.set_yticklabels([f"Word {i+1}" for i in range(n_tokens)], fontsize=9)
    ax.set_xlabel("Decoding Step ($t$)", fontsize=10)
    ax.set_ylabel("Constraint Words", fontsize=10)
    
    dir_title = "EN→TR" if direction == "en_tr" else "TR→EN"
    ax.set_title(f"Analytical Simulation of Dynamic Anchor Boost $\\delta_t$ ({dir_title})", fontsize=11)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Applied Boost $\\delta_t$ (logits)", fontsize=9)

    ax.axvspan(-0.5, grace_period - 0.5, alpha=0.15, color="royalblue",
               label=f"Grace period ($\\tau={grace_period}$)")

    for t_idx, step_num in [(1, 8), (2, 15), (3, 10), (3, 11)]:
        ax.text(step_num - 1, t_idx, "*", color="black", ha="center", va="center",
                fontsize=16, fontweight="bold")

    ax.plot([], [], marker="*", color="black", linestyle="None",
            label="Natural Fit (Sweet Rank)")
    ax.legend(fontsize=8.5, loc="upper left")

    fig.tight_layout()
    save(fig, f"fig3_anchor_heatmap_{direction}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Length Ratio comparison (incl. DBA baseline)
# ─────────────────────────────────────────────────────────────────────────────

def fig_length_ratio(direction):
    agg_path = _latest(f"aggregate_{direction}_*.json")
    with open(agg_path) as f:
        agg = json.load(f)

    mode_labels = {
        "hard_exclusion":  "Hard\nExclusion",
        "hard_inclusion":  "Hard\nInclusion",
        "hard_combined":   "Hard\nCombined",
        "soft_penalty":    "Soft\nPenalty",
        "soft_reward":     "Soft\nReward",
        "soft_combined":   "Soft\nCombined",
        "huggingface_dba": "HF DBA\n(Baseline)",
    }
    modes  = [m for m in mode_labels if m in agg]
    ratios = [agg[m]["avg_length_ratio"] for m in modes]
    labels = [mode_labels[m] for m in modes]

    def _color(m):
        if m == "huggingface_dba":
            return DBA_COLOR
        return HARD_COLOR if "hard" in m else SOFT_COLOR

    colors = [_color(m) for m in modes]
    max_ratio = max(ratios)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(modes))
    ax.bar(x, ratios, color=colors, width=0.55, edgecolor="white",
           linewidth=0.8, zorder=3)
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="-",
           zorder=4, label="Unconstrained baseline (1.0)")

    for xi, r in zip(x, ratios):
        ax.text(xi, r + 0.01, f"{r:.3f}", ha="center",
                va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Avg. Output Length Ratio\n(constrained / unconstrained)", fontsize=9.5)
    ax.set_ylim(0.9, max_ratio * 1.12)
    
    dir_title = "EN→TR" if direction == "en_tr" else "TR→EN"
    ax.set_title(f"Output Length Ratio per Decoding Mode ({dir_title})\n"
                 "(Ratio > 1 indicates output inflation; DBA shown for comparison)", fontsize=10.5)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    hard_patch = mpatches.Patch(color=HARD_COLOR, label="Hard modes")
    soft_patch = mpatches.Patch(color=SOFT_COLOR, label="Soft modes")
    dba_patch  = mpatches.Patch(color=DBA_COLOR,  label="HuggingFace DBA")
    baseline   = plt.Line2D([0], [0], color="black", linewidth=1.5,
                            label="Unconstrained (1.0)")
    ax.legend(handles=[hard_patch, soft_patch, dba_patch, baseline], fontsize=8.5)

    fig.tight_layout()
    save(fig, f"fig4_length_ratio_{direction}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Decoding Latency comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_latency(direction):
    """
    Horizontal bar chart of average decoding latency per sentence (ms)
    for all modes including the DBA baseline. Highlights the speed advantage
    of logit-level approaches.
    """
    agg_path = _latest(f"aggregate_{direction}_*.json")
    with open(agg_path) as f:
        agg = json.load(f)

    mode_labels = {
        "unconstrained":   "Unconstrained",
        "hard_exclusion":  "Hard Exclusion",
        "hard_inclusion":  "Hard Inclusion",
        "hard_combined":   "Hard Combined",
        "soft_penalty":    "Soft Penalty",
        "soft_reward":     "Soft Reward (Escalation)",
        "soft_combined":   "Soft Combined",
        "huggingface_dba": "HuggingFace DBA (Baseline)",
    }

    modes   = [m for m in mode_labels if m in agg]
    latency = [agg[m]["avg_latency_ms"] for m in modes]
    labels  = [mode_labels[m] for m in modes]

    def _color(m):
        if m == "unconstrained":
            return BASE_COLOR
        if m == "huggingface_dba":
            return DBA_COLOR
        return HARD_COLOR if "hard" in m else SOFT_COLOR

    colors = [_color(m) for m in modes]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(modes))

    bars = ax.barh(y, latency, color=colors, edgecolor="white",
                   linewidth=0.8, zorder=3, height=0.6)

    for bar, val in zip(bars, latency):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
                f"{val:.0f} ms", va="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Average Latency per Sentence (ms)", fontsize=10)
    
    dir_title = "EN→TR" if direction == "en_tr" else "TR→EN"
    ax.set_title(f"Decoding Latency Comparison ({dir_title})\n(lower is better; DBA shown for reference)",
                 fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.set_xlim(0, max(latency) * 1.22)

    hard_patch = mpatches.Patch(color=HARD_COLOR, label="Hard modes")
    soft_patch = mpatches.Patch(color=SOFT_COLOR, label="Soft modes")
    dba_patch  = mpatches.Patch(color=DBA_COLOR,  label="HuggingFace DBA")
    ax.legend(handles=[hard_patch, soft_patch, dba_patch],
              fontsize=8.5, loc="lower right")

    fig.tight_layout()
    save(fig, f"fig6_latency_{direction}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures...")
    fig_architecture()
    for direction in ["en_tr", "tr_en"]:
        print(f"Generating figures for direction: {direction}...")
        fig_quality_overview(direction)
        fig_anchor_heatmap(direction)
        fig_length_ratio(direction)
        fig_latency(direction)
    print(f'Done. All figures saved to: "{OUT_DIR}"')
