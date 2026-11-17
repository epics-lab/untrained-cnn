"""
Renders a schematic diagram of (1) the forward/backward pipeline shared by
every backbone in this repo and (2) how each of the 3 attribution methods
(--attribution-method in calculate_gradcam.py) turns that gradient into a
voxel saliency map. Purely illustrative -- it draws labeled boxes/arrows from
the architecture already documented in train_linear.py/calculate_gradcam.py,
it does not load any checkpoint or run a real forward/backward pass.

Usage:
    python3 plot_gradient_flow_diagram.py --output ../docs/figures/gradient_flow_diagram.png

See docs/attribution_methods.md for the formal equations this diagram
illustrates, and docs/known_limitations.md's preprocessing table for the
per-model input shapes reproduced in the top panel.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FORWARD_COLOR = "#2b6cb0"
BACKWARD_COLOR = "#c53030"
BOX_FACE = "#ebf8ff"
BOX_EDGE = "#2c5282"
SPECIAL_BOX_FACE = "#fffaf0"
SPECIAL_BOX_EDGE = "#b7791f"

# (model, input shape "(C,H,W,D)", backbone, postprocess) -- from
# train_linear.py's make_feature_spec() FeatureSpec entries.
BACKBONES = [
    ("optim_uncnn", "(3, 91, 109, 91)", "CNN3D_CovPool (untrained, frozen, cov-pooled)", "none"),
    ("brainiac", "(1, 96, 96, 96)", "BrainIAC ViT (pretrained, frozen)", "CLS token isolation"),
    ("swinbrain", "(3, 128, 128, 64)", "Swin UNETR encoder10 (pretrained, frozen)", "hook + AdaptiveAvgPool3d"),
    ("simclr", "(1, 150, 192, 192)", "SimCLR 3D encoder (pretrained, frozen)", "none"),
    ("anatcl", "(1, 121, 128, 121)", "AnatCL fold-averaged encoder (pretrained, frozen)", "5-fold average"),
]


def box(ax, xy, w, h, text, face=BOX_FACE, edge=BOX_EDGE, fontsize=9.5, fontweight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=1.4, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, zorder=3, wrap=True)
    return x, y, w, h


def arrow(ax, start, end, color, style="-", label=None, label_dy=0.0, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
        color=color, linestyle=style, connectionstyle=connectionstyle, zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=8.5, color=color, zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))


def draw_pipeline_panel(ax):
    """
    Shared forward (blue, downward) / backward (red, upward) pipeline every
    model in BrainAgePipeline.forward (train_linear.py) goes through. The
    signed-sqrt/cov-mask box is drawn in the "special" color because it's a
    no-op passthrough for every model except optim_uncnn (cov_mask is
    all-False elsewhere) -- see docs/known_limitations.md's signed-sqrt
    gradient-amplification caveat.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    ax.set_title(
        "Forward pipeline & gradient flow (BrainAgePipeline.forward, train_linear.py)\n"
        "shared by all 6 backbone/feature variants -- backbone-specific details in the table below",
        fontsize=12, fontweight="bold", pad=10,
    )

    stages = [
        ("Input  x\n(C, H, W, D)\nmodel-specific preprocessed volume", BOX_FACE, BOX_EDGE),
        ("Frozen backbone\n(no_grad params, but input still\ntracks grad -- see requires_grad below)", BOX_FACE, BOX_EDGE),
        ("Postprocess\n(model-specific: CLS token / pool /\nfold-average / identity)", BOX_FACE, BOX_EDGE),
        ("Flatten\n-> feature vector", BOX_FACE, BOX_EDGE),
        ("signed-sqrt(cov_mask)\nsign(f)*sqrt(|f|+eps) on covariance\nfeatures only -- no-op except optim_uncnn", SPECIAL_BOX_FACE, SPECIAL_BOX_EDGE),
        ("StandardScaler\n(f - mean) / scale", BOX_FACE, BOX_EDGE),
        ("Linear head\n(fit RidgeCV, frozen at inference)", BOX_FACE, BOX_EDGE),
        ("Predicted age  F(x)\n(scalar)", BOX_FACE, BOX_EDGE),
    ]

    n = len(stages)
    box_h = 0.85
    gap = (10.0 - box_h * n) / (n + 1)
    box_w = 6.0
    x0 = 0.4
    ys = []
    y = 10.6 - gap
    for text, face, edge in stages:
        y -= box_h
        ys.append(y)
        box(ax, (x0, y), box_w, box_h, text, face=face, edge=edge)
        y -= gap

    # Forward pass: solid blue arrows down the left/center of the boxes.
    for i in range(n - 1):
        top_y = ys[i]
        bot_y = ys[i + 1] + box_h
        arrow(ax, (x0 + box_w * 0.32, top_y), (x0 + box_w * 0.32, bot_y), FORWARD_COLOR)
    ax.text(x0 + box_w * 0.32 - 0.35, (ys[0] + ys[-1] + box_h) / 2, "forward\npass",
            rotation=90, ha="center", va="center", fontsize=9, color=FORWARD_COLOR, fontweight="bold")

    # Backward pass: dashed red arrows back up, one per stage, all the way
    # to the input -- one autograd backward() call, not per-box passes.
    for i in range(n - 1):
        top_y = ys[i]
        bot_y = ys[i + 1] + box_h
        arrow(ax, (x0 + box_w * 0.72, bot_y), (x0 + box_w * 0.72, top_y), BACKWARD_COLOR, style="--")
    ax.text(x0 + box_w * 0.72 + 0.35, (ys[0] + ys[-1] + box_h) / 2, "backward()\nd F(x) / d x",
            rotation=90, ha="center", va="center", fontsize=9, color=BACKWARD_COLOR, fontweight="bold")

    # Per-backbone table to the right of the pipeline.
    tx = x0 + box_w + 0.6
    tw = 10.0 - tx - 0.2  # noqa: unused width var kept for clarity of layout math
    header_y = ys[0] + box_h - 0.1
    ax.text(tx, header_y, "Per-backbone specifics", fontsize=10.5, fontweight="bold")
    row_h = 0.62
    ax.text(tx, header_y - row_h, "model", fontsize=8.5, fontweight="bold")
    ax.text(tx + 1.55, header_y - row_h, "input shape", fontsize=8.5, fontweight="bold")
    ax.text(tx + 3.55, header_y - row_h, "postprocess", fontsize=8.5, fontweight="bold")
    for i, (model, shape, backbone, postprocess) in enumerate(BACKBONES):
        ry = header_y - row_h * (i + 2)
        ax.text(tx, ry, model, fontsize=8.5, family="monospace")
        ax.text(tx + 1.55, ry, shape, fontsize=8, family="monospace")
        ax.text(tx + 3.55, ry, postprocess, fontsize=8)
        ax.text(tx, ry - 0.27, backbone, fontsize=7.3, color="#4a5568", style="italic")
    ax.text(
        tx, header_y - row_h * (len(BACKBONES) + 2.6),
        "optim_uncnn is the only model with a nontrivial\ncov_mask -- signed-sqrt is a no-op elsewhere.",
        fontsize=7.8, color=SPECIAL_BOX_EDGE, style="italic",
    )


def draw_pass_icon(ax, cx, cy, w=0.9, h=0.4, alpha_label=None):
    """One small forward(blue)+backward(red) pass icon used to tally passes."""
    box(ax, (cx - w / 2, cy - h / 2), w, h, "F", face=BOX_FACE, edge=BOX_EDGE, fontsize=8)
    arrow(ax, (cx - w / 2 - 0.35, cy + 0.28), (cx - w / 2, cy + h * 0.15), FORWARD_COLOR)
    arrow(ax, (cx + w / 2, cy - h * 0.15), (cx + w / 2 + 0.35, cy - 0.28), BACKWARD_COLOR, style="--")
    if alpha_label:
        ax.text(cx, cy - h / 2 - 0.28, alpha_label, ha="center", va="top", fontsize=7, color="#4a5568")


def draw_method_panel(ax, title, n_passes, passes_label, formula_lines, note_lines, extra_input_arrow=None):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 9.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor="white", edgecolor="#a0aec0", linewidth=1.2, zorder=0))
    ax.text(5, 9.25, title, ha="center", fontsize=11.5, fontweight="bold", clip_on=True)
    ax.text(5, 8.65, passes_label, ha="center", va="top", fontsize=8.5, color="#4a5568",
            clip_on=True, wrap=True)

    # Row of pass icons, one (or a representative few + "...") per forward/backward pass.
    icon_y = 7.0
    shown = min(n_passes, 6)
    total_w = 8.6
    spacing = total_w / max(shown, 1)
    start_x = 5 - total_w / 2 + spacing / 2
    if extra_input_arrow:
        ax.text(5, 7.85, extra_input_arrow, ha="center", va="top", fontsize=7.6, color="#2f855a",
                clip_on=True, wrap=True)
    for i in range(shown):
        alpha_label = None
        if n_passes > 1:
            k = i + 1 if shown == n_passes else (i + 1 if i < shown - 1 else n_passes)
            alpha_label = f"α={k}/{n_passes}"
        draw_pass_icon(ax, start_x + i * spacing, icon_y, alpha_label=alpha_label)
        if i < shown - 1:
            gap_x = start_x + i * spacing + spacing / 2
            if shown < n_passes and i == shown - 2:
                ax.text(gap_x, icon_y, "...", ha="center", va="center", fontsize=13, fontweight="bold")

    ax.text(5, 5.9, "\n".join(formula_lines), ha="center", va="top", fontsize=10, family="monospace",
            bbox=dict(facecolor="#f7fafc", edgecolor="#cbd5e0", boxstyle="round,pad=0.4"))
    ax.text(5, 2.1, "\n".join(note_lines), ha="center", va="top", fontsize=8.2, color="#2d3748", wrap=True)


def draw_methods_row(fig, gs, row):
    ax_grad = fig.add_subplot(gs[row, 0])
    draw_method_panel(
        ax_grad,
        "1. Gradient",
        n_passes=1,
        passes_label="1 forward + 1 backward pass",
        formula_lines=["S_v = d F(x) / d x_v"],
        note_lines=[
            "Sensitivity map at x itself (Simonyan et al., 2014).",
            "Nonzero almost everywhere, including background --",
            "not constrained to respect x's zero-valued voxels.",
        ],
    )

    ax_gxi = fig.add_subplot(gs[row, 1])
    draw_method_panel(
        ax_gxi,
        "2. Gradient x Input",
        n_passes=1,
        passes_label="1 forward + 1 backward pass,\nthen elementwise x input",
        formula_lines=["A_v = x_v * d F(x) / d x_v"],
        note_lines=[
            "Shrikumar et al. (2016) / Ancona et al. (2018).",
            "Forced to exactly 0 wherever x_v = 0 -- reveals the",
            "true (masked) extent of the data, unlike Gradient.",
        ],
    )

    ax_ig = fig.add_subplot(gs[row, 2])
    draw_method_panel(
        ax_ig,
        "3. Integrated Gradients",
        n_passes=50,
        passes_label="--ig-steps forward+backward passes\n(default 50) along x0 -> x",
        formula_lines=[
            "IG_v ~= (x_v - x0_v) * (1/m) *",
            "        sum_{k=1..m} dF(x0 + (k/m)(x-x0))/dx_v",
        ],
        note_lines=[
            "Sundararajan et al. (2017); Riemann-sum path integral.",
            "x0: --ig-baseline zero (all-zero) or reference",
            "(per-model population mean, compute_reference_baselines.py).",
            "Exact path integral is complete; finite sum residual is reported.",
        ],
        extra_input_arrow="input: x0 + alpha*(x-x0), alpha=k/m",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "figures" / "gradient_flow_diagram.png",
    )
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 11.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.08, wspace=0.15)

    ax_pipeline = fig.add_subplot(gs[0, :])
    draw_pipeline_panel(ax_pipeline)

    draw_methods_row(fig, gs, row=1)

    fig.suptitle(
        "Gradient flow & voxel-attribution computation -- untrained-CNN-saliency\n"
        "(see docs/attribution_methods.md for the formal definitions/references this diagram illustrates)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
