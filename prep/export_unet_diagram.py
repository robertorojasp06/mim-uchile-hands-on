"""Draw the architecture of the U-Net used in activity 4A (UNet300, DeepPath) in the style of Ronneberger et al. 2015.

Writes notebooks/img/S4A_unet300.png. Each box is a feature map: its height follows the spatial size and its
width the number of channels; white boxes are the copies brought by the skip connections.

Usage:
    python prep/export_unet_diagram.py [--out notebooks/img/S4A_unet300.png]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LEVEL_Y = {0: 0.0, 1: -3.0, 2: -5.4, 3: -7.3}             # vertical position of each resolution level
SIZE = {0: 300, 1: 150, 2: 75, 3: 37}
HEIGHT = {0: 2.6, 1: 1.9, 2: 1.4, 3: 1.0}
COLORS = {"conv": "#1f77b4", "pool": "#d62728", "up": "#2ca02c", "skip": "#7f7f7f", "out": "#17becf"}


def width(ch):
    return 0.1 + 0.006 * ch


def box(ax, x, level, ch, fc="#9ecae1", label=True, size_label=False):
    y, h, wd = LEVEL_Y[level], HEIGHT[level], width(ch)
    ax.add_patch(Rectangle((x, y - h / 2), wd, h, fc=fc, ec="black", lw=1))
    if label:
        ax.text(x + wd / 2, y + h / 2 + 0.12, str(ch), ha="center", va="bottom", fontsize=10)
    if size_label:
        ax.text(x - 0.08, y, f"{SIZE[level]}×{SIZE[level]}", ha="right", va="center", fontsize=9, rotation=90)
    return x + wd


def arrow(ax, x0, y0, x1, y1, kind, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=COLORS[kind],
                                 lw=2, ls=ls, shrinkA=1, shrinkB=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "notebooks" / "img" / "S4A_unet300.png")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7.6))
    top = {lv: LEVEL_Y[lv] + HEIGHT[lv] / 2 for lv in LEVEL_Y}
    bottom = {lv: LEVEL_Y[lv] - HEIGHT[lv] / 2 for lv in LEVEL_Y}
    # encoder: input, then one conv per level and max pooling between levels
    e0 = box(ax, 0.0, 0, 3, fc="#fdd0a2", size_label=True)
    ax.text(width(3) / 2, bottom[0] - 0.15, "parche\nRGB", ha="center", va="top", fontsize=9)
    enc = {0: (1.3, 16), 1: (2.9, 32), 2: (4.5, 64), 3: (6.6, 64)}
    arrow(ax, e0, 0, enc[0][0], 0, "conv")
    for lv, (x, ch) in enc.items():
        box(ax, x, lv, ch, size_label=lv > 0)
        if lv < 3:
            nx = enc[lv + 1][0]
            arrow(ax, x + width(ch) / 2, bottom[lv], nx + width(enc[lv + 1][1]) / 2, top[lv + 1], "pool")
    # decoder: upsampled map (blue) next to the copy from the encoder (white), then one conv
    dec = {2: (8.5, 64, 32), 1: (10.6, 32, 16), 0: (12.7, 16, 16)}     # x, channels upsampled, channels out
    prev_x, prev_ch, prev_lv = enc[3][0], 64, 3
    for lv, (x, ch_up, ch_out) in dec.items():
        arrow(ax, prev_x + width(prev_ch) / 2, top[prev_lv], x + width(ch_up) / 2, bottom[lv], "up")
        eu = box(ax, x, lv, ch_up, label=False)
        ch_skip = enc[lv][1]
        es = box(ax, eu, lv, ch_skip, fc="white", label=False)
        ax.text((x + es) / 2, top[lv] + 0.12, f"{ch_up}+{ch_skip}", ha="center", va="bottom", fontsize=10)
        xs = enc[lv][0] + width(ch_skip)
        arrow(ax, xs, LEVEL_Y[lv], x, LEVEL_Y[lv], "skip", ls="--")
        ax.text((xs + x) / 2, LEVEL_Y[lv] - 0.1, f"copia y concatena\n({ch_up + ch_skip} canales)", ha="center",
                va="top", fontsize=9, color=COLORS["skip"])
        arrow(ax, es, LEVEL_Y[lv], es + 0.55, LEVEL_Y[lv], "conv")
        box(ax, es + 0.55, lv, ch_out)
        prev_x, prev_ch, prev_lv = es + 0.55, ch_out, lv
    xo = prev_x + width(16) + 0.55
    arrow(ax, prev_x + width(16), 0, xo, 0, "out")
    eo = box(ax, xo, 0, 3, fc="#c7e9c0")
    ax.text(xo + width(3) / 2, bottom[0] - 0.15, "logits\nz", ha="center", va="top", fontsize=9)
    arrow(ax, eo, 0, eo + 1.0, 0, "out", ls=":")
    ax.text(eo + 0.5, 0.12, "softmax", ha="center", va="bottom", fontsize=9, color=COLORS["out"])
    ep = box(ax, eo + 1.0, 0, 3, fc="white")
    ax.text(eo + 1.0 + width(3) / 2, bottom[0] - 0.15, "P(fondo)\nP(tejido)\nP(artefacto)", ha="center", va="top",
            fontsize=9)
    x1, x4, xd2, xd0 = enc[0][0], enc[3][0], dec[2][0], dec[0][0]
    ax.text(3.0, bottom[3] - 0.3, "codificador:\n¿qué hay en la imagen?", ha="center", va="top", fontsize=12,
            fontweight="bold")
    ax.text(10.4, bottom[3] - 0.3, "decodificador:\n¿dónde está cada cosa?", ha="center", va="top", fontsize=12,
            fontweight="bold")
    leyenda = [Line2D([], [], color=COLORS["conv"], lw=2.5, label="conv 3×3 + ReLU"),
               Line2D([], [], color=COLORS["pool"], lw=2.5, label="max pooling 2×2 (reduce a la mitad)"),
               Line2D([], [], color=COLORS["up"], lw=2.5, label="interpolación bilineal ×2 (duplica)"),
               Line2D([], [], color=COLORS["skip"], lw=2.5, ls="--", label="conexión de salto (skip)"),
               Line2D([], [], color=COLORS["out"], lw=2.5, label="conv 1×1 → 3 clases")]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=10, frameon=False)
    ax.set_xlim(-0.6, ep + 0.6)
    ax.set_ylim(bottom[3] - 1.4, top[0] + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("U-Net de este notebook (UNet300): 111.315 parámetros. El número sobre cada caja es la cantidad "
                 "de canales;\na la izquierda, el tamaño en píxeles de cada nivel", fontsize=11, loc="left")
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print(args.out, args.out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
