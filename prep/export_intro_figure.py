"""Make the introductory figure of activity 4A: a whole colon H&E slide with its artifacts labelled.

From one colon NDPI slide (DeepPath, 40x, 0.2265 um/px) writes notebooks/img/S4A_lamina_colon.jpg:
(a) the upper part of the glass at 1.25x with tissue, background and artifacts labelled,
(b) one tissue fragment at 5x, (c) a 300 x 300 px patch at 20x, the input size of the U-Net.

Only the upper part of the glass is kept: the lower part has handwritten marks that could carry a
case number. Associated images (macro, label) are never read.
Needs openslide and matplotlib (e.g. the deeppath-ml-tissue-segmentation .venv).

Usage:
    python prep/export_intro_figure.py [--slide FILE] [--out notebooks/img/S4A_lamina_colon.jpg]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import openslide  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SLIDE = Path("/media/robber/Nuevo vol1/deeppath-ml-tissue-segmentation/data/colon/slides/evaluation/unlabelled/"
             "586-UCH-1-ID - 2021-09-14 09.14.51.ndpi")
REPO = Path(__file__).resolve().parents[1]
KEEP_HEIGHT = 39040            # level-0 rows kept (handwriting starts below)
OVERVIEW_LEVEL = 5             # 1.25x
FRAGMENT = (46400, 0, 3, 1800, 1400)       # level-0 x, y, level (5x), width, height
PATCH = (51200, 5400, 1, 300)              # level-0 x, y, level (20x), side
LABELS = [  # text, level-0 point, level-0 text position, colour
    ("tejido", (55500, 28500), (44000, 35500), "#c2185b"),
    ("fondo (vidrio)", (80000, 22000), (80000, 31000), "#555555"),
    ("burbuja de aire\nbajo el cubreobjetos", (12000, 7000), (14000, 30000), "#1976d2"),
    ("burbuja de aire", (201000, 8000), (178000, 29000), "#1976d2"),
    ("borde del\nportaobjetos", (120000, 600), (120000, 9000), "#1976d2"),
    ("sombra", (208000, 18000), (194000, 36000), "#1976d2"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide", type=Path, default=SLIDE)
    ap.add_argument("--out", type=Path, default=REPO / "notebooks" / "img" / "S4A_lamina_colon.jpg")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    s = openslide.OpenSlide(str(args.slide))
    mpp = float(s.properties["openslide.mpp-x"])
    width0, height0 = s.level_dimensions[0]
    ds = s.level_downsamples[OVERVIEW_LEVEL]
    overview = s.read_region((0, 0), OVERVIEW_LEVEL, (int(width0 / ds), int(KEEP_HEIGHT / ds))).convert("RGB")
    fx, fy, flevel, fw, fh = FRAGMENT
    fragment = s.read_region((fx, fy), flevel, (fw, fh)).convert("RGB")
    px, py, plevel, side = PATCH
    patch = s.read_region((px, py), plevel, (side, side)).convert("RGB")

    fig = plt.figure(figsize=(14, 6.6))
    ax = fig.add_axes([0.01, 0.53, 0.98, 0.41])
    ax.imshow(overview, extent=(0, width0, KEEP_HEIGHT, 0))
    for text, (x, y), (tx, ty), color in LABELS:
        ax.annotate(text, (x, y), (tx, ty), color=color, fontsize=11, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    fd = s.level_downsamples[flevel]
    ax.add_patch(Rectangle((fx, fy), fw * fd, fh * fd, fill=False, ec="black", lw=1.5, ls="--"))
    ax.text(fx + fw * fd + 800, 1500, "(b)", fontsize=12, fontweight="bold", va="top")
    mm = 1000 / mpp                     # level-0 pixels per mm
    ax.plot([3000, 3000 + 5 * mm], [KEEP_HEIGHT - 2500] * 2, color="black", lw=4)
    ax.text(3000 + 2.5 * mm, KEEP_HEIGHT - 3300, "5 mm", ha="center", fontsize=11)
    ax.set_axis_off()
    ax.set_title(f"(a) Lámina completa de colon, H&E (parte superior). A 40x mide {width0:,} × {height0:,} píxeles "
                 f"≈ {width0 * height0 / 1e9:.0f} gigapíxeles".replace(",", "."), fontsize=12, loc="left")

    ax = fig.add_axes([0.04, 0.03, 0.45, 0.42])
    ax.imshow(fragment, extent=(0, fw * fd * mpp / 1000, fh * fd * mpp / 1000, 0))
    pd_ = s.level_downsamples[plevel]
    ax.add_patch(Rectangle(((px - fx) * mpp / 1000, (py - fy) * mpp / 1000), side * pd_ * mpp / 1000,
                           side * pd_ * mpp / 1000, fill=False, ec="black", lw=2))
    ax.set_xlabel("mm")
    ax.tick_params(labelsize=9)
    ax.set_title("(b) Un fragmento de tejido a 5x: mucosa con criptas", fontsize=12, loc="left")

    ax = fig.add_axes([0.55, 0.03, 0.4, 0.42])
    ax.imshow(patch)
    ax.set_xticks([0, 150, 299])
    ax.set_yticks([0, 150, 299])
    ax.tick_params(labelsize=9)
    ax.set_title(f"(c) Parche de 300 × 300 px a 20x ({side * pd_ * mpp:.0f} µm):\nlo que ve la U-Net",
                 fontsize=12, loc="left")
    fig.savefig(args.out, dpi=90, bbox_inches="tight", pil_kwargs={"quality": 85})
    print(args.out, args.out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
