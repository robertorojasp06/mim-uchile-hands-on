"""Make the figure of activity 4A that shows a whole colon slide segmented by the 10x U-Net.

Uses the stitched output of deeppath-ml-tissue-segmentation (scripts/segment_slides.py) for the same
slide as export_intro_figure.py: stitched/image.tif and stitched/mask.tif (10x tiles of 300 px,
stitched at 1/4, i.e. 1/16 of level 0; mask 0 = fondo, 1 = tejido, 2 = artefacto).
Writes notebooks/img/S4A_lamina_segmentada.jpg:
(a) the upper part of the glass in H&E, (b) its segmentation, (c, d) a zoom of the left end.

Only the upper part of the glass is kept: the lower part has handwritten marks that could carry a
case number. Needs tifffile and matplotlib (e.g. the deeppath-ml-tissue-segmentation .venv).

Usage:
    python prep/export_segmented_figure.py [--seg DIR] [--out notebooks/img/S4A_lamina_segmentada.jpg]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

SEG = Path("/media/robber/Nuevo vol1/deeppath-ml-tissue-segmentation/results/colon/slide_segmentation/10x/"
           "586-UCH-1-ID - 2021-09-14 09.14.51")
REPO = Path(__file__).resolve().parents[1]
DOWN = 16                      # stitched pixel = 16 level-0 pixels
UM_PER_PX = 0.2265 * DOWN
WIDTH0, KEEP_HEIGHT = 208896, 39040   # level-0 slide width; rows kept (handwriting starts below)
ZOOM = (0, 0, 3600, 2440)      # stitched x, y, width, height
CLASSES = ["fondo", "tejido", "artefacto"]
CMAP = ListedColormap(["#eeeeee", "#c2185b", "#1976d2"])   # same colours as the notebook


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", type=Path, default=SEG)
    ap.add_argument("--out", type=Path, default=REPO / "notebooks" / "img" / "S4A_lamina_segmentada.jpg")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    w, h = WIDTH0 // DOWN, KEEP_HEIGHT // DOWN
    image = tifffile.imread(args.seg / "stitched" / "image.tif")[:h, :w]
    mask = tifffile.imread(args.seg / "stitched" / "mask.tif")[:h, :w]
    n_tiles = len(list((args.seg / "masks").glob("*.tif")))
    times = pd.read_csv(args.seg.parent / "slides_processing_times.csv")
    cpu_min = times.loc[times.Filename.str.startswith(args.seg.name), "CPU Segmentation inference execution time (min)"].iloc[0]
    share = {c: 100 * (mask == i).mean() for i, c in enumerate(CLASSES)}

    fig = plt.figure(figsize=(14, 9.4))
    for k, (data, title) in enumerate([
            (image, "(a) Lámina de colon, H&E (parte superior)"),
            (mask, f"(b) Segmentación de la U-Net de 10x: {n_tiles:,} parches de 300 × 300 px, "
                   f"~{cpu_min:.0f} min en CPU (lámina completa)".replace(",", "."))]):
        ax = fig.add_axes([0.01, 0.73 - 0.27 * k, 0.98, 0.22])
        if data.ndim == 3:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        zx, zy, zw, zh = ZOOM
        ax.add_patch(Rectangle((zx, zy), zw, zh, fill=False, ec="black", lw=1.5, ls="--"))
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, loc="left")
        if k == 0:
            mm = 1000 / UM_PER_PX
            ax.plot([60, 60 + 5 * mm], [h - 120] * 2, color="black", lw=4)
            ax.text(60 + 2.5 * mm, h - 180, "5 mm", ha="center", fontsize=11)
            ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(h - 0.5, -0.5)
    ax.legend(handles=[Patch(color=CMAP(i), label=f"{c} ({share[c]:.0f} %)") for i, c in enumerate(CLASSES)],
              loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=3, fontsize=11, frameon=False)

    zx, zy, zw, zh = ZOOM
    for k, title in enumerate(["(c) Zoom: extremo izquierdo", "(d) Su segmentación"]):
        ax = fig.add_axes([0.01 + 0.5 * k, 0.0, 0.48, 0.36])
        ax.imshow(image[zy:zy + zh, zx:zx + zw])
        if k:
            ax.imshow(mask[zy:zy + zh, zx:zx + zw], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest", alpha=0.65)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, loc="left")
    fig.savefig(args.out, dpi=90, bbox_inches="tight", pil_kwargs={"quality": 85})
    print(args.out, args.out.stat().st_size // 1024, "KB", {c: round(v, 1) for c, v in share.items()})


if __name__ == "__main__":
    main()
