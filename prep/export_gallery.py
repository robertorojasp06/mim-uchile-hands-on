"""Export the nnU-Net error gallery for activity 5-s2.

Source: renders of nnU-Net predictions on HCUCH CT (liver lesions and lung nodules): an axial slice with
the reference contour (red) and the prediction (yellow fill), and 3D renders of reference and prediction.
Writes data/galeria/:
- caso_XX_corte.png, caso_XX_ref3d.jpg, caso_XX_pred3d.jpg
- casos.csv   caso, organo, px_por_cm (for the scale bar)

The axial renders are flipped to radiological orientation (anterior up) and their burned-in scale bar is
cropped out (it is redrawn in the notebook). Files are renamed: the source names are DICOM series UIDs.

Usage:
    python prep/export_gallery.py [--src DIR] [--out data/galeria]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SRC = Path.home() / "nnunet-final-predictions"
REPO = Path(__file__).resolve().parents[1]
ORGANS = {"liver": "hígado (lesiones)", "lung": "pulmón (nódulos)"}


def scale_bar(img):
    """Length in px of the longest white horizontal run in the bottom 20 % (the 10 cm bar), and its top row."""
    a = np.asarray(img.convert("L"))
    h = a.shape[0]
    best, row = 0, h
    for r in range(int(0.8 * h), h):
        white = np.r_[0, (a[r] > 200).astype(int), 0]
        d = np.diff(white)
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts):
            run = (ends - starts).max()
            if run > best:
                best, row = run, r
    return best, row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "galeria")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, k = [], 0
    for organ, organ_es in ORGANS.items():
        for f in sorted((args.src / organ / "slice-all-overlaid").glob("*.png")):
            k += 1
            caso = f"caso_{k:02d}"
            img = Image.open(f).convert("RGB")
            bar_px, bar_row = scale_bar(img)
            top = bar_row - int(0.09 * img.height)            # cut above the "10 cm" label and the bar
            sl = img.crop((0, 0, img.width, top)).transpose(Image.FLIP_TOP_BOTTOM)
            factor = 900 / sl.width
            sl.resize((900, int(sl.height * factor)), Image.LANCZOS).save(args.out / f"{caso}_corte.png", optimize=True)
            for kind, name in (("ct-gt-overlaid", "ref3d"), ("ct-prediction-overlaid", "pred3d")):
                Image.open(args.src / organ / kind / f.name).convert("RGB").resize((560, 560), Image.LANCZOS).save(
                    args.out / f"{caso}_{name}.jpg", quality=90)
            rows.append({"caso": caso, "organo": organ_es, "px_por_cm": bar_px * factor / 10})
    pd.DataFrame(rows).to_csv(args.out / "casos.csv", index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
