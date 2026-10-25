"""Export CT slices segmented by two radiologists for activity 4-s2.

Source: HCUCH PET-CT studies (CT in HU, 2.04 x 2.04 x 3 mm) where the same structures were
segmented independently by two experts. For each structure, writes data/ct/<structure>.npz with

- ct      (n, H, W) int16, HU, axial slices in radiological orientation (anterior up, patient's right on the left)
- exp1    (n, H, W) uint8, expert 1 mask
- exp2    (n, H, W) uint8, expert 2 mask
- z       (n,) slice index in the original volume
- spacing (row_mm, col_mm, slice_mm)
- vol_*   3D metrics on the full volume (Dice, IoU, volumes in ml)

Study identifiers are replaced by structure names; experts are anonymised as "Experto 1/2".
Needs nibabel (e.g. ~/anaconda3/envs/nnunet/bin/python).

Usage:
    python prep/export_ct.py [--src DIR] [--out data/ct] [--max-slices 8]
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

SRC = Path("/media/robber/Nuevo vol2/datasets/fondef-ID23I10337/hcuch/annotation-on-PETCT/original")
REPO = Path(__file__).resolve().parents[1]
EXPERT1, EXPERT2 = "Gonzalo", "Maria Fernanda"
# structure -> (study, labels expert 1, labels expert 2)
STRUCTURES = {
    "masa_pulmonar": ("13b40a817b", [3], [3]),
    "lesiones_hepaticas_grandes": ("15a205ffcc", [2], [1]),
    "lesiones_hepaticas_pequenas": ("13b40a817b", [2], [1]),
    "adenopatias": ("6a3477cd9a", [1], [1, 2]),
}


def load(expert, study, labels):
    d = next((SRC / expert).glob(f"PETCT_{study}*"))
    seg = nib.as_closest_canonical(nib.load(d / "segmentation.nii.gz"))
    img = nib.as_closest_canonical(nib.load(d / "imaging.nii.gz"))
    return np.asarray(img.dataobj), np.isin(np.asarray(seg.dataobj), labels), img.header.get_zooms()


def radiological(a):
    """(x, y) slice in RAS -> (rows, cols) with anterior up and patient's right on the left."""
    return a.T[::-1, ::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "ct")
    ap.add_argument("--max-slices", type=int, default=8)
    ap.add_argument("--margin", type=int, default=30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, (study, l1, l2) in STRUCTURES.items():
        ct, m1, zooms = load(EXPERT1, study, l1)
        _, m2, _ = load(EXPERT2, study, l2)
        both = m1 & m2
        voxel_ml = float(np.prod(zooms)) / 1000.0
        dice3d = 2 * both.sum() / (m1.sum() + m2.sum())
        iou3d = both.sum() / (m1 | m2).sum()

        # slices annotated by both experts, spread over the extent of the structure
        zs = np.flatnonzero(both.any(axis=(0, 1)))
        z = np.unique(zs[np.linspace(0, len(zs) - 1, min(args.max_slices, len(zs))).round().astype(int)])

        sl = lambda v: np.stack([radiological(v[:, :, k]) for k in z])
        CT, E1, E2 = sl(ct), sl(m1), sl(m2)
        rows, cols = np.nonzero((E1 | E2).any(0))
        r0, r1 = max(rows.min() - args.margin, 0), min(rows.max() + args.margin + 1, CT.shape[1])
        c0, c1 = max(cols.min() - args.margin, 0), min(cols.max() + args.margin + 1, CT.shape[2])
        crop = (slice(None), slice(r0, r1), slice(c0, c1))

        np.savez_compressed(
            args.out / f"{name}.npz",
            ct=np.clip(CT[crop], -1024, 3071).astype(np.int16),
            exp1=E1[crop].astype(np.uint8), exp2=E2[crop].astype(np.uint8), z=z,
            spacing=np.array([zooms[1], zooms[0], zooms[2]], dtype=np.float32),
            vol_dice=dice3d, vol_iou=iou3d,
            vol_ml_exp1=m1.sum() * voxel_ml, vol_ml_exp2=m2.sum() * voxel_ml,
        )
        print(f"{name}: slices={len(z)} crop={CT[crop].shape[1:]} Dice3D={dice3d:.3f} "
              f"vol1={m1.sum() * voxel_ml:.1f} ml vol2={m2.sum() * voxel_ml:.1f} ml")


if __name__ == "__main__":
    main()
