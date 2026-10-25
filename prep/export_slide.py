"""Export a whole-slide image (WSI) in light form for activity 5B.

From one colon H&E NDPI slide (DeepPath, Hamamatsu NanoZoomer, 40x, 0.2265 um/px) writes data/lamina/:
- miniatura.jpg      the slide at 0.625x (pyramid level 6)
- region_A.jpg, region_B.jpg   3000 x 3000 px regions at 10x (level 2)
- info.json          pyramid levels, pixel size, region positions (level-0 coordinates)
- unet_colon_10x.pt  the 10x colon U-Net (same as in data/wsi)

Only the left part of the glass is kept: the right part has handwritten marks that could carry a
case number. Associated images (macro, label) are never read.
Needs openslide (e.g. the deeppath-ml-tissue-segmentation .venv).

Usage:
    python prep/export_slide.py [--slide FILE] [--out data/lamina]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import openslide

SLIDE = Path("/media/robber/Nuevo vol1/deeppath-ml-tissue-segmentation/data/colon/slides/evaluation/unlabelled/"
             "586-UCH-2-ID - 2021-09-09 10.44.34.ndpi")
REPO = Path(__file__).resolve().parents[1]
KEEP_WIDTH = 135680            # level-0 columns kept (handwriting starts to the right)
THUMB_LEVEL, REGION_LEVEL, REGION_SIZE = 6, 2, 3000
REGIONS = {"A": (35840, 95104), "B": (47360, 36224)}   # level-0 top-left corners


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide", type=Path, default=SLIDE)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "lamina")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = openslide.OpenSlide(str(args.slide))
    mpp = float(s.properties["openslide.mpp-x"])
    obj = float(s.properties["openslide.objective-power"])
    height0 = s.level_dimensions[0][1]

    ds = s.level_downsamples[THUMB_LEVEL]
    thumb = s.read_region((0, 0), THUMB_LEVEL, (int(KEEP_WIDTH / ds), int(height0 / ds))).convert("RGB")
    thumb.save(args.out / "miniatura.jpg", quality=92)

    for name, (x0, y0) in REGIONS.items():
        assert x0 + REGION_SIZE * s.level_downsamples[REGION_LEVEL] <= KEEP_WIDTH
        s.read_region((x0, y0), REGION_LEVEL, (REGION_SIZE, REGION_SIZE)).convert("RGB").save(
            args.out / f"region_{name}.jpg", quality=92)

    levels = [{"nivel": i, "reduccion": float(d), "aumento": obj / d, "um_por_pixel": mpp * d,
               "ancho": int(np.ceil(KEEP_WIDTH / d)), "alto": int(np.ceil(height0 / d))}
              for i, d in enumerate(s.level_downsamples)]
    info = {"um_por_pixel_nivel0": mpp, "aumento_nivel0": obj, "niveles": levels,
            "miniatura": {"nivel": THUMB_LEVEL, "reduccion": float(ds)},
            "regiones": {n: {"x0": x, "y0": y, "nivel": REGION_LEVEL, "lado": REGION_SIZE,
                             "reduccion": float(s.level_downsamples[REGION_LEVEL])} for n, (x, y) in REGIONS.items()}}
    shutil.copy(args.slide.parents[5] / "models" / "colon" / "unet_300x300" / "10x" / "final" / "2021_06_07" / "model_cpu.pth",
                args.out / "unet_colon_10x.pt")
    (args.out / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(thumb.size, [f.name for f in args.out.iterdir()])


if __name__ == "__main__":
    main()
