"""Export a small, fixed subset of H&E patches and the colon U-Nets for activity 4A.

Writes to data/wsi/:

- patches/<organo>/<aumento>/<region>_<k>.png and ..._mask.png   (RGB uint8; mask 0/1/2)
- index.csv                        organo, aumento, region, categoria, k, image, mask
- models/unet_colon_<aumento>.pt   state_dict of the 300x300 colon U-Net trained at that magnification

Patches come from the *testing* split of deeppath-ml-tissue-segmentation, so the colon models
never saw them. Original file names (which carry ROI identifiers) are not kept.

Usage:
    python prep/export_wsi.py [--src DIR] [--out data/wsi] [--per-region 3]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SRC = Path("/media/robber/Nuevo vol1/deeppath-ml-tissue-segmentation")
REPO = Path(__file__).resolve().parents[1]
MAGS = ["10x", "20x", "40x"]
ORGANS = {"colon": "colon", "gastric": "gastrico"}
MODELS = {"10x": "10x/final/2021_06_07", "20x": "20x/final/2021_06_05", "40x": "40x/final/2021_06_05"}
REGIONS = {  # source name -> (spanish name, category)
    "mucosa": ("mucosa", "tejido"), "submucosa": ("submucosa", "tejido"),
    "muscularis": ("muscular", "tejido"), "adipose": ("adiposo", "tejido"),
    "edges": ("bordes_tejido", "tejido"), "background": ("fondo", "fondo"),
    "air_bubbles": ("burbujas_aire", "artefacto"), "chinese_ink": ("tinta_china", "artefacto"),
    "out_of_focus": ("desenfoque", "artefacto"), "pen_marks": ("marcas_lapiz", "artefacto"),
    "shadows": ("sombras", "artefacto"), "slide_holder_edges": ("borde_portaobjetos", "artefacto"),
    "tissue_folds": ("pliegues_tejido", "artefacto"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "wsi")
    ap.add_argument("--per-region", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for organ, organ_es in ORGANS.items():
        for mag in MAGS:
            base = args.src / "data" / organ / "patches" / "300x300" / mag / "testing"
            for region, (region_es, cat) in REGIONS.items():
                samples = sorted((base / region / "samples").glob("*.tif"))
                if not samples:
                    continue
                for k, s in enumerate(rng.choice(samples, args.per_region, replace=False)):
                    m = s.parent.parent / "masks" / s.name.replace("sample", "mask", 1)
                    out = args.out / "patches" / organ_es / mag
                    out.mkdir(parents=True, exist_ok=True)
                    img, msk = f"{region_es}_{k}.png", f"{region_es}_{k}_mask.png"
                    Image.fromarray(np.asarray(Image.open(s))[..., :3]).save(out / img, optimize=True)
                    Image.fromarray(np.asarray(Image.open(m)).astype(np.uint8)).save(out / msk, optimize=True)
                    rows.append(dict(organo=organ_es, aumento=mag, region=region_es, categoria=cat, k=k,
                                     image=f"patches/{organ_es}/{mag}/{img}",
                                     mask=f"patches/{organ_es}/{mag}/{msk}"))
    pd.DataFrame(rows).to_csv(args.out / "index.csv", index=False)

    (args.out / "models").mkdir(parents=True, exist_ok=True)
    for mag, sub in MODELS.items():
        shutil.copy(args.src / "models" / "colon" / "unet_300x300" / sub / "model_cpu.pth",
                    args.out / "models" / f"unet_colon_{mag}.pt")
    print(pd.DataFrame(rows).groupby(["organo", "aumento"]).size())


if __name__ == "__main__":
    main()
