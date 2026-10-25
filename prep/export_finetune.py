"""Export the data for activity 5C (adapting the colon U-Net to gastric tissue) and precompute
reference learning curves.

Writes to data/ajuste/:
- datos.npz            train_x/train_y: pool of gastric 20x training patches, ordered round-robin over
                       regions (the first N patches are always the same, balanced over regions);
                       test_x/test_y: gastric 20x test patches; *_region: region of each patch
- unet_colon_20x.pt    starting model (trained on colon only)
- curvas.csv           Dice on the test set vs number of training patches, for fine-tuning and
                       training from scratch, 3 seeds (same training function as the notebook)

Usage:
    python prep/export_finetune.py [--src DIR] [--out data/ajuste] [--skip-curves]
"""
import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

SRC = Path("/media/robber/Nuevo vol1/deeppath-ml-tissue-segmentation")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "src"))
from tissue_segmentation.deeplearning.cnn_architectures import Unet2dSegmentation300x300 as UNet  # noqa: E402

REGIONS_ES = {"mucosa": "mucosa", "submucosa": "submucosa", "muscularis": "muscular", "adipose": "adiposo",
              "edges": "bordes_tejido", "background": "fondo", "air_bubbles": "burbujas_aire",
              "chinese_ink": "tinta_china", "out_of_focus": "desenfoque", "shadows": "sombras",
              "slide_holder_edges": "borde_portaobjetos", "tissue_folds": "pliegues_tejido"}
N_VALUES = [0, 4, 8, 16, 32, 64]
EPOCHS, LR, BATCH = 8, 3e-4, 4


def sample(base, per_region, rng):
    """Round-robin over regions: index k*R..(k+1)*R-1 holds the k-th patch of every region."""
    picked = {r: rng.choice(sorted((base / r / "samples").glob("*.tif")), per_region, replace=False)
              for r in REGIONS_ES}
    x, y, reg = [], [], []
    for k in range(per_region):
        for r, files in picked.items():
            s = files[k]
            m = s.parent.parent / "masks" / s.name.replace("sample", "mask", 1)
            x.append(np.asarray(Image.open(s))[..., :3]); y.append(np.asarray(Image.open(m)).astype(np.uint8))
            reg.append(REGIONS_ES[r])
    return np.stack(x), np.stack(y), np.array(reg)


def dice(pred, ref, c):
    a, b = pred == c, ref == c
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def train_and_eval(args):
    """Same procedure as the notebook: Adam, cross-entropy, random flips/rotations, fixed epochs."""
    n, strategy, seed, data, model_path = args
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = UNet()
    if strategy == "ajuste fino":
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    d = np.load(data)
    x = torch.tensor(d["train_x"][:n]).permute(0, 3, 1, 2).float() / 255
    y = torch.tensor(d["train_y"][:n]).long()
    if n:
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        for _ in range(EPOCHS):
            net.train()
            order = rng.permutation(n)
            for i in range(0, n, BATCH):
                b = order[i:i + BATCH]
                xb, yb = x[b], y[b]
                k = int(rng.integers(4))
                xb, yb = torch.rot90(xb, k, (2, 3)), torch.rot90(yb, k, (1, 2))
                if rng.random() < 0.5:
                    xb, yb = xb.flip(3), yb.flip(2)
                opt.zero_grad()
                F.cross_entropy(net(xb), yb).backward()
                opt.step()
    net.eval()
    tx = torch.tensor(d["test_x"]).permute(0, 3, 1, 2).float() / 255
    with torch.inference_mode():
        pred = torch.cat([net(tx[i:i + 16]).argmax(1) for i in range(0, len(tx), 16)]).numpy()
    ref = d["test_y"]
    return {"n_parches": n, "estrategia": strategy, "semilla": seed,
            "dice_tejido": dice(pred, ref, 1), "dice_artefacto": dice(pred, ref, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "ajuste")
    ap.add_argument("--skip-curves", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    base = args.src / "data" / "gastric" / "patches" / "300x300" / "20x"
    trx, try_, trr = sample(base / "training", 6, rng)     # 72 patches
    tex, tey, ter = sample(base / "testing", 4, rng)       # 48 patches
    np.savez_compressed(args.out / "datos.npz", train_x=trx, train_y=try_, train_region=trr,
                        test_x=tex, test_y=tey, test_region=ter)
    model = args.out / "unet_colon_20x.pt"
    shutil.copy(args.src / "models" / "colon" / "unet_300x300" / "20x" / "final" / "2021_06_05" / "model_cpu.pth", model)
    print("train", trx.shape, "test", tex.shape)

    if args.skip_curves:
        return
    jobs = [(n, s, seed, str(args.out / "datos.npz"), str(model))
            for n in N_VALUES for s in ("ajuste fino", "desde cero") for seed in (0, 1, 2)
            if not (n == 0 and s == "desde cero")]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(train_and_eval, jobs))
    curves = pd.DataFrame(rows)
    curves.to_csv(args.out / "curvas.csv", index=False)
    print(curves.groupby(["estrategia", "n_parches"])[["dice_tejido", "dice_artefacto"]].mean().round(3))


if __name__ == "__main__":
    main()
