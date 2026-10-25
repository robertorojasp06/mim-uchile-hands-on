"""Export MedSAM image embeddings and a TorchScript box-prompt decoder for activity 5A.

MedSAM (Ma et al., Nat Commun 2024; weights: huggingface.co/wanglab/medsam-vit-base, Apache-2.0)
splits into a heavy image encoder (ViT-B, ~18 s per image on CPU) and a light prompt encoder +
mask decoder (~50 ms). We run the encoder here, once, and ship its embeddings; students only run
the decoder, so the activity works on CPU without installing anything.

Writes to data/medsam/:
- decoder.pt          TorchScript: (embedding (1,256,64,64), box (1,4) in 1024-px coords) -> logits (1,256,256)
- <structure>.npz     ct (n,H,W) int16 HU crop, exp1/exp2 (n,H,W) uint8, emb (n,256,64,64) float16,
                      spacing, z (slice index), crop (row0, col0, size)

Needs: torch==2.5.1, transformers==4.46.3, nibabel (a separate venv; see README).
Usage:
    python prep/export_medsam.py [--src DIR] [--out data/medsam]
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from transformers import SamModel

SRC = Path("/media/robber/Nuevo vol2/datasets/fondef-ID23I10337/hcuch/annotation-on-PETCT/original")
REPO = Path(__file__).resolve().parents[1]
EXPERT1, EXPERT2 = "Gonzalo", "Maria Fernanda"
# structure -> (study, labels expert 1, labels expert 2, keep only the largest 3D component)
STRUCTURES = {
    "masa_pulmonar": ("13b40a817b", [3], [3], False),
    "lesion_hepatica": ("13b40a817b", [2], [1], True),
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BoxDecoder(torch.nn.Module):
    """Prompt encoder + mask decoder of SAM/MedSAM for a single box prompt."""

    def __init__(self, sam):
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        with torch.no_grad():
            self.register_buffer("pe", sam.get_image_wide_positional_embeddings())

    def forward(self, emb, box):
        sparse, dense = self.prompt_encoder(input_points=None, input_labels=None,
                                            input_boxes=box[None], input_masks=None)
        low, _, _ = self.mask_decoder(image_embeddings=emb, image_positional_embeddings=self.pe,
                                      sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                      multimask_output=False)
        return low[:, 0, 0]


def load(expert, study, labels):
    d = next((SRC / expert).glob(f"PETCT_{study}*"))
    img = nib.as_closest_canonical(nib.load(d / "imaging.nii.gz"))
    seg = nib.as_closest_canonical(nib.load(d / "segmentation.nii.gz"))
    return np.asarray(img.dataobj), np.isin(np.asarray(seg.dataobj), labels), img.header.get_zooms()


def largest_component(mask):
    lab, n = ndimage.label(mask)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else mask


def radiological(a):
    return a.T[::-1, ::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "medsam")
    ap.add_argument("--margin", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    sam = SamModel.from_pretrained("wanglab/medsam-vit-base").eval()
    for name, (study, l1, l2, largest) in STRUCTURES.items():
        ct, m1, zooms = load(EXPERT1, study, l1)
        _, m2, _ = load(EXPERT2, study, l2)
        if largest:
            comp = largest_component(m1 | m2)
            m1, m2 = m1 & comp, m2 & comp
        z = np.flatnonzero((m1 | m2).any(axis=(0, 1)))
        CT = np.stack([radiological(ct[:, :, k]) for k in z])
        E1 = np.stack([radiological(m1[:, :, k]) for k in z])
        E2 = np.stack([radiological(m2[:, :, k]) for k in z])

        rows, cols = np.nonzero((E1 | E2).any(0))
        size = int(max(np.ptp(rows), np.ptp(cols))) + 2 * args.margin
        r0 = int(np.clip((rows.min() + rows.max()) // 2 - size // 2, 0, CT.shape[1] - size))
        c0 = int(np.clip((cols.min() + cols.max()) // 2 - size // 2, 0, CT.shape[2] - size))
        crop = (slice(None), slice(r0, r0 + size), slice(c0, c0 + size))
        CT, E1, E2 = CT[crop], E1[crop], E2[crop]

        emb = []
        for i, sl in enumerate(CT):
            x = torch.tensor(np.clip((sl + 160) / 400, 0, 1), dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
            x = (F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False) - MEAN) / STD
            with torch.no_grad():
                emb.append(sam.get_image_embeddings(x)[0].numpy().astype(np.float16))
            print(f"{name}: slice {i + 1}/{len(CT)}", flush=True)

        np.savez_compressed(args.out / f"{name}.npz", ct=np.clip(CT, -1024, 3071).astype(np.int16),
                            exp1=E1.astype(np.uint8), exp2=E2.astype(np.uint8), emb=np.stack(emb), z=z,
                            spacing=np.array([zooms[1], zooms[0], zooms[2]], dtype=np.float32),
                            crop=np.array([r0, c0, size]))

    dec = BoxDecoder(sam).eval()
    example = (torch.zeros(1, 256, 64, 64), torch.tensor([[100.0, 100.0, 400.0, 400.0]]))
    traced = torch.jit.trace(dec, example)
    traced.save(str(args.out / "decoder.pt"))
    print("decoder:", (args.out / "decoder.pt").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
