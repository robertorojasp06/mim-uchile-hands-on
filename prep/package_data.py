"""Zip each dataset in data/ into dist/<name>.zip, the files the notebooks download from DATA_URL.

Each zip contains a top-level folder <name>/ (e.g. holter/records.csv), so the notebooks can
extract it next to other datasets.

Usage:
    python prep/package_data.py
"""
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["holter", "wsi", "ct"]


def main():
    dist = REPO / "dist"
    dist.mkdir(exist_ok=True)
    for name in DATASETS:
        src = REPO / "data" / name
        out = dist / f"{name}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(src.parent))
        print(f"{out.relative_to(REPO)}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
