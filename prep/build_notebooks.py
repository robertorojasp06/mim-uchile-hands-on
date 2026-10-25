"""Build Colab notebooks (notebooks/*.ipynb) from percent-format sources (notebooks/src/*.py).

Source format:
    # %% [markdown]          markdown cell; each line starts with "# " (or is "#")
    # %%                     code cell
    # INCLUDE: _setup.py     (inside a code cell) replaced by notebooks/src/_setup.py

Code cells whose first line is a Colab form title ("#@title ...") are marked with
cellView=form, so Colab shows the form and hides the code until the student opens it.

Usage:
    python prep/build_notebooks.py
"""
import re
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "notebooks" / "src"
OUT = REPO / "notebooks"


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if not body:
            return
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cell = nbformat.v4.new_code_cell(body)
            if body.lstrip().startswith("#@title"):
                cell.metadata["cellView"] = "form"
            cells.append(cell)

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif line.startswith("# INCLUDE: "):
            buf.extend((SRC / line.split(": ", 1)[1].strip()).read_text().splitlines())
        else:
            buf.append(line)
    flush()
    return cells


def main():
    for src in sorted(SRC.glob("S*.py")):
        nb = nbformat.v4.new_notebook()
        nb.cells = parse(src.read_text())
        nb.metadata = {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        }
        out = OUT / f"{src.stem}.ipynb"
        nbformat.write(nb, out)
        print(f"{out.relative_to(REPO)}: {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
