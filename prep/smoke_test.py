"""Run a notebook source (notebooks/src/*.py) headless, as a quick check before class.

Every ipywidgets interact/interact_manual call is executed once with the widgets' default values
(plus optional overrides), printed outputs go to the terminal and figures are saved as PNG.
Uses data/ if present (run from notebooks/), otherwise downloads from sasiba and needs the
password in the environment variable CLAVE_DATOS.

Usage (from notebooks/):
    python ../prep/smoke_test.py src/S4A_unet_patologia.py [OUT_DIR] ["{'inferencia': [{'fila': 20, 'columna': 280}]}"]
"""
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import ipywidgets as w  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

src = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("smoke_test_figs")
extra = eval(sys.argv[3]) if len(sys.argv) > 3 else {}   # {function name: [overrides, ...]}
out.mkdir(parents=True, exist_ok=True)
n_fig = 0


def show(*args, **kwargs):
    global n_fig
    for n in plt.get_fignums():
        n_fig += 1
        plt.figure(n).savefig(out / f"{src.stem}_{n_fig:02d}.png", dpi=60)
    plt.close("all")


def display(*objs, **kwargs):
    for o in objs:
        if hasattr(o, "to_string"):
            print(o.to_string()[:2000])
        elif hasattr(o, "data"):        # pandas Styler: render it to catch format errors, print the data
            o.to_html()
            print(o.data.to_string()[:2000])
        else:
            print(repr(o)[:500])


def interact(f, **widgets):
    for over in [{}] + extra.get(f.__name__, []):
        kwargs = {k: v.value for k, v in widgets.items()}
        kwargs.update(over)
        t = time.time()
        f(**kwargs)
        show()
        print(f"[interact {f.__name__} {over} -> {time.time() - t:.2f} s]")


plt.show = show
w.interact = w.interact_manual = interact
text = re.sub(r"^# INCLUDE: (.*)$", lambda m: (src.parent / m.group(1)).read_text(), src.read_text(), flags=re.M)
text = re.sub(r'^CLAVE_DATOS = ""', f'CLAVE_DATOS = {os.environ.get("CLAVE_DATOS", "")!r}', text, flags=re.M)
ns = {"display": display, "__name__": "__main__"}
for i, cell in enumerate(re.split(r"^# %%.*$", text, flags=re.M)):
    if all(line.startswith("#") or not line.strip() for line in cell.splitlines()):
        continue
    cell = cell.replace("from IPython.display import clear_output, display", "from IPython.display import clear_output")
    t = time.time()
    exec(compile(cell, f"{src.name}:cell{i}", "exec"), ns)
    show()
    print(f"[cell {i} -> {time.time() - t:.1f} s]")
