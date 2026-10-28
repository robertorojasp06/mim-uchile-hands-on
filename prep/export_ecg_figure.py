"""Make the introductory figure of activity 4B: ~10 s of a 3-channel Holter ECG with its beats labelled.

Reads one ITMed/Pathfinder record (ISHNE .ecg + *_beats.csv) from the same source as export_holter.py
and writes notebooks/img/S4B_holter_segmento.png. The signal is read from byte 0, the convention that
aligns the beat annotations; the segment is far from the ISHNE header, which is never decoded except for
the amplitude resolution. The record is shown with its course ID (H01..H24) and time from the start of
the recording, never the clock time.

Usage:
    python prep/export_ecg_figure.py [--record 0198] [--start 23040.6] [--out notebooks/img/S4B_holter_segmento.png]
    python prep/export_ecg_figure.py --record 0198 --candidates DIR    # draw candidate segments to choose one
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_holter import SRC, clean_channel, read_header  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RECORD, START, LENGTH = "0198", 23040.6, 10.0     # train record H08; start in s from the beginning of the file
COLORS = {"N": "#4d4d4d", "S": "#e08214", "V": "#c51b7d"}
NAMES = {"N": "N · normal", "S": "S · extrasístole supraventricular", "V": "V · extrasístole ventricular"}


def load(record):
    d = SRC / record
    fs = int(json.loads((d / f"{record}_metadata.json").read_text())["fs"])
    amp = read_header(d / f"{record}.ecg")["amp_nv"] * 1e-6                  # mV per count
    raw = np.fromfile(d / f"{record}.ecg", "<i2").reshape(-1, 3).astype(np.float64)
    beats = pd.read_csv(d / f"{record}_beats.csv")
    return fs, amp, raw, beats


def course_id(record):
    ids = pd.read_csv(REPO / "prep" / "private" / "holter_id_map.csv", dtype=str)
    return ids.set_index("original").id[record]


def draw(fs, amp, raw, beats, start, title, out):
    a, b = int(start * fs), int((start + LENGTH) * fs)
    pad = 10 * fs                                                            # filter on a wider span, then crop
    seg = np.stack([clean_channel(raw[a - pad:b + pad, c] * amp, fs)[pad:-pad] for c in range(3)])
    t = np.arange(a, b) / fs - start
    bt = beats[(beats.sample_index >= a) & (beats.sample_index < b)]

    fig, ejes = plt.subplots(3, 1, figsize=(14, 6.4), sharex=True)
    for c, e in enumerate(ejes):
        y = seg[c]
        lo, hi = y.min(), y.max()
        e.set_ylim(lo - 0.25 * (hi - lo), hi + 0.3 * (hi - lo))
        for k in np.arange(0, LENGTH + 1e-9, 0.2):                         # ECG paper: 0.2 s major grid
            e.axvline(k, color="#f4c7c3", lw=0.6, zorder=0)
        e.grid(axis="y", color="#f4c7c3", lw=0.6)
        e.plot(t, y, color="black", lw=0.9)
        e.set_ylabel(f"canal {c + 1}\n(mV)")
        for _, r in bt.iterrows():
            x = r.sample_index / fs - start
            e.axvline(x, color=COLORS[r.code], lw=1 if r.code == "N" else 1.6, alpha=0.35 if r.code == "N" else 0.8,
                      ls="--", zorder=1)
            if c == 0:
                e.text(x, e.get_ylim()[1], r.code, ha="center", va="top", fontsize=13, fontweight="bold",
                       color=COLORS[r.code])
    rr = np.diff(bt.sample_index.to_numpy()) / fs
    for (x0, x1), v in zip(zip(bt.sample_index.to_numpy()[:-1] / fs - start, bt.sample_index.to_numpy()[1:] / fs - start), rr):
        ejes[-1].text((x0 + x1) / 2, ejes[-1].get_ylim()[0], f"{v:.2f} s", ha="center", va="bottom", fontsize=8,
                      color="#555555")
    ejes[-1].set_xlabel("tiempo (s)")
    ejes[-1].set_xlim(0, LENGTH)
    handles = [plt.Line2D([], [], color=COLORS[k], lw=2, ls="--", label=NAMES[k]) for k in "NSV" if k in set(bt.code)]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955), frameon=False, fontsize=11)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--start", type=float, default=START)
    ap.add_argument("--out", type=Path, default=REPO / "notebooks" / "img" / "S4B_holter_segmento.png")
    ap.add_argument("--candidates", type=Path, help="draw up to 12 candidate segments with N, S and V beats here")
    args = ap.parse_args()
    fs, amp, raw, beats = load(args.record)
    rid = course_id(args.record)

    if args.candidates:
        args.candidates.mkdir(parents=True, exist_ok=True)
        t, code = beats.sample_index.to_numpy() / fs, beats.code.to_numpy()
        found, last = [], -np.inf
        for i in range(len(t)):
            m = (t >= t[i] - 1) & (t < t[i] + LENGTH - 1)
            cc = code[m]
            if t[i] - last > 60 and not set(cc) - set("NSV") and {"S", "V"} <= set(cc) and (cc != "N").sum() <= 4:
                found.append(t[i] - 1); last = t[i]
        for s in found[:: max(1, len(found) // 12)][:12]:
            draw(fs, amp, raw, beats, s, f"{rid} · inicio {s:.1f} s", args.candidates / f"{args.record}_{s:.1f}.png")
        print(len(found), "candidates")
        return

    title = (f"Holter de 3 canales, paciente {rid}: {LENGTH:.0f} s a {fs} Hz con sus latidos anotados "
             f"(letra arriba; bajo el último canal, el intervalo RR entre latidos)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw(fs, amp, raw, beats, args.start, title, args.out)
    print(args.out, args.out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
