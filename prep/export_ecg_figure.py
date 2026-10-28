"""Make the introductory figures of activity 4B from the Holter records.

- notebooks/img/S4B_latido.png: one normal beat (channel 1) with its waves (P, Q, R, S, T) and intervals
  (PR, QRS, QT). The fiducial points are found with simple rules on the beat, upsampled 8x for display.
- notebooks/img/S4B_holter_segmento.png: ~10 s of a 3-channel Holter ECG with its beats labelled.

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
from scipy import signal  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_holter import SRC, clean_channel, read_header  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RECORD, START, LENGTH = "0198", 23040.6, 10.0     # train record H08; start in s from the beginning of the file
BEAT_RECORD, BEAT_TIME = "0135", 10905.19          # train record H24 (normal group): a typical N beat, channel 1
UP = 8                                             # upsampling factor for the single-beat figure
COLORS = {"N": "#7f7f7f", "S": "#1f77b4", "V": "#d62728"}   # same colours as notebook 4B
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


def fiducial_points(y, r, fs):
    """Waves and limits of one beat (y upsampled, sampling rate fs, R peak near index r), in samples."""
    ms = lambda v: int(round(v * fs / 1000))
    r = r - ms(40) + int(np.argmax(y[r - ms(40):r + ms(40)]))
    q = r - ms(60) + int(np.argmin(y[r - ms(60):r]))
    s = r + int(np.argmin(y[r:r + ms(80)]))
    p = r - ms(300) + int(np.argmax(y[r - ms(300):r - ms(80)]))
    tp = r + ms(150) + int(np.argmax(y[r + ms(150):r + ms(450)]))
    base = np.median(y[p + ms(60):q - ms(10)]) if q - ms(10) > p + ms(60) else np.median(y[:ms(80)])
    lim = lambda i, step, frac: next(j for j in range(i, i + step * ms(200), step)
                                     if y[j] - base < frac * (y[i] - base))
    d = np.gradient(y)
    thr = 0.1 * np.abs(d[q:s]).max()
    on = q - next(k for k in range(ms(6), ms(80)) if abs(d[q - k]) < thr)      # Q and S are extremes (slope 0):
    off = s + next(k for k in range(ms(6), ms(120)) if abs(d[s + k]) < thr)    # start a few ms away from them
    k = tp + int(np.argmin(d[tp:tp + ms(150)]))                        # steepest descent of the T wave
    tend = int(k + (base - y[k]) / d[k])                                # tangent meets the baseline
    return dict(P=p, Q=q, R=r, S=s, T=tp, P_ini=lim(p, -1, 0.15), P_fin=lim(p, 1, 0.15), QRS_ini=on, QRS_fin=off,
                T_fin=tend, base=base)


def draw_beat(record, when, out):
    fs, amp, raw, beats = load(record)
    i = int(beats.sample_index.iloc[(beats.sample_index - when * fs).abs().argmin()])
    pre, post = int(0.45 * fs), int(0.6 * fs)
    y = clean_channel(raw[i - 10 * fs:i + 10 * fs, 0] * amp, fs)[10 * fs - pre:10 * fs + post]
    y = signal.resample(y, len(y) * UP)
    fu = fs * UP
    t = (np.arange(len(y)) - pre * UP) / fu * 1000                       # ms from the annotated R peak
    f = fiducial_points(y, pre * UP, fu)
    ms = lambda a, b: (f[b] - f[a]) / fu * 1000

    fig, ax = plt.subplots(figsize=(11, 5.6))
    for k in np.arange(-440, 600, 40):                                    # ECG paper: 40 ms and 0.1 mV
        ax.axvline(k, color="#f4c7c3", lw=0.5 if k % 200 else 1.0, zorder=0)
    for v in np.arange(-0.6, 1.4, 0.1):
        ax.axhline(v, color="#f4c7c3", lw=0.5 if round(v * 10) % 5 else 1.0, zorder=0)
    ax.plot(t, y, color="black", lw=1.8)
    top, low = y.max(), y.min()
    for name, off in (("P", 0.07), ("Q", -0.09), ("R", 0.05), ("S", -0.09), ("T", 0.07)):
        j = f[name]
        ax.plot(t[j], y[j], "o", color="#1f77b4", ms=7, zorder=3)
        ax.text(t[j], y[j] + off, name, ha="center", va="bottom" if off > 0 else "top", fontsize=15,
                fontweight="bold", color="#1f77b4")
    span = top - low
    for k, (a, b, label, color) in enumerate([("P_ini", "QRS_ini", "PR", "#2ca02c"),
                                               ("QRS_ini", "QRS_fin", "QRS", "#d62728"),
                                               ("QRS_ini", "T_fin", "QT", "#9467bd")]):
        yy = low - (0.2 + 0.13 * k) * span
        for j in (f[a], f[b]):
            ax.plot([t[j], t[j]], [yy, y[j]], color=color, lw=0.8, ls=":")
        ax.annotate("", (t[f[a]], yy), (t[f[b]], yy), arrowprops=dict(arrowstyle="<->", color=color, lw=1.8))
        ax.text((t[f[a]] + t[f[b]]) / 2, yy + 0.015 * span, f"{label} = {ms(a, b):.0f} ms", ha="center", va="bottom",
                fontsize=11, color=color, fontweight="bold")
    ax.set_ylim(low - 0.58 * span, top + 0.18 * span)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("tiempo desde el pico R (ms)")
    ax.set_ylabel("mV")
    ax.set_title(f"Un latido normal (N), canal 1 del paciente {course_id(record)}. Cuadrícula: 40 ms × 0,1 mV",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return {k: round(ms("R", k) if k != "base" else f[k], 1) for k in f}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--start", type=float, default=START)
    ap.add_argument("--out", type=Path, default=REPO / "notebooks" / "img" / "S4B_holter_segmento.png")
    ap.add_argument("--candidates", type=Path, help="draw up to 12 candidate segments with N, S and V beats here")
    args = ap.parse_args()
    if not args.candidates:
        beat_out = args.out.parent / "S4B_latido.png"
        print(beat_out, draw_beat(BEAT_RECORD, BEAT_TIME, beat_out))
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
