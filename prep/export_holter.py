"""Export a de-identified Holter dataset for the session 4 hands-on (4B, 4C, 4-s1).

Reads the ITMed/Pathfinder records (ISHNE .ecg + *_beats.csv + *_metadata.json),
and writes to data/holter/:

- records.csv        one row per record: new ID, group, split, age band, sex, recorder, counts
- windows.npz        4-s ECG windows (channel 1, mV, float16) centred on N/S/V beats, for 4B
- features.parquet   hand-crafted beat features for 4C/4-s1:
                     train/val records -> sampled beats; test records -> every beat (24 h)
- pacientes.npz      per-patient histogram of out-of-fold beat scores (all 24 records), for 5-s1

The ISHNE header (patient name, ID, birth date) is never copied: the signal is read with the
same convention used to align the annotations (from byte 0), and every sample that overlaps
the header is discarded. Original record names are replaced by H01..H24; the mapping is
written to prep/private/holter_id_map.csv (git-ignored).

Usage:
    python prep/export_holter.py [--src DIR] [--out data/holter]
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier

SRC = Path("/media/robber/Nuevo vol1/ISATEC/projects/mvp-holter/repositories/"
           "holter-itmed-dataset/data/itmed-pathfinder/final")
REPO = Path(__file__).resolve().parents[1]

# Selection: robust annotations (report and beat file agree, low artifact, no pacemaker, no AF),
# age/sex-matched between groups. Split is by patient.
RECORDS = {
    # abnormal (many S and/or V beats)
    "0158": ("anormal", "train"), "0176": ("anormal", "train"), "0046": ("anormal", "train"),
    "0164": ("anormal", "train"), "0133": ("anormal", "train"), "0103": ("anormal", "train"),
    "0198": ("anormal", "train"),
    "0132": ("anormal", "val"), "0205": ("anormal", "val"),
    "0143": ("anormal", "test"), "0213": ("anormal", "test"), "0138": ("anormal", "test"),
    # normal (< 50 S+V beats in ~24 h)
    "0209": ("normal", "train"), "0073": ("normal", "train"), "0074": ("normal", "train"),
    "0135": ("normal", "train"), "0071": ("normal", "train"), "0242": ("normal", "train"),
    "0210": ("normal", "train"),
    "0171": ("normal", "val"), "0234": ("normal", "val"),
    "0082": ("normal", "test"), "0062": ("normal", "test"), "0247": ("normal", "test"),
}

FEATURES = ["rr_prev_ms", "rr_next_ms", "prematuridad", "pausa_compensatoria",
            "ancho_qrs_ms", "amplitud_relativa", "correlacion_plantilla"]
WIN_PRE, WIN_POST = 320, 192   # 4-s window (2.5 s before, 1.5 s after the R peak) at 128 Hz, for 4B
CAP_WINDOWS = 600        # max windows per class and record (4B)
CAP_FEATURES = 4000      # max beats per class and record in train/val (4C)
SEED = 0


def read_header(path):
    h = open(path, "rb").read(300)
    return dict(offset_ecg=struct.unpack_from("<i", h, 22)[0],
                sex={1: "M", 2: "F"}.get(struct.unpack_from("<h", h, 128)[0], "?"),
                amp_nv=struct.unpack_from("<h", h, 206)[0],
                recorder=h[232:272].split(b"\0")[0].decode("latin1").strip())


def clean_channel(x, fs):
    """Baseline wander (0.5 Hz high-pass) and 50 Hz mains notch, zero phase."""
    b, a = signal.butter(2, 0.5 / (fs / 2), "highpass")
    y = signal.filtfilt(b, a, x)
    b, a = signal.iirnotch(50.0 / (fs / 2), 30.0)
    return signal.filtfilt(b, a, y)


def beat_features(y, idx, fs, rng):
    """Hand-crafted features per beat. idx: R-peak sample of every annotated beat (any code)."""
    rr = np.diff(idx) / fs * 1000.0
    rr_prev = np.r_[np.nan, rr]
    rr_next = np.r_[rr, np.nan]
    rr_local = pd.Series(rr_prev).rolling(11, center=True, min_periods=5).median().to_numpy()

    pre, post = int(0.20 * fs), int(0.40 * fs)
    q = int(0.10 * fs)
    ok = (idx - pre >= 0) & (idx + post < len(y))
    W = np.full((len(idx), pre + post), np.nan, np.float32)
    W[ok] = y[idx[ok, None] + np.arange(-pre, post)]
    W -= np.nanmedian(W[:, :q], axis=1, keepdims=True)   # isoelectric reference before the QRS

    pick = rng.choice(np.flatnonzero(ok), size=min(3000, ok.sum()), replace=False)
    template = np.median(W[pick], axis=0)
    Wc = W - np.nanmean(W, 1, keepdims=True)
    tc = template - template.mean()
    corr = (Wc @ tc) / (np.linalg.norm(Wc, axis=1) * np.linalg.norm(tc) + 1e-9)

    qrs = slice(pre - q, pre + q)
    ptp = np.nanmax(W[:, qrs], 1) - np.nanmin(W[:, qrs], 1)
    amp_rel = ptp / (template[qrs].max() - template[qrs].min())

    d = np.abs(np.diff(W[:, pre - int(0.15 * fs): pre + int(0.15 * fs)], axis=1))
    above = d > 0.25 * np.nanmax(d, axis=1, keepdims=True)
    first = above.argmax(1)
    last = above.shape[1] - 1 - above[:, ::-1].argmax(1)
    width = (last - first + 1) / fs * 1000.0

    return pd.DataFrame({
        "rr_prev_ms": rr_prev, "rr_next_ms": rr_next,
        "prematuridad": rr_prev / rr_local, "pausa_compensatoria": rr_next / rr_prev,
        "ancho_qrs_ms": width, "amplitud_relativa": amp_rel, "correlacion_plantilla": corr,
    })


def sample_per_class(df, cap, rng):
    parts = [g.iloc[rng.permutation(len(g))[:cap]] for _, g in df.groupby("code")]
    return pd.concat(parts).sort_values("sample_index")


def patient_scores(all_nsv, records, out):
    """Out-of-fold random forest score for every N/S/V beat (6 folds by patient, 2 normal + 2 abnormal
    each), summarised per patient as a histogram so any aggregation rule can be evaluated."""
    rng = np.random.default_rng(SEED + 1)
    by_group = {g: list(rng.permutation(records.loc[records.grupo == g, "record"])) for g in ("normal", "anormal")}
    folds = [by_group["normal"][2 * i:2 * i + 2] + by_group["anormal"][2 * i:2 * i + 2] for i in range(6)]
    score = np.empty(len(all_nsv), np.float32)
    for fold in folds:
        test = all_nsv.record.isin(fold).to_numpy()
        fit = pd.concat([sample_per_class(g, CAP_FEATURES, rng) for _, g in all_nsv[~test].groupby("record")])
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight="balanced",
                                    n_jobs=-1, random_state=SEED).fit(fit[FEATURES], fit.code.isin(["S", "V"]))
        score[test] = rf.predict_proba(all_nsv.loc[test, FEATURES])[:, 1]
    bins = np.linspace(0, 1, 101)
    rec = records.set_index("record")
    ids = sorted(all_nsv.record.unique())
    hist, smax, smean, n, n_ect = [], [], [], [], []
    for r in ids:
        m = (all_nsv.record == r).to_numpy()
        hist.append(np.histogram(score[m], bins=bins)[0])
        smax.append(score[m].max()); smean.append(score[m].mean())
        n.append(m.sum()); n_ect.append(int(all_nsv.code[m].isin(["S", "V"]).sum()))
    np.savez_compressed(out / "pacientes.npz", record=np.array(ids, dtype="U3"),
                        grupo=rec.loc[ids, "grupo"].to_numpy().astype("U7"), horas=rec.loc[ids, "horas"].to_numpy(),
                        n_latidos=np.array(n), n_ectopicos=np.array(n_ect), hist=np.stack(hist), bins=bins,
                        score_max=np.array(smax), score_medio=np.array(smean))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "holter")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (REPO / "prep" / "private").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    order = list(RECORDS)
    rng.shuffle(order)
    new_id = {rec: f"H{i + 1:02d}" for i, rec in enumerate(order)}
    pd.DataFrame({"original": list(new_id), "id": list(new_id.values())}).to_csv(
        REPO / "prep" / "private" / "holter_id_map.csv", index=False)

    rows, win_x, win_y, win_rec, feats, all_nsv = [], [], [], [], [], []
    for rec in sorted(RECORDS, key=new_id.get):
        group, split = RECORDS[rec]
        rid = new_id[rec]
        d = args.src / rec
        meta = json.loads((d / f"{rec}_metadata.json").read_text())
        fs = int(meta["fs"])
        hdr = read_header(d / f"{rec}.ecg")
        first_valid = int(np.ceil(hdr["offset_ecg"] / 6)) + 2 * fs   # header samples + 2 s margin

        raw = np.fromfile(d / f"{rec}.ecg", "<i2").reshape(-1, 3)[:, 0]
        y = clean_channel(raw.astype(np.float64) * hdr["amp_nv"] * 1e-6, fs).astype(np.float32)  # mV

        beats = pd.read_csv(d / f"{rec}_beats.csv")
        beats = beats[(beats.sample_index >= first_valid) &
                      (beats.sample_index < len(y) - WIN_POST)].reset_index(drop=True)
        f = beat_features(y, beats.sample_index.to_numpy(), fs, rng)
        f.insert(0, "code", beats.code.to_numpy())
        f.insert(0, "time_s", ((beats.sample_index - first_valid) / fs).round(3).to_numpy())
        f.insert(0, "sample_index", beats.sample_index.to_numpy())
        f.insert(0, "split", split)
        f.insert(0, "record", rid)

        nsv = f[f.code.isin(["N", "S", "V"])].dropna(subset=FEATURES)
        all_nsv.append(nsv)
        keep = nsv if split == "test" else sample_per_class(nsv, CAP_FEATURES, rng)
        feats.append(keep)

        w = sample_per_class(nsv, CAP_WINDOWS, rng)
        idx = w.sample_index.to_numpy()
        win_x.append(y[idx[:, None] + np.arange(-WIN_PRE, WIN_POST)].astype(np.float16))
        win_y.append(w.code.to_numpy())
        win_rec.append(np.full(len(w), rid))

        age = int(meta["StartTime"][:4]) - int(meta["DateOfBirth"][:4])
        band = "<65" if age < 65 else ("65-79" if age < 80 else ">=80")
        counts = beats.code.value_counts()
        rows.append(dict(record=rid, grupo=group, split=split, edad=band, sexo=hdr["sex"],
                         equipo=hdr["recorder"], horas=round(len(y) / fs / 3600, 1),
                         total_latidos=len(beats), **{f"n_{c}": int(counts.get(c, 0)) for c in "NSVXU"}))
        print(rid, group, split, dict(counts.reindex(list("NSVXU")).fillna(0).astype(int)))

    records = pd.DataFrame(rows)
    records.to_csv(args.out / "records.csv", index=False)

    feats = pd.concat(feats, ignore_index=True)
    feats["ectopico"] = feats.code.isin(["S", "V"]).astype(int)
    # Reference model for 4-s1: random forest trained on train+val, scored on the test records.
    fit = feats[feats.split != "test"]
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                n_jobs=-1, random_state=SEED).fit(fit[FEATURES], fit.ectopico)
    feats["score_rf"] = np.where(feats.split == "test", rf.predict_proba(feats[FEATURES])[:, 1], np.nan)
    for c in FEATURES + ["score_rf"]:
        feats[c] = feats[c].astype(np.float32)
    feats.drop(columns="sample_index").to_parquet(args.out / "features.parquet", index=False)

    np.savez_compressed(args.out / "windows.npz", x=np.concatenate(win_x),
                        y=np.concatenate(win_y).astype("U1"), record=np.concatenate(win_rec).astype("U3"), fs=128, pre=WIN_PRE)
    patient_scores(pd.concat(all_nsv, ignore_index=True), records, args.out)
    print("features:", feats.groupby(["split", "code"]).size().to_dict())
    print("windows:", sum(len(v) for v in win_y))


if __name__ == "__main__":
    main()
