r"""
Step 4 (locked 8 kHz task): the scorecard of ../playback_rules.py for 8 kHz models.
test_birds (524 calls, 16 species, Mockingbird not scored) and test_zero (frog, cricket, cicada, cat, rain, wind,
thunder; 4 user-excluded bird-like spots) on the GPX10 board recordings, 1-window rule first, then 2-of-3, at 0.5.
Board audio: the 16 kHz GPX10 recordings decimated to 8 kHz (bird_preprocess_8k.preprocess_16k) until the
8 kHz board recordings exist - pass them with  --birds <wav> --zero <wav>  (8 kHz files are used as they are;
they must follow the same playback schedule, alignment csv next to them as for the 16 kHz files).
Windows: 2 s every 1 s, time = window end (as the live dashboard).

    python score_board_8k.py r8k_1 [r8k_2 ...]
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
os.chdir(ROOT)

import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402
import soundfile as sf             # noqa: E402

import bird_preprocess_8k as b8    # noqa: E402
import stream_eval as se           # noqa: E402
import train_bird_model as t       # noqa: E402


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


models = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--birds", "--zero")]
B = (arg("--birds", "playback/test_bird_gpx_16khz.wav"), "test_birds")
Z = (arg("--zero", "playback/test_zero_gpx_16khz.wav"), "test_zero")


def windows8(rec):
    y, sr = sf.read(rec, dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    y8 = b8.from_16k(y) if sr == 16000 else b8.resample(y, sr, 8000) if sr != 8000 else y
    st = np.arange(0, len(y8) - b8.CLIP + 1, 8000)
    X = np.stack([b8.quantize_int8(b8.preprocess(y8[a:a + b8.CLIP])) for a in st])
    return (st + b8.CLIP) / 8000, X


def load(rec, nm):
    tt, X = windows8(rec)
    s = pd.read_csv(rec.replace(".wav", f"_{nm}_aligned.csv"))
    return tt, s, {r: t.run_tflite(HERE / "runs" / r / "model_int8.tflite", X) for r in models}


tb, sb, Pb = load(*B)
tz, sz, Pz = load(*Z)
DSF = HERE / "features_final" / "drop_species.json"            # clean_8k.py: Wren dropped from the target?
DROP = ["Northern Mockingbird"] + (__import__("json").loads(DSF.read_text()) if DSF.exists() else [])
allb = sb[(sb.kind == "bird") & sb.rec_start_s.notna()]
calls = np.c_[allb.rec_start_s.values, allb.rec_end_s.values]       # every played bird (no gap false alarm there)
BXF = "playback/test_birds_excluded_8k.csv"      # calls with no bird left at 8 kHz (user listened 2026-11-12)
BX = set(pd.read_csv(BXF).source_file) if os.path.exists(BXF) else set()
b = allb[~allb.cls.isin(DROP) & ~allb.source.astype(str).str.replace("\\", "/").str.split("/").str[-1].isin(BX)]
on = np.where(b.onset_s.notna(), b.onset_s, b.rec_start_s)
en = b.rec_end_s.values
z = sz[sz.kind == "zero"]
ZC = ["frog", "cricket", "cicada", "cat", "rain", "wind", "thunder"]
EXF = "playback/test_zero_excluded.csv"
EX = pd.read_csv(EXF)[["start_s", "end_s"]].values if os.path.exists(EXF) else np.zeros((0, 2))


def keep(o):
    return np.array([x for x in o if not ((x >= EX[:, 0]) & (x <= EX[:, 1])).any()])


def state(p, k, n, thr=0.5):
    return np.convolve((p >= thr).astype(int), np.ones(n, int))[:len(p)] >= k


print(f"8 kHz scorecard | board audio: {B[0]}, {Z[0]}")
print(f"{len(b)} test calls, {b.cls.nunique()} species (not scored: {DROP}); {len(BX)} calls with no bird at 8 kHz excluded; test_zero: {len(EX)} bird-like spots excluded")
print(f"{'model':10s} {'rule':7s} {'calls':>12s} {'gapFA':>5s}  " + " ".join(f"{c:>7s}" for c in ZC) + "  zeroTOTAL")
mk = sb[sb.kind.str.startswith("marker")]
for r in models:
    for rule, (k, n) in (("1-win", (1, 1)), ("2-of-3", (2, 3))):
        a = state(Pb[r], k, n)
        caught = np.array([a[(tb >= o + 0.3) & (tb <= e + 2.5)].any() for o, e in zip(on, en)])
        ob = np.array([tb[i] for i in se.alarm_onsets(a)])
        gap = [x for x in ob if not ((x >= calls[:, 0] - 1) & (x <= calls[:, 1] + 4)).any()
               and not ((x >= mk.rec_start_s.values - 0.5) & (x <= mk.rec_end_s.values + 2.5)).any()]
        az = state(Pz[r], k, n)
        oz = keep([tz[i] for i in se.alarm_onsets(az)])
        fz = {c: int(sum(((oz >= a0 + 2) & (oz <= e + 1)).sum() for a0, e in z[z.cls == c][["rec_start_s", "rec_end_s"]].values))
              if len(oz) else 0 for c in ZC}
        print(f"{r:10s} {rule:7s} {int(caught.sum()):4d}/{len(b)} {caught.mean() * 100:5.1f}% {len(gap):5d}  "
              + " ".join(f"{fz[c]:7d}" for c in ZC) + f"  {sum(fz.values()):5d}", flush=True)
        if rule == "1-win":
            miss = b[~caught]
            print("   1-win misses by species:", miss.cls.value_counts().to_dict())
