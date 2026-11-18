r"""
8 kHz FINAL model (LOCKED task 2026-11-11, memory locked-task-8k): the device supports only 8 kHz.
Data   bird_voice_8khz/features_final/*.npz (extract_final_8k.py: final cleaned dataset, 8 kHz audio, 126 x 40 int8)
       minus the rows in features_final/drop_8k.csv (8 kHz BirdNET check: bird lost at 8 kHz) and DROP_SPECIES.
Class  output index 0 = wake-word (bird), index 1 = not-wake-word  (p_bird = softmax[:, 0]).
Model  "mid150" Conv2D net with the first time stride 2 instead of 4 (126 frames of 16 ms): same layer sizes and
       parameters as run15 (147.0k), ~1.32 s on the GPX10 (device_cost.py), heap as run15 (~193 KB).
Weights
  x3 (user): cicada, cricket, insects, cat, frog, conversation, monkey, rain, office ambient (office_ambient +
       office_data), traffic, puppy_dog, whistle. run15 board rows: max(their run15 weight, category weight).
  bird species balanced: a species with more than SPECIES_CAP clips gets weight SPECIES_CAP / count (Jay, Song Thrush).
  round 2 (HARD_FROM=<round-1 run>): every train negative that run alarms on (int8 P(bird) >= 0.5) x HARD_W.
Then bird / not-bird class balance on the weighted mass.

    python train_8k_final.py r8k_1                       # round 1
    HARD_FROM=r8k_1 python train_8k_final.py r8k_2       # round 2 (hard negatives of round 1)
Output bird_voice_8khz/runs/<name>/: model.h5, model_int8.tflite, training_log.csv, report.txt, hard_weights.npy
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import tensorflow as tf       # noqa: E402

import train_bird_model as t  # noqa: E402  (helpers only; settings overridden below)

FF = HERE / "features_final"
t.RUNS = HERE / "runs"
t.N_FRAMES = 126
t.ARCH = "mid150"
t.MID150 = [(16, (5, 5), (2, 2)), (32, (3, 3), (2, 2)), (80, (3, 3), (2, 2)), (128, (3, 3), (2, 2)),
            (192, (1, 1), (1, 1))]                     # time stride 2 (not 4): 126 frames of 16 ms
t.MAX_SHIFT = 30                                       # run15: 60 frames x 8 ms = 0.48 s -> 30 x 16 ms
t.TIME_MASKS = (2, 12)                                 # ~0.2 s, as before in seconds
t.DISTILL, t.QUIET_SET = [], ""
t.EPOCHS = 1 if os.environ.get("SMOKE", "") == "1" else int(os.environ.get("BIRD_EPOCHS", 60))
t.LR, t.PATIENCE = 2e-3, 15

CAT_W = {c: 3.0 for c in ("cicada", "cricket", "insects", "cat", "frog", "conversation", "monkey", "rain",
                          "office_ambient", "office_data", "traffic", "puppy_dog", "whistle")}
for _kv in filter(None, os.environ.get("CAT_BOOST", "").split(",")):      # round 3: e.g. CAT_BOOST=cat=6,frog=6
    _c, _v = _kv.split("=")
    CAT_W[_c] = float(_v)
DROP_SPECIES = json.loads(os.environ.get("DROP_SPECIES", "null") or "null") or (       # clean_8k.py: Wren decision
    json.loads((FF / "drop_species.json").read_text()) if (FF / "drop_species.json").exists() else [])
SMOKE = os.environ.get("SMOKE", "") == "1"             # quick pipeline test: 1 % of the rows, 1 epoch
if SMOKE:
    t.REP_SAMPLES = 200
SPECIES_CAP = 2500
HARD_FROM = os.environ.get("HARD_FROM", "")
HARD_W = float(os.environ.get("HARD_W", 3.0))
THR = 0.5                                              # locked threshold

SETS = {"train": ["train", "extra_neg_train", "insect_frog_train", "extra_birds_train", "new_species_train",
                  "board_birds_train", "playback_train"],
        "val": ["val", "extra_neg_val", "insect_frog_val", "extra_birds_val", "new_species_val",
                "board_birds_val", "playback_val"],
        "test": ["test", "extra_neg_test", "insect_frog_test", "extra_birds_test", "new_species_test",
                 "board_birds_test", "playback_test", "replace_aug_test", "inat_test", "peadetect_test",
                 "peacock_dl_test"]}
KEYS = ["X", "y", "cls", "files", "group", "offset_ms", "weight", "set"]


def drop_rows():
    f = FF / "drop_8k.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    return {s: set(g.row) for s, g in d.groupby("set")}


def load_split(split, drops):
    parts = []
    names = list(SETS[split])
    run15 = sorted(p.stem for p in FF.glob("run15__*.npz") if not p.stem.endswith("office_sound"))
    for name in names + [f"{r}#{split}" for r in run15] + (["run15__20261110_110606_office_sound"] if split == "test" else []):
        stem, sp = (name.split("#") + [None])[:2]
        z = np.load(FF / f"{stem}.npz", allow_pickle=True)
        n = len(z["y"])
        m = np.ones(n, bool)
        if sp:                                                  # run15 windows: their own train / val / test split
            m &= z["split"].astype(str) == sp
        if stem in drops:
            m[list(drops[stem])] = False
        if SMOKE:
            m &= np.random.default_rng(0).random(n) < 0.01
        c = z["cls"].astype(str)
        for s in DROP_SPECIES:
            m &= c != s
        part = dict(X=z["X"][m], y=z["y"][m].astype(np.int64), cls=c[m], files=z["files"].astype(str)[m],
                    group=(z["group"].astype(str)[m] if "group" in z.files else np.full(m.sum(), stem)),
                    offset_ms=(z["offset_ms"][m].astype(np.int64) if "offset_ms" in z.files else np.full(m.sum(), -1)),
                    weight=(z["weight"][m].astype(np.float32) if stem.startswith("run15__") and "weight" in z.files
                            else np.zeros(m.sum(), np.float32)),
                    set=np.full(m.sum(), stem))
        parts.append(part)
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def main():
    name = sys.argv[1]
    out = t.RUNS / name
    out.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(t.SEED)
    drops = drop_rows()
    print(f"dropped rows (8 kHz BirdNET): {sum(len(v) for v in drops.values())}; species dropped: {DROP_SPECIES}",
          flush=True)
    tr, va, te = (load_split(s, drops) for s in ("train", "val", "test"))
    for s, d in (("train", tr), ("val", va), ("test", te)):
        print(f"{s}: {len(d['y'])} (bird {int((d['y'] == 0).sum())}, not bird {int((d['y'] == 1).sum())})", flush=True)
    ytr, ctr = tr["y"], tr["cls"]
    w = np.ones(len(ytr), np.float32)
    neg = ytr == 1
    cw = np.array([CAT_W.get(c, 1.0) for c in ctr], np.float32)
    r15 = np.char.startswith(tr["set"].astype(str), "run15__")
    w[neg & ~r15] *= cw[neg & ~r15]
    w[r15] = np.where(neg[r15], np.maximum(tr["weight"][r15], cw[r15]), np.maximum(tr["weight"][r15], 3.0))
    print("category weights: " + ", ".join(f"{c} {int(((ctr == c) & neg).sum())} x{CAT_W[c]:g}" for c in CAT_W), flush=True)
    sp = pd.Series(ctr[~neg]).value_counts()
    big = sp[sp > SPECIES_CAP]
    for c, n in big.items():
        w[(~neg) & (ctr == c) & ~r15] *= SPECIES_CAP / n
    print("species capped to ~%d: %s" % (SPECIES_CAP, ", ".join(f"{c} {n}" for c, n in big.items())), flush=True)
    for _kv in filter(None, os.environ.get("BIRD_BOOST", "").split(",")):   # round 4: weak species, no cap
        c, v = _kv.split("=")
        m = (~neg) & (ctr == c)
        w[m & ~r15] = float(v)
        w[m & r15] = np.maximum(w[m & r15], float(v))
        print(f"bird boost: {c} {int(m.sum())} clips -> x{float(v):g}", flush=True)
    if HARD_FROM:
        p = np.max([t.run_tflite(t.RUNS / r / "model_int8.tflite", tr["X"][neg]) for r in HARD_FROM.split(",")], 0)
        hit = np.flatnonzero(neg)[p >= THR]          # alarms of any listed run (HARD_FROM=r8k_1,r8k_2)
        w[hit] *= HARD_W
        print(f"round 2: {HARD_FROM} alarms on {len(hit)} of {int(neg.sum())} train negatives -> x{HARD_W}; by category: "
              + ", ".join(f"{c} {n}" for c, n in pd.Series(ctr[hit]).value_counts().head(15).items()), flush=True)
    np.save(out / "hard_weights.npy", w)
    mass = np.array([w[ytr == 0].sum(), w[ytr == 1].sum()])
    cwt = (len(ytr) / (2 * mass)).astype(np.float32)
    sw = w * cwt[ytr]

    model = t.build_model()
    model.summary(print_fn=lambda s: print(s, flush=True))
    steps = t.EPOCHS * int(np.ceil(len(ytr) / t.BATCH))
    model.compile(optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(t.LR, steps)),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=t.LABEL_SMOOTH),
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    t0 = time.time()
    model.fit(t.Batches(tr["X"], ytr, sw, train=True), validation_data=t.Batches(va["X"], va["y"]), epochs=t.EPOCHS,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=t.PATIENCE, restore_best_weights=True),
                         tf.keras.callbacks.CSVLogger(str(out / "training_log.csv"))],
              workers=4, use_multiprocessing=False, verbose=2)
    minutes = (time.time() - t0) / 60
    model.save(out / "model.h5")
    tfl = out / "model_int8.tflite"
    t.to_int8_tflite(model, tr["X"], tfl, ytr)
    info = t.check_tflite(tfl)
    del tr
    report(name, out, tfl, info, model.count_params(), minutes, va, te)


def report(name, out, tfl, info, n_params, minutes, va, te):
    pv, pt = t.run_tflite(tfl, va["X"]), t.run_tflite(tfl, te["X"])
    pd.DataFrame({"set": te["set"], "file": te["files"], "cls": te["cls"], "label": te["y"], "group": te["group"],
                  "offset_ms": te["offset_ms"], "p_bird_int8": pt.round(4)}).to_csv(out / "test_predictions.csv", index=False)
    L = [f"run {name} (8 kHz, 126x40, index 0 = wake-word): {n_params} parameters, {minutes:.0f} min training",
         "TFLite int8: " + json.dumps(info), ""]
    for tag, p, d in (("val", pv, va), ("test", pt, te)):
        b, n = d["y"] == 0, d["y"] == 1
        L.append(f"[{tag} int8 @ {THR}] bird recall {(p[b] >= THR).mean():.4f}  false-positive rate {(p[n] >= THR).mean():.4f}")
    b = te["y"] == 0
    g = pd.DataFrame({"c": te["cls"][b], "h": pt[b] >= THR}).groupby("c").h.agg(["size", "mean"]).sort_values("mean")
    L += ["", f"TEST bird recall per species @ {THR}:"] + [f"  {c:32s} {int(r['size']):6d} {r['mean']:.4f}" for c, r in g.iterrows()]
    n = te["y"] == 1
    g = pd.DataFrame({"c": te["cls"][n], "h": pt[n] >= THR}).groupby("c").h.agg(["size", "mean"]).sort_values("mean", ascending=False)
    L += ["", f"TEST false-positive rate per not-bird category @ {THR} (worst first):"] + \
         [f"  {c:32s} {int(r['size']):6d} {r['mean']:.4f}" for c, r in g.iterrows()]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for title, m in (("test-recording negatives (extra_neg + insect_frog TEST)",
                      np.isin(te["set"], ["extra_neg_test", "insect_frog_test"])),
                     ("OFFICE test session 20261110_110606 (GPX10, 16 kHz board audio decimated)",
                      te["set"] == "run15__20261110_110606_office_sound")):
        m &= te["y"] == 1
        if m.any():
            L += ["", f"{title}: {int(m.sum())} windows, {t.WINDOWS_PER_HOUR} windows/hour"]
            L += t.false_alarm_table(pt[m], te["group"][m], te["offset_ms"][m], thresholds)
    for s in ("board_birds_test", "playback_test", "new_species_test"):
        m = te["set"] == s
        yb, yn = m & (te["y"] == 0), m & (te["y"] == 1)
        L.append(f"{s}: bird recall {(pt[yb] >= THR).mean():.4f} ({int(yb.sum())})"
                 + (f", false-positive rate {(pt[yn] >= THR).mean():.4f} ({int(yn.sum())})" if yn.any() else ""))
    text = "\n".join(L)
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
