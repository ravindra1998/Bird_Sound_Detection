r"""
Phase 1: train the bird / not-bird DS-CNN on the PCEN features and export a fully int8 TFLite model.

Run in the tf210 environment:
    C:\Users\Ravindra\anaconda3\envs\tf210\python.exe train_bird_model.py run1

Input  features/<train|val|test>.npz (extract_features.py): X int8 (N, 251, 40) = quantize_int8(PCEN features)
Model  input (1, 251, 40) float = dequantize(X)  ->  Reshape (251, 40, 1)
       Conv2D 32 (5x3, stride 2) - 5 depthwise-separable blocks (64, 64, 96, 96, 128) - AveragePooling2D over
       the whole 16x3 map - Flatten - dropout - Dense 2 softmax (class 0 = bird, class 1 = not bird);
       BatchNorm + ReLU after every conv. Device layers only: Conv2D, DepthwiseConv2D, AveragePooling2D, Dense.
       tf.keras.Sequential; heap (peak activation arena + model) must stay <= HEAP_LIMIT (284 KB), checked
Train  + EXTRA_SETS: every unused clean non-bird clip from train recordings (features/extra_neg_train.npz) and
       the added insects / frogs (features/insect_frog_train.npz, add_insect_frog.py); clips the earlier model
       scored as bird (p_prev) get HARD_NEG weight; class-balanced weights, label smoothing, SpecAugment (time / frequency masks) + time shift,
       Adam with cosine decay, early stopping on validation loss
Export full integer TFLite: int8 input (1, 1, 251, 40) and int8 output, int8 weights and activations
       (int32 biases, as in every int8 TFLite), int8-only ops - conversion fails if anything stays float.
       Input quantisation must equal bird_preprocess.py (scale 0.0194852948, zero point -128); checked.
Test   float and int8 model on val and test; the int8 model gets the stored int8 features unchanged,
       exactly as the device. Threshold for P(bird): the lowest whose smoothed false alarms per hour on all
       val-recording negatives stay <= FA_BUDGET (catch as many birds as the alarm budget allows), applied once
       to test. False alarms per hour on all test-recording negatives (never trained on), per 2 s window and with
       SMOOTH (k of n consecutive windows) on clips that follow each other in the same recording.

Output runs/<name>/: model.h5, model_int8.tflite, training_log.csv, report.txt, test_predictions.csv
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

PROJECT  = Path(r"C:\Users\Ravindra\Desktop\Bird_sound_detection_PCEN")
FEATURES = PROJECT / "features"
RUNS     = PROJECT / "runs"

# ---- settings ---------------------------------------------------------------------------------
EPOCHS       = int(os.environ.get("BIRD_EPOCHS", 60))   # BIRD_EPOCHS=1 for a smoke test
BATCH        = 256
LR           = float(os.environ.get("BIRD_LR", 2e-3))
INIT_FROM    = os.environ.get("BIRD_INIT", "")             # run9: warm start from runs/<INIT_FROM>/model.h5
HARD_POS     = float(os.environ.get("BIRD_HARD_POS", 1.0))  # birds BIRD_HARD_POS_RUN scores < 0.5 get this weight
HARD_POS_RUN = os.environ.get("BIRD_HARD_POS_RUN", "")
CLS_WEIGHT   = json.loads(os.environ.get("BIRD_CLS_WEIGHT", "{}"))   # {"Duck": 2, "monkey": 3, ...}
PATIENCE     = int(os.environ.get("BIRD_PATIENCE", 10))
LABEL_SMOOTH = 0.05
DROPOUT      = 0.2
TIME_MASKS   = (2, 25)     # number of masks, max width in frames
FREQ_MASKS   = (2, 5)      # number of masks, max width in mel bands
MAX_SHIFT    = int(os.environ.get("BIRD_MAX_SHIFT", 20))   # frames (8 ms), zero-filled; run14: 60 (~0.5 s)
STEM         = 32
BLOCKS       = [(64, (2, 2)), (64, (2, 2)), (96, (2, 2)), (96, (1, 1)), (128, (1, 1))]   # depthwise: equal strides only
BLOCKS_WIDE  = [(64, (2, 2)), (80, (2, 2)), (96, (2, 2)), (112, (1, 1)), (136, (1, 1))]  # "--wide": 48,658 params
ARCH         = os.environ.get("BIRD_ARCH", "dscnn")   # "fast": plain Conv2D net below (device time ~6x lower)
FAST         = [(8, (5, 5), (4, 2)), (24, (3, 3), (2, 2)), (48, (3, 3), (2, 2)), (96, (3, 3), (2, 2)),
                (192, (1, 1), (1, 1))]               # (channels, kernel, stride): Conv2D strides may be unequal
MID          = [(16, (5, 5), (4, 2)), (32, (3, 3), (2, 2)), (64, (3, 3), (2, 2)), (96, (3, 3), (2, 1)),
                (160, (1, 1), (1, 1))]               # run11 "mid": ~1.32 s on GPX10 (device_cost.py P1), 95.9k params
MID150       = [(16, (5, 5), (4, 2)), (32, (3, 3), (2, 2)), (80, (3, 3), (2, 2)), (128, (3, 3), (2, 2)),
                (192, (1, 1), (1, 1))]               # run15 "mid150": ~1.32 s (device_cost.py), 147.0k params
MID_DW       = [("conv", 16, (5, 5), (4, 2)), ("conv", 32, (3, 3), (2, 2)), ("dw", (2, 2)), ("pw", 64),
                ("conv", 96, (3, 3), (2, 1)), ("dw", (1, 1)), ("pw", 144)]   # run12d "middw": D3, ~1.47 s, 79.5k
DISTILL      = [r for r in os.environ.get("BIRD_DISTILL", "").split(",") if r]   # teacher runs (float model.h5)
DISTILL_A    = float(os.environ.get("BIRD_DISTILL_ALPHA", 0.5))   # weight of the teacher in the target
DISTILL_T    = float(os.environ.get("BIRD_DISTILL_T", 2.0))       # temperature on the averaged teacher logit
REP_SAMPLES  = 6000        # representative clips for the int8 calibration: half birds, half non-birds, all sources
HEAP_LIMIT   = 284 * 1024  # device heap: activation arena + model must fit (checked after export)
SEED         = 42
EXTRA_SETS   = ["extra_neg", "insect_frog"]   # features/<set>_<split>.npz, all non-bird; [] = balanced set only
HARD_NEG     = (0.5, 3.0)  # extra negatives the earlier model scored P(bird) >= 0.5 get 3x weight
SMOOTH       = (2, 3)      # alarm when 2 of 3 consecutive 2 s windows say bird
WINDOWS_PER_HOUR = 1800    # 2 s windows, no overlap
FA_BUDGET    = 1.5         # threshold = lowest one whose smoothed false alarms / hour on val stay <= this
COMPARE_RUN  = "run7q"       # earlier run scored on the same test negatives for comparison ("" = none)
QUIET_SET    = ""          # make_quiet_aug.py quiet copies: run4 only (they taught "quiet = bird" - not used again)
MIXED_SETS   = ["extra_birds", "category_copies", "board_copies", "board_birds"] + \
               (["playback"] if os.environ.get("BIRD_PLAYBACK", "0") == "1" else [])   # run14: GPX10 playback recordings
               # run8: measured board response
               # copies + board-recorded birds; gpx_copies / speaker_copies (guessed speaker) dropped   # labelled sets (birds and/or non-birds): train; val/test reported
HARD_RUN     = "run7q"      # its P(bird) >= HARD_NEG[0] on train negatives -> HARD_NEG weight ("" = off)
HARD_ALL     = True        # True: HARD_RUN scores EVERY train negative (all sets), not only the balanced ones
SET_WEIGHT   = {"board_birds": 3.0, "playback": float(os.environ.get("BIRD_PLAYBACK_W", 3.0))}   # extra weight of a MIXED set (few real board birds vs many copies)
BOARD_NEG_W  = 2.0       # real GPX10 recordings (group GPXREC:*) among the extra negatives
RUN15        = os.environ.get("BIRD_RUN15", "0") == "1"   # run15: + run15_data/windows/*.npz (cleaned live sessions)
RUN15_DIR    = PROJECT / "run15_data"
DEVICE_TIME_SPLIT = {"Animal_chip/cat.wav": 0.7}   # device recording (all test before): first 70 % of its time
                           # moves to training, the rest stays in val/test (balanced and extra_neg sets)
# -------------------------------------------------------------------------------------------------

IN_SCALE, IN_ZP = 0.0194852948, -128          # bird_preprocess.py
OUT_SCALE, OUT_ZP = 1 / 256, -128
N_FRAMES, N_MELS = 251, 40


def dequant(q):
    return (q.astype(np.float32) - IN_ZP) * IN_SCALE


DROP_CLASSES = ["Northern Mockingbird"]   # 2026-11-11: mimics frogs / insects / whistles -> false alarms; not trained


def keep_mask(cls, files):
    """False for rows of a DROP_CLASSES species (by class name or by the clip path: copies, playback windows)."""
    c, f = np.asarray(cls).astype(str), np.asarray(files).astype(str)
    m = np.ones(len(c), bool)
    for n in DROP_CLASSES:
        m &= (np.char.find(c, n) < 0) & (np.char.find(f, n) < 0)
        if n == "Northern Mockingbird":
            m &= np.char.find(f, "Mimus_polyglottos") < 0
    return m


def load(split):
    d = np.load(FEATURES / f"{split}.npz", allow_pickle=False)
    m = keep_mask(d["cls"], d["files"])
    return d["X"][m], d["y"][m], d["cls"][m], d["files"][m], d["made"][m]


def load_extra(split, sets=None):
    """All EXTRA_SETS of one split joined; p_prev = P(bird) of the model that mined the set (-1 = none)."""
    keys = ["X", "y", "cls", "files", "group", "offset_ms", "p_prev"]
    parts = {k: [] for k in keys}
    for s in (EXTRA_SETS if sets is None else sets):
        d = dict(np.load(FEATURES / f"{s}_{split}.npz", allow_pickle=False))
        d.setdefault("p_prev", d.pop("p_run1", None))
        m = keep_mask(d["cls"], d["files"])
        for k in keys:
            parts[k].append(d[k][m])
    if not EXTRA_SETS:
        return {k: np.zeros((0, N_FRAMES, N_MELS) if k == "X" else 0, np.int8 if k == "X" else np.float32)
                for k in keys}
    return {k: np.concatenate(v) for k, v in parts.items()}


def load_run15(sub="windows"):
    """run15_data/<sub>/*.npz joined (run15_add_recording.py / run15_centroid_clips.py --add). Per-window weight =
    the recording's weight (x3 whistle, monkey, ...); windows run14aq/bq scored as bird (hard) get at least HARD_NEG[1]."""
    keys = ["X", "y", "cls", "files", "group", "offset_ms", "p_prev", "split", "hard", "weight"]
    parts = {k: [] for k in keys}
    for f in sorted((RUN15_DIR / sub).glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        m = keep_mask(d["cls"], d["files"])
        for k in keys:
            parts[k].append((d[k].astype(str) if d[k].dtype.kind == "U" else d[k])[m])
    d = {k: np.concatenate(v) for k, v in parts.items()}
    d["y"] = d["y"].astype(np.int64)
    d["w"] = np.where(d["hard"], np.maximum(d["weight"], HARD_NEG[1]), d["weight"]).astype(np.float32)
    return d


def teacher_targets(X, y):
    """Soft P(bird) targets (run11 distillation). Label smoothed as LABEL_SMOOTH would; where the teachers'
    averaged, temperature-softened P(bird) is on the same side of 0.5 as the label, the target becomes
    (1 - DISTILL_A) * label + DISTILL_A * teacher. Where the teachers disagree (e.g. new board recordings they
    never saw) the label alone is used."""
    lab = np.where(y == 0, 1 - LABEL_SMOOTH / 2, LABEL_SMOOTH / 2).astype(np.float32)
    if not DISTILL:
        return lab
    key = f"{'_'.join(DISTILL)}_{len(X)}_{int(X[::997, ::25, ::5].astype(np.int64).sum())}_{int(y.sum())}"
    cache = FEATURES / "distill_cache" / f"{key}.npy"        # averaged teacher logit sum, same data -> reuse
    if cache.exists():
        lg = np.load(cache)
        print(f"teacher logits from cache {cache.name}", flush=True)
    else:
        lg = None
    for r in (DISTILL if lg is None else []):
        lg = np.zeros(len(X), np.float64) if lg is None else lg
        m = tf.keras.models.load_model(str(RUNS / r / "model.h5"), compile=False)
        p = np.clip(m.predict(Batches(X, y), verbose=0)[:, 0].astype(np.float64), 1e-6, 1 - 1e-6)
        lg += np.log(p / (1 - p))
        print(f"teacher {r}: recall {(p[y == 0] >= 0.5).mean():.4f} fp {(p[y == 1] >= 0.5).mean():.4f}", flush=True)
        del m
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, lg)
    pt = 1 / (1 + np.exp(-lg / len(DISTILL) / DISTILL_T))
    agree = (pt >= 0.5) == (y == 0)
    soft = np.where(agree, (1 - DISTILL_A) * lab + DISTILL_A * pt, lab).astype(np.float32)
    print(f"distill {DISTILL} alpha {DISTILL_A} T {DISTILL_T}: teachers agree with the label on "
          f"{agree.mean():.4f} (birds {agree[y == 0].mean():.4f}, non-birds {agree[y == 1].mean():.4f}); "
          f"mean soft target birds {soft[y == 0].mean():.3f}, non-birds {soft[y == 1].mean():.3f}", flush=True)
    return soft


class Batches(tf.keras.utils.Sequence):
    def __init__(self, X, y, weights=None, train=False, soft=None):
        self.X, self.y, self.w, self.train, self.soft = X, y, weights, train, soft
        self.idx = np.arange(len(X))
        self.rng = np.random.default_rng(SEED)
        if train:
            self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.X) / BATCH))

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.idx)

    def __getitem__(self, b):
        i = np.sort(self.idx[b * BATCH:(b + 1) * BATCH])
        x = dequant(self.X[i])
        if self.train:
            x = self.augment(x)
        if self.soft is None:
            y = tf.keras.utils.to_categorical(self.y[i], 2)
        else:                                                # P(bird) target: column 0 = bird
            y = np.stack([self.soft[i], 1 - self.soft[i]], 1).astype(np.float32)
        x = x[:, None, :, :]
        if self.w is None:
            return x, y
        return x, y, self.w[i]

    def augment(self, x):
        r = self.rng
        for k in range(len(x)):
            s = r.integers(-MAX_SHIFT, MAX_SHIFT + 1)
            if s:
                x[k] = np.roll(x[k], s, axis=0)
                if s > 0:
                    x[k, :s] = 0
                else:
                    x[k, s:] = 0
            for _ in range(TIME_MASKS[0]):
                w = r.integers(0, TIME_MASKS[1] + 1)
                t0 = r.integers(0, N_FRAMES - w + 1)
                x[k, t0:t0 + w] = 0
            for _ in range(FREQ_MASKS[0]):
                w = r.integers(0, FREQ_MASKS[1] + 1)
                f0 = r.integers(0, N_MELS - w + 1)
                x[k, :, f0:f0 + w] = 0
        return x


def build_model():
    """tf.keras.Sequential (required by the device toolchain)."""
    L = tf.keras.layers
    m = tf.keras.Sequential(name="bird_dscnn")
    m.add(L.InputLayer((1, N_FRAMES, N_MELS), name="pcen_int8_dequant"))
    m.add(L.Reshape((N_FRAMES, N_MELS, 1)))
    if ARCH in ("fast", "mid", "mid150"):                   # run10 fast: ~0.74 s (F4); run11 mid: ~1.32 s (P1)
        for ch, k, st in {"fast": FAST, "mid": MID, "mid150": MID150}[ARCH]:
            m.add(L.Conv2D(ch, k, strides=st, padding="same", use_bias=False))
            m.add(L.BatchNormalization())
            m.add(L.ReLU())
        m.add(L.AveragePooling2D(pool_size=tuple(m.output_shape[1:3])))
        m.add(L.Flatten())
        m.add(L.Dropout(DROPOUT))
        m.add(L.Dense(2, activation="softmax", name="bird_notbird"))
        return m
    if ARCH == "middw":                                     # run12d: Conv2D early, depthwise+pointwise late (D3)
        for layer in MID_DW:
            if layer[0] == "conv":
                m.add(L.Conv2D(layer[1], layer[2], strides=layer[3], padding="same", use_bias=False))
            elif layer[0] == "dw":
                m.add(L.DepthwiseConv2D(3, strides=layer[1], padding="same", use_bias=False))
            else:
                m.add(L.Conv2D(layer[1], 1, use_bias=False))
            m.add(L.BatchNormalization())
            m.add(L.ReLU())
        m.add(L.AveragePooling2D(pool_size=tuple(m.output_shape[1:3])))
        m.add(L.Flatten())
        m.add(L.Dropout(DROPOUT))
        m.add(L.Dense(2, activation="softmax", name="bird_notbird"))
        return m
    m.add(L.Conv2D(STEM, (5, 3), strides=(2, 2), padding="same", use_bias=False))
    m.add(L.BatchNormalization())
    m.add(L.ReLU())
    for ch, stride in BLOCKS:
        m.add(L.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False))
        m.add(L.BatchNormalization())
        m.add(L.ReLU())
        m.add(L.Conv2D(ch, 1, use_bias=False))
        m.add(L.BatchNormalization())
        m.add(L.ReLU())
    # AveragePooling2D over the whole map (no GlobalAveragePooling2D: not supported on the device)
    m.add(L.AveragePooling2D(pool_size=tuple(m.output_shape[1:3])))
    m.add(L.Flatten())
    m.add(L.Dropout(DROPOUT))
    m.add(L.Dense(2, activation="softmax", name="bird_notbird"))
    return m


def to_int8_tflite(model, Xtrain, path, ytrain=None):
    rng = np.random.default_rng(SEED)
    if ytrain is None:
        pick = rng.choice(len(Xtrain), REP_SAMPLES, replace=False)
    else:                                                    # stratified: half birds, half non-birds
        pick = np.concatenate([rng.choice(np.flatnonzero(ytrain == c), REP_SAMPLES // 2, replace=False) for c in (0, 1)])

    def rep():
        # real clips only: 99 % of them contain both int8 extremes (-128 and 127), so the calibrated input
        # range is exactly [0, 255 * IN_SCALE] = bird_preprocess's quantisation (checked in check_tflite)
        for i in pick:
            yield [dequant(Xtrain[i])[None, None]]

    # fixed batch 1 (the device runs one clip at a time): no SHAPE / PACK / STRIDED_SLICE ops from a dynamic batch
    fn = tf.function(lambda x: model(x, training=False)).get_concrete_function(
        tf.TensorSpec([1, 1, N_FRAMES, N_MELS], tf.float32))
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn], model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    path.write_bytes(conv.convert())


def check_tflite(path):
    it = tf.lite.Interpreter(str(path))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    types = {}
    for t in it.get_tensor_details():
        types[np.dtype(t["dtype"]).name] = types.get(np.dtype(t["dtype"]).name, 0) + 1
    ops = sorted({op["op_name"] for op in it._get_ops_details()})
    info = dict(input_shape=i["shape"].tolist(), input_dtype=np.dtype(i["dtype"]).name,
                input_scale=float(i["quantization"][0]), input_zero_point=int(i["quantization"][1]),
                output_shape=o["shape"].tolist(), output_dtype=np.dtype(o["dtype"]).name,
                output_scale=float(o["quantization"][0]), output_zero_point=int(o["quantization"][1]),
                tensor_types=types, ops=ops, size_bytes=path.stat().st_size)
    info["input_matches_bird_preprocess"] = bool(abs(info["input_scale"] - IN_SCALE) < 1e-7
                                                 and info["input_zero_point"] == IN_ZP)
    info["no_float_tensors"] = "float32" not in types
    arena, peak_op = activation_arena(it)
    info["activation_arena_bytes"] = arena
    info["activation_peak_at"] = peak_op
    info["heap_conservative_bytes"] = arena + info["size_bytes"]      # if the model itself is also held in RAM
    info["heap_limit_bytes"] = HEAP_LIMIT
    info["heap_ok"] = info["heap_conservative_bytes"] <= HEAP_LIMIT
    return info


def activation_arena(it):
    """Peak bytes of activation tensors alive at the same time (what the TFLite Micro arena must hold):
    every tensor lives from the op that writes it (model input: from the start) to the last op that reads it."""
    ops = it._get_ops_details()
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    produced = {o for op in ops for o in op["outputs"]}
    inputs = {d["index"] for d in it.get_input_details()}
    act = produced | inputs
    start = {t: -1 for t in inputs}
    end = {t: len(ops) for t in (d["index"] for d in it.get_output_details())}
    for k, op in enumerate(ops):
        for o in op["outputs"]:
            start[o] = k
        for i in op["inputs"]:
            if i in act:
                end[i] = max(end.get(i, k), k)
    size = {t: int(np.prod(tensors[t]["shape"])) * np.dtype(tensors[t]["dtype"]).itemsize for t in act}
    peak, at = 0, ""
    for k, op in enumerate(ops):
        live = sum(size[t] for t in act if start.get(t, -1) <= k <= end.get(t, k))
        if live > peak:
            peak, at = live, f"op {k} {op['op_name']}"
    return int(peak), at


def run_tflite(path, Xq):
    """P(bird) from the int8 model, fed the stored int8 features exactly as the device does."""
    it = tf.lite.Interpreter(str(path))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    p = np.empty(len(Xq), np.float32)
    for k in range(len(Xq)):                                   # batch 1, as on the device
        it.set_tensor(i["index"], Xq[k][None, None])
        it.invoke()
        p[k] = (float(it.get_tensor(o["index"])[0, 0]) - o["quantization"][1]) * o["quantization"][0]
    return p


def scores(p_bird, y, cls, thr):
    pred = np.where(p_bird >= thr, 0, 1)
    bird, neg = y == 0, y == 1
    r = dict(threshold=round(float(thr), 3), accuracy=float((pred == y).mean()),
             bird_recall=float((pred[bird] == 0).mean()), false_positive_rate=float((pred[neg] == 0).mean()))
    per_bird = {c: float((pred[bird & (cls == c)] == 0).mean()) for c in np.unique(cls[bird])}
    per_neg = {c: float((pred[neg & (cls == c)] == 0).mean()) for c in np.unique(cls[neg])}
    return r, per_bird, per_neg


def pick_threshold(p, y):
    """Best balanced accuracy on val; ties -> higher threshold (fewer false alarms)."""
    best, best_t = -1, 0.5
    for t in np.arange(0.30, 0.96, 0.01):
        pred = np.where(p >= t, 0, 1)
        ba = 0.5 * ((pred[y == 0] == 0).mean() + (pred[y == 1] == 1).mean())
        if ba >= best:
            best, best_t = ba, t
    return best_t


def pick_threshold_fa(p_neg, group, offset):
    """Lowest threshold (= most birds caught) whose smoothed false alarms per hour on val stay <= FA_BUDGET."""
    for t in np.arange(0.30, 0.96, 0.01):
        if smoothed_rate(p_neg, group, offset, t)[0] * WINDOWS_PER_HOUR <= FA_BUDGET:
            return round(float(t), 2)
    return 0.95


def smoothed_rate(p, group, offset, thr, k=SMOOTH[0], n=SMOOTH[1], step=2000):
    """Share of runs of n back-to-back clips (same recording, offsets step ms apart) in which >= k say bird."""
    d = pd.DataFrame({"g": group, "o": offset, "a": p >= thr})
    d = d[d.o >= 0].sort_values(["g", "o"])
    runs = alarms = 0
    for _, r in d.groupby("g", sort=False):
        o, a = r.o.values, r.a.values.astype(int)
        for j in range(len(o) - n + 1):
            if o[j + n - 1] - o[j] == (n - 1) * step:
                runs += 1
                alarms += a[j:j + n].sum() >= k
    return alarms / max(1, runs), runs


def false_alarm_table(p, group, offset, thresholds):
    sm_name = f"fp {SMOOTH[0]}-of-{SMOOTH[1]}"
    L = [f"  {'threshold':>9s} {'fp/window':>10s} {'FA/hour':>8s}   {sm_name:>10s} {'FA/hour':>8s} {'(runs)':>8s}"]
    for thr in thresholds:
        fp = float((p >= thr).mean())
        sm, n = smoothed_rate(p, group, offset, thr)
        L.append(f"  {thr:9.2f} {fp:10.5f} {fp * WINDOWS_PER_HOUR:8.2f}   {sm:10.5f} {sm * WINDOWS_PER_HOUR:8.2f} {n:8d}")
    return L


def main():
    global BLOCKS
    name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("run_%Y%m%d_%H%M")
    if "--wide" in sys.argv:
        BLOCKS = BLOCKS_WIDE
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(SEED)
    Xtr, ytr, ctr, _, _ = load("train")
    ctr = ctr.astype(str)
    Xva, yva, cva, fva, mva = load("val")
    Xte, yte, cte, fte, _ = load("test")
    hard = np.ones(len(ytr), np.float32)
    if HARD_RUN and not HARD_ALL:                            # balanced-train negatives the earlier model calls bird
        neg = np.flatnonzero(ytr == 1)
        ph = run_tflite(RUNS / HARD_RUN / "model_int8.tflite", Xtr[neg])
        hard[neg[ph >= HARD_NEG[0]]] = HARD_NEG[1]
        print(f"{HARD_RUN} false alarms among balanced-train negatives: {int((ph >= HARD_NEG[0]).sum())} "
              f"-> weight {HARD_NEG[1]}", flush=True)
    if EXTRA_SETS:
        e = load_extra("train")
        Xtr = np.concatenate([Xtr, e["X"]])
        ytr = np.concatenate([ytr, e["y"]])
        ctr = np.concatenate([ctr, e["cls"].astype(str)])
        w = np.where(e["p_prev"] >= HARD_NEG[0], HARD_NEG[1], 1).astype(np.float32)
        board = np.char.startswith(e["group"].astype(str), "GPXREC")
        w[board] *= BOARD_NEG_W                              # real board audio counts more than any copy
        print(f"board-recorded extra negatives: {int(board.sum())} (weight x{BOARD_NEG_W})", flush=True)
        hard = np.concatenate([hard, w])
        print(f"extra train negatives {len(e['y'])} ({int((e['p_prev'] >= HARD_NEG[0]).sum())} hard, "
              f"weight {HARD_NEG[1]}): " + ", ".join(f"{c} {n}" for c, n in zip(*np.unique(e["cls"], return_counts=True))
                                                     if c in ("cricket", "cicada", "frog", "insects", "cat",
                                                              "office_ambient", "conversation")), flush=True)
        del e
    if QUIET_SET:
        e = load_extra("train", [QUIET_SET])
        Xtr, ytr = np.concatenate([Xtr, e["X"]]), np.concatenate([ytr, e["y"]])
        ctr = np.concatenate([ctr, e["cls"].astype(str)])
        hard = np.concatenate([hard, np.ones(len(e["y"]), np.float32)])
        print(f"quiet copies {len(e['y'])} (bird {int((e['y'] == 0).sum())}, not bird {int((e['y'] == 1).sum())})",
              flush=True)
        del e
    for s in MIXED_SETS:
        e = load_extra("train", [s])
        Xtr, ytr = np.concatenate([Xtr, e["X"]]), np.concatenate([ytr, e["y"]])
        ctr = np.concatenate([ctr, e["cls"].astype(str)])
        hard = np.concatenate([hard, np.full(len(e["y"]), SET_WEIGHT.get(s, 1.0), np.float32)])
        print(f"{s} train {len(e['y'])} (bird {int((e['y'] == 0).sum())}, not bird {int((e['y'] == 1).sum())})",
              flush=True)
        del e
    ev, et = load_extra("val"), load_extra("test")
    # device recordings split by time: the early part of a held-out recording moves into training
    if DEVICE_TIME_SPLIT:
        man = pd.read_csv(PROJECT / "Bird_Balanced_Dataset" / "balanced_manifest.csv", keep_default_na=False).set_index("path")
        moved = []

        def early(groups, offsets_ms):
            g = pd.Series(groups).astype(str).str.replace("\\", "/")
            m = np.zeros(len(g), bool)
            for key, frac in DEVICE_TIME_SPLIT.items():
                hit = g.str.contains(key, regex=False).values
                if hit.any():
                    dur = sf.info(g[hit].iloc[0]).duration
                    m |= hit & (np.asarray(offsets_ms) < frac * dur * 1000)
            return m, g.str.contains("|".join(DEVICE_TIME_SPLIT), regex=False).values

        for nm in ("val", "test"):
            X_, y_, c_, f_ = (Xva, yva, cva, fva) if nm == "val" else (Xte, yte, cte, fte)
            src = man.loc[f_]
            off = src.source.str.extract(r"_(\d+)ms\.wav$")[0].fillna(-1).astype(float).values
            mv, _ = early(src.group.values, off)
            Xtr, ytr = np.concatenate([Xtr, X_[mv]]), np.concatenate([ytr, y_[mv]])
            ctr = np.concatenate([ctr, c_[mv].astype(str)])
            hard = np.concatenate([hard, np.ones(int(mv.sum()), np.float32)])
            moved.append(int(mv.sum()))
            keep = ~mv
            if nm == "val":
                Xva, yva, cva, fva = X_[keep], y_[keep], c_[keep], f_[keep]
            else:
                Xte, yte, cte, fte = X_[keep], y_[keep], c_[keep], f_[keep]
        for nm, e in (("val", ev), ("test", et)):
            mv, _ = early(e["group"], e["offset_ms"])
            Xtr, ytr = np.concatenate([Xtr, e["X"][mv]]), np.concatenate([ytr, e["y"][mv]])
            ctr = np.concatenate([ctr, e["cls"][mv].astype(str)])
            hard = np.concatenate([hard, np.ones(int(mv.sum()), np.float32)])
            moved.append(int(mv.sum()))
            for k in list(e):
                e[k] = e[k][~mv]
        print(f"device time split {list(DEVICE_TIME_SPLIT)}: moved to training {moved} "
              "(balanced val, balanced test, extra val, extra test)", flush=True)
    if RUN15:                                                # cleaned GPX10 live sessions: train rows -> train, val rows -> val
        r = load_run15()
        tr, va = r["split"] == "train", r["split"] == "val"
        ytr = ytr.astype(np.int64)
        Xtr, ytr = np.concatenate([Xtr, r["X"][tr]]), np.concatenate([ytr, r["y"][tr]])
        ctr = np.concatenate([ctr, np.char.add("r15:", r["cls"][tr])])
        hard = np.concatenate([hard, r["w"][tr]])
        Xva, yva = np.concatenate([Xva, r["X"][va]]), np.concatenate([yva.astype(np.int64), r["y"][va]])
        cva = np.concatenate([cva.astype(str), np.char.add("r15:", r["cls"][va])])
        fva = np.concatenate([fva.astype(str), r["files"][va]])
        for c in np.unique(r["cls"]):
            m = r["cls"] == c
            print(f"run15 {c:16s} y={int(r['y'][m][0])}: train {int((m & tr).sum()):5d} val {int((m & va).sum()):4d} "
                  f"weighted {r['w'][m & tr].sum():8.0f} hard {int(r['hard'][m].sum())}", flush=True)
        print(f"run15 total: train {int(tr.sum())} (bird {int((tr & (r['y'] == 0)).sum())}), val {int(va.sum())}", flush=True)
        del r
    if HARD_RUN and HARD_ALL:                                # every train negative the earlier model calls bird
        neg = np.flatnonzero(ytr == 1)
        ph = run_tflite(RUNS / HARD_RUN / "model_int8.tflite", Xtr[neg])
        hard[neg[ph >= HARD_NEG[0]]] = np.maximum(hard[neg[ph >= HARD_NEG[0]]], HARD_NEG[1])
        print(f"{HARD_RUN} false alarms among all {len(neg)} train negatives: {int((ph >= HARD_NEG[0]).sum())} "
              f"-> weight {HARD_NEG[1]}", flush=True)
    if HARD_POS_RUN and HARD_POS != 1.0:                    # train birds the earlier model misses
        pos = np.flatnonzero(ytr == 0)
        pp = run_tflite(RUNS / HARD_POS_RUN / "model_int8.tflite", Xtr[pos])
        hard[pos[pp < HARD_NEG[0]]] *= HARD_POS
        print(f"{HARD_POS_RUN} missed train birds: {int((pp < HARD_NEG[0]).sum())} -> weight x{HARD_POS}", flush=True)
    if CLS_WEIGHT:                                           # run10: per-category weight (all sets, cc:/bc: too)
        base = np.char.replace(np.char.replace(ctr, "cc:", ""), "bc:", "")
        assert len(base) == len(ytr)
        for c, wc in CLS_WEIGHT.items():
            hard[base == c] *= wc
        print("category weights: " + ", ".join(f"{c} x{wc} ({int((base == c).sum())})" for c, wc in CLS_WEIGHT.items()),
              flush=True)
    counts = np.bincount(ytr, minlength=2)
    mass = np.array([hard[ytr == 0].sum(), hard[ytr == 1].sum()])
    cw = (len(ytr) / (2 * mass)).astype(np.float32)          # both classes carry the same total weight
    sw = hard * cw[ytr]
    print(f"train {len(ytr)} (bird {counts[0]}, not {counts[1]}) class weights {cw.round(3)}", flush=True)

    soft = teacher_targets(Xtr, ytr) if DISTILL else None    # before the student: teachers are loaded and freed
    model = build_model()
    if INIT_FROM:
        model.load_weights(str(RUNS / INIT_FROM / "model.h5"))
        print(f"warm start from {INIT_FROM}, lr {LR}, epochs {EPOCHS}", flush=True)
    model.summary(print_fn=lambda s: print(s, flush=True))
    n_params = model.count_params()
    steps = EPOCHS * int(np.ceil(len(Xtr) / BATCH))
    model.compile(optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(LR, steps)),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0 if DISTILL else LABEL_SMOOTH),
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    t0 = time.time()
    model.fit(Batches(Xtr, ytr, sw, train=True, soft=soft), validation_data=Batches(Xva, yva), epochs=EPOCHS,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True),
                         tf.keras.callbacks.CSVLogger(str(out / "training_log.csv"))],
              workers=4, use_multiprocessing=False, verbose=2)
    minutes = (time.time() - t0) / 60
    model.save(out / "model.h5")

    tfl = out / "model_int8.tflite"
    to_int8_tflite(model, Xtr, tfl, ytr)
    info = check_tflite(tfl)

    pf_va = model.predict(Batches(Xva, yva), verbose=0)[:, 0]
    pf_te = model.predict(Batches(Xte, yte), verbose=0)[:, 0]
    pq_va, pq_te = run_tflite(tfl, Xva), run_tflite(tfl, Xte)
    pq_ev, pq_et = run_tflite(tfl, ev["X"]), run_tflite(tfl, et["X"])
    pf_ev = model.predict(Batches(ev["X"], ev["y"]), verbose=0)[:, 0]
    thr_f = pick_threshold_fa(pf_ev, ev["group"], ev["offset_ms"])
    thr_q = pick_threshold_fa(pq_ev, ev["group"], ev["offset_ms"])
    cmp_tfl = RUNS / COMPARE_RUN / "model_int8.tflite"
    p_cmp = run_tflite(cmp_tfl, et["X"]) if COMPARE_RUN and cmp_tfl.exists() else None
    res = {}
    for tag, p, thr in [("float", pf_te, thr_f), ("int8", pq_te, thr_q)]:
        res[tag] = {"at_0.5": scores(p, yte, cte, 0.5), "at_val_threshold": scores(p, yte, cte, thr)}

    pd.DataFrame({"file": fte, "cls": cte, "label": yte, "p_bird_float": pf_te.round(4),
                  "p_bird_int8": pq_te.round(4)}).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame({"file": et["files"], "cls": et["cls"], "group": et["group"], "offset_ms": et["offset_ms"],
                  "p_bird_int8": pq_et.round(4),
                  f"p_bird_{COMPARE_RUN or 'prev'}": (p_cmp if p_cmp is not None else et["p_prev"]).round(4)}).to_csv(
        out / "test_extra_negatives.csv", index=False)
    L = [f"run {name}: {n_params} parameters, {minutes:.0f} min training", "",
         "TFLite int8: " + json.dumps(info, indent=1), ""]
    for tag in ("float", "int8"):
        for k in ("at_0.5", "at_val_threshold"):
            r, pb, pn = res[tag][k]
            L.append(f"[{tag} {k}] threshold {r['threshold']}  accuracy {r['accuracy']:.4f}  "
                     f"bird recall {r['bird_recall']:.4f}  false-positive rate {r['false_positive_rate']:.4f}")
    r, pb, pn = res["int8"]["at_val_threshold"]
    L += ["", "int8 at val threshold - bird recall per class:"]
    L += [f"  {c:32s} {v:.3f}" for c, v in sorted(pb.items(), key=lambda kv: kv[1])]
    L += ["", "int8 at val threshold - false-positive rate per non-bird category (worst first):"]
    L += [f"  {c:32s} {v:.3f}" for c, v in sorted(pn.items(), key=lambda kv: -kv[1])]
    thresholds = sorted({0.3, 0.4, 0.5, 0.7, 0.8, 0.9, round(float(thr_q), 2)})
    L += ["", f"int8 false alarms on ALL test-recording negatives never used anywhere ({len(pq_et)} clips, "
              f"{len(np.unique(et['group']))} recordings), {WINDOWS_PER_HOUR} windows/hour:"]
    L += false_alarm_table(pq_et, et["group"], et["offset_ms"], thresholds)
    if p_cmp is not None:
        L += ["", f"same clips, {COMPARE_RUN} int8 model (for comparison):"]
        L += false_alarm_table(p_cmp, et["group"], et["offset_ms"], thresholds)
    L += ["", f"chosen threshold {thr_q:.2f}: lowest with <= {FA_BUDGET} smoothed false alarms/hour on val negatives",
          "int8 bird recall on test at these thresholds:"]
    L += [f"  {thr:5.2f}  {float((pq_te[yte == 0] >= thr).mean()):.4f}" for thr in thresholds]
    fpc = pd.DataFrame({"cls": et["cls"], "fp": pq_et >= thr_q,
                        "cmp": (p_cmp if p_cmp is not None else et["p_prev"]) >= thr_q}).groupby("cls")
    fpc = pd.DataFrame({"size": fpc.size(), "mean": fpc.fp.mean(), "cmp": fpc.cmp.mean()})
    L += ["", f"int8 false-positive rate per category on test-recording negatives at {thr_q:.2f} (worst first)"
              f"   [{COMPARE_RUN or 'prev'} at the same threshold]:"]
    L += [f"  {c:32s} {int(r['size']):6d} {r['mean']:.4f}   [{r['cmp']:.4f}]"
          for c, r in fpc.sort_values("mean", ascending=False).iterrows()]
    if QUIET_SET:                                            # quiet copies (-50 .. -75 dBFS) of held-out test clips
        q = load_extra("test", [QUIET_SET])
        pq_q = run_tflite(tfl, q["X"])
        pc_q = run_tflite(cmp_tfl, q["X"]) if p_cmp is not None else None
        L += ["", f"QUIET test copies ({int((q['y'] == 0).sum())} bird, {int((q['y'] == 1).sum())} not bird; "
                  f"clips scaled to -50 .. -75 dBFS + noise floor): bird recall / false-positive rate"
                  + (f"   [{COMPARE_RUN}]" if pc_q is not None else "")]
        for thr in thresholds:
            line = f"  {thr:5.2f}  recall {float((pq_q[q['y'] == 0] >= thr).mean()):.4f}  fp {float((pq_q[q['y'] == 1] >= thr).mean()):.4f}"
            if pc_q is not None:
                line += f"   [recall {float((pc_q[q['y'] == 0] >= thr).mean()):.4f}  fp {float((pc_q[q['y'] == 1] >= thr).mean()):.4f}]"
            L.append(line)
    cmp_ok = COMPARE_RUN and cmp_tfl.exists()
    for s in MIXED_SETS:                                     # extra held-out birds, GPX-simulated test copies
        q = load_extra("test", [s])
        pq_q = run_tflite(tfl, q["X"])
        pc_q = run_tflite(cmp_tfl, q["X"]) if cmp_ok else None
        yb, yn = q["y"] == 0, q["y"] == 1
        L += ["", f"{s} TEST ({int(yb.sum())} bird, {int(yn.sum())} not bird): bird recall / false-positive rate"
                  + (f"   [{COMPARE_RUN}]" if cmp_ok else "")]
        for thr in thresholds:
            line = f"  {thr:5.2f}  recall {float((pq_q[yb] >= thr).mean()):.4f}" + (
                f"  fp {float((pq_q[yn] >= thr).mean()):.4f}" if yn.any() else "")
            if cmp_ok:
                line += f"   [recall {float((pc_q[yb] >= thr).mean()):.4f}" + (
                    f"  fp {float((pc_q[yn] >= thr).mean()):.4f}]" if yn.any() else "]")
            L.append(line)
        pcls = pd.DataFrame({"c": q["cls"][yb], "h": pq_q[yb] >= thr_q, "hc": (pc_q[yb] if cmp_ok else pq_q[yb]) >= thr_q})
        g = pcls.groupby("c")[["h", "hc"]].mean().sort_values("h")
        L += [f"  recall per class at {thr_q:.2f}   [{COMPARE_RUN}]:"]
        L += [f"    {c:36s} {r.h:.3f}   [{r.hc:.3f}]" for c, r in g.iterrows()]
    for title, key in (("GPX10 office recording, held-out minutes", "GPXREC"),
                       ("device cat recording (Animal_chip/cat.wav), held-out last 30 %", "Animal_chip/cat.wav")):
        m = pd.Series(et["group"]).astype(str).str.replace("\\", "/").str.contains(key, regex=False).values
        if m.any():
            L += ["", f"{title}: {int(m.sum())} clips, {WINDOWS_PER_HOUR} windows/hour"]
            L += false_alarm_table(pq_et[m], et["group"][m], et["offset_ms"][m], thresholds)
            if p_cmp is not None:
                L += [f"  same clips, {COMPARE_RUN}:"]
                L += false_alarm_table(p_cmp[m], et["group"][m], et["offset_ms"][m], thresholds)
    if RUN15 and any((RUN15_DIR / "test").glob("*.npz")):    # held-out live sessions (office 110606, never trained on)
        q = load_run15("test")
        pq_q = run_tflite(tfl, q["X"])
        pc_q = run_tflite(cmp_tfl, q["X"]) if cmp_ok else None
        sess = np.array([s.split("#")[0] for s in q["group"]])
        for g in np.unique(sess):
            m = sess == g
            L += ["", f"run15 TEST live session {g}: {int(m.sum())} windows, "
                      f"1-window alarms at 0.5: {int((pq_q[m] >= 0.5).sum())}"
                      + (f"   [{COMPARE_RUN}: {int((pc_q[m] >= 0.5).sum())}]" if cmp_ok else "")]
            L += false_alarm_table(pq_q[m], q["group"][m], q["offset_ms"][m], thresholds)
    gap = abs(res["float"]["at_0.5"][0]["false_positive_rate"] - res["int8"]["at_0.5"][0]["false_positive_rate"])
    L += ["", f"float vs int8 at 0.5: false-positive rate {res['float']['at_0.5'][0]['false_positive_rate']:.4f} vs "
              f"{res['int8']['at_0.5'][0]['false_positive_rate']:.4f}, bird recall "
              f"{res['float']['at_0.5'][0]['bird_recall']:.4f} vs {res['int8']['at_0.5'][0]['bird_recall']:.4f} "
              f"(gap in false-positive rate {gap:.4f})"]
    text = "\n".join(L)
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
