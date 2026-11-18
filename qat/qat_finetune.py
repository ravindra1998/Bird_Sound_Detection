r"""
Quantization-aware fine-tuning: start from a trained float run, train a few epochs with simulated int8
(tfmot.quantization.keras.quantize_model), export the int8 TFLite. Fixes the post-training int8 loss
(run7: float 99.3 % recall / 0.77 % false positives -> int8 97.3 % / 1.49 %).

    C:\Users\Ravindra\anaconda3\envs\tf210\python.exe qat_finetune.py run7 run7q
Training data: the same sets as train_bird_model (balanced train + EXTRA_SETS + MIXED_SETS); weights: both
classes the same total, negatives the float model scores >= 0.5 x HARD_NEG[1]. Output runs/<out>/:
model_int8.tflite, qat_log.csv, qat_report.txt (float vs QAT-int8 on val, device checks).
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import train_bird_model as t

import os
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 5))
QAT_LR     = float(os.environ.get("QAT_LR", 1e-4))
HARD_POS   = float(os.environ.get("QAT_HARD_POS", 1.0))   # birds the float model scores < 0.5 get this weight
BIRD_MASS  = float(os.environ.get("QAT_BIRD_MASS", 1.0))  # total bird weight / total non-bird weight
Q = tfmot.quantization.keras


class SameScale(Q.QuantizeConfig):
    """No quantizers: output keeps the input's int8 scale. Default QAT gives AveragePooling2D its own output
    scale, which the converter turns into a QUANTIZE op - not supported by the device runtime."""
    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


def quantize(model):
    """QAT model: default 8-bit scheme, pooling / flatten / dropout share the scale of their input."""
    same = (tf.keras.layers.AveragePooling2D, tf.keras.layers.Flatten, tf.keras.layers.Dropout)
    ann = tf.keras.models.clone_model(model, clone_function=lambda l: l.__class__.from_config(l.get_config())
                                      if isinstance(l, tf.keras.layers.InputLayer) else
                                      Q.quantize_annotate_layer(l.__class__.from_config(l.get_config()),
                                                                SameScale() if isinstance(l, same) else None))
    ann.set_weights(model.get_weights())                     # clone_model starts from fresh weights
    x = np.random.default_rng(0).uniform(0, 4.9, (4, 1, t.N_FRAMES, t.N_MELS)).astype(np.float32)
    assert np.allclose(ann(x, training=False), model(x, training=False), atol=1e-5), "weights not copied"
    with Q.quantize_scope({"SameScale": SameScale}):
        return Q.quantize_apply(ann)


def train_data(with_cls=False):
    b = t.load("train")
    X, y, c = [b[0]], [b[1]], [b[2].astype(str)]
    for s in t.EXTRA_SETS + t.MIXED_SETS:
        e = t.load_extra("train", [s])
        X.append(e["X"])
        y.append(e["y"])
        c.append(e["cls"].astype(str))
    w = [np.ones(sum(len(v) for v in y), np.float32)]
    if t.RUN15:                                              # run15 live sessions + playback: train rows, own weights
        r = t.load_run15()
        tr = r["split"] == "train"
        X.append(r["X"][tr]); y.append(r["y"][tr]); c.append(np.char.add("r15:", r["cls"][tr])); w.append(r["w"][tr])
        print(f"run15 train rows {int(tr.sum())} (bird {int((r['y'][tr] == 0).sum())})", flush=True)
    if with_cls:
        return np.concatenate(X), np.concatenate(y), np.concatenate(c), np.concatenate(w)
    return np.concatenate(X), np.concatenate(y)


def convert(qmodel, path, Xrep):
    fn = tf.function(lambda x: qmodel(x, training=False)).get_concrete_function(
        tf.TensorSpec([1, 1, t.N_FRAMES, t.N_MELS], tf.float32))
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn], qmodel)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # required by the converter for full-int8 from a concrete function; the ranges come from the QAT fake-quant
    conv.representative_dataset = lambda: ([t.dequant(x)[None, None]] for x in Xrep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    path.write_bytes(conv.convert())


def main():
    src, dst = t.RUNS / sys.argv[1], t.RUNS / sys.argv[2]
    dst.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(t.SEED)
    model = tf.keras.models.load_model(src / "model.h5", compile=False)
    Xtr, ytr, ctr, base = train_data(with_cls=True)
    pf = model.predict(t.Batches(Xtr, ytr), batch_size=t.BATCH, verbose=0)[:, 0]
    hard = np.where((ytr == 1) & (pf >= t.HARD_NEG[0]), t.HARD_NEG[1], 1.0).astype(np.float32)
    hard[(ytr == 0) & (pf < t.HARD_NEG[0])] = HARD_POS
    hard *= base                                             # run15 per-recording weights (whistle x3, frog x5, ...)
    if t.CLS_WEIGHT:
        base = np.char.replace(np.char.replace(ctr, "cc:", ""), "bc:", "")
        for c, wc in t.CLS_WEIGHT.items():
            hard[base == c] *= wc
        print("category weights: " + str(t.CLS_WEIGHT), flush=True)
    mass = np.array([hard[ytr == 0].sum(), hard[ytr == 1].sum()])
    cw = len(ytr) / (2 * mass) * np.array([2 * BIRD_MASS / (1 + BIRD_MASS), 2 / (1 + BIRD_MASS)])
    sw = hard * cw.astype(np.float32)[ytr]
    print(f"train {len(ytr)} (bird {int((ytr == 0).sum())}), hard negatives {int(((ytr == 1) & (hard > 1)).sum())}, "
          f"missed birds {int(((ytr == 0) & (pf < t.HARD_NEG[0])).sum())} x{HARD_POS}, bird mass x{BIRD_MASS}, "
          f"epochs {QAT_EPOCHS}, lr {QAT_LR}", flush=True)
    Xva, yva = t.load("val")[:2]
    soft = t.teacher_targets(Xtr, ytr) if t.DISTILL else None   # run11: same distillation targets as training

    q = quantize(model)
    steps = QAT_EPOCHS * int(np.ceil(len(Xtr) / t.BATCH))
    q.compile(optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(QAT_LR, steps)),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0 if t.DISTILL else t.LABEL_SMOOTH),
              metrics=["accuracy"])
    t0 = time.time()
    q.fit(t.Batches(Xtr, ytr, sw, train=True, soft=soft), validation_data=t.Batches(Xva, yva), epochs=QAT_EPOCHS,
          callbacks=[tf.keras.callbacks.CSVLogger(str(dst / "qat_log.csv"))], workers=4, verbose=2)
    q.save_weights(str(dst / "qat_weights.h5"))
    tfl = dst / "model_int8.tflite"
    rng = np.random.default_rng(t.SEED)
    convert(q, tfl, Xtr[np.concatenate([rng.choice(np.flatnonzero(ytr == c), 1000, replace=False) for c in (0, 1)])])
    shutil.copy(src / "model.h5", dst / "model_float.h5")
    info = t.check_tflite(tfl)

    L = [f"QAT of {src.name}: {QAT_EPOCHS} epochs, lr {QAT_LR}, {(time.time() - t0) / 60:.0f} min", str(info), ""]
    for name, X, y in (("val (balanced)", Xva, yva),
                       ("val device copies", *[np.load(t.FEATURES / "category_copies_val.npz")[k] for k in ("X", "y")])):
        pfl = model.predict(t.Batches(X, y), verbose=0)[:, 0]
        p8_ptq = t.run_tflite(src / "model_int8.tflite", X)
        p8 = t.run_tflite(tfl, X)
        for thr in (0.5, 0.6, 0.7, 0.8):
            row = []
            for lab, p in (("float", pfl), ("int8 PTQ", p8_ptq), ("int8 QAT", p8)):
                row.append(f"{lab} recall {(p[y == 0] >= thr).mean():.4f} fp {(p[y == 1] >= thr).mean():.4f}")
            L.append(f"{name:18s} @{thr}: " + "  |  ".join(row))
        L.append("")
    text = "\n".join(L)
    (dst / "qat_report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
