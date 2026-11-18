r"""
Quantization-aware fine-tune (QAT) of an 8 kHz run (user 2026-11-12: "we can do QAT as well so it won't lose accuracy").
Same method as ../qat_finetune.py (run15q): start from runs/<src>/model.h5, train QAT_EPOCHS epochs with simulated int8
(tfmot quantize_model, pooling keeps its input scale), low learning rate, then export the int8 TFLite.
Data and per-row weights are exactly the source round's (train_8k_final.load_split + runs/<src>/hard_weights.npy),
so a round-3 QAT keeps the round-3 emphasis (cat/frog x6, hard negatives x5, ...).
Output runs/<dst>/: model_int8.tflite, model.h5 (float source copy), qat_log.csv, qat_weights.h5,
report.txt (same format as train_8k_final) + qat_compare.txt (float vs PTQ int8 vs QAT int8, val and test).

    python qat_8k.py r8k_3 r8k_3q
"""
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

import train_8k_final as T          # sets the 8 kHz model / feature settings on train_bird_model
import qat_finetune as qf           # quantize(), convert() - shares the same train_bird_model module

t = T.t
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 5))
QAT_LR = float(os.environ.get("QAT_LR", 1e-4))


def main():
    src, dst = t.RUNS / sys.argv[1], t.RUNS / sys.argv[2]
    dst.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(t.SEED)
    model = tf.keras.models.load_model(src / "model.h5", compile=False)
    drops = T.drop_rows()
    tr, va, te = (T.load_split(s, drops) for s in ("train", "val", "test"))
    ytr = tr["y"]
    w = np.ones(len(ytr), np.float32) if T.SMOKE else np.load(src / "hard_weights.npy")
    assert len(w) == len(ytr), f"weights {len(w)} != train rows {len(ytr)} (data changed since {src.name})"
    mass = np.array([w[ytr == 0].sum(), w[ytr == 1].sum()])
    sw = (w * (len(ytr) / (2 * mass)).astype(np.float32)[ytr]).astype(np.float32)
    print(f"QAT {src.name} -> {dst.name}: train {len(ytr)} (bird {int((ytr == 0).sum())}), epochs {QAT_EPOCHS}, "
          f"lr {QAT_LR}, weights from {src.name}/hard_weights.npy", flush=True)

    q = qf.quantize(model)
    steps = QAT_EPOCHS * int(np.ceil(len(ytr) / t.BATCH))
    q.compile(optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(QAT_LR, steps)),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=t.LABEL_SMOOTH), metrics=["accuracy"])
    t0 = time.time()
    q.fit(t.Batches(tr["X"], ytr, sw, train=True), validation_data=t.Batches(va["X"], va["y"]), epochs=QAT_EPOCHS,
          callbacks=[tf.keras.callbacks.CSVLogger(str(dst / "qat_log.csv"))], workers=4, verbose=2)
    minutes = (time.time() - t0) / 60
    q.save_weights(str(dst / "qat_weights.h5"))
    tfl = dst / "model_int8.tflite"
    rng = np.random.default_rng(t.SEED)
    rep = np.concatenate([rng.choice(np.flatnonzero(ytr == c), min(1000, int((ytr == c).sum())), replace=False) for c in (0, 1)])
    qf.convert(q, tfl, tr["X"][rep])
    shutil.copy(src / "model.h5", dst / "model.h5")
    info = t.check_tflite(tfl)
    del tr

    L = [f"QAT of {src.name}: {QAT_EPOCHS} epochs, lr {QAT_LR}, {minutes:.0f} min", ""]
    for tag, d in (("val", va), ("test", te)):
        pfl = model.predict(t.Batches(d["X"], d["y"]), verbose=0)[:, 0]
        for lab, p in (("float", pfl), ("int8 PTQ", t.run_tflite(src / "model_int8.tflite", d["X"])),
                       ("int8 QAT", t.run_tflite(tfl, d["X"]))):
            b, n = d["y"] == 0, d["y"] == 1
            L.append(f"{tag:5s} {lab:9s} @ {T.THR}: bird recall {(p[b] >= T.THR).mean():.4f}  "
                     f"false-positive rate {(p[n] >= T.THR).mean():.4f}")
        L.append("")
    (dst / "qat_compare.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)
    T.report(dst.name, dst, tfl, info, model.count_params(), minutes, va, te)


if __name__ == "__main__":
    main()
