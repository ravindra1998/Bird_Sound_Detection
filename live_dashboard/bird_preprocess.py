"""
Bird sound detection front end: log-mel + PCEN.

Two classes: Wake-word (bird call, class 0) and Not-wake-word (class 1).

Turns 2 seconds of 16 kHz audio into a (251, 40) feature map, then into the int8 tensor the
bird detection model expects. This is the pipeline the model was trained on; features produced
any other way will not be read correctly by it.

    audio (any rate / channels)
      -> mono float32 @ 16 kHz
      -> last 32000 samples (2.00 s), left-padded with zeros if shorter
      -> gated peak normalisation to 0.95
      -> mel spectrogram: n_fft 256, hop 128, 40 mel bands, |STFT|^2, periodic Hann, centred
      -> x 2**31
      -> PCEN
      -> (251, 40) float32, values in roughly [0, 5]
      -> int8: q = clip(round(P / 0.0194852948) - 128, -128, 127)

Requires numpy, librosa and soundfile.

    from bird_preprocess import preprocess, preprocess_pcm16, model_input
    F = preprocess("clip.wav")          # (251, 40) float32
    x = model_input(F)                  # (1, 1, 251, 40) int8, channels_first
    F = preprocess_pcm16(mic_int16)     # live path from an int16 mic buffer

    python bird_preprocess.py                         # extract both classes under DATASET
    python bird_preprocess.py --dataset D:/other_data # same, for another dataset folder
    python bird_preprocess.py clip.wav                # print shape and stats
    python bird_preprocess.py wavs/ -o feats.npy      # batch a folder -> (N, 251, 40)
    python bird_preprocess.py --selftest              # check against the golden vector
"""
from __future__ import annotations

import numpy as np

# ---- dataset paths: change these to extract another dataset ---------------------------------
DATASET    = r"C:\Users\Ravindra\Desktop\Bird_sound_detection_PCEN\dataset"            # contains the class folders below
OUTPUT_DIR = r"C:\Users\Ravindra\Desktop\Bird_sound_detection_PCEN\features"   # features are saved here
CLASSES    = {"wake-word": 0, "Not-wake-word": 1}   # folder name -> label
AUDIO_EXTS = ("wav", "flac", "mp3", "ogg")
LEN_TOL    = 16             # batch extraction keeps only 2 s files, +/- 1 ms of resampling rounding
# ---------------------------------------------------------------------------------------------

__all__ = [
    "preprocess", "preprocess_pcm16", "preprocess_batch", "preprocess_training",
    "model_input", "load_audio", "take_window", "normalize", "logmel_pcen", "pcen",
    "fix_length_center", "peak_normalize", "quantize_int8", "dequantize", "probability",
    "mel_filterbank", "selftest", "golden_input",
    "SR", "CLIP", "N_FFT", "HOP", "N_MELS", "N_FRAMES",
    "IN_SCALE", "IN_ZP", "OUT_SCALE", "OUT_ZP",
]

SR       = 16000
CLIP     = 32000         # 2.00 s window
N_FFT    = 256
HOP      = 128
N_MELS   = 40
POWER    = 2.0
PRE_GAIN = 2.0 ** 31     # PCEN's bias and eps are absolute, so this sets where the signal
                         # sits on the compression curve. Removing it changes the scale ~18x.
N_FRAMES = 1 + CLIP // HOP      # 251: frames are centred, so the STFT pads n_fft/2 each side

PCEN_GAIN   = 0.98
PCEN_BIAS   = 2.0
PCEN_POWER  = 0.5
PCEN_TCONST = 0.4               # seconds
PCEN_EPS    = 1e-6
_T          = PCEN_TCONST * SR / HOP
PCEN_B      = (np.sqrt(1 + 4 * _T ** 2) - 1) / (2 * _T ** 2)     # 0.019800999975

RMS_GATE    = 0.005
PEAK_GATE   = 1e-4
PEAK_TARGET = 0.95

IN_SCALE  = 0.0194852948        # model input quantisation
IN_ZP     = -128
OUT_SCALE = 0.00390625          # model output quantisation, = 1/256
OUT_ZP    = -128
FEAT_MAX  = IN_SCALE * 255.0    # 4.96875, the largest representable feature value


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Any audio file -> mono float32 in [-1, 1] at `sr`. Resamples and down-mixes."""
    import soundfile as sf
    import librosa
    y, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="polyphase")
    return np.ascontiguousarray(y, dtype=np.float32)


def take_window(y, clip: int = CLIP) -> np.ndarray:
    """Keep the last `clip` samples, padding on the LEFT so audio stays flush with the end
    of the window. Inference uses this placement."""
    y = np.asarray(y, dtype=np.float32)
    return y[-clip:] if len(y) >= clip else np.pad(y, (clip - len(y), 0))


def fix_length_center(y, n: int = CLIP) -> np.ndarray:
    """Pad or crop to `n` samples, centred. The training corpus was cut this way."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == n:
        return y
    if len(y) < n:
        pad = n - len(y)
        return np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
    start = (len(y) - n) // 2
    return y[start:start + n]


def normalize(y) -> np.ndarray:
    """Peak-normalise to 0.95, but only above the gate. Near-silence passes through so the
    noise floor is never amplified. This is what makes the pipeline insensitive to mic gain:
    features are identical across a 90x range of input level."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    rms = float(np.sqrt(np.mean(y ** 2)))
    pk = float(np.max(np.abs(y)))
    if rms > RMS_GATE and pk > PEAK_GATE:
        return (y / pk * PEAK_TARGET).astype(np.float32)
    return y


def peak_normalize(y, peak: float = PEAK_TARGET) -> np.ndarray:
    """Ungated peak normalisation, as used when building the training set."""
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y if m < 1e-8 else (y / m * peak).astype(np.float32)


def mel_filterbank() -> np.ndarray:
    """(40, 129) Slaney-normalised mel filters. Per-band peaks are deliberately unequal --
    do not substitute unit-peak triangles."""
    import librosa
    return librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                               fmin=0.0, fmax=SR / 2, htk=False, norm="slaney")


def pcen(S) -> np.ndarray:
    """Per-channel energy normalisation without librosa. S: (n_mels, n_frames).

        M[0] = b*S[0] + (1 - b)
        M[t] = b*S[t] + (1 - b)*M[t-1]
        P    = (S / (eps + M)**gain + bias)**power - bias**power

    The initial condition is the steady state for a unit-step input, which is what
    scipy.signal.lfilter_zi returns for this filter. Starting from S[:, 0] instead is the
    usual porting bug and skews the first frames of the 251.
    """
    S = np.asarray(S, dtype=np.float64)
    M = np.empty_like(S)
    M[:, 0] = PCEN_B * S[:, 0] + (1.0 - PCEN_B)
    for t in range(1, S.shape[1]):
        M[:, t] = PCEN_B * S[:, t] + (1.0 - PCEN_B) * M[:, t - 1]
    smooth = np.exp(-PCEN_GAIN * (np.log(PCEN_EPS) + np.log1p(M / PCEN_EPS)))   # (eps+M)**-gain
    return (PCEN_BIAS ** PCEN_POWER) * np.expm1(PCEN_POWER * np.log1p(S * smooth / PCEN_BIAS))


def logmel_pcen(w, use_librosa_pcen: bool = True) -> np.ndarray:
    """Windowed, normalised 16 kHz mono audio -> (251, 40) float32."""
    import librosa
    x = take_window(w)
    S = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=SR, n_fft=N_FFT,
                                       hop_length=HOP, n_mels=N_MELS, power=POWER)
    P = librosa.pcen(S * PRE_GAIN, sr=SR, hop_length=HOP) if use_librosa_pcen \
        else pcen(S.astype(np.float64) * PRE_GAIN)
    return P.T.astype(np.float32)


def preprocess(source, do_normalize: bool = True) -> np.ndarray:
    """Path, or 16 kHz mono float array -> (251, 40) float32. The complete inference front end."""
    y = load_audio(source) if isinstance(source, (str, bytes)) else np.asarray(source, dtype=np.float32)
    x = take_window(y)
    if do_normalize:
        x = normalize(x)
    return logmel_pcen(x)


def preprocess_pcm16(pcm_i16, do_normalize: bool = True) -> np.ndarray:
    """Live path: int16 PCM already at 16 kHz -> (251, 40) float32."""
    return preprocess(np.asarray(pcm_i16, dtype=np.float32) / 32768.0, do_normalize)


def preprocess_training(source, shift_s: float = 0.0, rng=None) -> np.ndarray:
    """Reproduce the training-set window placement: centre the clip in 2 s, optionally shift
    it, peak-normalise ungated, then featurise. Shifts used were +/-0.15 s for positives and
    +/-0.4 s for negatives. The corpus also applied noise/reverb/low-pass augmentation between
    the shift and the normalisation; that is augmentation, not preprocessing."""
    y = load_audio(source) if isinstance(source, (str, bytes)) else np.asarray(source, dtype=np.float32)
    y = fix_length_center(y, CLIP)
    if shift_s:
        m = int(shift_s * SR)
        r = rng if rng is not None else np.random.default_rng()
        off = int(r.integers(-m, m + 1))
        if off:
            out = np.zeros_like(y)
            if off > 0:
                out[off:] = y[:CLIP - off]
            else:
                out[:CLIP + off] = y[-off:]
            y = out
    return logmel_pcen(peak_normalize(y))


class WrongLength(ValueError):
    """Audio is not 2 s long; batch extraction skips it instead of padding or cropping."""


def preprocess_exact(path: str) -> np.ndarray:
    """Path -> (251, 40) float32, only for audio that is exactly 2 s (+/- LEN_TOL samples
    after resampling to 16 kHz). Shorter or longer files raise WrongLength."""
    y = load_audio(path)
    if abs(len(y) - CLIP) > LEN_TOL:
        raise WrongLength(f"{len(y) / SR:.3f} s, need {CLIP / SR:.2f} s")
    return preprocess(y)


def preprocess_batch(paths, progress: bool = False) -> np.ndarray:
    """List of paths -> (N, 251, 40) float32. Unreadable files and files that are not 2 s long
    are skipped with a message."""
    out = []
    for i, p in enumerate(paths):
        try:
            out.append(preprocess_exact(p))
        except Exception as exc:
            print(f"  skip {p}: {type(exc).__name__}: {exc}")
            continue
        if progress and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)
    return np.stack(out).astype(np.float32) if out else np.empty((0, N_FRAMES, N_MELS), np.float32)


def quantize_int8(F) -> np.ndarray:
    """(..., 251, 40) float -> int8 model input values."""
    q = np.rint(np.asarray(F, dtype=np.float64) / IN_SCALE) + IN_ZP
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(q) -> np.ndarray:
    return ((np.asarray(q, dtype=np.float32) - IN_ZP) * IN_SCALE).astype(np.float32)


def model_input(F) -> np.ndarray:
    """(251, 40) -> (1, 1, 251, 40) int8, channels_first, ready for the tflite interpreter."""
    return quantize_int8(F)[None, None, :, :]


def probability(raw_int8_class0) -> float:
    """Model output value for class 0 -> P(wake word). Class 0 is Wake-word (bird), class 1 is Not-wake-word."""
    return (float(raw_int8_class0) - OUT_ZP) * OUT_SCALE


def golden_input() -> np.ndarray:
    """Fixed 2 s float32 test signal: three tones plus noise from a portable LCG, so the same
    vector can be produced in any language."""
    n = np.arange(CLIP)
    sig = (0.30 * np.sin(2 * np.pi * 440 * n / SR)
           + 0.15 * np.sin(2 * np.pi * 1200 * n / SR)
           + 0.07 * np.sin(2 * np.pi * 2600 * n / SR))
    s, noise = 12345, np.empty(CLIP)
    for i in range(CLIP):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        noise[i] = ((s % 601) - 300) / 30000.0
    return np.clip(sig + noise, -1.0, 1.0).astype(np.float32)


GOLDEN = {"shape": (N_FRAMES, N_MELS), "float_sum": 5094.2715, "int8_sum": -1028403}


def selftest() -> bool:
    y = golden_input()
    F = preprocess(y)
    q = quantize_int8(F)
    fs, qs = float(F.sum()), int(q.astype(np.int64).sum())
    print(f"golden input : {len(y)} samples  rms={np.sqrt(np.mean(y**2)):.6f}  peak={np.max(np.abs(y)):.6f}")
    print(f"  features   : {F.shape} {F.dtype}  range [{F.min():.6f}, {F.max():.6f}]  mean {F.mean():.6f}")
    print(f"  float sum  : {fs:.4f}   (expected {GOLDEN['float_sum']} +/- 0.05)")
    print(f"  int8  sum  : {qs}   (expected {GOLDEN['int8_sum']} +/- 40)")
    ok = (F.shape == GOLDEN["shape"]
          and abs(fs - GOLDEN["float_sum"]) < 0.05
          and abs(qs - GOLDEN["int8_sum"]) < 40)
    d = float(np.max(np.abs(logmel_pcen(normalize(take_window(y)), use_librosa_pcen=False) - F)))
    print(f"  librosa-free PCEN agrees to {d:.3e}")
    print(f"  clipped above {FEAT_MAX:.5f}: {100*(F > FEAT_MAX).mean():.4f} %")
    print(f"GOLDEN MATCH: {ok}")
    return bool(ok and d < 1e-5)


def list_audio(folder) -> list:
    import glob
    import os
    return sorted(f for ext in AUDIO_EXTS
                  for f in glob.glob(os.path.join(folder, "**", "*." + ext), recursive=True))


def extract_dataset(dataset: str = DATASET, out_dir: str = OUTPUT_DIR) -> int:
    """Extract features for every class folder under `dataset` in one run.

    Saves to `out_dir`:
        <class>.npy     (N, 251, 40) float32, one per class folder
        features.npz    X (N, 251, 40) float32, y (N,) int64 labels, files (N,) source paths
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    X_all, y_all, f_all = [], [], []
    for name, label in CLASSES.items():
        folder = os.path.join(dataset, name)
        if not os.path.isdir(folder):
            print(f"missing class folder: {folder}")
            return 1
        files = list_audio(folder)
        print(f"[{name}] label {label}: {len(files)} files", flush=True)
        feats, kept, n_len = [], [], 0
        for i, p in enumerate(files):
            try:
                feats.append(preprocess_exact(p))
                kept.append(p)
            except WrongLength as exc:
                print(f"  skip {p}: {exc}")
                n_len += 1
            except Exception as exc:
                print(f"  skip {p}: {type(exc).__name__}: {exc}")
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
        F = np.stack(feats).astype(np.float32) if feats else np.empty((0, N_FRAMES, N_MELS), np.float32)
        np.save(os.path.join(out_dir, f"{name}.npy"), F)
        print(f"  saved {name}.npy  {F.shape}  skipped {len(files) - len(kept)} "
              f"({n_len} not 2 s, {len(files) - len(kept) - n_len} unreadable)")
        X_all.append(F)
        y_all.append(np.full(len(F), label, dtype=np.int64))
        f_all.extend(kept)
    X, y = np.concatenate(X_all), np.concatenate(y_all)
    np.savez(os.path.join(out_dir, "features.npz"), X=X, y=y, files=np.array(f_all))
    print(f"saved features.npz  X {X.shape}  y {y.shape}  "
          + "  ".join(f"{n}={int((y == l).sum())}" for n, l in CLASSES.items()))
    print(f"output: {out_dir}")
    return 0


def _main(argv=None) -> int:
    import argparse
    import os
    ap = argparse.ArgumentParser(description="Bird sound detection preprocessing (log-mel + PCEN)")
    ap.add_argument("input", nargs="?", help="audio file, or a directory to batch")
    ap.add_argument("-o", "--out", help="save features to this .npy")
    ap.add_argument("--dataset", default=DATASET, help="dataset folder holding the class folders")
    ap.add_argument("--out-dir", default=OUTPUT_DIR, help="where the dataset features are saved")
    ap.add_argument("--int8", action="store_true", help="save int8 model-input values instead of float")
    ap.add_argument("--no-normalize", action="store_true", help="skip the gated peak normalisation")
    ap.add_argument("--selftest", action="store_true", help="check against the golden vector")
    a = ap.parse_args(argv)

    if a.selftest:
        return 0 if selftest() else 1
    if not a.input:
        return extract_dataset(a.dataset, a.out_dir)

    if os.path.isdir(a.input):
        files = list_audio(a.input)
        if not files:
            print(f"no audio files under {a.input}")
            return 1
        print(f"{len(files)} files -> features")
        F = preprocess_batch(files, progress=True)
        print(f"batch {F.shape} float32  range [{F.min():.4f}, {F.max():.4f}]  mean {F.mean():.4f}")
    else:
        F = preprocess(a.input, do_normalize=not a.no_normalize)
        print(f"{a.input}")
        print(f"  features {F.shape} {F.dtype}  range [{F.min():.6f}, {F.max():.6f}]  mean {F.mean():.6f}")
        print(f"  model input {model_input(F).shape} int8   clipped {100*(F > FEAT_MAX).mean():.3f} %")

    if a.out:
        data = quantize_int8(F) if a.int8 else F.astype(np.float32)
        np.save(a.out, data)
        print(f"saved {a.out}  {data.shape} {data.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
