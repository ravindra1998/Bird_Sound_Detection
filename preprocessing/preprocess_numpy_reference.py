#!/usr/bin/env python3
"""
Reference pre-processing for AUD_BirdSound_8k_INT8_v1.0 (bird sound detection, 8 kHz).

The model input is the newest 2 s of sound at 8 kHz turned into 126 frames x 40 mel bands (0 to 4 kHz) with
PCEN (per-channel energy normalisation), quantised with THIS model's int8 scale and zero point.
This file is a numpy-only copy of the training front end (which uses librosa);
on the three samples in raw/ both give exactly the same int8 values.

Usage:
    python3 preprocess.py raw/sample_01_bird_nightingale.wav --out processed/sample_01_bird_nightingale_int8.csv
    python3 preprocess.py raw/sample_01_bird_nightingale.wav --out processed/sample_01_bird_nightingale_int8.csv --check

Output: 126 rows (frames, oldest first, one every 16 ms) x 40 columns (mel bands, lowest first) of int8 values.
Flattened row by row (index = frame * 40 + band) this is the model input tensor [1, 1, 126, 40].

Input: 16-bit PCM WAV, mono, 8 kHz, at least 16,000 samples (2 s). Only the newest 16,000 samples are used.

Steps:
    1  int16 -> float32: divide by 32768
    2  window: the newest 16,000 samples (2.00 s)
    3  gated peak normalisation: if RMS > 0.005 and peak > 1e-4, scale so the peak is 0.95; otherwise unchanged
    4  STFT: n_fft 256, hop 128, periodic Hann window, centred (128 zeros added at both ends) -> 126 frames,
       power |X|^2 (129 bins, 0 to 4 kHz)
    5  mel: 40 Slaney-scale bands 0 to 4 kHz, Slaney area normalisation (librosa.filters.mel default)
    6  x 2^31 (PRE_GAIN)
    7  PCEN per band: M[t] = b*E[t] + (1-b)*M[t-1], M[-1] = 1 (so M[0] = b*E[0] + 1 - b), b = 0.0392...
       (time constant 0.4 s = 25 frames)
       out = sqrt(2) * (sqrt(1 + E * (1e-6 + M)^-0.98 / 2) - 1)   [gain 0.98, bias 2, power 0.5, eps 1e-6]
    8  int8: q = clip(round(out / 0.0194852948) - 128, -128, 127)

Only numpy is needed:  pip install numpy
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- PARAMETERS
SAMPLE_RATE_HZ = 8000
WINDOW_SAMPLES = 16000          # 2.00 s
PCM_SCALE = 32768.0             # int16 -> float: divide by 32768
N_FFT = 256                     # 32 ms
HOP = 128                       # 16 ms
N_MELS = 40
N_FRAMES = 1 + WINDOW_SAMPLES // HOP          # 126
F_MIN_HZ, F_MAX_HZ = 0.0, 4000.0
RMS_GATE, PEAK_GATE, PEAK_TARGET = 0.005, 1e-4, 0.95
PRE_GAIN = 2.0 ** 31
PCEN_GAIN, PCEN_BIAS, PCEN_POWER, PCEN_EPS, PCEN_TCONST = 0.98, 2.0, 0.5, 1e-6, 0.4
_T = PCEN_TCONST * SAMPLE_RATE_HZ / HOP       # 25 frames
PCEN_B = (np.sqrt(1 + 4 * _T ** 2) - 1) / (2 * _T ** 2)
INPUT_SCALE = 0.0194852948      # the model's int8 input scale
INPUT_ZERO_POINT = -128         # the model's int8 input zero point


# ----------------------------------------------------------------------------- helpers
def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit(f"Expected 16-bit PCM, got {sw * 8}-bit")
    data = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if ch > 1:
        data = data.reshape(-1, ch)[:, 0]
    return data, sr


def _hz_to_mel(f):                       # Slaney mel scale (librosa htk=False)
    f = np.asanyarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank() -> np.ndarray:
    """[40, 129] Slaney mel filters 0-4 kHz with Slaney area normalisation (= librosa.filters.mel), float32."""
    fft_hz = np.linspace(0, SAMPLE_RATE_HZ / 2, N_FFT // 2 + 1)
    mel_hz = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN_HZ), _hz_to_mel(F_MAX_HZ), N_MELS + 2))
    fdiff = np.diff(mel_hz)
    ramps = mel_hz[:, None] - fft_hz[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_hz[2:N_MELS + 2] - mel_hz[:N_MELS]))[:, None]
    return w.astype(np.float32)


# ----------------------------------------------------------------------------- the pipeline
def preprocess(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Return the int8 model input for ONE window: shape [126, 40]."""
    if sr != SAMPLE_RATE_HZ:
        raise SystemExit(f"Expected {SAMPLE_RATE_HZ} Hz audio, got {sr} Hz.")
    if len(pcm) < WINDOW_SAMPLES:
        raise SystemExit(f"Need at least {WINDOW_SAMPLES} samples (2 s), got {len(pcm)}.")
    x = np.asarray(pcm[-WINDOW_SAMPLES:], dtype=np.float32) / np.float32(PCM_SCALE)      # steps 1-2
    rms, pk = float(np.sqrt(np.mean(x ** 2))), float(np.max(np.abs(x)))                   # step 3
    if rms > RMS_GATE and pk > PEAK_GATE:
        x = (x / pk * PEAK_TARGET).astype(np.float32)
    xp = np.pad(x, N_FFT // 2)                                                             # step 4
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)   # periodic Hann
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(N_FRAMES)[:, None]
    spec = np.fft.rfft(xp[idx] * win, n=N_FFT, axis=1).astype(np.complex64)
    power = (np.abs(spec) ** 2).astype(np.float32)                                          # [126, 129]
    E = (mel_filterbank() @ power.T).astype(np.float64) * PRE_GAIN                          # steps 5-6, [40, 126]
    M = np.empty_like(E)                                                                    # step 7
    M[:, 0] = PCEN_B * E[:, 0] + (1.0 - PCEN_B)
    for t in range(1, N_FRAMES):
        M[:, t] = PCEN_B * E[:, t] + (1.0 - PCEN_B) * M[:, t - 1]
    smooth = np.exp(-PCEN_GAIN * (np.log(PCEN_EPS) + np.log1p(M / PCEN_EPS)))
    P = (PCEN_BIAS ** PCEN_POWER) * np.expm1(PCEN_POWER * np.log1p(E * smooth / PCEN_BIAS))
    P = P.T.astype(np.float32)                                                              # [126, 40]
    q = np.clip(np.rint(P.astype(np.float64) / INPUT_SCALE) + INPUT_ZERO_POINT, -128, 127)  # step 8
    return q.astype(np.int8)


def write_csv(path: Path, mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        for row in np.asarray(mat):
            w.writerow([int(v) for v in row])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path, help="raw .wav file (16-bit PCM, mono, 8 kHz, at least 2 s)")
    ap.add_argument("--out", type=Path, required=True, help="processed .csv or .npy to write")
    ap.add_argument("--check", action="store_true", help="compare with the existing --out file instead of overwriting")
    args = ap.parse_args()

    audio, sr = read_wav(args.raw)
    feats = preprocess(audio, sr)
    print(f"{args.raw.name}: {len(audio) / sr:.2f} s @ {sr} Hz -> features {feats.shape} {feats.dtype}")
    if args.check:
        if not args.out.exists():
            print("MISSING", args.out)
            return 1
        ref = np.load(args.out) if args.out.suffix == ".npy" else np.loadtxt(args.out, delimiter=",")
        ref, got = np.asarray(ref).astype(np.int64).reshape(-1), feats.astype(np.int64).reshape(-1)
        if ref.size == got.size and np.array_equal(ref, got):
            print("MATCH with", args.out)
            return 0
        d = np.abs(ref - got) if ref.size == got.size else None
        print("MISMATCH with", args.out, f"({ref.size} values, expected {got.size})" if d is None
              else f"({int((d > 0).sum())} of {d.size} values differ, max {int(d.max())} LSB)")
        return 1
    if args.out.suffix == ".npy":
        np.save(args.out, feats)
    else:
        write_csv(args.out, feats)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
