"""
Bird sound detection front end at 8 kHz: log-mel + PCEN. The 8 kHz twin of ../bird_preprocess.py (16 kHz).

LOW-POWER front end: the same STFT sizes in samples as the 16 kHz front end (n_fft 256, hop 128), so at 8 kHz
the 2 s window gives 126 frames instead of 251 -> about half the model compute on the GPX10 (run7q architecture:
~2.25 s instead of ~4.47 s per window, device_cost.py).

                       16 kHz (run7q)                 8 kHz (this file)
    window             32000 samples = 2.00 s         16000 samples = 2.00 s
    n_fft / hop        256 / 128  (16 ms / 8 ms)      256 / 128  (32 ms / 16 ms)
    FFT bin width      62.5 Hz, 129 bins 0-8 kHz      31.25 Hz, 129 bins 0-4 kHz
    mel bands          40 over 0-8 kHz                40 over 0-4 kHz
    frames             251                            126
    PCEN               same constants (time constant 0.4 s = 25 frames here, 50 at 16 kHz)
    int8 input         scale 0.0194852948, zero point -128 (same)
    model input        (1, 1, 251, 40) int8           (1, 1, 126, 40) int8

(2026-11-10: replaces the first 8 kHz version, n_fft 128 / hop 64 / 251 frames = no power saving.)

    audio (any rate / channels)
      -> mono float32 @ 8 kHz (polyphase resampling)
      -> last 16000 samples (2.00 s), left-padded with zeros if shorter
      -> gated peak normalisation to 0.95
      -> mel spectrogram: n_fft 256, hop 128, 40 mel bands 0-4 kHz, |STFT|^2, periodic Hann, centred
      -> x PRE_GAIN
      -> PCEN
      -> (126, 40) float32
      -> int8: q = clip(round(P / 0.0194852948) - 128, -128, 127)

    from bird_preprocess_8k import preprocess, model_input, from_16k
    F = preprocess("clip.wav")              # any file -> (251, 40) float32
    F = preprocess(from_16k(y16))           # 16 kHz array (e.g. the GPX10 mic) -> decimate -> features
    python bird_preprocess_8k.py clip.wav   # print shape and stats
    python bird_preprocess_8k.py --selftest # golden vector
"""
from __future__ import annotations

import numpy as np

SR       = 8000
CLIP     = 16000         # 2.00 s window
N_FFT    = 256
HOP      = 128
N_MELS   = 40
POWER    = 2.0
PRE_GAIN = 2.0 ** 31            # same n_fft as 16 kHz -> same |STFT|^2 scale for a tone (checked on real clips)
N_FRAMES = 1 + CLIP // HOP      # 126

PCEN_GAIN   = 0.98
PCEN_BIAS   = 2.0
PCEN_POWER  = 0.5
PCEN_TCONST = 0.4               # seconds
PCEN_EPS    = 1e-6
_T          = PCEN_TCONST * SR / HOP          # 25 frames = 0.4 s, as at 16 kHz
PCEN_B      = (np.sqrt(1 + 4 * _T ** 2) - 1) / (2 * _T ** 2)

RMS_GATE    = 0.005
PEAK_GATE   = 1e-4
PEAK_TARGET = 0.95

IN_SCALE  = 0.0194852948
IN_ZP     = -128
OUT_SCALE = 0.00390625
OUT_ZP    = -128
FEAT_MAX  = IN_SCALE * 255.0
LEN_TOL   = 8


def resample(y, orig_sr: int, target_sr: int = SR) -> np.ndarray:
    """Polyphase resampling (anti-alias low-pass included), as librosa res_type='polyphase'."""
    if orig_sr == target_sr:
        return np.ascontiguousarray(y, dtype=np.float32)
    import librosa
    return np.ascontiguousarray(librosa.resample(np.asarray(y, np.float32), orig_sr=orig_sr,
                                                 target_sr=target_sr, res_type="polyphase"), dtype=np.float32)


def from_16k(y16) -> np.ndarray:
    """16 kHz float audio -> 8 kHz (what an 8 kHz device path would do: low-pass + decimate by 2)."""
    return resample(y16, 16000, SR)


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    import soundfile as sf
    y, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return resample(y, file_sr, sr)


def take_window(y, clip: int = CLIP) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return y[-clip:] if len(y) >= clip else np.pad(y, (clip - len(y), 0))


def fix_length_center(y, n: int = CLIP) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if len(y) == n:
        return y
    if len(y) < n:
        pad = n - len(y)
        return np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
    start = (len(y) - n) // 2
    return y[start:start + n]


def normalize(y) -> np.ndarray:
    """Gated peak normalisation, as the 16 kHz front end."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    rms = float(np.sqrt(np.mean(y ** 2)))
    pk = float(np.max(np.abs(y)))
    if rms > RMS_GATE and pk > PEAK_GATE:
        return (y / pk * PEAK_TARGET).astype(np.float32)
    return y


def mel_filterbank() -> np.ndarray:
    """(40, 129) Slaney-normalised mel filters, 0-4 kHz."""
    import librosa
    return librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=SR / 2, htk=False, norm="slaney")


def pcen(S) -> np.ndarray:
    """librosa-free PCEN, S: (n_mels, n_frames); same recursion as bird_preprocess.pcen."""
    S = np.asarray(S, dtype=np.float64)
    M = np.empty_like(S)
    M[:, 0] = PCEN_B * S[:, 0] + (1.0 - PCEN_B)
    for t in range(1, S.shape[1]):
        M[:, t] = PCEN_B * S[:, t] + (1.0 - PCEN_B) * M[:, t - 1]
    smooth = np.exp(-PCEN_GAIN * (np.log(PCEN_EPS) + np.log1p(M / PCEN_EPS)))
    return (PCEN_BIAS ** PCEN_POWER) * np.expm1(PCEN_POWER * np.log1p(S * smooth / PCEN_BIAS))


def logmel_pcen(w, use_librosa_pcen: bool = True) -> np.ndarray:
    """Windowed, normalised 8 kHz mono audio -> (126, 40) float32."""
    import librosa
    x = take_window(w)
    S = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, power=POWER, fmin=0.0, fmax=SR / 2)
    P = librosa.pcen(S * PRE_GAIN, sr=SR, hop_length=HOP, time_constant=PCEN_TCONST, gain=PCEN_GAIN,
                     bias=PCEN_BIAS, power=PCEN_POWER, eps=PCEN_EPS) if use_librosa_pcen \
        else pcen(S.astype(np.float64) * PRE_GAIN)
    return P.T.astype(np.float32)


def preprocess(source, do_normalize: bool = True) -> np.ndarray:
    """Path, or 8 kHz mono float array -> (126, 40) float32. The complete inference front end."""
    y = load_audio(source) if isinstance(source, (str, bytes)) else np.asarray(source, dtype=np.float32)
    x = take_window(y)
    if do_normalize:
        x = normalize(x)
    return logmel_pcen(x)


def preprocess_pcm16(pcm_i16, do_normalize: bool = True) -> np.ndarray:
    """Live path: int16 PCM already at 8 kHz -> (126, 40) float32."""
    return preprocess(np.asarray(pcm_i16, dtype=np.float32) / 32768.0, do_normalize)


def preprocess_16k(y16, do_normalize: bool = True) -> np.ndarray:
    """16 kHz float audio (2 s) -> 8 kHz -> features. Used for every clip of the 8 kHz training set."""
    return preprocess(from_16k(y16), do_normalize)


def quantize_int8(F) -> np.ndarray:
    q = np.rint(np.asarray(F, dtype=np.float64) / IN_SCALE) + IN_ZP
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(q) -> np.ndarray:
    return ((np.asarray(q, dtype=np.float32) - IN_ZP) * IN_SCALE).astype(np.float32)


def model_input(F) -> np.ndarray:
    """(126, 40) -> (1, 1, 126, 40) int8, channels_first."""
    return quantize_int8(F)[None, None, :, :]


def golden_input() -> np.ndarray:
    """Fixed 2 s 8 kHz test signal: the 16 kHz golden tones (440, 1200, 2600 Hz) plus LCG noise."""
    n = np.arange(CLIP)
    sig = (0.30 * np.sin(2 * np.pi * 440 * n / SR)
           + 0.15 * np.sin(2 * np.pi * 1200 * n / SR)
           + 0.07 * np.sin(2 * np.pi * 2600 * n / SR))
    s, noise = 12345, np.empty(CLIP)
    for i in range(CLIP):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        noise[i] = ((s % 601) - 300) / 30000.0
    return np.clip(sig + noise, -1.0, 1.0).astype(np.float32)


GOLDEN = {"shape": (N_FRAMES, N_MELS), "float_sum": 2432.2769, "int8_sum": -520307}


def selftest() -> bool:
    y = golden_input()
    F = preprocess(y)
    q = quantize_int8(F)
    d = float(np.max(np.abs(logmel_pcen(normalize(take_window(y)), use_librosa_pcen=False) - F)))
    print(f"golden input : {len(y)} samples @ {SR} Hz")
    print(f"  features   : {F.shape}  range [{F.min():.6f}, {F.max():.6f}]  mean {F.mean():.6f}")
    print(f"  float sum  : {float(F.sum()):.4f}   int8 sum {int(q.astype(np.int64).sum())}")
    print(f"  librosa-free PCEN agrees to {d:.3e}")
    print(f"  clipped above {FEAT_MAX:.5f}: {100 * (F > FEAT_MAX).mean():.4f} %")
    ok = F.shape == GOLDEN["shape"] and d < 1e-4
    if GOLDEN["float_sum"] is not None:
        ok = ok and abs(float(F.sum()) - GOLDEN["float_sum"]) < 0.05 \
             and abs(int(q.astype(np.int64).sum()) - GOLDEN["int8_sum"]) < 40
    print(f"SELFTEST: {ok}")
    return bool(ok)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        raise SystemExit(0 if selftest() else 1)
    for p in sys.argv[1:]:
        F = preprocess(p)
        print(f"{p}\n  features {F.shape}  range [{F.min():.4f}, {F.max():.4f}]  mean {F.mean():.4f}  "
              f"clipped {100 * (F > FEAT_MAX).mean():.3f} %")
