r"""
Live test dashboard for the 8 kHz low-power bird model (run7q_8k126, input 126 x 40) - the same dashboard as
../live_dashboard.py (imported unchanged: same page, GPX10 serial / microphone / file sources, UART stretch fix,
2-of-3 alarm rule, alarm clips, labelling, missed-bird button), with only the front end swapped:

    audio -> 16 kHz ring (as the 16 kHz dashboard; the GPX10 board streams 16 kHz) -> 2 s window every 1 s
          -> decimate to 8 kHz -> bird_preprocess_8k (n_fft 256, hop 128, 40 mels 0-4 kHz, PCEN) -> int8 -> model

This is exactly how every 8 kHz training clip was made (bird_preprocess_8k.preprocess_16k on the 16 kHz clip).

    C:\Users\Ravindra\anaconda3\envs\tf210\python.exe bird_voice_8khz\live_dashboard_8k.py      # http://127.0.0.1:8766
    ... live_dashboard_8k.py --model bird_voice_8khz\runs\run7q_8k126\model_int8.tflite --threshold 0.5 --rule 1win

Port 8766 by default so it can run next to the 16 kHz dashboard (8765). Only one program can open the GPX10 COM
port at a time. Sessions are saved in bird_voice_8khz/live_sessions/<date_time>/ (session.wav and alarm clips at
16 kHz, as received).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import bird_preprocess as b16       # noqa: E402
import bird_preprocess_8k as b8     # noqa: E402
import live_dashboard as ld         # noqa: E402  (root dashboard, used read-only)

MODEL = HERE / "runs" / "r8k_3" / "model_int8.tflite"     # round 3 (2026-11-12)


class FrontEnd8k:
    """Stands in for bird_preprocess inside live_dashboard: 16 kHz window in, 8 kHz (126, 40) features out."""
    SR, CLIP = b16.SR, b16.CLIP                      # the audio path stays 16 kHz

    @staticmethod
    def preprocess(w16, do_normalize=True):
        return b8.preprocess_16k(b16.take_window(w16), do_normalize)

    quantize_int8 = staticmethod(b8.quantize_int8)


ld.bp = FrontEnd8k
ld.SESSIONS = HERE / "live_sessions"
ld.PAGE = (ld.PAGE
           .replace("F=251,", f"F={b8.N_FRAMES},")
           .replace("<title>Bird Detector Live</title>", "<title>Bird Detector Live 8 kHz</title>")
           .replace("Bird Detector &mdash; Live Test", "Bird Detector 8 kHz (low power) &mdash; Live Test")
           .replace("current 2 s window (40 mel bands)", "current 2 s window at 8 kHz (126 frames, 40 mel bands 0&ndash;4 kHz)"))
assert f"F={b8.N_FRAMES}," in ld.PAGE, "live_dashboard.PAGE changed - feature canvas width not patched"


def check_model(path):
    import tensorflow as tf
    it = tf.lite.Interpreter(str(path))
    shape = tuple(it.get_input_details()[0]["shape"])
    if shape[-2:] != (b8.N_FRAMES, b8.N_MELS):
        sys.exit(f"{path}: input {shape} is not an 8 kHz model ({b8.N_FRAMES}, {b8.N_MELS})")


def main():
    args = sys.argv[1:]
    # --board_sr 8000: board firmware streams 8 kHz -> upsampled x2 into the 16 kHz ring (decimated back to 8 kHz
    # for the features, so the model still sees 8 kHz). Default 16000 = the I2s_Mic_data_collection 16 kHz stream.
    if "--board_sr" in args:
        i = args.index("--board_sr")
        sr = int(args[i + 1])
        del args[i:i + 2]
        if sr == 8000 and "--uart_stretch" not in args:
            args += ["--uart_stretch", "2.0"]
    defaults = {"--model": str(MODEL), "--threshold": "0.5", "--port": "8766"}
    for k, v in defaults.items():
        if k not in args:
            args += [k, v]
    sys.argv = [sys.argv[0]] + args
    check_model(args[args.index("--model") + 1])
    ld.main()


if __name__ == "__main__":
    main()
