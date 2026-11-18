r"""
Live test dashboard for the bird / not-bird int8 model.

Audio (microphone, or a recording played through the same path) -> 16 kHz -> a 2 s window every HOP_S seconds ->
bird_preprocess.preprocess (gated normalisation + log-mel + PCEN, the device front end) -> quantize_int8 ->
int8 TFLite model -> P(bird). Alarm when K of the last N windows are >= threshold (as stream_eval.py), one alarm
per event (REFRACT_S). Every alarm is saved as a 4 s clip to listen to and label (bird / false alarm); a "missed
bird" button logs birds the model did not catch, so the page shows live false alarms per hour and recall.

    C:\Users\Ravindra\anaconda3\envs\tf210\python.exe live_dashboard.py            # then open http://127.0.0.1:8765
    ... live_dashboard.py --model runs/run2/model_int8.tflite --threshold 0.5 --port 8766

Each session is saved in live_sessions/<date_time>/:
    windows.csv   every window: time, P(bird), level, alarm state, threshold
    alarms.csv    every alarm: id, time, peak P(bird), label (bird / false_alarm / unsure), missed-bird marks
    alarm_<id>.wav  4 s around each alarm (16 kHz)
    session.wav   the whole input at 16 kHz (when "record session" is on) - real device-like data for training
"""
import argparse
import base64
import csv
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import bird_preprocess as bp

PROJECT   = Path(__file__).resolve().parent
SESSIONS  = PROJECT / "live_sessions"
SR, CLIP  = bp.SR, bp.CLIP
HOP_S     = 1.0
K, N      = 2, 3
REFRACT_S = 5.0
CLIP_S    = 4.0          # saved audio around an alarm
RING_S    = 30           # seconds of input kept in memory
SERIAL_BAUD = 921600     # GPX10 I2s_Mic_data_collection trace UART
UART_STRETCH = (2257, 2000)   # --uart_stretch: received ~14,175 samples/s -> resampled to true 16 kHz
# The GPX10 trace UART runs slightly slow against the CP210x at 921600: the low 4 bits of every character arrive
# intact, bit 5 lands in bit 6 (and bit 4 is noise) - "," -> "\" or "L", "-" -> "]" or "M", digits -> 0x6X / 0x7X.
# The stream only holds digits, "," and "-", so the character is recovered from its low nibble and bit 5.
SLIP_FIX = bytes((0x30 | (c & 0xF)) if (c & 0x60) == 0x60 and (c & 0xF) <= 9 else
                 (0x20 | (c & 0xF)) if (c & 0x60) == 0x40 and (c & 0xF) in (0xC, 0xD) else c
                 for c in range(256))


class Ring:
    """Last RING_S seconds of 16 kHz audio; `total` = samples written since start."""
    def __init__(self, sr=SR):
        self.buf = np.zeros(RING_S * sr, np.float32)
        self.total = 0
        self.lock = threading.Lock()

    def write(self, x):
        with self.lock:
            n = len(x)
            if n >= len(self.buf):
                self.buf[:] = x[-len(self.buf):]
            else:
                self.buf = np.roll(self.buf, -n)
                self.buf[-n:] = x
            self.total += n

    def last(self, n, end):
        """n samples ending at absolute sample `end` (zeros before the start of the session)."""
        with self.lock:
            back = self.total - end
            hi = len(self.buf) - back
            lo = hi - n
            out = np.zeros(n, np.float32)
            src_lo = max(lo, 0)
            if hi > src_lo:
                out[n - (hi - src_lo):] = self.buf[src_lo:hi]
            return out


class Model:
    def __init__(self, path):
        import tensorflow as tf
        self.path = str(path)
        self.it = tf.lite.Interpreter(self.path)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]

    def p_bird(self, q):
        self.it.set_tensor(self.i["index"], q[None, None])
        self.it.invoke()
        scale, zp = self.o["quantization"]
        return (float(self.it.get_tensor(self.o["index"])[0, 0]) - zp) * scale


class Engine:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        self.subscribers = []
        self.lock = threading.Lock()
        self.running = False
        self.source = None
        self.session = None

    # ---------------------------------------------------------------- session control
    def start(self, source, device=None, file=None, speed=1.0, record=True):
        self.stop()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        d = SESSIONS / stamp
        d.mkdir(parents=True, exist_ok=True)
        self.session = dict(dir=d, name=stamp, source=source, device=device, file=file, started=time.time(),
                            windows=0, alarms=[], missed=[], seconds=0.0)
        self.ring = Ring()
        self.pos = 0                      # end sample of the last window analysed
        self.hits = []                    # recent single-window decisions (for K of N)
        self.last_on = None               # time of the last window with the alarm on
        self.wcsv = open(d / "windows.csv", "w", newline="")
        self.ww = csv.writer(self.wcsv)
        self.ww.writerow(["time_s", "p_bird", "level_db", "alarm_on", "threshold"])
        self.rec = sf.SoundFile(str(d / "session.wav"), "w", SR, 1, "PCM_16") if record else None
        self.__dict__.pop("_st", None)                      # fresh UART resampler state
        if source == "serial":                               # later analysis must know the audio is already corrected
            (d / "audio_info.txt").write_text(f"serial audio stretched x{UART_STRETCH[0]}/{UART_STRETCH[1]} "
                                              f"(UART sample drop corrected, true speed / pitch)\n")
        self.running = True
        self.serial_rate = None
        if source == "mic":
            self.source = threading.Thread(target=self._mic, args=(device,), daemon=True)
        elif source == "serial":
            self.source = threading.Thread(target=self._serial, args=(file,), daemon=True)
        else:
            self.source = threading.Thread(target=self._file, args=(file, speed), daemon=True)
        self.source.start()
        self.worker = threading.Thread(target=self._analyse, daemon=True)
        self.worker.start()
        self.publish(dict(type="session", **self.summary()))

    def stop(self):
        with self.lock:
            if not self.session or self.session.get("closed"):
                return
            self.session["closed"] = True
        self.running = False
        for t in (self.source, self.worker):
            if t is not None and t is not threading.current_thread():
                t.join(timeout=3)
        self.wcsv.close()
        if self.rec is not None:
            self.rec.close()
        self._write_alarms()
        self.publish(dict(type="stopped", **self.summary()))

    # ---------------------------------------------------------------- audio sources (write 16 kHz to the ring)
    def _push(self, x16):
        # 16-bit rounding: the GPX10 delivers int16 PCM, so every source is analysed at the same resolution
        x16 = (np.round(np.clip(x16, -1.0, 32767 / 32768) * 32768) / 32768).astype(np.float32)
        self.ring.write(x16)
        if self.rec is not None:
            self.rec.write(x16)

    def _stretch_push(self, x):
        """UART audio back to true speed: the board's UART delivers only ~14,175 of 16,000 samples/s with the drops
        spread evenly, so the raw stream is ~11 % short and +12.9 % in pitch (1 kHz playback beep -> 1128.5 Hz,
        measured 2026-11-10). Resampled by UART_STRETCH (up/down) in blocks of `down` samples with `down` samples of
        real audio on both sides (as _mic: same as resampling the whole stream, no clicks, ~0.3 s delay)."""
        up, down = UART_STRETCH
        if up == down:
            self._push(x)
            return
        st = self.__dict__.setdefault("_st", dict(pend=np.zeros(0, np.float32), prev=np.zeros(2 * down, np.float32)))
        st["pend"] = np.concatenate([st["pend"], x])
        while len(st["pend"]) >= down:
            z = np.concatenate([st["prev"], st["pend"][:down]])
            st["pend"] = st["pend"][down:]
            y = resample_poly(z, up, down).astype(np.float32)
            self._push(y[up:2 * up])                        # the middle `down` input samples -> `up` output samples
            st["prev"] = z[-2 * down:]

    def _serial(self, port):
        """GPX10 board running I2s_Mic_data_collection: int16 samples printed as "%d," on the UART at 921600 baud
        (16 kHz). Banner / text lines are skipped. The received sample rate is reported: below 16000 /s means the
        UART cannot keep up and the firmware drops samples."""
        import re
        import serial
        num = re.compile(rb"-?\d+")
        try:
            with serial.Serial(port, SERIAL_BAUD, timeout=0.1) as sp:
                sp.reset_input_buffer()
                tail, t0, count = b"", time.time(), 0
                while self.running:
                    data = sp.read(8192)
                    if not data:
                        continue
                    if sum(0x4C <= c <= 0x79 for c in data) > 0.9 * len(data):
                        data = data.translate(SLIP_FIX)          # GPX10 UART bit slip, see SLIP_FIX
                    buf = tail + data
                    cut = buf.rfind(b",")                    # complete tokens end with ","
                    if cut < 0:
                        tail = buf[-64:]
                        continue
                    body, tail = buf[:cut], buf[cut + 1:]
                    vals = [int(t) for t in re.split(rb"[,\s]+", body) if num.fullmatch(t)]   # text lines skipped
                    if vals:
                        x = np.clip(np.array(vals, np.float32), -32768, 32767) / 32768
                        self._stretch_push(x)
                        count += len(vals)
                    if time.time() - t0 >= 5:
                        self.serial_rate = round(count / (time.time() - t0))
                        exp = 16000 * UART_STRETCH[1] / UART_STRETCH[0]
                        self.publish(dict(type="info", message=f"{port}: {self.serial_rate} samples/s received, "
                                          f"stretched x{UART_STRETCH[0] / UART_STRETCH[1]:.4f} to 16 kHz"
                                          + ("" if abs(self.serial_rate - exp) < 0.02 * exp else
                                             f" - WARNING: expected ~{exp:.0f}/s, set --uart_stretch "
                                             f"{16000 / max(self.serial_rate, 1):.4f}")))
                        t0, count = time.time(), 0
        except Exception as e:
            self.publish(dict(type="error", message=f"serial {port}: {type(e).__name__}: {e}"))
            self.running = False

    def _mic(self, device):
        import sounddevice as sd
        info = sd.query_devices(device, "input")
        sr_in = int(info["default_samplerate"])
        g = np.gcd(SR, sr_in)
        up, down = SR // g, sr_in // g
        blk = ctx = sr_in // 10                             # 100 ms blocks; 1600 samples at 16 kHz for 44.1 / 48 kHz
        prev = np.zeros(2 * ctx, np.float32)
        # each block is resampled with 100 ms of real audio on BOTH sides (output delayed by 100 ms):
        # same result as resampling the whole stream at once, no block-edge clicks
        try:
            with sd.InputStream(device=device, channels=1, samplerate=sr_in, blocksize=blk, dtype="float32") as st:
                while self.running:
                    x, _ = st.read(blk)
                    z = np.concatenate([prev, x[:, 0]])
                    y = resample_poly(z, up, down).astype(np.float32)
                    a = ctx * up // down
                    self._push(y[a:a + blk * up // down])
                    prev = z[-2 * ctx:]
        except Exception as e:
            self.publish(dict(type="error", message=f"microphone: {type(e).__name__}: {e}"))
            self.running = False

    def _file(self, path, speed):
        import librosa
        try:
            y = librosa.load(path, sr=SR, mono=True, res_type="polyphase")[0]
        except Exception as e:
            self.publish(dict(type="error", message=f"file: {type(e).__name__}: {e}"))
            self.running = False
            return
        step = SR // 10
        t0 = time.time()
        for s in range(0, len(y), step):
            if not self.running:
                return
            while self.running and self.ring.total - self.pos > 10 * SR:   # never run ahead of the analysis
                time.sleep(0.005)
            self._push(y[s:s + step])
            if speed > 0:                                   # 0 = as fast as possible
                wait = t0 + (s + step) / SR / speed - time.time()
                if wait > 0:
                    time.sleep(wait)
        time.sleep(0.5)
        self.publish(dict(type="info", message="file finished"))
        threading.Thread(target=self.stop, daemon=True).start()

    # ---------------------------------------------------------------- analysis
    def _analyse(self):
        hop = int(HOP_S * SR)
        while self.running:
            if self.ring.total - self.pos < hop or self.ring.total < CLIP:
                time.sleep(0.02)
                continue
            self.pos = max(self.pos + hop, CLIP)
            w = self.ring.last(CLIP, self.pos)
            q = bp.quantize_int8(bp.preprocess(w))
            p = self.model.p_bird(q)
            t = self.pos / SR
            level = float(20 * np.log10(np.sqrt(np.mean(w[-hop:] ** 2)) + 1e-9))
            thr = self.threshold
            self.hits = (self.hits + [p >= thr])[-N:]
            on = sum(self.hits) >= K
            alarm = None
            if on:
                if self.last_on is None or t - self.last_on >= REFRACT_S:
                    alarm = self._new_alarm(t, p)
                else:
                    a = self.session["alarms"][-1]
                    a["p_max"] = max(a["p_max"], round(p, 3))
                self.last_on = t
            self.session["windows"] += 1
            self.session["seconds"] = t
            self.ww.writerow([f"{t:.2f}", f"{p:.4f}", f"{level:.1f}", int(on), thr])
            if self.session["windows"] % 10 == 0:
                self.wcsv.flush()
            self.publish(dict(type="window", t=round(t, 2), p=round(p, 4), level=round(level, 1), on=bool(on),
                              thr=thr, feat=base64.b64encode(q.astype(np.int8).tobytes()).decode(),
                              alarm=alarm, stats=self.stats()))

    def _new_alarm(self, t, p):
        a = dict(id=len(self.session["alarms"]) + 1, time=round(t, 2), p_max=round(p, 3), label="",
                 clock=time.strftime("%H:%M:%S"))
        clip = self.ring.last(int(CLIP_S * SR), self.pos)
        sf.write(self.session["dir"] / f"alarm_{a['id']:04d}.wav", clip, SR, subtype="PCM_16")
        self.session["alarms"].append(a)
        self._write_alarms()
        return a

    def _write_alarms(self):
        s = self.session
        with open(s["dir"] / "alarms.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "id", "time_s", "clock", "p_max", "label"])
            for a in s["alarms"]:
                w.writerow(["alarm", a["id"], a["time"], a["clock"], a["p_max"], a["label"]])
            for m in s["missed"]:
                w.writerow(["missed_bird", "", m["time"], m["clock"], "", "missed"])

    def label(self, aid, label):
        for a in self.session["alarms"]:
            if a["id"] == aid:
                a["label"] = label
        self._write_alarms()
        self.publish(dict(type="stats", stats=self.stats()))

    def missed(self):
        if not self.session:
            return
        self.session["missed"].append(dict(time=round(self.session["seconds"], 2), clock=time.strftime("%H:%M:%S")))
        self._write_alarms()
        self.publish(dict(type="stats", stats=self.stats()))

    def stats(self):
        s = self.session
        if not s:
            return {}
        hours = max(s["seconds"], 1e-9) / 3600
        lab = [a["label"] for a in s["alarms"]]
        birds, fa = lab.count("bird"), lab.count("false_alarm")
        caught_and_missed = birds + len(s["missed"])
        return dict(seconds=round(s["seconds"], 1), windows=s["windows"], alarms=len(s["alarms"]),
                    alarms_per_hour=round(len(s["alarms"]) / hours, 2) if s["seconds"] > 60 else None,
                    labelled_bird=birds, labelled_false=fa, unlabelled=lab.count(""), missed=len(s["missed"]),
                    false_per_hour=round(fa / hours, 2) if s["seconds"] > 60 else None,
                    recall=round(birds / caught_and_missed, 3) if caught_and_missed else None)

    def summary(self):
        s = self.session or {}
        return dict(running=self.running, session=s.get("name"), dir=str(s.get("dir", "")), source=s.get("source"),
                    model=self.model.path, threshold=self.threshold, k=K, n=N, hop=HOP_S,
                    alarms=s.get("alarms", []), stats=self.stats())

    # ---------------------------------------------------------------- server-sent events
    def publish(self, msg):
        data = json.dumps(msg)
        with self.lock:
            for q in list(self.subscribers):
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass

    def subscribe(self):
        q = queue.Queue(maxsize=500)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


def serial_ports():
    try:
        import serial.tools.list_ports as lp
        return [dict(id=p.device, name=f"{p.device} - {p.description}") for p in lp.comports()]
    except Exception:
        return []


def input_devices():
    try:
        import sounddevice as sd
        default = sd.default.device[0]
        out, seen = [], set()
        for i, d in enumerate(sd.query_devices()):
            api = sd.query_hostapis(d["hostapi"])["name"]
            if d["max_input_channels"] > 0 and api == "MME" and d["name"] not in seen:
                seen.add(d["name"])
                out.append(dict(id=i, name=d["name"], default=i == default))
        return out
    except Exception as e:
        return [dict(id=None, name=f"no audio input ({type(e).__name__})", default=True)]


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/state":
                return self._send(200, json.dumps(dict(engine.summary(), devices=input_devices(), ports=serial_ports())))
            if self.path.startswith("/clip/"):
                try:
                    aid = int(self.path.split("/")[-1].split(".")[0])
                    f = engine.session["dir"] / f"alarm_{aid:04d}.wav"
                    return self._send(200, f.read_bytes(), "audio/wav")
                except Exception:
                    return self._send(404, "{}")
            if self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                q = engine.subscribe()
                try:
                    while True:
                        try:
                            data = q.get(timeout=15)
                            self.wfile.write(f"data: {data}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    engine.unsubscribe(q)
                return
            self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/start":
                    dev = body.get("device")
                    engine.start(body.get("source", "mic"), device=int(dev) if dev not in (None, "") else None,
                                 file=body.get("file"), speed=float(body.get("speed", 1)),
                                 record=bool(body.get("record", True)))
                elif self.path == "/stop":
                    engine.stop()
                elif self.path == "/threshold":
                    engine.threshold = round(float(body["threshold"]), 2)
                elif self.path == "/label":
                    engine.label(int(body["id"]), body["label"])
                elif self.path == "/missed":
                    engine.missed()
                else:
                    return self._send(404, "{}")
                return self._send(200, json.dumps(engine.summary()))
            except Exception as e:
                return self._send(400, json.dumps(dict(error=f"{type(e).__name__}: {e}")))
    return H


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bird Detector Live</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mute:#667085;--line:#e3e6eb;--accent:#2563eb;--bird:#16a34a;--alarm:#dc2626;--warn:#d97706}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#171b23;--ink:#e6e9ef;--mute:#98a2b3;--line:#2a303b;--accent:#60a5fa;--bird:#22c55e;--alarm:#f87171;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card)}
h1{font-size:18px;margin:0}.sub{color:var(--mute);font-size:12px}
main{display:grid;gap:14px;padding:16px 20px;grid-template-columns:repeat(12,1fr);max-width:1400px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.c3{grid-column:span 3}.c4{grid-column:span 4}.c5{grid-column:span 5}.c7{grid-column:span 7}.c8{grid-column:span 8}.c12{grid-column:span 12}
@media (max-width:900px){.c3,.c4,.c5,.c7,.c8{grid-column:span 12}}
h2{font-size:13px;margin:0 0 10px;color:var(--mute);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input[type=text],input[type=number]{background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:6px 12px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.danger{border-color:var(--alarm);color:var(--alarm)}
button:disabled{opacity:.5;cursor:default}
#status{display:flex;align-items:center;justify-content:center;flex-direction:column;height:150px;border-radius:10px;font-size:34px;font-weight:700;transition:background .2s}
#status.idle{background:var(--bg);color:var(--mute)}#status.quiet{background:var(--bg);color:var(--ink)}#status.bird{background:var(--bird);color:#fff}
#status small{font-size:13px;font-weight:500;opacity:.85}
.kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.kpi .v{font-size:22px;font-weight:700}.kpi .l{color:var(--mute);font-size:12px}
canvas{width:100%;display:block;border-radius:6px;background:var(--bg)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px;border-bottom:1px solid var(--line)}th{color:var(--mute);font-weight:600}
td audio{height:28px;width:200px}.lab button{padding:3px 8px;font-size:12px}.lab .on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tag{padding:2px 7px;border-radius:99px;font-size:12px}.tag.bird{background:var(--bird);color:#fff}.tag.false_alarm{background:var(--alarm);color:#fff}.tag.unsure{background:var(--warn);color:#fff}
#msg{color:var(--alarm);font-size:13px;min-height:18px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.mute{color:var(--mute)}
</style></head><body>
<header><div><h1>Bird Detector &mdash; Live Test</h1><div class="sub" id="modelinfo">loading&hellip;</div></div>
<div class="controls">
 <select id="source"><option value="mic">Microphone</option><option value="serial">GPX10 board (serial)</option><option value="file">Audio file</option></select>
 <select id="port" style="display:none"></select>
 <select id="device"></select>
 <input type="text" id="file" placeholder="C:\path\to\recording.wav or .mp3" size="34" style="display:none">
 <select id="speed" style="display:none"><option value="1">1x real time</option><option value="4">4x</option><option value="0">as fast as possible</option></select>
 <label class="mute"><input type="checkbox" id="record" checked> record session</label>
 <button class="primary" id="start">Start</button><button class="danger" id="stop" disabled>Stop</button>
</div></header>
<main>
 <section class="card c4"><h2>Now</h2><div id="status" class="idle">IDLE<small>press Start</small></div>
  <div class="row" style="margin-top:12px"><span class="mute">Threshold</span>
   <input type="range" id="thr" min="0.30" max="0.95" step="0.01" style="flex:1"><b id="thrv"></b></div>
  <div class="row mute" style="margin-top:6px" id="rule"></div>
  <div class="row" style="margin-top:12px"><button id="missed">I heard a bird &mdash; it was missed</button>
   <label class="mute"><input type="checkbox" id="beep"> beep on alarm</label></div>
  <div id="msg"></div></section>
 <section class="card c8"><h2>P(bird) &mdash; last 120 s</h2><canvas id="plot" height="230"></canvas>
  <div class="row mute" style="margin-top:6px;font-size:12px">blue = P(bird) per 2 s window &middot; dashed = threshold &middot; red bars = alarm on &middot; grey = input level</div></section>
 <section class="card c4"><h2>Session</h2><div class="kpis">
  <div class="kpi"><div class="v" id="k_time">0:00</div><div class="l">listened</div></div>
  <div class="kpi"><div class="v" id="k_alarms">0</div><div class="l">alarms (<span id="k_aph">&ndash;</span>/h)</div></div>
  <div class="kpi"><div class="v" id="k_fph">&ndash;</div><div class="l">labelled false alarms / hour</div></div>
  <div class="kpi"><div class="v" id="k_recall">&ndash;</div><div class="l">recall (labelled birds vs missed)</div></div>
  <div class="kpi"><div class="v" id="k_lab">0 / 0</div><div class="l">labelled bird / false alarm</div></div>
  <div class="kpi"><div class="v" id="k_unl">0</div><div class="l">alarms to label</div></div></div>
  <div class="mute" style="font-size:12px;margin-top:10px" id="sessdir"></div></section>
 <section class="card c8"><h2>Model input &mdash; PCEN features of the current 2 s window (40 mel bands)</h2><canvas id="feat" height="160"></canvas></section>
 <section class="card c12"><h2>Alarms</h2><table><thead><tr><th>#</th><th>clock</th><th>time</th><th>peak P</th><th>listen</th><th>label</th></tr></thead><tbody id="alarms"></tbody></table>
  <div class="mute" id="noalarms" style="padding:8px 0">No alarms yet.</div></section>
</main>
<script>
const $=id=>document.getElementById(id);let hist=[],running=false,thr=0.7,K=2,N=3,alarms=new Map();
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});const j=await r.json();if(j.error)$('msg').textContent=j.error;return j}
function fmt(s){s=Math.floor(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return(h?h+':'+String(m).padStart(2,'0'):m)+':'+String(x).padStart(2,'0')}
function setRunning(r){running=r;$('start').disabled=r;$('stop').disabled=!r;['source','device','port','file','speed','record'].forEach(i=>$(i).disabled=r);
 if(!r){$('status').className='idle';$('status').innerHTML='IDLE<small>press Start</small>'}}
function showSource(){const v=$('source').value,f=v==='file';$('file').style.display=f?'':'none';$('speed').style.display=f?'':'none';$('device').style.display=v==='mic'?'':'none';$('port').style.display=v==='serial'?'':'none'}
$('source').onchange=showSource;
$('start').onclick=async()=>{$('msg').textContent='';hist=[];alarms.clear();renderAlarms();
 const j=await post('/start',{source:$('source').value,device:$('device').value,file:$('source').value==='serial'?$('port').value:$('file').value.trim(),speed:$('speed').value,record:$('record').checked});
 if(!j.error){setRunning(true);$('sessdir').textContent='saved to '+j.dir}};
$('stop').onclick=()=>post('/stop');
$('thr').oninput=e=>{$('thrv').textContent=(+e.target.value).toFixed(2)};
$('thr').onchange=e=>{thr=+e.target.value;post('/threshold',{threshold:thr})};
$('missed').onclick=()=>{if(running)post('/missed')};
function stats(s){if(!s||s.seconds===undefined)return;$('k_time').textContent=fmt(s.seconds);$('k_alarms').textContent=s.alarms;
 $('k_aph').textContent=s.alarms_per_hour??'\u2013';$('k_fph').textContent=s.false_per_hour??'\u2013';
 $('k_recall').textContent=s.recall==null?'\u2013':(100*s.recall).toFixed(1)+'%';$('k_lab').textContent=s.labelled_bird+' / '+s.labelled_false;$('k_unl').textContent=s.unlabelled}
function renderAlarms(){const tb=$('alarms');tb.innerHTML='';$('noalarms').style.display=alarms.size?'none':'';
 [...alarms.values()].sort((a,b)=>b.id-a.id).forEach(a=>{const tr=document.createElement('tr');
  tr.innerHTML=`<td>${a.id}</td><td>${a.clock}</td><td>${fmt(a.time)}</td><td>${a.p_max.toFixed(2)}</td><td><audio controls preload="none" src="/clip/${a.id}.wav"></audio></td><td class="lab"></td>`;
  const td=tr.lastChild;[['bird','Bird'],['false_alarm','False alarm'],['unsure','Unsure']].forEach(([k,t])=>{const b=document.createElement('button');b.textContent=t;if(a.label===k)b.className='on';
   b.onclick=()=>{a.label=k;post('/label',{id:a.id,label:k});renderAlarms()};td.appendChild(b)});tb.appendChild(tr)})}
function beep(){try{const c=new AudioContext(),o=c.createOscillator(),g=c.createGain();o.frequency.value=880;g.gain.value=.08;o.connect(g);g.connect(c.destination);o.start();o.stop(c.currentTime+.15)}catch(e){}}
function drawPlot(){const cv=$('plot'),dpr=devicePixelRatio||1,W=cv.clientWidth,H=230;cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);
 const L=34,R=8,T=8,B=20,w=W-L-R,h=H-T-B;g.clearRect(0,0,W,H);g.font='11px system-ui';g.fillStyle=css('--mute');g.strokeStyle=css('--line');
 for(const v of[0,.25,.5,.75,1]){const y=T+h*(1-v);g.beginPath();g.moveTo(L,y);g.lineTo(W-R,y);g.stroke();g.fillText(v.toFixed(2),2,y+4)}
 if(!hist.length)return;const tEnd=hist[hist.length-1].t,t0=tEnd-120,x=t=>L+w*(t-t0)/120;
 g.fillStyle=css('--alarm');g.globalAlpha=.25;hist.forEach(d=>{if(d.on)g.fillRect(x(d.t-1),T,Math.max(2,w/120),h)});g.globalAlpha=1;
 g.strokeStyle=css('--mute');g.globalAlpha=.6;g.beginPath();hist.forEach((d,i)=>{const v=Math.min(1,Math.max(0,(d.level+80)/80)),y=T+h*(1-v);i?g.lineTo(x(d.t),y):g.moveTo(x(d.t),y)});g.stroke();g.globalAlpha=1;
 g.setLineDash([5,4]);g.strokeStyle=css('--warn');const ty=T+h*(1-thr);g.beginPath();g.moveTo(L,ty);g.lineTo(W-R,ty);g.stroke();g.setLineDash([]);
 g.strokeStyle=css('--accent');g.lineWidth=2;g.beginPath();hist.forEach((d,i)=>{const y=T+h*(1-d.p);i?g.lineTo(x(d.t),y):g.moveTo(x(d.t),y)});g.stroke();g.lineWidth=1;
 g.fillStyle=css('--mute');for(let s=Math.max(0,Math.ceil(t0/20)*20);s<=tEnd;s+=20)g.fillText(fmt(s),x(s)-12,H-4)}
function drawFeat(b64){const raw=atob(b64),F=251,M=40,cv=$('feat'),g=cv.getContext('2d');cv.width=F;cv.height=M;const im=g.createImageData(F,M);
 for(let t=0;t<F;t++)for(let m=0;m<M;m++){let v=raw.charCodeAt(t*M+m);v=v>127?v-256:v;const u=(v+128)/255,i=((M-1-m)*F+t)*4;
  im.data[i]=Math.round(255*Math.min(1,Math.max(0,1.6*u-.2)));im.data[i+1]=Math.round(255*Math.min(1,u*u*1.3));im.data[i+2]=Math.round(255*(0.35+0.5*(1-u)));im.data[i+3]=255}
 g.putImageData(im,0,0);cv.style.imageRendering='pixelated'}
const es=new EventSource('/events');
es.onmessage=e=>{const m=JSON.parse(e.data);
 if(m.type==='window'){hist.push(m);while(hist.length&&hist[hist.length-1].t-hist[0].t>122)hist.shift();thr=m.thr;
  $('status').className=m.on?'bird':'quiet';$('status').innerHTML=(m.on?'BIRD':'no bird')+`<small>P(bird) ${m.p.toFixed(2)} &middot; level ${m.level.toFixed(0)} dBFS</small>`;
  if(m.alarm){alarms.set(m.alarm.id,m.alarm);renderAlarms();if($('beep').checked)beep()}else if(m.on&&alarms.size){const a=[...alarms.values()].pop();if(m.p>a.p_max){a.p_max=m.p}}
  stats(m.stats);drawPlot();drawFeat(m.feat)}
 else if(m.type==='stats')stats(m.stats);
 else if(m.type==='stopped'){setRunning(false);stats(m.stats)}
 else if(m.type==='session'){setRunning(true)}
 else if(m.type==='error'){$('msg').textContent=m.message;setRunning(false)}
 else if(m.type==='info'){$('msg').textContent=m.message}};
(async()=>{const s=await (await fetch('/state')).json();thr=s.threshold;K=s.k;N=s.n;$('thr').value=thr;$('thrv').textContent=thr.toFixed(2);
 $('rule').textContent=`alarm when ${s.k} of the last ${s.n} windows \u2265 threshold \u00b7 window 2 s every ${s.hop} s`;
 $('modelinfo').textContent='model: '+s.model;(s.ports||[]).forEach(d=>{const o=document.createElement('option');o.value=d.id;o.textContent=d.name;$('port').appendChild(o)});s.devices.forEach(d=>{const o=document.createElement('option');o.value=d.id??'';o.textContent=d.name+(d.default?' (default)':'');if(d.default)o.selected=true;$('device').appendChild(o)});
 showSource();setRunning(s.running);(s.alarms||[]).forEach(a=>alarms.set(a.id,a));renderAlarms();stats(s.stats);drawPlot()})();
addEventListener('resize',drawPlot);
</script></body></html>"""


def main():
    global SERIAL_BAUD, K, N, UART_STRETCH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", default="2of3", choices=["1win", "2of3"],
                    help="alarm rule: 1win = any single window >= threshold, 2of3 = 2 of the last 3 windows")
    ap.add_argument("--model", default=str(PROJECT / "runs" / "run6" / "model_int8.tflite"))
    ap.add_argument("--threshold", type=float, default=0.37, help="P(bird) threshold (run5a: lowest with <= 1.5 false alarms/h on val; 99.0 %% test recall)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD, help="GPX10 UART baud rate (must match the firmware)")
    ap.add_argument("--uart_stretch", type=float, default=1.1285,
                    help="serial audio stretch to true speed (16000 / received samples per s; 1 = off)")
    a = ap.parse_args()
    SERIAL_BAUD = a.baud
    from fractions import Fraction
    fr = Fraction(a.uart_stretch).limit_denominator(2000)
    UART_STRETCH = (fr.numerator, fr.denominator)
    K, N = (1, 1) if a.rule == "1win" else (2, 3)
    engine = Engine(Model(a.model), a.threshold)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(engine))
    srv.daemon_threads = True
    print(f"Bird detector live dashboard: http://127.0.0.1:{a.port}   (model {a.model}, threshold {a.threshold}, "
          f"alarm rule {K} of {N})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        engine.stop()


if __name__ == "__main__":
    main()
