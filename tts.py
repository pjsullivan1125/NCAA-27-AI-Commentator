"""Local text-to-speech. Kokoro first, Piper as backup.
   Speaks on a background thread, and starts playing the FIRST SENTENCE while the rest is still being synthesized."""
import os
import queue
import subprocess
import tempfile
import threading
import time
import warnings

import config

# Everything runs on this machine. Without this, the voice library pings huggingface.co at startup just to ask whether
# its already-downloaded files are up to date. (If you ever change KOKORO_VOICE to a new voice, set ALLOW_NETWORK = True
# in config.py for one run so it can download it.)
if not getattr(config, "ALLOW_NETWORK", False):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

warnings.filterwarnings("ignore", category=UserWarning)     # harmless notices printed by the voice libraries
warnings.filterwarnings("ignore", category=FutureWarning)


class _Kokoro:
    def __init__(self):
        from kokoro import KPipeline
        device = getattr(config, "KOKORO_DEVICE", "cpu")             # "cuda" needs the CUDA build of PyTorch
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("[tts] KOKORO_DEVICE is 'cuda' but this PyTorch has no CUDA support; using the CPU")
                device = "cpu"
        self.pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
        self.sr = 24000

    def stream(self, text):
        """Yield (audio, sample_rate) one sentence at a time."""
        import numpy as np
        try:
            gen = self.pipe(text, voice=config.KOKORO_VOICE, speed=config.KOKORO_SPEED, split_pattern=r"(?<=[.!?])\s+")
        except TypeError:                                     # older versions without split_pattern
            gen = self.pipe(text, voice=config.KOKORO_VOICE, speed=config.KOKORO_SPEED)
        for _, _, audio in gen:
            yield (audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)), self.sr


class _Piper:
    def stream(self, text):
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        subprocess.run([config.PIPER_EXE, "--model", config.PIPER_MODEL, "--output_file", path],
                       input=text.encode("utf-8"), check=True)
        data, sr = sf.read(path)
        yield data, sr


class Speaker:
    def __init__(self):
        self.engine = _Kokoro() if config.TTS_BACKEND == "kokoro" else _Piper()
        self.q = queue.Queue(maxsize=1)
        self.busy = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text):
        try:  # keep only the newest line; stale commentary is worse than none
            self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put(text)

    def _worker(self):
        import sounddevice as sd
        while True:
            text = self.q.get()
            self.busy.set()
            try:
                t0 = time.time()
                chunks = queue.Queue(maxsize=4)

                def produce():
                    try:
                        for item in self.engine.stream(text):
                            chunks.put(item)
                    except Exception as e:
                        print(f"[tts error] {e}")
                    finally:
                        chunks.put(None)

                threading.Thread(target=produce, daemon=True).start()
                first = True
                while True:
                    item = chunks.get()
                    if item is None:
                        break
                    audio, sr = item
                    if first:
                        print(f"  [voice] first audio after {time.time() - t0:.1f}s")
                        first = False
                    sd.play(audio, sr)
                    sd.wait()
            except Exception as e:
                print(f"[tts error] {e}")
            finally:
                self.busy.clear()
