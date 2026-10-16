"""Run the same frames through each model ONE AT A TIME (no VRAM swapping).
   python compare_models.py screens gemma4:12b qwen3-vl:8b-instruct
   Optional: screens/expected.json, e.g. {"frame.png": [7, 22]}, lists the numbers clearly readable in each frame.
   Scores: numbers found, and numbers the model reported that are NOT in your list (made-up or unlisted)."""
import json
import sys
from pathlib import Path

import requests
from PIL import Image

import config
import vision


def to_int(n):
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


folder = Path(sys.argv[1])
models = sys.argv[2:] or ["gemma4:12b", "qwen3-vl:8b-instruct"]
expected = {}
exp_path = folder / "expected.json"
if exp_path.exists():
    try:
        expected = json.loads(exp_path.read_text())
    except Exception:
        print("[!] expected.json could not be read, so no scoring this run")
images = [(p, Image.open(p).convert("RGB")) for p in sorted(folder.glob("*.*"))
          if p.suffix.lower() in (".png", ".jpg", ".jpeg")]

summary = {}
for m in models:
    print(f"\n##### {m}: loading (this first call is not timed)")
    vision.describe_frame(images[0][1], model=m)
    s = {"secs": [], "hits": 0, "want": 0, "wrong": 0, "bad": 0}
    for p, img in images:
        obs, secs = vision.describe_frame(img, model=m)
        s["secs"].append(secs)
        if not obs:
            s["bad"] += 1
            print(f"{p.name} | {secs:.1f}s | UNPARSEABLE REPLY")
            continue
        players = [pl for pl in (obs.get("players") or []) if isinstance(pl, dict)]
        seen = {to_int(pl.get("number")) for pl in players} - {None}
        if p.name in expected:
            want = set(expected[p.name])
            s["want"] += len(want)
            s["hits"] += len(want & seen)
            s["wrong"] += len(seen - want)
        who = [f"{pl.get('team', '?')}#{to_int(pl.get('number'))}" for pl in players if to_int(pl.get('number')) is not None]
        print(f"{p.name} | {secs:.1f}s | who={who} | {obs.get('action')}")
    summary[m] = s
    # unload this model so the next one gets the whole GPU
    requests.post(config.OLLAMA_URL.replace("/api/chat", "/api/generate"), json={"model": m, "keep_alive": 0})

print("\nSUMMARY")
for m, s in summary.items():
    avg = sum(s["secs"]) / max(1, len(s["secs"]))
    score = (f"{s['hits']}/{s['want']} listed numbers found, {s['wrong']} reported that are not in your list"
             if s["want"] else ("no scored frames (every reply was unparseable)" if expected else "no expected.json"))
    print(f"{m}: avg {avg:.1f}s/frame, {score}, unparseable: {s['bad']}")
