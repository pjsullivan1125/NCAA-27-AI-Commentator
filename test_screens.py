"""Check screen detection + scoreboard reading on saved screenshots. No Ollama needed.
   python test_screens.py screens"""
import sys
import time
from pathlib import Path

from PIL import Image

from screen_state import Reader, parse_scoreboard

reader = Reader()
folder = Path(sys.argv[1] if len(sys.argv) > 1 else "screens")
for p in sorted(folder.glob("*.*")):
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        continue
    img = Image.open(p).convert("RGB")
    t0 = time.time()
    state, text = reader.classify(img)
    sb = parse_scoreboard(reader, img)
    secs = time.time() - t0
    print(f"{p.name}: {state.value}  ({secs:.1f}s)\n   scoreboard: {sb}")
    print(f"   text seen: {text[:160]!r}")
