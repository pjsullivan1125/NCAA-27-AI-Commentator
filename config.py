"""All tunable settings. Edit this file, not the other modules."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
ROSTER_DIR = BASE_DIR / "rosters"
OVERRIDES_PATH = ROSTER_DIR / "overrides.json"   # your own fixes where the game's roster differs from the real one

# ---------------- Models (all run locally through Ollama) ----------------
OLLAMA_URL = "http://localhost:11434/api/chat"
VISION_MODEL = "qwen3-vl:8b-instruct"   # "qwen3-vl:4b-instruct" is roughly 2x faster, at some cost in number reading
TEXT_MODEL = "qwen3-vl:8b-instruct"     # writes the line and fact-checks it. Keep it the same as VISION_MODEL (no VRAM swapping)
OLLAMA_THINK = False                    # "thinking" models answer slowly and can return empty text for vision
KEEP_ALIVE = "30m"                      # lower this (e.g. "5m") to free the GPU sooner after you stop
REQUEST_TIMEOUT = 60

# ---------------- Screen capture ----------------
WINDOW_TITLE_CONTAINS = "Projector"     # OBS Windowed Projector window title contains this
CAPTURE_REGION = None                   # or (left, top, width, height) to grab a fixed screen area
CROP_MARGINS = (0, 0, 0, 0)             # pixels trimmed off the window (left, top, right, bottom)
TRIM_LETTERBOX = True                   # cut black bars above/below the game picture automatically
CAPTURE_INTERVAL = 2.0                  # seconds between frames
VISION_MAX_WIDTH = 1280                 # frames are downscaled to this width before the model sees them
FRAME_DIFF_THRESHOLD = 6.0              # how much the picture must change before a frame is analyzed

# ---------------- Pacing ----------------
MIN_SECONDS_BETWEEN_LINES = 4.0         # ordinary play lines; big events ignore this
VISION_COOLDOWN_SECONDS = 6.0           # GPU rest between frame analyses, so OBS keeps up
STALE_FRAME_SECONDS = 6.0               # a frame older than this is dropped instead of analyzed
FILLER_AFTER_SECONDS = 30.0             # nothing said for this long -> say something grounded (score, situation, ...)
MAX_WORDS_SPOKEN = 24

# ---------------- Accuracy / safety ----------------
FACT_CHECK = True                       # second model pass that rejects lines claiming anything not in the facts
MAX_ZOOM_PLAYERS = 2                    # players per frame that get a zoomed second look at their number (~1.5 s each)
ZOOM_CACHE_SECONDS = 25.0               # a player confirmed by a zoomed read stays confirmed this long
ALLOW_NETWORK = False                   # False = the voice library never contacts huggingface.co

# ---------------- The game ----------------
LEFT_TEAM = "akron"     # team on the LEFT of the scoreboard (away)
RIGHT_TEAM = "msu"      # team on the RIGHT of the scoreboard (home)
TEAMS = {
    "msu": {"name": "Michigan State", "nickname": "Spartans", "roster": ROSTER_DIR / "msu.json"},
    "akron": {"name": "Akron", "nickname": "Zips", "roster": ROSTER_DIR / "akron.json"},
}
ROI_BANNER = (0.0, 0.75, 1.0, 1.0)      # where the penalty FLAG banner appears (fractions of the frame)

# ---------------- Voice ----------------
TTS_BACKEND = "kokoro"                  # "kokoro" or "piper"
KOKORO_VOICE = "am_adam"
KOKORO_SPEED = 1.1
KOKORO_DEVICE = "cpu"                   # "cuda" is much faster but needs the CUDA build of PyTorch (see README)
PIPER_EXE = r"C:\piper\piper.exe"
PIPER_MODEL = r"C:\piper\en_US-ryan-high.onnx"
