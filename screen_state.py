"""Figures out WHAT KIND of screen this is, and reads the scoreboard. Plain OCR + text matching, no LLM.
   v2: reads the whole bottom strip once at full size and uses text POSITIONS to find scores,
       and tolerates common OCR misspellings ('Ist', 'lst810')."""
import re
from enum import Enum

import numpy as np
from PIL import Image

import config


class State(str, Enum):
    GAMEPLAY = "gameplay"          # scoreboard visible, play running or just finished
    PRE_SNAP = "pre_snap"
    PLAY_SELECT = "play_select"
    PENALTY = "penalty"
    REPLAY = "replay"
    PAUSE = "pause"
    UNKNOWN = "unknown"


STRIP = (0.0, 0.68, 1.0, 1.0)            # bottom of the screen: scoreboard + button hints
CENTER_X = (0.43, 0.57)                  # scoreboard middle block (quarter, clock, down & distance)
SCORE_LEFT_ZONE = (0.375, 0.435, 0.41)   # (min x, max x, expected x) of the left team's score
SCORE_RIGHT_ZONE = (0.565, 0.625, 0.592)

CLOCK_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
DOWN_RE = re.compile(r"(?<!\d)([1-4])\s*(?:ST|ND|RD|TH)\s*(?:[&8B\u65e5]|[^\w\s])?\s*(\d{1,2}|GOAL|INCHES)(?![\d:])", re.I)
KICKOFF_RE = re.compile(r"KIC\w{0,3}F{1,2}", re.I)          # also catches OCR slips like KICROFF / KICHOFF
QTR_RE = re.compile(r"(?<!\d)([1-4])\s*(?:ST|ND|RD|TH)\b", re.I)

PAUSE_CUES = ("PAUSED", "EXIT TO MAIN MENU", "CHALLENGE PLAY", "VIEW HIGHLIGHTS")
SELECT_CUES = ("SELECT A PLAY", "COACH AD", "SUPER SIM", "PLAYCALL", "THE TRENCH", "COACH SUGGESTIONS",
               "PERSONNEL", "KICKOFF LEFT", "KICKOFF MIDDLE", "KICKOFF RIGHT", "STUNTS", "FAVORITE", "BLITZ")
PRESNAP_CUES = ("PLAYER LOCK", "ADJUST CAMERA", "FULL MENU", "PRE-PL", "CONTROLS")


def _norm(t):
    """OCR often reads '1st' as 'Ist' or 'lst'. Fix that before parsing."""
    return re.sub(r"(?<![A-Za-z0-9])[lI|!]\s*(?=[sS][tT])", "1", t)


def _rows(dets, gap=0.015):
    """Group detected text into visual rows (top to bottom), each row left to right."""
    rows, cur = [], []
    for d in sorted(dets, key=lambda d: d["y"]):
        if cur and d["y"] - cur[-1]["y"] > gap:
            rows.append(cur)
            cur = []
        cur.append(d)
    if cur:
        rows.append(cur)
    return [sorted(r, key=lambda d: d["x"]) for r in rows]


def _join(dets):
    return " ".join(d["text"] for row in _rows(dets) for d in row)


class Reader:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # CPU OCR, keeps the GPU free for the LLM
        self.engine = RapidOCR()
        self._strip_img = None
        self._strip = []

    def detect(self, img, box=None):
        """OCR a region. Returns [{'text','x','y','conf'}] with x,y as fractions of the WHOLE image."""
        w, h = img.size
        x0, y0, x1, y1 = box or (0.0, 0.0, 1.0, 1.0)
        crop = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        result, _ = self.engine(np.array(crop.convert("RGB"))[:, :, ::-1])
        out = []
        for pts, text, conf in (result or []):
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = 0.0
            out.append({"text": str(text),
                        "x": x0 + (sum(xs) / len(xs) / crop.width) * (x1 - x0),
                        "y": y0 + (sum(ys) / len(ys) / crop.height) * (y1 - y0),
                        "conf": c})
        return out

    def text(self, img, box=None, scale=1.0):
        return _join(self.detect(img, box))

    def strip(self, img):
        """The bottom strip, OCR'd once per frame and reused."""
        if self._strip_img is not img:
            self._strip = self.detect(img, STRIP)
            self._strip_img = img
        return self._strip

    def classify(self, img):
        small = img if img.width <= 960 else img.resize((960, int(img.height * 960 / img.width)))
        big_text = _join(self.detect(small))              # large menu text (PAUSED, play cards, ...)
        t = (big_text + " " + _join(self.strip(img))).upper()  # small text at full resolution
        if any(c in t for c in PAUSE_CUES):
            return State.PAUSE, t
        if ("ROTATE" in t and "DOF" in t) or ("DOF" in t and "HIDE" in t):
            return State.REPLAY, t
        if any(c in t for c in SELECT_CUES):
            return State.PLAY_SELECT, t
        if any(c in t for c in PRESNAP_CUES):
            return State.PRE_SNAP, t
        if re.search(r"\bFLAG\b", t) or flag_from_plate(img, self.strip(img)):
            return State.PENALTY, t
        if CLOCK_RE.search(t) and any(k in t for k in ("AKRON", "MICHIG", "EASPORTS", "SPORTS")):
            return State.GAMEPLAY, t
        return State.UNKNOWN, t


def _score(dets, zone):
    lo, hi, target = zone
    best = None
    for d in dets:
        if lo <= d["x"] <= hi:
            t = d["text"].strip().replace("O", "0").replace("o", "0")
            if re.fullmatch(r"\d{1,2}", t):
                dist = abs(d["x"] - target)
                if best is None or dist < best[0]:
                    best = (dist, int(t))
    return best[1] if best else None


def flag_from_plate(img, dets):
    """During a penalty the scoreboard's down-and-distance plate turns bright yellow (with the word FLAG)."""
    clock = None
    for d in dets:
        if CENTER_X[0] <= d["x"] <= CENTER_X[1] and CLOCK_RE.search(d["text"]):
            clock = d
    if clock is None:
        return False
    w, h = img.size
    box = (max(0, int((clock["x"] - 0.045) * w)), max(0, int((clock["y"] + 0.029 - 0.008) * h)),
           min(w, int((clock["x"] + 0.045) * w)), min(h, int((clock["y"] + 0.029 + 0.008) * h)))
    reg = np.asarray(img.crop(box).convert("RGB")).reshape(-1, 3).astype(int)
    if len(reg) < 20:
        return False
    r, g, b = reg[:, 0], reg[:, 1], reg[:, 2]
    return float(((r > 170) & (g > 140) & (b < 110) & (r - b > 90)).mean()) > 0.35


def possession_from_plate(img, dets):
    """The 'down & distance' plate on the scoreboard is coloured by the team with the ball:
       navy = Akron, green = Michigan State. Reads that colour straight from the pixels. None if unsure."""
    plate = None
    for d in dets:
        if CENTER_X[0] <= d["x"] <= CENTER_X[1] and (DOWN_RE.search(_norm(d["text"])) or KICKOFF_RE.search(d["text"])):
            plate = d
    if plate is None:
        return None
    w, h = img.size
    box = (max(0, int((plate["x"] - 0.045) * w)), max(0, int((plate["y"] - 0.010) * h)),
           min(w, int((plate["x"] + 0.045) * w)), min(h, int((plate["y"] + 0.010) * h)))
    reg = np.asarray(img.crop(box).convert("RGB")).reshape(-1, 3).astype(int)
    dark = reg[reg.sum(axis=1) < 330]                       # drop the white text and bright edges
    if len(dark) < 10:
        return None
    r, g, b = dark.mean(axis=0)
    if b - g > 12:
        return "akron"
    if b - g < 0:
        return "msu"
    return None


def parse_scoreboard(reader, img):
    dets = reader.strip(img)
    center = _norm(_join([d for d in dets if CENTER_X[0] <= d["x"] <= CENTER_X[1]]))
    sb = {"quarter": None, "clock_sec": None, "down_dist": None,
          "score_left": _score(dets, SCORE_LEFT_ZONE), "score_right": _score(dets, SCORE_RIGHT_ZONE)}
    m = DOWN_RE.search(center)
    if m:
        sb["down_dist"] = f"{m.group(1)} & {m.group(2)}"
        center = center.replace(m.group(0), " ")
    q = QTR_RE.search(center)
    if q:
        sb["quarter"] = int(q.group(1))
    for c in CLOCK_RE.finditer(center):
        if int(c.group(2)) < 60:
            sb["clock_sec"] = int(c.group(1)) * 60 + int(c.group(2))
            break
    sb["kickoff"] = bool(KICKOFF_RE.search(center)) and not sb["down_dist"]
    # plate colour = team with the ball (or the team kicking off)
    sb["possession"] = possession_from_plate(img, dets) if (sb["down_dist"] or sb["kickoff"]) else None
    return sb
