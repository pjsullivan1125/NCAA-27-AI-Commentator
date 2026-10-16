"""Second look at individual players.
   1) team_from_pixels: decides Akron (navy) vs Michigan State (white) from the jersey's actual colors, in code.
   2) read_number: re-reads a jersey number from a zoomed-in crop, so we can require TWO independent reads to agree."""
import numpy as np
from PIL import Image

import config
import vision


def valid_box(box):
    """Model boxes are [x1, y1, x2, y2] on a 0-1000 scale. Returns a clean box or None."""
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    if x2 - x1 < 8 or y2 - y1 < 12:      # too tiny to mean anything
        return None
    return [x1, y1, x2, y2]


def crop_box(img, box, margin=0.0):
    w, h = img.size
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) / 1000 * w, (y2 - y1) / 1000 * h
    left = max(0, int(x1 / 1000 * w - bw * margin)); right = min(w, int(x2 / 1000 * w + bw * margin))
    top = max(0, int(y1 / 1000 * h - bh * margin)); bottom = min(h, int(y2 / 1000 * h + bh * margin))
    return img.crop((left, top, right, bottom))


def _grass_level(img):
    """Brightness of the grass in THIS frame (changes with sun, clouds, rain, night). Falls back to the frame median."""
    small = img.resize((200, max(1, int(200 * img.height / img.width))))
    a = np.asarray(small.convert("HSV")).astype(np.float32)
    h, s, v = a[..., 0], a[..., 1], a[..., 2]
    g = (h > 45) & (h < 115) & (s > 55)
    return float(np.median(v[g])) if g.mean() > 0.03 else float(np.median(v))


def color_stats(img, box):
    """ratio = how bright the uniform is compared with the grass in the same frame.
       White jerseys are brighter than grass and navy jerseys are darker, in any weather or time of day."""
    box = valid_box(box)
    if not box:
        return None
    arr = np.asarray(crop_box(img, box).convert("HSV")).astype(np.float32)
    if arr.shape[0] < 6 or arr.shape[1] < 4:
        return None
    h_, w_ = arr.shape[:2]                                   # central region: helmet + jersey, less background
    arr = arr[int(0.04 * h_): int(0.85 * h_), int(0.15 * w_): int(0.85 * w_)]
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    keep = ~((h > 45) & (h < 115) & (s > 55))                # ignore grass inside the box
    n = int(keep.sum())
    if n < 10:
        return None
    gold = ((h > 18) & (h < 42) & (s > 80) & keep).sum() / n  # Akron's gold helmet and numbers
    return {"ratio": float(np.median(v[keep]) / max(_grass_level(img), 1.0)), "gold": float(gold), "n": n}


AKRON_BELOW = 0.85    # darker than the grass  -> navy -> Akron
MSU_ABOVE = 0.98      # brighter than the grass -> white -> Michigan State
# anything in between is "unknown", which means no player name is used


def team_from_pixels(img, box):
    st = color_stats(img, box)
    if not st:
        return "unknown"
    if st["ratio"] < AKRON_BELOW:
        return "akron"
    if st["ratio"] > MSU_ABOVE:
        return "msu"
    return "unknown"


READ_PROMPT = """This is a close crop of ONE football player from a video game. Read the jersey number printed on him (chest, back or sleeve).
Reply ONLY with JSON: {"number": <integer or null>, "nameplate": <name printed on the back or null>}
If the number is not clearly readable, use null. Never guess."""


def read_number(img, box, model=None):
    """Returns (number or None, nameplate or None) from a zoomed crop of the player."""
    box = valid_box(box)
    if not box:
        return None, None
    crop = crop_box(img, box, margin=0.25)
    scale = max(1.0, 512 / max(1, crop.height))
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    out = vision.chat(model or config.VISION_MODEL,
                      [{"role": "user", "content": READ_PROMPT, "images": [vision.encode_image(crop)]}],
                      as_json=True, temperature=0.0, num_predict=60)
    d = vision.parse_json(out) or {}
    try:
        number = int(d.get("number"))
    except (TypeError, ValueError):
        number = None
    plate = d.get("nameplate") if isinstance(d.get("nameplate"), str) and d.get("nameplate").strip() else None
    return number, plate
