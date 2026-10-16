"""Talks to Ollama. Vision step = 'what do you SEE' -> structured JSON. No commentary here."""
import base64
import io
import json
import re
import time

import requests

import config

VISION_PROMPT = """You are analyzing one frame from the video game EA College Football 27.
Return ONLY a JSON object with exactly these keys:
{{
 "action": "one factual sentence about what is happening right now. Say 'the offense' and 'the defense'. NEVER use school names, team names or uniform colors. If players are just standing or lined up, say that; do not invent movement.",
 "players": [{{"box": [x1, y1, x2, y2], "number": <int or null>, "role": "passer | ball_carrier | receiver | tackler | other"}}]
}}
Rules:
- "box" is a tight box around that one player: four integers on a 0-1000 scale relative to the image (x1,y1 = top-left, x2,y2 = bottom-right).
- Only list a player if you can clearly read the jersey number on THAT player. Small, blurry, hidden or far-away numbers must be null. NEVER guess a number. An empty players list is fine and expected on wide shots.
- List at most 2 players: the ball carrier or passer first.
- Do not describe the scoreboard or any menus."""


import re as _re
_TEAM_WORDS = _re.compile(r"(?i)\b(akron|zips|michigan state|spartans|msu|(?:navy|white|gold|green)(?:[- ]and[- ]\w+)?[- ](?:team|uniforms?|jerseys?|clad|squad))\b")


def normalize(obs, img=None, sent_size=None):
    """Code, not the model, decides which TEAM a player is on (from the pixels inside his box)."""
    import zoom
    if not isinstance(obs, dict):
        return obs
    players = [pl for pl in (obs.get("players") or []) if isinstance(pl, dict)]
    # Work out the box scale: asked for 0-1000, but some models answer in pixels or 0-1 fractions.
    vals = []
    for pl in players:
        try:
            vals += [float(x) for x in pl.get("box")]
        except (TypeError, ValueError):
            pass
    mode = "1000"
    if vals and max(vals) <= 1.0:
        mode = "fraction"
    elif vals and max(vals) > 1000 and sent_size:
        mode = "pixels"
    obs["_box_mode"] = mode
    for pl in players:
        raw = pl.get("box")
        try:
            b = [float(x) for x in raw]
            if mode == "fraction":
                b = [x * 1000 for x in b]
            elif mode == "pixels":
                b = [b[0] / sent_size[0] * 1000, b[1] / sent_size[1] * 1000, b[2] / sent_size[0] * 1000, b[3] / sent_size[1] * 1000]
        except (TypeError, ValueError, IndexError):
            b = None
        pl["box"] = zoom.valid_box(b)
        pl["team"] = zoom.team_from_pixels(img, pl["box"]) if (img is not None and pl["box"]) else "unknown"
    obs["players"] = players
    obs["ball_team"] = "unknown"
    action = str(obs.get("action") or "")
    if _TEAM_WORDS.search(action):                          # the model named a team anyway: its claim is unreliable, drop the sentence
        action = ""
    obs["action"] = action
    return obs
    for pl in obs.get("players") or []:
        if isinstance(pl, dict):
            pl["team"] = UNIFORM_TO_TEAM.get(str(pl.get("uniform", "")).lower(), "unknown")
    obs["ball_team"] = UNIFORM_TO_TEAM.get(str(obs.get("ball_uniform", "")).lower(), "unknown")
    action = str(obs.get("action") or "")
    action = re.sub(r"(?i)\b(?:the\s+)?navy[ -]team\b", config.TEAMS["akron"]["name"], action)
    action = re.sub(r"(?i)\b(?:the\s+)?white[ -]team\b", config.TEAMS["msu"]["name"], action)
    obs["action"] = action
    return obs


def encode_image(img):
    if img.width > config.VISION_MAX_WIDTH:
        img = img.resize((config.VISION_MAX_WIDTH, int(img.height * config.VISION_MAX_WIDTH / img.width)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def chat(model, messages, as_json=False, temperature=0.3, num_predict=300):
    payload = {"model": model, "messages": messages, "stream": False, "keep_alive": config.KEEP_ALIVE,
               "options": {"temperature": temperature, "num_predict": num_predict}}
    if as_json:
        payload["format"] = "json"
    if config.OLLAMA_THINK is not None:
        payload["think"] = config.OLLAMA_THINK
    r = requests.post(config.OLLAMA_URL, json=payload, timeout=config.REQUEST_TIMEOUT)
    if r.status_code == 400 and "think" in payload:  # model doesn't support the think flag
        payload.pop("think")
        r = requests.post(config.OLLAMA_URL, json=payload, timeout=config.REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None


def describe_frame(img, model=None):
    """Returns (observation dict or None, seconds taken)."""
    model = model or config.VISION_MODEL
    prompt = VISION_PROMPT
    w_s = min(img.width, config.VISION_MAX_WIDTH)
    h_s = int(img.height * w_s / img.width)
    t0 = time.time()
    out = chat(model, [{"role": "user", "content": prompt, "images": [encode_image(img)]}],
               as_json=True, temperature=0.1, num_predict=260)
    return normalize(parse_json(out), img, (w_s, h_s)), time.time() - t0
