"""Dead-air filler: when nothing has been said for a while, pick a topic we can state with certainty
   (score, situation, drive, a player we verified, the setting). Each topic comes with facts and a safe template line."""
import random
import time
from collections import deque

import config
from fallback import DOWNS, QUARTERS, other, score_phrase, spoken_down, team, words

POS_WORDS = {"QB": "quarterback", "RB": "running back", "WR": "wide receiver", "TE": "tight end", "OL": "offensive lineman",
             "OT": "offensive tackle", "OG": "guard", "C": "center", "DL": "defensive lineman", "DE": "defensive end",
             "DT": "defensive tackle", "LB": "linebacker", "CB": "cornerback", "S": "safety", "DB": "defensive back",
             "K": "kicker", "P": "punter", "K/P": "kicker"}
SETTING = ["We're at Spartan Stadium in East Lansing, Akron against Michigan State.",
           "Spartans and Zips, here at Spartan Stadium.",
           "Akron visiting Michigan State today, at Spartan Stadium."]
ORDER = ["score", "situation", "player", "drive", "setting"]


class Filler:
    def __init__(self, rng=random):
        self.i, self.rng = 0, rng
        self.mentioned = {}
        self.setting_i = 0
        self.recent_lines = deque(maxlen=6)

    def next(self, sb, possession, seen, drive_plays, down_fresh):
        for k in range(len(ORDER)):
            topic = ORDER[(self.i + k) % len(ORDER)]
            res = getattr(self, "_" + topic)(sb, possession, seen, drive_plays, down_fresh)
            if res and res["template"] not in self.recent_lines:
                self.i = (self.i + k + 1) % len(ORDER)
                self.recent_lines.append(res["template"])
                return res
        return None

    def _score(self, sb, possession, seen, drive_plays, down_fresh):
        phrase = score_phrase(sb)
        q = sb.get("quarter")
        if not phrase or not q:
            return None
        where = "in the second half" if q >= 3 else f"in the {QUARTERS[q]} quarter"
        return {"topic": "score", "facts": f"{phrase} It is {where}.", "template": f"{phrase[:-1]} {where}."}

    def _situation(self, sb, possession, seen, drive_plays, down_fresh):
        if not down_fresh or possession not in config.TEAMS or not sb.get("down_dist") or sb.get("kickoff"):
            return None
        try:
            d, dist = sb["down_dist"].split(" & ")
            spoken = spoken_down(int(d), dist)
        except ValueError:
            return None
        t = team(possession)
        return {"topic": "situation", "facts": f"{t} has the ball, {spoken}.", "template": f"{t} has the ball, {spoken}."}

    def _player(self, sb, possession, seen, drive_plays, down_fresh):
        now = time.time()
        cands = [(ts, name, tm, pos) for name, (tm, pos, ts) in (seen or {}).items()
                 if now - ts < 240 and now - self.mentioned.get(name, 0) > 300 and pos in POS_WORDS]
        if not cands:
            return None
        _, name, tm, pos = max(cands)
        self.mentioned[name] = now
        last = name.replace(",", " ").split()
        last = [w for w in last if w.lower().strip(".") not in ("jr", "sr", "ii", "iii", "iv")][-1]
        return {"topic": "player", "facts": f"{name} is {team(tm)}'s {POS_WORDS[pos]}.",
                "template": f"Keep an eye on {team(tm)}'s {POS_WORDS[pos]}, {last}."}

    def _drive(self, sb, possession, seen, drive_plays, down_fresh):
        if possession not in config.TEAMS or drive_plays < 2 or sb.get("kickoff"):
            return None
        t = team(possession)
        return {"topic": "drive", "facts": f"{t} has run {drive_plays} plays on this drive.",
                "template": f"{t} is {words(drive_plays)} plays into this drive."}

    def _setting(self, sb, possession, seen, drive_plays, down_fresh):
        line = SETTING[self.setting_i % len(SETTING)]
        self.setting_i += 1
        return {"topic": "setting", "facts": line, "template": line}
