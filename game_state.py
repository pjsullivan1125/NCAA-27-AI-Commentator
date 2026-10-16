"""Watches scoreboard readings over time and turns changes into events (touchdown, quarter ending, ...).
   The scoreboard is read by OCR, which sometimes misreads a digit, so every change is sanity-checked."""
import time

import config

VALID_SCORE_JUMPS = {1, 2, 3, 6, 7, 8}     # extra point, safety/2-pt, field goal, touchdown (+ conversions)
ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


class Tracker:
    def __init__(self):
        self.sb = {}
        self._pending = {}
        self._noise = {}
        self._base = {}
        self._ending_fired = set()
        self._kickoff_done = False
        self.last_result = None        # e.g. "gain of 3 yards, now 2nd & 6" (from the scoreboard, never invented)
        self.last_result_data = None   # the same thing as numbers, for fallback lines
        self.drive_plays = 0
        self.last_read = {}            # when each value was last read successfully

    # ---------- reading helpers ----------
    def _confirm(self, key, val):
        """Only accept a changed value after seeing it twice in a row (OCR misreads happen)."""
        if val is None:
            return False, self.sb.get(key)
        cur = self.sb.get(key)
        if cur == val:
            self._pending.pop(key, None)
            return False, cur
        if cur is None:
            self.sb[key] = val
            return False, None
        if self._pending.get(key) == val:
            self._pending.pop(key)
            self.sb[key] = val
            return True, cur
        self._pending[key] = val
        return False, cur

    def _confirm_score(self, key, val):
        """Like _confirm, but scores can only go up by a real scoring play. Anything else is an OCR misread."""
        if val is None:
            return False, self.sb.get(key)
        cur = self.sb.get(key)
        if cur is None:                                             # first reading: it must repeat 3 times before we believe it
            seen, n = self._base.get(key, (None, 0))
            n = n + 1 if seen == val else 1
            self._base[key] = (val, n)
            if n >= 3:
                self.sb[key] = val
                self._base.pop(key, None)
            return False, None
        if val == cur:
            self._pending.pop(key, None)
            self._noise.pop(key, None)
            return False, cur
        if val - cur in VALID_SCORE_JUMPS:
            self._noise.pop(key, None)
            if self._pending.get(key) == val:
                self._pending.pop(key)
                self.sb[key] = val
                return True, cur
            self._pending[key] = val
            return False, cur
        seen, n = self._noise.get(key, (None, 0))                  # impossible jump: ignore it...
        n = n + 1 if seen == val else 1
        self._noise[key] = (val, n)
        if n >= 5:                                                  # ...unless it persists, then our old baseline was the misread
            self.sb[key] = val
            self._noise.pop(key, None)
        return False, cur

    # ---------- main update ----------
    def update(self, sb):
        events = []
        now = time.time()
        self.sb["kickoff"] = bool(sb.get("kickoff"))
        if self.sb["kickoff"]:
            self.sb["down_dist"] = None       # a kickoff has no down & distance; don't carry the old one into it
            self.drive_plays = 0
        for key, team in (("score_left", config.LEFT_TEAM), ("score_right", config.RIGHT_TEAM)):
            changed, old = self._confirm_score(key, sb.get(key))
            if changed and old is not None:
                delta = self.sb[key] - old
                kind = {6: "touchdown", 7: "touchdown", 8: "touchdown", 3: "field_goal",
                        2: "two_points_or_safety", 1: "extra_point"}.get(delta, "score")
                events.append({"type": "score", "team": team, "kind": kind, "delta": delta})
        changed, _ = self._confirm("quarter", sb.get("quarter"))
        if changed:
            events.append({"type": "quarter_start", "quarter": self.sb["quarter"]})
            self._kickoff_done = True
        if sb.get("possession"):                 # the plate colour is read from pixels, so trust it at once
            if self.sb.get("possession") not in (None, sb["possession"]):
                self.drive_plays = 0             # the ball changed hands: a new drive
            self.sb["possession"] = sb["possession"]
        old_dd = self.sb.get("down_dist")
        changed, _ = self._confirm("down_dist", sb.get("down_dist"))
        if sb.get("down_dist"):
            self.last_read["down"] = now
        if changed:
            data = self._parse_result(old_dd, self.sb["down_dist"])
            self.last_result = data["text"] if data else None
            self.last_result_data = data
            if data:
                self.drive_plays += 1
            events.append({"type": "play_complete"})

        q, sec = self.sb.get("quarter"), sb.get("clock_sec")
        if sec is not None and sec < 290:
            self._kickoff_done = True            # the game is already under way: never announce a "opening kickoff"
        if q and q >= 2:
            self._kickoff_done = True
        if q and sec is not None:
            if sec < 30 and q not in self._ending_fired:
                self._ending_fired.add(q)
                kind = {2: "half_ending", 4: "game_ending"}.get(q, "quarter_ending")
                events.append({"type": kind, "quarter": q})
            if q == 1 and sec >= 295 and not self._kickoff_done:
                self._kickoff_done = True
                events.append({"type": "kickoff"})
        return events

    # ---------- results from down & distance ----------
    @staticmethod
    def _parse_result(old, new):
        try:
            od, odist = [int(x) for x in old.split(" & ")]
            nd, ndist = [int(x) for x in new.split(" & ")]
        except (AttributeError, ValueError):
            return None                      # goal-to-go, inches, or unreadable: say nothing rather than guess
        if nd == 1:
            return {"kind": "first_down", "yards": None, "down": nd, "dist": ndist, "text": f"first down, now 1st & {ndist}"}
        if nd == od + 1:
            gain = odist - ndist
            if gain > 0:
                return {"kind": "gain", "yards": gain, "down": nd, "dist": ndist,
                        "text": f"gain of {gain} yard{'s' if gain != 1 else ''}, now {ORDINAL[nd]} & {ndist}"}
            if gain < 0:
                return {"kind": "loss", "yards": -gain, "down": nd, "dist": ndist,
                        "text": f"loss of {-gain} yard{'s' if gain != -1 else ''}, now {ORDINAL[nd]} & {ndist}"}
            return {"kind": "no_gain", "yards": 0, "down": nd, "dist": ndist, "text": f"no gain, now {ORDINAL[nd]} & {ndist}"}
        return None

    @staticmethod
    def _summarize(old, new):
        data = Tracker._parse_result(old, new)
        return data["text"] if data else None

    def down_is_fresh(self, seconds=25):
        return time.time() - self.last_read.get("down", 0) < seconds

    def situation_text(self):
        s = self.sb
        names = {k: config.TEAMS[k]["name"] for k in config.TEAMS}
        parts = []
        if s.get("score_left") is not None and s.get("score_right") is not None:
            parts.append(f"{names[config.LEFT_TEAM]} {s['score_left']}, {names[config.RIGHT_TEAM]} {s['score_right']}")
        if s.get("quarter"):
            parts.append(f"quarter {s['quarter']}")
        if s.get("kickoff"):
            if s.get("possession") in names:
                parts.append(f"kickoff: {names[s['possession']]} is kicking (from the scoreboard, always correct)")
            else:
                parts.append("kickoff (which team is kicking is unknown)")
        else:
            if s.get("down_dist"):
                parts.append(s["down_dist"])
            if s.get("possession") in names:
                parts.append(f"{names[s['possession']]} has the ball (from the scoreboard, always correct)")
        return "; ".join(parts) or "unknown"
