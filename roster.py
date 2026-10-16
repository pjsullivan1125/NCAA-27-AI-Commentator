"""Roster lookup. Code (not the AI) turns team + number into a name, so names can't be hallucinated."""
import json
import difflib
from pathlib import Path

OFFENSE = {"QB", "RB", "FB", "HB", "WR", "TE", "OL", "OT", "OG", "C", "G", "T"}
DEFENSE = {"DL", "DE", "DT", "NT", "EDGE", "LB", "DB", "CB", "S", "FS", "SS"}
SPECIAL = {"K", "P", "LS", "K/P", "PK"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def side_of(pos):
    p = (pos or "").upper()
    if p in OFFENSE:
        return "offense"
    if p in DEFENSE:
        return "defense"
    if p in SPECIAL:
        return "special"
    return "unknown"


def last_name(full):
    parts = full.replace(",", " ").split()
    while len(parts) > 1 and parts[-1].lower().strip(".") in SUFFIXES:
        parts.pop()
    return parts[-1] if parts else full


class Rosters:
    def __init__(self, teams_cfg, overrides_path):
        self.players = {}
        for key, t in teams_cfg.items():
            p = Path(t["roster"])
            self.players[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        op = Path(overrides_path)
        self.overrides = json.loads(op.read_text(encoding="utf-8")) if op.exists() else {}

    def _override_entries(self, team):
        out = []
        for num, v in self.overrides.get(team, {}).items():
            for e in (v if isinstance(v, list) else [v]):
                out.append({"number": int(num), "name": e["name"], "pos": e.get("pos", "")})
        return out

    def candidates(self, team, number):
        ov = self.overrides.get(team, {}).get(str(number))
        if ov:  # an override REPLACES whatever the website roster says for that number
            return [ov] if isinstance(ov, dict) else list(ov)
        return [p for p in self.players.get(team, []) if p["number"] == number]

    def resolve(self, team, number=None, side=None, role=None, nameplate=None):
        out = {"team": team, "number": number, "name": None, "pos": None, "status": "unverified"}
        if team not in self.players:
            return out

        # 1) a legible jersey nameplate beats the number
        if nameplate:
            pool = self.players[team] + self._override_entries(team)
            best, best_score = None, 0.0
            for p in pool:
                s = difflib.SequenceMatcher(None, nameplate.lower(), last_name(p["name"]).lower()).ratio()
                if s > best_score:
                    best, best_score = p, s
            if best and best_score >= 0.85:
                out.update(name=best["name"], pos=best.get("pos"), status="verified")
                return out

        # 2) number lookup
        try:
            number = int(number)
        except (TypeError, ValueError):
            return out
        is_override = str(number) in self.overrides.get(team, {})
        cands = self.candidates(team, number)
        if is_override:   # you said this number is this player; the model's side/role guesses must not veto it
            if len(cands) == 1:
                out.update(name=cands[0]["name"], pos=cands[0].get("pos"), status="verified")
            return out
        sides = ({side} if isinstance(side, str) else set(side or [])) & {"offense", "defense", "special"}
        if sides:
            narrowed = [c for c in cands if side_of(c.get("pos")) in sides]
            if cands and not narrowed:   # roster position contradicts what was seen: don't risk a wrong name
                out["status"] = "mismatch"
                return out
            cands = narrowed
        if role == "passer":  # a passer must be a QB, otherwise the roster doesn't match the game
            cands = [c for c in cands if (c.get("pos") or "").upper() == "QB"]
        if len(cands) == 1:
            out.update(name=cands[0]["name"], pos=cands[0].get("pos"), status="verified")
        elif len(cands) > 1:
            out["status"] = "ambiguous"
        return out
