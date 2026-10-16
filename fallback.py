"""Plain, always-correct lines built straight from the scoreboard. Used when the writer model stays silent or is
   rejected on an important moment (touchdown, quarter ending, ...), and as the base for 'dead air' filler."""
import random

import config

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
DOWNS = {1: "first", 2: "second", 3: "third", 4: "fourth"}
QUARTERS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def words(n):
    if n < 20:
        return UNITS[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else f" {UNITS[n % 10]}")
    return str(n)


def team(k):
    return config.TEAMS[k]["name"]


def other(k):
    return "msu" if k == "akron" else "akron"


def score_phrase(sb):
    a, b = sb.get("score_left"), sb.get("score_right")
    if a is None or b is None:
        return None
    L, R = team(config.LEFT_TEAM), team(config.RIGHT_TEAM)
    if a == b:
        return f"It's tied, {words(a)} apiece."
    lead, trail = (L, R) if a > b else (R, L)
    hi, lo = max(a, b), min(a, b)
    return f"{lead} leads {trail}, {words(hi)} to {words(lo)}."


def spoken_down(down, dist):
    d = DOWNS.get(int(down), str(down))
    if isinstance(dist, str) and not dist.isdigit():
        return f"{d} and {dist.lower()}"
    return f"{d} and {words(int(dist))}"


def fallback_line(events, sb, possession, rng=random):
    """One safe line for the most important event in the list, or None."""
    by = {e["type"]: e for e in events}
    if "score" in by:
        e = by["score"]
        t = team(e["team"])
        lead = {"touchdown": rng.choice([f"Touchdown, {t}!", f"{t} finds the end zone!"]),
                "field_goal": f"The kick is good. Field goal for {t}.",
                "extra_point": f"The extra point is good for {t}.",
                "two_points_or_safety": f"{t} picks up two."}.get(e["kind"], f"{t} puts points on the board.")
        tail = score_phrase(sb)
        return f"{lead} {tail}" if tail else lead
    if "game_ending" in by:
        return "Thirty seconds left in the game."
    if "half_ending" in by:
        return "Thirty seconds to go until halftime."
    if "quarter_ending" in by:
        q = by["quarter_ending"]["quarter"]
        return f"Under thirty seconds left in the {QUARTERS.get(q, 'quarter')} quarter."
    if "quarter_start" in by:
        q = by["quarter_start"]["quarter"]
        return "We're into the second half." if q == 3 else f"We're underway in the {QUARTERS.get(q, 'next')} quarter."
    if "kickoff" in by:
        return "Akron at Michigan State, here at Spartan Stadium. Kickoff is coming up."
    if "play_result" in by:
        d = by["play_result"].get("data") or {}
        t = team(possession) if possession in config.TEAMS else None
        down = spoken_down(d["down"], d["dist"]) if d.get("down") else None
        if d.get("kind") == "first_down":
            return f"First down, {t}!" if t else "First down!"
        if d.get("kind") == "gain" and down:
            return f"{t} picks up {words(d['yards'])} yard{'s' if d['yards'] != 1 else ''}, now {down}." if t else f"A gain of {words(d['yards'])}, now {down}."
        if d.get("kind") == "loss" and down:
            return f"{t} loses {words(d['yards'])}, now {down}." if t else f"Loss of {words(d['yards'])}, now {down}."
        if d.get("kind") == "no_gain" and down:
            return f"No gain on the play, now {down}."
    if "penalty" in by:
        return "There's a flag on the play."
    return None
