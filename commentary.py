"""Second step: turns verified facts into one announcer line. Text-only call, so it's quick."""
import difflib
import re

import config
import vision

LAST_NOTE = ""        # why the last call produced no line (printed in the log)

SYSTEM = """You are the live TV play-by-play announcer for a college football game: Akron at Michigan State, Spartan Stadium. Energetic, natural and concise: one or two short sentences, at most {max_words} words. No emojis, no stage directions.
HARD RULES:
1. Only use player names from the VERIFIED PLAYERS list. Never invent or guess a name or number.
2. When a verified player is clearly involved in what happened, name him by last name. For any other player, do not name him; say something natural like "couldn't see who it was, but..." and vary that phrasing every time.
3. Never repeat or closely paraphrase anything in RECENT LINES. If there is nothing new worth saying, reply with exactly SKIP.
4. Never mention screenshots, AI, models, or on-screen graphics.
5. Do not narrate play-calling menus. GAME SITUATION says which team has the ball; never contradict it.
6. Never state a fact that is not given to you: no formations (shotgun, under center), routes, targets, kick distances, tackles, stops, turnovers, or results beyond what the EVENT and WHAT HAPPENED lines say. Never describe where players are looking or what they are thinking. WHAT HAPPENED can be imprecise about timing, so keep it general.
7. If WHAT HAPPENED is empty, talk ONLY about the EVENT and GAME SITUATION. Never describe a play, a kick, or any player action."""

EVENT_TEXT = {
    "touchdown": "TOUCHDOWN just scored by {team}.",
    "field_goal": "FIELD GOAL just made by {team}.",
    "extra_point": "Extra point is good for {team}.",
    "two_points_or_safety": "{team} just scored 2 points.",
    "score": "{team} just scored.",
}


def events_to_text(events):
    lines = []
    for e in events:
        t = e["type"]
        if t == "score":
            team = config.TEAMS[e["team"]]["name"]
            lines.append(EVENT_TEXT.get(e["kind"], EVENT_TEXT["score"]).format(team=team))
        elif t == "quarter_ending":
            lines.append(f"Quarter {e['quarter']} is about to end (under 30 seconds left).")
        elif t == "half_ending":
            lines.append("The first half is about to end (under 30 seconds left).")
        elif t == "game_ending":
            lines.append("The game is about to end (under 30 seconds left in the 4th quarter).")
        elif t == "quarter_start":
            lines.append(f"Quarter {e['quarter']} is starting.")
        elif t == "kickoff":
            lines.append("The opening kickoff is about to happen. No play has been run yet. Do not say who is kicking or receiving.")
        elif t == "penalty":
            lines.append("A penalty flag was just thrown. You do NOT know which team committed it or what the penalty is. Say only that there is a flag on the play.")
        elif t == "filler":
            lines.append(f"No play is happening right now. Make ONE short, natural comment about this and nothing else: {e['facts']}")
        elif t == "play_result":
            lines.append(f"The play just ended. Scoreboard result: {e.get('detail')}. Say ONLY that result in one short sentence. Do not describe formations or how it happened.")
    return " ".join(lines) or "Normal play in progress."


RISKY = ["seam", "route", "slant", "screen pass", "play action", "play-action", "blitz", "sack", "fumble", "intercept",
         "shotgun", "under center", "scramble", "touchdown", "field goal", "punt", "penalty", "flag", "turnover",
         "audible", "eyes", "scans", "stares", "glances", "looks off"]


def unsupported(line, allowed):
    """Words an announcer might invent. Allowed only if they already appear in what we were told."""
    l, a = line.lower(), allowed.lower()
    bad = [w for w in RISKY if w in l and w not in a]
    said = {(d, x.lower()) for d, x in re.findall(r"\b([1-4])(?:st|nd|rd|th)\s*(?:&|and)\s*(\d{1,2}|goal)\b", l)}
    told = {(d, x.lower()) for d, x in re.findall(r"\b([1-4])(?:st|nd|rd|th)?\s*&\s*(\d{1,2}|goal)\b", a)}
    bad += [f"{d} and {x}" for d, x in said - told]                 # e.g. "2nd and 10" when the scoreboard says 1st & 10
    for n in re.findall(r"\b\d+\b", l):                      # yard lines, distances, scores: only if we were told them
        if not re.search(rf"(?<!\d){n}(?!\d)", a):
            bad.append(n)
    return bad


def name_teams(action, possession):
    """The video model says 'the offense' / 'the defense'. The scoreboard says who actually has the ball."""
    if possession not in config.TEAMS:
        return action                                       # unknown: leave it generic, never guess
    off = config.TEAMS[possession]["name"]
    dfn = config.TEAMS["msu" if possession == "akron" else "akron"]["name"]
    action = re.sub(r"(?i)\b(the )?offen[sc]e\b", off, action)
    return re.sub(r"(?i)\b(the )?defen[sc]e\b", dfn, action)


CHECK_PROMPT = """You check an announcer's line against the ONLY facts we have.
FACTS:
{facts}

LINE: {line}

Does the LINE claim anything that is NOT supported by the FACTS? Watch for: a team or player not in the facts, which team is kicking or has the ball, a yard line or distance, a location such as the sideline, an action or result nobody stated, a formation. General excitement and opinion are fine.
Answer with exactly one word: OK or BAD."""


def fact_check(line, facts):
    if not getattr(config, "FACT_CHECK", True):
        return True
    try:
        out = vision.chat(config.TEXT_MODEL, [{"role": "user", "content": CHECK_PROMPT.format(facts=facts, line=line)}],
                          temperature=0.0, num_predict=4)
    except Exception:
        return True                                            # a failed check must not silence the show
    return out.strip().upper().startswith("OK")


_FORMATION = re.compile(r",?\s*\b(?:(?:out of|in) the shotgun|under center|in shotgun|shotgun(?: formation)?)\b", re.I)


def scrub(line):
    """The model likes to claim a formation ('under center'). We can't verify it, so cut the phrase out."""
    out = _FORMATION.sub("", line)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*,", ",", out)
    return re.sub(r"\s{2,}", " ", out).strip(" ,")


def too_similar(line, recent, threshold=0.72):
    return any(difflib.SequenceMatcher(None, line.lower(), r.lower()).ratio() > threshold for r in recent)


def generate(observation, resolved, situation, events, recent, must_speak=False):
    verified = [f"#{p['number']} {p['name']} ({config.TEAMS[p['team']]['name']} {p['pos']})"
                for p in resolved if p["status"] == "verified"]
    unverified = [f"a player on {config.TEAMS.get(p['team'], {'name': 'an unknown team'})['name']} (identity not confirmed)"
                  for p in resolved if p["status"] != "verified"]
    user = (
        f"GAME SITUATION: {situation}\n"
        f"EVENT: {events_to_text(events)}\n"
        f"WHAT HAPPENED (from video): {observation.get('action', '')} {observation.get('note') or ''}\n"
        f"VERIFIED PLAYERS: {'; '.join(verified) or 'none'}\n"
        f"UNVERIFIED PLAYERS (do not name): {'; '.join(unverified) or 'none'}\n"
        f"RECENT LINES: {' | '.join(recent) or 'none'}\n\n"
        + ("Write the next announcer line." if must_speak else "Write the next announcer line, or SKIP.")
    )
    allowed = f"{observation.get('action', '')} {observation.get('note') or ''} {events_to_text(events)} {situation}"
    facts = (f"GAME SITUATION: {situation}\nEVENT: {events_to_text(events)}\n"
             f"WHAT HAPPENED: {observation.get('action', '')} {observation.get('note') or ''}\n"
             f"VERIFIED PLAYERS: {'; '.join(verified) or 'none'}")
    global LAST_NOTE
    notes = ""
    LAST_NOTE = ""
    for attempt in range(2):
        system = SYSTEM.format(max_words=getattr(config, "MAX_WORDS_SPOKEN", 24))
        if must_speak:
            system = system.replace(" If there is nothing new worth saying, reply with exactly SKIP.", "")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user + notes}]
        out = scrub(vision.chat(config.TEXT_MODEL, messages, temperature=0.8, num_predict=90).strip().strip('"'))
        if not out or out.upper().startswith("SKIP"):
            LAST_NOTE = f"writer chose to say nothing ({out!r})"
            return None
        if too_similar(out, recent):
            LAST_NOTE = f"too similar to a recent line: {out!r}"
            return None
        bad = unsupported(out, allowed)
        if bad:
            LAST_NOTE = f"guard blocked {bad} in {out!r}"
            notes = f"\n\nDo NOT use these words or numbers: {', '.join(bad)}."
            continue
        if not fact_check(out, facts):
            LAST_NOTE = f"fact-check said BAD: {out!r}"
            notes = "\n\nYour previous line made claims that are not in the facts. Use ONLY the facts above."
            continue
        LAST_NOTE = ""
        return out
    return None
