"""Live commentator loop.
   Live:     python main.py
   Offline:  python main.py --images screens --no-tts     (test with saved screenshots, no OBS needed)"""
import argparse
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import commentary
import config
import fallback
import vision
import zoom
from filler import Filler
from game_state import Tracker
from roster import Rosters
from screen_state import Reader, State, parse_scoreboard

SILENT = {State.PAUSE, State.REPLAY, State.UNKNOWN, State.PRE_SNAP}
PRIORITY = {"score", "quarter_ending", "half_ending", "game_ending", "quarter_start", "kickoff", "penalty", "play_result"}


def frame_diff(a, b):
    ga = np.asarray(a.convert("L").resize((64, 36)), dtype=np.float32)
    gb = np.asarray(b.convert("L").resize((64, 36)), dtype=np.float32)
    return float(np.abs(ga - gb).mean())


def frames(args):
    if args.images:
        for p in sorted(Path(args.images).glob("*.*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                print(f"\n=== {p.name}")
                yield Image.open(p).convert("RGB")
    else:
        import capture
        while True:
            t0 = time.time()
            try:
                yield capture.grab()
            except RuntimeError as e:
                print(f"[capture] {e}")
            time.sleep(max(0.0, config.CAPTURE_INTERVAL - (time.time() - t0)))


ROLE_ORDER = {"passer": 0, "ball_carrier": 1, "tackler": 2, "receiver": 3}


_ZOOM_OK = {}      # (team, number) -> time until which a recent zoomed read still counts


def resolve_players(obs, frame, rosters, possession=None, kickoff=False):
    """For the most important players: team from pixels, number re-read on a zoomed crop.
       A name is only used when the wide-shot number and the zoomed number AGREE (or a nameplate matches)."""
    players = [p for p in (obs.get("players") or []) if isinstance(p, dict)]
    players.sort(key=lambda p: ROLE_ORDER.get(p.get("role"), 9))
    out, zoomed, seen_ids = [], 0, set()
    for p in players[:3]:
        key = (p.get("team"), p.get("number"))
        if key in seen_ids:                       # the model listed the same number twice
            continue
        seen_ids.add(key)
        team = p.get("team") if p.get("team") in config.TEAMS else "unknown"
        number, plate = None, None
        try:
            first = int(p.get("number"))
        except (TypeError, ValueError):
            first = None
        nameable = first is not None and bool(rosters.candidates(team, first))   # no roster entry = can't be named anyway
        now = time.time()
        if team != "unknown" and nameable and _ZOOM_OK.get((team, first), 0) > now:
            number = first                        # a zoomed read confirmed this player a moment ago
        elif team != "unknown" and p.get("box") and nameable and zoomed < getattr(config, "MAX_ZOOM_PLAYERS", 2):
            zoomed += 1
            try:
                n2, plate = zoom.read_number(frame, p["box"])
            except Exception as e:
                print(f"  [zoom error] {e}")
                n2, plate = None, None
            if n2 is not None and n2 == first:
                number = n2
                _ZOOM_OK[(team, first)] = time.time() + getattr(config, "ZOOM_CACHE_SECONDS", 25.0)
        # which side of the ball a player is on comes from the SCOREBOARD (who has the ball) + the pixel-based team,
        # not from the model's guess
        if kickoff or possession not in config.TEAMS or team == "unknown":
            side = None
        elif team == possession:
            side = ("offense", "special")
        else:
            side = ("defense", "special")
        r = rosters.resolve(team, number, side, p.get("role"), plate)
        r["role"] = p.get("role", "other")
        r["seen"] = p.get("number")
        out.append(r)
    return out


class Worker(threading.Thread):
    """Does the slow work (vision, zoom, writing the line) in the background,
       so the screen watcher keeps reading the scoreboard and never stalls."""

    def __init__(self, handle):
        super().__init__(daemon=True)
        self.handle, self.cv = handle, threading.Condition()
        self.events, self.frame_job, self.busy = deque(maxlen=3), None, False

    def submit(self, job):
        with self.cv:
            if job["priority"]:
                self.events.append(job)       # event lines (touchdown, recap, ...) are never replaced by frames
            else:
                self.frame_job = job          # a newer frame replaces an older one that is still waiting
            self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while not self.events and self.frame_job is None:
                    self.cv.wait()
                if self.events:
                    job = self.events.popleft()
                else:
                    job, self.frame_job = self.frame_job, None
                self.busy = True
            try:
                self.handle(job)
            except Exception as e:
                print(f"  [error] {e}")
            finally:
                self.busy = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="folder of screenshots to run through instead of live capture")
    ap.add_argument("--no-tts", action="store_true")
    args = ap.parse_args()

    rosters = Rosters(config.TEAMS, config.OVERRIDES_PATH)
    reader, tracker = Reader(), Tracker()
    speaker = None
    if not args.no_tts:
        from tts import Speaker
        speaker = Speaker()

    min_gap = 0.0 if args.images else config.MIN_SECONDS_BETWEEN_LINES
    stale_after = getattr(config, "STALE_FRAME_SECONDS", 6.0)
    recent = deque(maxlen=8)
    lock = threading.Lock()
    shared = {"last_spoken": 0.0, "gpu_free_at": 0.0, "seen": {}, "start": time.time(), "last_filler": 0.0}
    filler = Filler()
    filler_after = getattr(config, "FILLER_AFTER_SECONDS", 30.0)
    cooldown = 0.0 if args.images else getattr(config, "VISION_COOLDOWN_SECONDS", 6.0)

    def handle(job):
        priority, frame, events = job["priority"], job["frame"], job["events"]
        if not priority:
            if time.time() - job["t"] > stale_after:                       # the moment has passed
                return
            if time.time() - shared["last_spoken"] < min_gap or (speaker and speaker.busy.is_set()):
                return
        if not priority and time.time() < shared["gpu_free_at"]:       # let the GPU rest so OBS can keep up
            return
        if frame is None:                      # event-only line (kickoff, score, recap): no vision needed
            obs, v_s = {"action": "", "players": []}, 0.0
        else:
            obs, v_s = vision.describe_frame(frame)
        if not obs:
            print("  [vision] unparseable reply, skipping")
            return
        obs["action"] = commentary.name_teams(obs.get("action", ""), job["possession"])
        if not priority and not obs["action"]:
            return                                          # nothing reliable to say about this frame
        tz = time.time()
        resolved = resolve_players(obs, frame, rosters, job["possession"], job["kickoff"]) if frame is not None else []
        z_s = time.time() - tz
        t1 = time.time()
        with lock:
            recent_now = list(recent)
        line = commentary.generate(obs, resolved, job["situation"], events, recent_now, must_speak=priority)
        origin = ""
        if not line:
            note = commentary.LAST_NOTE or "no line"
            if priority and events:               # important moments are never left silent: use the safe template line
                tmpl = next((e["template"] for e in events if e["type"] == "filler"), None) \
                    or fallback.fallback_line(events, job.get("sb", {}), job["possession"])
                if tmpl:
                    line, origin = tmpl, " (template)"
            print(f"  [no line] {note}" + (" -> using the safe template line" if origin else ""))
        for r in resolved:
            if r["status"] == "verified" and r.get("name"):
                shared["seen"][r["name"]] = (r["team"], r.get("pos") or "", time.time())
        txt_s = time.time() - t1
        print(f"  TIMING vision={v_s:.1f}s zoom={z_s:.1f}s text={txt_s:.1f}s | {time.time() - job['t']:.1f}s from capture to line | events={[e['type'] for e in events]}")
        if resolved:
            print(f"  players={[(r['team'], 'saw', r['seen'], 'agreed', r['number'], r['name'], r['status']) for r in resolved]}")
        if frame is not None:
            shared["gpu_free_at"] = time.time() + cooldown
        if line:
            with lock:
                recent.append(line)
            shared["last_spoken"] = time.time()
            print(f"  >>> {line}{origin}")
            if speaker:
                speaker.say(line)

    if args.images:
        dispatch = handle                      # saved screenshots: do everything in order, one at a time
    else:
        frame_worker, event_worker = Worker(handle), Worker(handle)      # events (touchdown, recap) never wait behind a vision job
        frame_worker.start()
        event_worker.start()
        dispatch = lambda job: (event_worker if (job["priority"] and job["frame"] is None) else frame_worker).submit(job)

    prev_state, last_analyzed = None, None
    for img in frames(args):
        t0 = time.time()
        state, _ = reader.classify(img)
        events = []
        if state in (State.GAMEPLAY, State.PRE_SNAP, State.PLAY_SELECT, State.PENALTY):
            raw = parse_scoreboard(reader, img)
            events = tracker.update(raw)
            clock = "--" if raw["clock_sec"] is None else f"{raw['clock_sec'] // 60}:{raw['clock_sec'] % 60:02d}"
            read = f"Q{raw['quarter']} {clock} {raw['down_dist']} score {raw['score_left']}-{raw['score_right']} ball={raw['possession']}{' KICKOFF' if raw['kickoff'] else ''}"
        else:
            read = "-"
        ocr_s = time.time() - t0
        print(f"[{state.value}] ocr={ocr_s:.1f}s read: {read} | events={[e['type'] for e in events]}")

        prio, frame = [], None
        for e in events:
            if e["type"] == "play_complete":          # the scoreboard changed: report the real result, whatever screen we are on
                if tracker.last_result:
                    prio.append({"type": "play_result", "detail": tracker.last_result, "data": tracker.last_result_data})
            elif e["type"] in PRIORITY:
                prio.append(e)
        if state == State.PENALTY and prev_state != State.PENALTY:
            prio.append({"type": "penalty"})                # nothing to look at: the flag is on the scoreboard
        prev_state = state

        snap = {"situation": tracker.situation_text(), "possession": tracker.sb.get("possession"), "kickoff": tracker.sb.get("kickoff"),
                "sb": dict(tracker.sb)}
        # no dead air: if nothing has been said for a while, talk about something we know for certain
        now = time.time()
        if (not prio and not args.images and state != State.PAUSE
                and now - max(shared["last_spoken"], shared["start"]) > filler_after
                and now - shared["last_filler"] > 12 and not (speaker and speaker.busy.is_set())):
            topic = filler.next(tracker.sb, tracker.sb.get("possession"), shared["seen"], tracker.drive_plays, tracker.down_is_fresh())
            if topic:
                shared["last_filler"] = now
                print(f"  [quiet for {filler_after:.0f}s: filler on '{topic['topic']}']")
                prio.append({"type": "filler", **topic})
        if prio:
            dispatch({"frame": frame, "events": prio, "priority": True, "t": t0, **snap})
        elif state == State.GAMEPLAY and (last_analyzed is None or frame_diff(img, last_analyzed) > config.FRAME_DIFF_THRESHOLD):
            last_analyzed = img
            dispatch({"frame": img, "events": [], "priority": False, "t": t0, **snap})


if __name__ == "__main__":
    main()
