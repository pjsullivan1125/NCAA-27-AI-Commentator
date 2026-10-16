"""Offline tests for the logic that decides what is true. No GPU, model, OBS or audio device needed:
       python -m unittest discover -s tests -v"""
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for name in ("requests", "mss"):                      # only needed at runtime; stubbed so tests run anywhere
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = types.ModuleType(name)

from PIL import Image, ImageDraw

import commentary
import fallback
import filler
import main
import screen_state as ss
import zoom
from game_state import Tracker
from roster import Rosters

BASE = dict(quarter=1, clock_sec=200, down_dist="1 & 10", possession="akron", kickoff=False)


class TrackerTests(unittest.TestCase):
    def test_impossible_score_jumps_are_ignored_but_real_scores_count(self):
        t, events = Tracker(), []
        for left, right in [(3, 6), (3, 6), (3, 22), (3, 22), (3, 6), (9, 6), (9, 6)]:
            events += t.update(dict(BASE, score_left=left, score_right=right))
        kinds = [(e["team"], e["kind"]) for e in events if e["type"] == "score"]
        self.assertEqual(kinds, [("akron", "touchdown")])       # 3-22 never announced; 3 -> 9 is one touchdown
        self.assertEqual((t.sb["score_left"], t.sb["score_right"]), (9, 6))

    def test_opening_kickoff_is_never_announced_mid_game(self):
        t, seen = Tracker(), []
        for q, sec in [(1, 100), (1, 20), (1, 300), (2, 300), (2, 298)]:
            seen += [e["type"] for e in t.update(dict(BASE, quarter=q, clock_sec=sec, score_left=3, score_right=6))]
        self.assertNotIn("kickoff", seen)
        self.assertIn("quarter_ending", seen)

    def test_a_first_scoreboard_reading_is_not_trusted_until_it_repeats(self):
        t = Tracker()
        t.update(dict(BASE, score_left=6, score_right=6))                  # the real left score is 9, misread once as 6
        self.assertNotIn("score_left", t.sb)
        for _ in range(3):
            t.update(dict(BASE, score_left=9, score_right=6))
        self.assertEqual(t.sb["score_left"], 9)

    def test_results_come_from_the_scoreboard(self):
        p = Tracker._parse_result
        self.assertEqual(p("1 & 10", "2 & 8")["text"], "gain of 2 yards, now 2nd & 8")
        self.assertEqual(p("2 & 8", "3 & 8")["kind"], "no_gain")
        self.assertEqual(p("3 & 4", "1 & 10")["kind"], "first_down")
        self.assertEqual(p("2 & 9", "3 & 12")["kind"], "loss")
        self.assertIsNone(p("1 & 10", "goal"))                   # unreadable: say nothing rather than guess

    def test_kickoff_clears_stale_down_and_names_the_kicking_team(self):
        t = Tracker()
        t.update(dict(BASE, score_left=3, score_right=0))
        t.update(dict(BASE, score_left=3, score_right=0, down_dist=None, kickoff=True, possession="msu"))
        self.assertIn("Michigan State is kicking", t.situation_text())
        self.assertNotIn("1 & 10", t.situation_text())


class SafeLinesTests(unittest.TestCase):
    def test_every_important_event_has_a_correct_fallback_line(self):
        sb = dict(score_left=3, score_right=6, quarter=1)
        line = fallback.fallback_line([{"type": "score", "team": "msu", "kind": "touchdown", "delta": 6}], sb, "msu")
        self.assertIn("Michigan State leads Akron, six to three", line)
        self.assertIn("thirty seconds", fallback.fallback_line([{"type": "quarter_ending", "quarter": 1}], sb, None).lower())
        gain = {"type": "play_result", "data": {"kind": "gain", "yards": 2, "down": 2, "dist": 8}}
        self.assertEqual(fallback.fallback_line([gain], sb, "akron"), "Akron picks up two yards, now second and eight.")

    def test_filler_rotates_topics_and_does_not_repeat(self):
        f = filler.Filler()
        sb = dict(score_left=3, score_right=6, quarter=1, down_dist="2 & 8", kickoff=False)
        seen = {"Example Player": ("akron", "QB", time.time() - 30)}
        got = [f.next(sb, "akron", seen, 3, True) for _ in range(5)]
        self.assertEqual([g["topic"] for g in got], ["score", "situation", "player", "drive", "setting"])
        self.assertEqual(len({g["template"] for g in got}), 5)


class GuardTests(unittest.TestCase):
    def test_invented_numbers_and_downs_are_caught(self):
        facts = "TOUCHDOWN just scored by Michigan State. Akron 3, Michigan State 6; quarter 1; 2 & Goal"
        self.assertTrue(commentary.unsupported("Touchdown puts them up 12-3.", facts))
        self.assertTrue(commentary.unsupported("They face 2nd and 10 from their own 31.", "first down, now 1st & 10"))
        self.assertFalse(commentary.unsupported("Michigan State scores, now ahead six to three.", facts))

    def test_unverifiable_formation_claims_are_cut_out_of_the_line(self):
        line = commentary.scrub("Michigan State lines up, QB Smith under center, ready to take the snap.")
        self.assertEqual(line, "Michigan State lines up, QB Smith, ready to take the snap.")
        self.assertFalse(commentary.unsupported(line, "The offense is lined up."))

    def test_team_names_come_from_the_scoreboard_not_the_model(self):
        said = "The offense is lined up while the defense waits."
        self.assertEqual(commentary.name_teams(said, "akron"), "Akron is lined up while Michigan State waits.")
        self.assertEqual(commentary.name_teams(said, None), said)           # unknown possession: stay generic


class RosterTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "a.json").write_text(json.dumps([{"number": 22, "name": "A Cornerback", "pos": "CB"},
                                                      {"number": 7, "name": "A Linebacker", "pos": "LB"}]))
        (self.dir / "m.json").write_text(json.dumps([{"number": 0, "name": "A Corner", "pos": "CB"},
                                                      {"number": 0, "name": "A Runner", "pos": "RB"}]))
        (self.dir / "o.json").write_text(json.dumps({"akron": {"7": {"name": "The QB", "pos": "QB"}}}))
        cfg = {"akron": {"roster": self.dir / "a.json"}, "msu": {"roster": self.dir / "m.json"}}
        self.r = Rosters(cfg, self.dir / "o.json")

    def test_override_beats_the_website_roster_and_the_models_side_guess(self):
        out = self.r.resolve("akron", 7, side="defense", role="tackler")
        self.assertEqual((out["name"], out["status"]), ("The QB", "verified"))

    def test_duplicate_numbers_are_resolved_by_side_of_ball(self):
        self.assertEqual(self.r.resolve("msu", 0, side=("offense", "special"))["name"], "A Runner")
        self.assertEqual(self.r.resolve("msu", 0)["status"], "ambiguous")

    def test_a_position_that_contradicts_the_side_is_never_named(self):
        out = self.r.resolve("akron", 22, side=("offense", "special"))
        self.assertEqual((out["name"], out["status"]), (None, "mismatch"))


class VisionRulesTests(unittest.TestCase):
    def frame(self):
        im = Image.new("RGB", (400, 300), (70, 120, 55))                        # grass
        d = ImageDraw.Draw(im)
        d.rectangle((100, 100, 140, 200), fill=(235, 235, 235))                  # white uniform
        d.rectangle((250, 100, 290, 200), fill=(15, 30, 70))                     # navy uniform
        return im

    def test_team_comes_from_uniform_brightness_relative_to_the_grass(self):
        im = self.frame()
        box = lambda x1, y1, x2, y2: [x1 / 400 * 1000, y1 / 300 * 1000, x2 / 400 * 1000, y2 / 300 * 1000]
        self.assertEqual(zoom.team_from_pixels(im, box(100, 100, 140, 200)), "msu")
        self.assertEqual(zoom.team_from_pixels(im, box(250, 100, 290, 200)), "akron")
        dim = Image.eval(im, lambda v: int(v * 0.5))                              # same scene, half the light
        self.assertEqual(zoom.team_from_pixels(dim, box(100, 100, 140, 200)), "msu")
        self.assertEqual(zoom.team_from_pixels(dim, box(250, 100, 290, 200)), "akron")

    def test_a_name_needs_two_independent_number_reads_to_agree(self):
        d = Path(tempfile.mkdtemp())
        (d / "a.json").write_text(json.dumps([{"number": 8, "name": "A Receiver", "pos": "WR"}]))
        (d / "m.json").write_text("[]")
        rosters = Rosters({"akron": {"roster": d / "a.json"}, "msu": {"roster": d / "m.json"}}, d / "none.json")
        obs = {"players": [{"team": "akron", "number": 8, "box": [10, 10, 90, 90], "role": "ball_carrier"}]}
        main._ZOOM_OK.clear()
        zoom.read_number = lambda img, box, model=None: (8, None)
        agree = main.resolve_players(obs, Image.new("RGB", (50, 50)), rosters, "akron", False)[0]
        main._ZOOM_OK.clear()
        zoom.read_number = lambda img, box, model=None: (21, None)
        disagree = main.resolve_players(obs, Image.new("RGB", (50, 50)), rosters, "akron", False)[0]
        self.assertEqual(agree["name"], "A Receiver")
        self.assertIsNone(disagree["name"])


class ScoreboardTests(unittest.TestCase):
    def reader(self, tokens):
        class Fake(ss.Reader):
            def __init__(self):
                self._strip_img, self._strip = None, []

            def detect(self, img, box=None):
                return [{"text": t, "x": x, "y": y, "conf": 0.9} for t, x, y in tokens] if box == ss.STRIP else []
        return Fake()

    def test_reads_tolerate_common_ocr_slips(self):
        toks = [("AKRON", 0.34, 0.865), ("0", 0.41, 0.865), ("19", 0.452, 0.838), ("Ist", 0.445, 0.857), ("DASPORTS", 0.5, 0.835),
                ("4:57", 0.5, 0.857), ("25", 0.551, 0.857), ("3", 0.592, 0.865), ("MICHIGRN STATE", 0.68, 0.865), ("lst\u65e510", 0.5, 0.883)]
        sb = ss.parse_scoreboard(self.reader(toks), Image.new("RGB", (1500, 950)))
        self.assertEqual((sb["quarter"], sb["clock_sec"], sb["down_dist"], sb["score_left"], sb["score_right"]),
                         (1, 297, "1 & 10", 0, 3))

    def test_a_flag_is_recognised_by_the_yellow_plate_even_if_ocr_misses_the_word(self):
        tokens = [("4:41", 0.503, 0.841), ("AKRON", 0.37, 0.855)]
        for colour, expected in (((235, 200, 30), "penalty"), ((14, 35, 62), "gameplay")):
            im = Image.new("RGB", (1500, 950))
            ImageDraw.Draw(im).rectangle((680, 790, 830, 830), fill=colour)       # the plate under the clock
            toks = tokens + [("EASPORTS", 0.5, 0.82)]
            self.assertEqual(self.reader(toks).classify(im)[0].value, expected)

    def test_possession_is_read_from_the_colour_of_the_down_and_distance_plate(self):
        for colour, expected in (((14, 35, 62), "akron"), ((20, 70, 50), "msu")):
            im = Image.new("RGB", (1500, 950))
            ImageDraw.Draw(im).rectangle((600, 810, 900, 880), fill=colour)
            self.assertEqual(ss.possession_from_plate(im, [{"text": "1st & 10", "x": 0.5, "y": 0.886, "conf": 0.9}]), expected)


if __name__ == "__main__":
    unittest.main()
