# Local AI Football Commentator

A live play-by-play commentator for a video game, built entirely from **local** components. It watches the game's video feed,
works out what is happening, and speaks about it through a local voice. No cloud APIs, no per-token costs.

> **Demo video:** _add link here_

It was built for EA College Football 27 (Michigan State at Spartan Stadium) as a portfolio project, but the ideas
(grounding a vision model, verifying what it claims, never letting it improvise facts) apply to any live-video AI system.

## The problem this really solves

Pointing a vision model at a video game and asking for commentary produces fluent text that is often **wrong**: it names the
wrong team as having the ball, invents jersey numbers, makes up yard lines and scores. The interesting engineering is everything
that stops that from reaching the microphone.

## How it works

```
OBS capture window ─► screen capture ─► OCR on the bottom strip ─► screen type + scoreboard ─► event tracker
                                                                         │                      (score, recap,
                                                                         │                       quarter end, kickoff)
              vision model: "what is happening" + a box per player      │                          │
                         │                                              │                          │
   team from uniform pixels ─► zoomed re-read of the number ─► roster lookup                        │
                         └──────────────────────────► writer model ◄────────────────────────────────┘
                                                         │
                       word/number/down guards ─► fact-check pass ─► (safe template if rejected)
                                                         │
                                         dead-air filler ─┘ ─► streaming local text-to-speech
```

| Stage | What it uses | Runs on |
|---|---|---|
| Screen type + scoreboard | RapidOCR + regexes tolerant of OCR slips | CPU |
| "What is happening" | Qwen3-VL via Ollama (vision) | GPU |
| Team of each player | Jersey brightness compared with the grass in the same frame | CPU |
| Who is the player | Number read twice (wide shot + zoomed crop) must agree, then a roster lookup | GPU + CPU |
| Who has the ball | Color of the scoreboard's down-and-distance plate | CPU |
| Commentary line | Text model, then guards and a second fact-check pass | GPU |
| Voice | Kokoro (or Piper), sentence-by-sentence streaming | CPU |

## Design decisions worth reading

- **The scoreboard is the source of truth.** Score, quarter, clock, down and distance, possession, and the result of each play
  all come from the scoreboard by OCR and pixel color, never from the vision model's opinion.
- **Code decides the teams.** The model only says "the offense" and "the defense". Code fills in the names from the scoreboard.
  A model sentence that names a team anyway is thrown away.
- **Names need two independent reads to agree.** A wide-shot number and a zoomed-crop number must match, the team (from pixels)
  must have a roster entry for it, and the roster position must fit the side of the ball. Otherwise the commentator says it
  "couldn't see who it was" instead of guessing.
- **Lighting-proof team check.** Fixed color ranges failed in simulated fog and warm light, so the check compares uniform
  brightness with the grass in the same frame (white is always brighter than grass, navy darker). Stress-tested on simulated
  dim, fog, rain, sunset and dusk versions of real frames; ambiguous cases return "unknown" rather than a wrong team.
- **OCR noise is expected.** Scores can only rise by 1, 2, 3, 6, 7 or 8, and the very first reading must repeat three times
  before it is believed. A reading of `3-22` is ignored, not announced.
- **Penalties are seen two ways:** the word FLAG, and the scoreboard plate turning yellow, so one OCR miss does not hide a flag.
- **Every line is checked.** A deterministic guard rejects invented jargon, numbers, and any down-and-distance the scoreboard
  did not give, and a second model pass compares the line to the facts. Important moments (touchdown, quarter ending, recap)
  fall back to a safe template line, so they are never silent and never made up.
- **No dead air.** After 30 seconds of quiet it rotates through things it knows for certain: score, situation, drive length,
  a verified player, the setting.
- **Latency-aware design.** Slow model work runs in a background worker while the scoreboard watcher keeps running; event
  lines (touchdowns, recaps) use their own lane so they never queue behind a vision job; stale frames are dropped.

## Measured on one machine (RTX 4070 12 GB, Linux Mint, X11)

| Step | Observed |
|---|---|
| Screen type + scoreboard (CPU OCR) | about 1.1 to 2.5 s per frame |
| Vision call | about 2.5 to 5.8 s depending on model size and frame |
| Zoomed number read | about 1.3 to 1.5 s per player |
| Scoreboard-driven recap, capture to line | about 2.2 to 3.1 s |
| Frame commentary, capture to line | about 6.5 to 11.5 s |

A small bake-off on three hand-labeled frames (21 clearly readable jersey numbers): Gemma 4 12B found 2 and reported 3
numbers that were not in the frame (about 1.6 s per frame); Qwen3-VL 8B instruct found 7 and reported 2 (about 2.4 s). That
sample is tiny and not a benchmark; it is why the design verifies numbers instead of trusting either model.

## Setup

Python 3.10 to 3.12. Tested on Linux Mint (X11). The Windows capture path is written but has not been tested live.

```bash
# system packages (Linux)
sudo apt install espeak-ng libportaudio2 xdotool

python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build; Kokoro runs on the CPU
pip install -r requirements.txt

ollama pull qwen3-vl:8b-instruct      # or qwen3-vl:4b-instruct for speed

python fetch_rosters.py               # downloads both rosters from the teams' public sites into rosters/
cp rosters/overrides.example.json rosters/overrides.json   # optional: fix players the game names differently
```

Run (OBS Windowed Projector open on the capture-card feed, and not covered by other windows):

```bash
python main.py --no-tts               # prints what it would say
python main.py                        # speaks
python main.py --images screens --no-tts   # replay saved screenshots through the whole pipeline
```

Optional, for a faster voice: install the CUDA build of PyTorch (use the command from pytorch.org/get-started/locally),
then set `KOKORO_DEVICE = "cuda"` in `config.py`.

Handy tools: `test_screens.py` (screen detection on saved images), `compare_models.py` (speed and number reading per model),
`debug_boxes.py` (draws the model's player boxes and shows how each was judged). Settings and their explanations are in
`config.py`.

## Tests

```bash
python -m unittest discover -s tests -v
```

18 offline tests cover the logic that decides what is true: score-noise filtering, kickoff and recap events, the invented-fact
guards, roster rules, the lighting-robust team check, scoreboard parsing, and possession from the plate color. They need no
GPU, model, OBS or audio device.

## Known limitations

- A single still frame cannot tell a play starting from one that just ended, so the model's description can be off in timing.
- Jersey numbers on wide shots are often unreadable; most lines therefore avoid names, by design.
- The game's roster can differ from the real roster (for example, players who did not sign NIL deals), so there are overrides.
- The commentary reacts to plays rather than keeping pace with them: roughly 6 to 10 seconds behind on frame-based lines.
- The lighting test used simulated weather on real frames, not real rain, snow or night games.
- Screen capture reads on-screen pixels, so other windows covering the projector window end up in the picture.

## Disclaimer

An unofficial hobby project, not affiliated with or endorsed by EA, Michigan State University, or the University of Akron.
No game footage, logos or roster data are included in this repository. Rosters are downloaded at run time from the
universities' public athletics sites; check their terms before reusing the data.
