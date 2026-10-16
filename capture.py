"""Grabs a frame from the OBS Windowed Projector window.
   Windows 11: finds the window by title (pygetwindow).
   Linux Mint: finds it with xdotool (sudo apt install xdotool), or set CAPTURE_REGION in config.py.
   The window must be visible (not minimized or covered)."""
import ctypes
import platform
import shutil
import subprocess

import mss
from PIL import Image

import config

try:  # Windows: make window coordinates match real pixels on scaled displays
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass


def parse_geometry(text):
    """Parse the output of `xdotool getwindowgeometry --shell`."""
    vals = dict(item.split("=", 1) for item in text.split() if "=" in item)
    return {"left": int(vals["X"]), "top": int(vals["Y"]), "width": int(vals["WIDTH"]), "height": int(vals["HEIGHT"])}


def _box():
    region = getattr(config, "CAPTURE_REGION", None)        # (left, top, width, height) if you want to set it by hand
    if region:
        return {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
    if platform.system() == "Windows":
        import pygetwindow as gw
        wins = [w for w in gw.getAllWindows() if config.WINDOW_TITLE_CONTAINS.lower() in w.title.lower() and w.width > 200]
        if not wins:
            raise RuntimeError(f"No window with '{config.WINDOW_TITLE_CONTAINS}' in its title. Is the OBS projector open?")
        w = wins[0]
        if w.isMinimized:
            raise RuntimeError("The projector window is minimized.")
        return {"left": w.left, "top": w.top, "width": w.width, "height": w.height}
    if shutil.which("xdotool"):
        ids = subprocess.run(["xdotool", "search", "--name", config.WINDOW_TITLE_CONTAINS],
                             capture_output=True, text=True).stdout.split()
        if ids:
            geo = subprocess.run(["xdotool", "getwindowgeometry", "--shell", ids[-1]], capture_output=True, text=True).stdout
            return parse_geometry(geo)
        raise RuntimeError(f"No window with '{config.WINDOW_TITLE_CONTAINS}' in its title. Is the OBS projector open?")
    raise RuntimeError("On Linux, install xdotool (sudo apt install xdotool) or set CAPTURE_REGION in config.py")


def trim_letterbox(img, dark=10, run=12):
    """The projector window is rarely exactly 16:9, so it shows black bars above/below the game.
       Cut them off so the scoreboard sits where the rest of the code expects it.
       The bottom bar is found as the first long run of black rows, so a window covering part of the bar can't fool it."""
    import numpy as np
    is_dark = np.median(np.asarray(img.convert("L")), axis=1) <= dark
    live = np.where(~is_dark)[0]
    if len(live) == 0:
        return img
    top = int(live[0])
    bottom = int(live[-1]) + 1
    count = 0
    for y in range(top + int(0.4 * img.height), img.height):      # look for the bottom bar in the lower part only
        count = count + 1 if is_dark[y] else 0
        if count >= run:
            bottom = y - run + 1
            break
    if bottom - top < 0.4 * img.height:      # mostly dark scene: don't risk cutting real content
        return img
    return img.crop((0, top, img.width, bottom))


def grab():
    box = _box()
    l, t, r, b = config.CROP_MARGINS
    box = {"left": box["left"] + l, "top": box["top"] + t, "width": box["width"] - l - r, "height": box["height"] - t - b}
    with mss.mss() as sct:
        shot = sct.grab(box)
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    return trim_letterbox(img) if getattr(config, "TRIM_LETTERBOX", True) else img
