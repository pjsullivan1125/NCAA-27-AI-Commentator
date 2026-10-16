"""See exactly what boxes the model draws and how each one is judged.
   python debug_boxes.py screens/2026-10-15_13-40.png screens/2026-10-15_13-39.png
   Saves copies with the boxes drawn on them into the debug/ folder."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import config
import vision
import zoom

Path("debug").mkdir(exist_ok=True)
for arg in sys.argv[1:]:
    img = Image.open(arg).convert("RGB")
    w_s = min(img.width, config.VISION_MAX_WIDTH)
    h_s = int(img.height * w_s / img.width)
    prompt = vision.VISION_PROMPT
    raw = vision.chat(config.VISION_MODEL, [{"role": "user", "content": prompt, "images": [vision.encode_image(img)]}],
                      as_json=True, temperature=0.1, num_predict=450)
    print(f"\n=== {arg}  (image {img.width}x{img.height}, model saw {w_s}x{h_s})\nRAW REPLY: {raw[:900]}")
    obs = vision.normalize(vision.parse_json(raw), img, (w_s, h_s))
    if not obs:
        print("  could not parse the reply")
        continue
    print(f"  box scale detected: {obs.get('_box_mode')}")
    draw = ImageDraw.Draw(img)
    for pl in obs["players"]:
        st = zoom.color_stats(img, pl["box"]) if pl["box"] else None
        print(f"  #{pl.get('number')} role={pl.get('role')} box={pl['box'] and [round(x) for x in pl['box']]} -> team={pl['team']} stats={st and {k: round(v, 2) for k, v in st.items() if k != 'n'}} (Akron if ratio<0.85, MSU if >0.98)")
        if pl["box"]:
            x1, y1, x2, y2 = pl["box"]
            r = (x1 / 1000 * img.width, y1 / 1000 * img.height, x2 / 1000 * img.width, y2 / 1000 * img.height)
            draw.rectangle(r, outline=(255, 0, 255), width=3)
            draw.text((r[0] + 4, r[1] + 4), f"#{pl.get('number')} {pl['team']}", fill=(255, 255, 0))
    out = Path("debug") / (Path(arg).stem + "_boxes.png")
    img.save(out)
    print(f"  saved {out}")
