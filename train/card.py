"""
Draw the social preview card (web/public/og.png, 1200x630) from training.json,
so links to the demo unfurl with the real loss curve and numbers.

    python card.py        # run after export.py
"""

import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PUBLIC = os.path.join(ROOT, "web", "public")
FONTS = "C:/Windows/Fonts"

W, H, S = 1200, 630, 2  # draw at 2x, downsample for smooth edges
BG, TEXT, MUTED, LINE = (11, 13, 18), (231, 233, 239), (139, 146, 165), (36, 42, 56)
AMBER, VIOLET = (245, 165, 36), (157, 140, 255)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size * S)


def main():
    with open(os.path.join(PUBLIC, "model", "training.json"), encoding="utf-8") as f:
        t = json.load(f)
    m = t["meta"]

    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    x0 = 72 * S

    d.text((x0, 64 * S), "A LANGUAGE MODEL, BUILT FROM SCRATCH", font=font("consolab.ttf", 20), fill=AMBER)
    d.text((x0 - 6 * S, 92 * S), "AARUL", font=font("consolab.ttf", 150), fill=TEXT)
    acro = [("A", "ttention "), ("A", "rchitecture, "), ("R", "ebuilt "), ("U", "sing a "), ("L", "aptop")]
    x, y = x0, 252 * S
    for cap, rest in acro:
        d.text((x, y), cap, font=font("consolab.ttf", 27), fill=TEXT)
        x += d.textlength(cap, font=font("consolab.ttf", 27))
        d.text((x, y), rest, font=font("consola.ttf", 27), fill=MUTED)
        x += d.textlength(rest, font=font("consola.ttf", 27))

    lines = ["Written from scratch, trained on a laptop GPU,", "running in your browser. Watch it think."]
    for i, line in enumerate(lines):
        d.text((x0, (338 + i * 38) * S), line, font=font("segoeui.ttf", 28), fill=TEXT)

    hours = (m.get("train_minutes") or 0) / 60
    stats = [
        (f"{m['params'] / 1e6:.1f}M", "parameters"),
        (f"{hours:.1f} h", "on a laptop"),
        (f"{m['file_mb']:.0f} MB", "in the browser"),
        ("$0", "cloud bill"),
    ]
    x = x0
    for big, small in stats:
        d.text((x, 470 * S), big, font=font("consolab.ttf", 40), fill=TEXT)
        d.text((x, 520 * S), small, font=font("segoeui.ttf", 20), fill=MUTED)
        x += 190 * S

    # Loss curve, top right: log-x like the page.
    cx0, cx1, cy0, cy1 = 820 * S, 1140 * S, 92 * S, 330 * S
    curve = [(0, t["val_loss"][0][1])] + [tuple(p) for p in t["train_loss"]]
    max_step = curve[-1][0]
    rand = math.log(t["config"]["model"]["vocab_size"])
    lo = min(l for _, l in curve) - 0.3
    sx = lambda s: cx0 + math.log10(1 + s / 10) / math.log10(1 + max_step / 10) * (cx1 - cx0)  # noqa: E731
    sy = lambda l: cy0 + (rand + 0.2 - l) / (rand + 0.2 - lo) * (cy1 - cy0)  # noqa: E731
    d.rounded_rectangle((cx0 - 24 * S, cy0 - 40 * S, cx1 + 24 * S, cy1 + 46 * S), 16 * S, outline=LINE, width=2 * S)
    d.text((cx0, cy0 - 30 * S), "loss while training", font=font("consola.ttf", 18), fill=MUTED)
    for gx in range(cx0, cx1, 14 * S):
        d.line((gx, sy(rand), gx + 6 * S, sy(rand)), fill=LINE, width=2 * S)
    d.text((cx1, sy(rand) + 6 * S), "random guessing", font=font("consola.ttf", 15), fill=MUTED, anchor="ra")
    d.line([(sx(s), sy(l)) for s, l in curve], fill=VIOLET, width=4 * S, joint="curve")
    end = (sx(curve[-1][0]), sy(curve[-1][1]))
    d.ellipse((end[0] - 7 * S, end[1] - 7 * S, end[0] + 7 * S, end[1] + 7 * S), fill=AMBER)
    d.text((cx1, cy1 + 12 * S), f"step 0 → {max_step:,}", font=font("consola.ttf", 16), fill=MUTED, anchor="ra")

    # A strip of "attention" heat along the bottom, as a visual signature.
    for i in range(48):
        a = 0.15 + 0.85 * abs(math.sin(i * 1.7) * math.cos(i * 0.6))
        col = tuple(int(BG[k] + (AMBER[k] - BG[k]) * a) for k in range(3))
        bx = (W * S) * i // 48
        d.rectangle((bx, H * S - 10 * S, (W * S) * (i + 1) // 48 - S, H * S), fill=col)

    img.resize((W, H), Image.LANCZOS).save(os.path.join(PUBLIC, "og.png"), optimize=True)
    print("wrote", os.path.join(PUBLIC, "og.png"))


if __name__ == "__main__":
    main()
