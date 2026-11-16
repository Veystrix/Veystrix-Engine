#!/usr/bin/env python3
"""Generate AppIcon.icns (original artwork) with Pillow + iconutil. Usage: python3 make_icon.py <out.icns>"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SCALE = 4  # supersample, then downsample with LANCZOS


def _gradient(size: int) -> Image.Image:
    """Deep indigo to violet background with a soft diagonal glow."""
    img = Image.new("RGBA", (size, size))
    d = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(1, size - 1)
        d.line([(0, y), (size, y)], fill=(
            int(22 + 64 * t),
            int(26 + 34 * t),
            int(74 + 132 * t),
            255,
        ))
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([int(size * 0.10), int(size * 0.04), int(size * 0.78), int(size * 0.66)],
               fill=(130, 150, 255, 70))
    glow = glow.filter(ImageFilter.GaussianBlur(size * 0.10))
    img.alpha_composite(glow)
    return img


def draw(size: int) -> Image.Image:
    s = size * SCALE

    # Rounded squircle mask (Apple-style continuous corner).
    pad = int(s * 0.055)
    radius = int(s * 0.235)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, s - pad, s - pad], radius=radius, fill=255)
    backdrop = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    backdrop.paste(_gradient(s), (0, 0), mask)

    # Soft drop shadow for the mark, drawn on its own layer.
    shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    w = int(s * 0.082)
    left, right = int(s * 0.30), int(s * 0.70)
    top, bottom, mid = int(s * 0.27), int(s * 0.70), s // 2
    sd.line([(left, top), (mid, bottom)], fill=(0, 0, 0, 120), width=w, joint="curve")
    sd.line([(right, top), (mid, bottom)], fill=(0, 0, 0, 120), width=w, joint="curve")
    shadow = shadow.filter(ImageFilter.GaussianBlur(s * 0.018))
    shadow.putalpha(shadow.getchannel("A").point(lambda a: int(a * 0.55)))
    shadow = Image.composite(shadow, Image.new("RGBA", (s, s), (0, 0, 0, 0)), mask)
    backdrop.alpha_composite(shadow)

    # The V: two crisp strokes, white and ice blue, with rounded terminals.
    d = ImageDraw.Draw(backdrop)
    d.line([(left, top), (mid, bottom)], fill=(252, 253, 255, 255), width=w, joint="curve")
    d.line([(right, top), (mid, bottom)], fill=(150, 222, 255, 255), width=w, joint="curve")
    for (x, y), color in (
        ((left, top), (252, 253, 255, 255)),
        ((right, top), (150, 222, 255, 255)),
        ((mid, bottom), (220, 240, 255, 255)),
    ):
        r = w // 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # Stream motif: three dots fading below the vertex, suggesting streamed weights.
    y = int(s * 0.80)
    for i, alpha in enumerate((200, 140, 85)):
        x = int(s * (0.42 + i * 0.08))
        r = int(s * 0.018)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(226, 240, 255, alpha))

    # Subtle top gloss.
    gloss = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(gloss).ellipse(
        [int(s * 0.16), int(-s * 0.30), int(s * 0.92), int(s * 0.46)],
        fill=(255, 255, 255, 26),
    )
    gloss = gloss.filter(ImageFilter.GaussianBlur(s * 0.06))
    backdrop.alpha_composite(Image.composite(gloss, Image.new("RGBA", (s, s), (0, 0, 0, 0)), mask))

    return backdrop.resize((size, size), Image.LANCZOS)


def main(out: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            draw(base).save(iconset / f"icon_{base}x{base}.png")
            draw(base * 2).save(iconset / f"icon_{base}x{base}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", out], check=True)


if __name__ == "__main__":
    main(sys.argv[1])
