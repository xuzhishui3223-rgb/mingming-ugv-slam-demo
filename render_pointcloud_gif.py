"""Render an ASCII PLY point cloud as a looping GitHub README GIF."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
PLY_PATH = ROOT / "building_3d_map.ply"
OUTPUT_PATH = ROOT / "pointcloud_demo.gif"
WIDTH, HEIGHT, FRAMES = 900, 520, 40


def load_ply(path):
    lines = path.read_text(encoding="ascii").splitlines()
    end = lines.index("end_header")
    vertex_line = next(line for line in lines[:end] if line.startswith("element vertex "))
    count = int(vertex_line.split()[2])
    return np.loadtxt(lines[end + 1:end + 1 + count], usecols=(0, 1, 2))


def palette(height_fraction):
    stops = np.array(((25, 125, 220), (35, 230, 125), (250, 215, 55), (255, 105, 45)))
    scaled = np.clip(height_fraction, 0, 1) * (len(stops) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.minimum(lower + 1, len(stops) - 1)
    mix = (scaled - lower)[:, None]
    return (stops[lower] * (1 - mix) + stops[upper] * mix).astype(np.uint8)


def font(size):
    candidates = (
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def render(points):
    minimum, maximum = points.min(0), points.max(0)
    center = (minimum + maximum) / 2
    centered = points - center
    span = maximum - minimum
    max_span = max(float(span.max()), 1.0)
    colors = palette((points[:, 2] - minimum[2]) / max(float(span[2]), 1e-6))
    camera, focal = max_span * 3.4, min(WIDTH, HEIGHT) * 1.55
    pitch = np.deg2rad(47.0)
    cp, sp = np.cos(pitch), np.sin(pitch)
    frames = []

    for frame_index in range(FRAMES):
        yaw = -0.65 + 2 * np.pi * frame_index / FRAMES
        cy, sy = np.cos(yaw), np.sin(yaw)
        x, y, z = centered.T
        x1, y1 = cy * x - sy * y, sy * x + cy * y
        depth = cp * y1 - sp * z
        z2 = sp * y1 + cp * z
        factor = focal / np.maximum(camera - depth, 0.1)
        screen_x = WIDTH * 0.54 + x1 * factor
        screen_y = HEIGHT * 0.56 - z2 * factor
        visible = ((screen_x >= 5) & (screen_x < WIDTH - 5) &
                   (screen_y >= 5) & (screen_y < HEIGHT - 5))
        order = np.argsort(depth[visible])
        px, py = screen_x[visible][order], screen_y[visible][order]
        pc = colors[visible][order]

        image = Image.new("RGB", (WIDTH, HEIGHT), (3, 9, 6))
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        for sx, sy_, color in zip(px[::2], py[::2], pc[::2]):
            glow_draw.ellipse((sx - 3, sy_ - 3, sx + 3, sy_ + 3),
                              fill=(int(color[0]), int(color[1]), int(color[2]), 75))
        glow = glow.filter(ImageFilter.GaussianBlur(2.2))
        image = Image.alpha_composite(image.convert("RGBA"), glow)
        draw = ImageDraw.Draw(image)
        for sx, sy_, color in zip(px, py, pc):
            draw.ellipse((sx - 1.1, sy_ - 1.1, sx + 1.1, sy_ + 1.1),
                         fill=(int(color[0]), int(color[1]), int(color[2]), 235))

        draw.rounded_rectangle((18, 17, 375, 77), radius=12,
                               fill=(5, 22, 13, 225), outline=(38, 125, 76, 255), width=1)
        draw.text((34, 28), "MINGMING UGV  |  3D SLAM MAP", font=font(21), fill=(93, 242, 145, 255))
        draw.text((35, 55), f"{len(points):,} LiDAR points  -  height colored", font=font(12), fill=(150, 205, 169, 255))
        draw.text((WIDTH - 176, HEIGHT - 31), "MuJoCo simulation", font=font(12), fill=(92, 140, 108, 255))
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))
    return frames


def main():
    points = load_ply(PLY_PATH)
    frames = render(points)
    frames[0].save(OUTPUT_PATH, save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True, disposal=2)
    print(f"Saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
