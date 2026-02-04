import argparse
import math
import matplotlib.pyplot as plt


# ======================================================
# Bézier + Rasterization
# ======================================================

def bezier_point(t, p0, p1, p2, p3):

    u = 1.0 - t

    x = (
        u * u * u * p0[0]
        + 3 * u * u * t * p1[0]
        + 3 * u * t * t * p2[0]
        + t * t * t * p3[0]
    )

    y = (
        u * u * u * p0[1]
        + 3 * u * u * t * p1[1]
        + 3 * u * t * t * p2[1]
        + t * t * t * p3[1]
    )

    return x, y


def bresenham(x0, y0, x1, y1):

    pts = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    x, y = x0, y0

    while True:

        pts.append((x, y))

        if x == x1 and y == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return pts


# ======================================================
# Circular Brush
# ======================================================

def circular_brush(radius):

    mask = []
    r2 = radius * radius

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):

            if dx * dx + dy * dy <= r2:
                mask.append((dx, dy))

    return mask


# ======================================================
# Rasterize at 2x Resolution
# ======================================================

def rasterize(p0, p1, p2, p3, samples, width):

    scale = 6

    p0 = (p0[0] * scale, p0[1] * scale)
    p1 = (p1[0] * scale, p1[1] * scale)
    p2 = (p2[0] * scale, p2[1] * scale)
    p3 = (p3[0] * scale, p3[1] * scale)

    radius = ((width - 0.5) / 2.0) * scale

    samples *= scale

    # --------------------------------------------------
    # Sample curve
    # --------------------------------------------------

    pts = []

    for i in range(samples + 1):

        t = i / samples
        x, y = bezier_point(t, p0, p1, p2, p3)

        pts.append((x, y))

    # --------------------------------------------------
    # Compute normals
    # --------------------------------------------------

    normals = []

    for i in range(len(pts)):

        if i == 0:
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]

        elif i == len(pts) - 1:
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]

        else:
            dx = pts[i+1][0] - pts[i-1][0]
            dy = pts[i+1][1] - pts[i-1][1]

        l = math.hypot(dx, dy)

        if l == 0:
            nx, ny = 0, 0
        else:
            nx = -dy / l
            ny = dx / l

        normals.append((nx, ny))


    # --------------------------------------------------
    # Build outline
    # --------------------------------------------------

    left = []
    right = []

    for (x, y), (nx, ny) in zip(pts, normals):

        left.append((
            x + nx * radius,
            y + ny * radius
        ))

        right.append((
            x - nx * radius,
            y - ny * radius
        ))

    # Square caps
    dx0 = pts[1][0] - pts[0][0]
    dy0 = pts[1][1] - pts[0][1]

    dl = math.hypot(dx0, dy0)
    dx0 /= dl
    dy0 /= dl

    dx1 = pts[-1][0] - pts[-2][0]
    dy1 = pts[-1][1] - pts[-2][1]

    dl = math.hypot(dx1, dy1)
    dx1 /= dl
    dy1 /= dl

    # Extend outline to make square ends
    left[0]  = (left[0][0]  - dx0*radius, left[0][1]  - dy0*radius)
    right[0] = (right[0][0] - dx0*radius, right[0][1] - dy0*radius)

    left[-1]  = (left[-1][0]  + dx1*radius, left[-1][1]  + dy1*radius)
    right[-1] = (right[-1][0] + dx1*radius, right[-1][1] + dy1*radius)


    # --------------------------------------------------
    # Build closed polygon
    # --------------------------------------------------

    poly = left + right[::-1]


    # --------------------------------------------------
    # Rasterize polygon (scanline fill)
    # --------------------------------------------------

    poly = [(int(round(x)), int(round(y))) for x, y in poly]

    ys = [p[1] for p in poly]
    miny, maxy = min(ys), max(ys)

    pixels = set()

    for y in range(miny, maxy + 1):

        inter = []

        for i in range(len(poly)):

            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]

            if y1 == y2:
                continue

            if y < min(y1, y2) or y >= max(y1, y2):
                continue

            t = (y - y1) / (y2 - y1)

            x = x1 + t * (x2 - x1)

            inter.append(x)

        inter.sort()

        for i in range(0, len(inter), 2):

            x0 = int(math.ceil(inter[i]))
            x1 = int(math.floor(inter[i+1]))

            for x in range(x0, x1 + 1):
                pixels.add((x, y))

    return pixels


# ======================================================
# Block Patterns
# ======================================================

# ======================================================
# Block Patterns (Flip Only, No Rotation)
# ======================================================

def flip_x(p):
    return [row[::-1] for row in p]


def flip_y(p):
    return p[::-1]


def variants(pattern):

    v0 = pattern
    v1 = flip_x(v0)
    v2 = flip_y(v0)
    v3 = flip_x(v2)

    out = [v0, v1, v2, v3]

    unique = []

    for v in out:
        if v not in unique:
            unique.append(v)

    return unique


# ======================================================
# 6x6 Block Patterns (Minecraft Parts)
# ======================================================

BASE_BLOCKS = {

    # Empty
    "empty": [
        "______",
        "______",
        "______",
        "______",
        "______",
        "______",
    ],

    # Full block
    "full": [
        "######",
        "######",
        "######",
        "######",
        "######",
        "######",
    ],

    # Shelf
    "shelf": [
        "##____",
        "##____",
        "##____",
        "##____",
        "##____",
        "##____",
    ],

    # Stair
    "stair": [
        "###___",
        "###___",
        "###___",
        "######",
        "######",
        "######",
    ],

    # Open trapdoor (vertical)
    "trapdoor_open": [
        "#_____",
        "#_____",
        "#_____",
        "#_____",
        "#_____",
        "#_____",
    ],

    # Closed trapdoor (horizontal)
    "trapdoor_closed": [
        "______",
        "______",
        "______",
        "______",
        "______",
        "######",
    ],
}


def pattern_to_bits(pat):
    """
    '#' -> 1
    '_' -> 0
    """

    out = []

    for row in pat:
        out.append([1 if c == "#" else 0 for c in row])

    return out

BLOCK_LIBRARY = []

for name, pat in BASE_BLOCKS.items():

    bits = pattern_to_bits(pat)

    for v in variants(bits):
        BLOCK_LIBRARY.append((name, v))

# ======================================================
# Tile Matching
# ======================================================

def tile_error(a, b):

    err = 0

    for y in range(6):
        for x in range(6):

            if a[y][x] != b[y][x]:
                err += 1

    return err


def best_block(tile):

    best = None
    best_err = 10**9

    for name, pat in BLOCK_LIBRARY:

        e = tile_error(tile, pat)

        if e < best_err:
            best_err = e
            best = name

    return best

# ======================================================
# Bitmap → Tiles
# ======================================================

def pixels_to_bitmap(pixels):

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    w = maxx - minx + 1
    h = maxy - miny + 1

    bmp = [[0] * w for _ in range(h)]

    for x, y in pixels:

        bx = x - minx
        by = y - miny

        bmp[by][bx] = 1

    return bmp


def bitmap_to_blocks(bmp):

    h = len(bmp)
    w = len(bmp[0])

    blocks = []

    TILE = 6

    for y in range(0, h, TILE):

        row = []

        for x in range(0, w, TILE):

            tile = [
                [
                    bmp[y + dy][x + dx]
                    if y + dy < h and x + dx < w else 0
                    for dx in range(TILE)
                ]
                for dy in range(TILE)
            ]

            name = best_block(tile)

            row.append(name)

        blocks.append(row)

    return blocks


# ======================================================
# Display
# ======================================================

def print_blocks(blocks):

    char_map = {
        "full": "F",
        "shelf": "H",
        "stair": "T",
        "trapdoor_open": "O",
        "trapdoor_closed": "C",
        "empty": " ",
    }
    # Reverse rows so higher Y is at top
    for row in reversed(blocks):

        print("".join(char_map[b] for b in row))

    print("F = Full")
    print("H = Shelf")
    print("T = Stair")
    print("O = Open Trapdoor")
    print("C = Closed Trapdoor")


# ======================================================
# CLI
# ======================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--p0", nargs=2, type=int, required=True)
    p.add_argument("--p1", nargs=2, type=int, required=True)
    p.add_argument("--p2", nargs=2, type=int, required=True)
    p.add_argument("--p3", nargs=2, type=int, required=True)

    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--width", type=float, default=1.0)

    return p.parse_args()


# ======================================================
# Main
# ======================================================

def main():

    args = parse_args()

    p0 = tuple(args.p0)
    p1 = tuple(args.p1)
    p2 = tuple(args.p2)
    p3 = tuple(args.p3)

    pixels = rasterize(
        p0, p1, p2, p3,
        args.samples,
        args.width
    )

    bmp = pixels_to_bitmap(pixels)

    blocks = bitmap_to_blocks(bmp)

    print_blocks(blocks)


if __name__ == "__main__":
    main()
