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

    return pixels, pts


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

    # Slab
    "slab": [
        "______",
        "______",
        "______",
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

VERTICAL_BLOCKS = {
    "shelf",
    "trapdoor_open",
}

HORIZONTAL_BLOCKS = {
    "trapdoor_closed",
    "slab",   # future-proof
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


def best_block(tile, dx, dy):

    best = None
    best_pat = None
    best_err = 10**9
    best_bias = -1

    ax = abs(dx)
    ay = abs(dy)

    prefer_vertical = ay > ax
    prefer_horizontal = ax > ay

    for name, pat in BLOCK_LIBRARY:

        err = tile_error(tile, pat)

        if err > best_err:
            continue

        bias = 0

        if prefer_vertical and name in VERTICAL_BLOCKS:
            bias = 1
        elif prefer_horizontal and name in HORIZONTAL_BLOCKS:
            bias = 1

        if err < best_err or (err == best_err and bias > best_bias):

            best = name
            best_pat = pat
            best_err = err
            best_bias = bias

    return best, best_pat

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
        bmp[y - miny][x - minx] = 1

    # Return offset too
    return bmp, minx, miny

def bitmap_to_blocks(bmp, pts):

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

            # Tile center (in bitmap coords)
            cx = x + TILE // 2
            cy = y + TILE // 2

            # Estimate tangent
            dx, dy = estimate_tangent(pts, cx, cy)

            name, pat = best_block(tile, dx, dy)
            row.append((name, pat))

        blocks.append(row)

    return blocks

def estimate_tangent(pts, cx, cy, radius=10):
    """
    Estimate tangent near (cx, cy) by finding nearby curve points.
    Returns (dx, dy).
    """

    close = []

    for x, y in pts:

        if abs(x - cx) <= radius and abs(y - cy) <= radius:
            close.append((x, y))

    if len(close) < 2:
        return 0, 0

    # Use first and last nearby point
    x0, y0 = close[0]
    x1, y1 = close[-1]

    return x1 - x0, y1 - y0

def is_vertical_edge_block(name):
    return name in ("full", "shelf", "trapdoor_open")

def is_horizontal_edge_block(name):
    return name in ("full", "slab", "trapdoor_closed")

def smooth_stair_edges(blocks, bmp, pts, max_extra_error=6):
    """
    Replace repeated stairs with continuation blocks
    to keep edges smooth (vertical + horizontal).
    """

    h = len(blocks)
    w = len(blocks[0])

    TILE = 6


    # ----------------------------------
    # Extract bitmap tile
    # ----------------------------------

    def get_tile(y, x):

        by = y * TILE
        bx = x * TILE

        return [
            [
                bmp[by + dy][bx + dx]
                if by + dy < len(bmp) and bx + dx < len(bmp[0])
                else 0
                for dx in range(TILE)
            ]
            for dy in range(TILE)
        ]


    # ----------------------------------
    # Error metric
    # ----------------------------------

    def tile_error(tile, pat):

        err = 0

        for yy in range(6):
            for xx in range(6):
                if tile[yy][xx] != pat[yy][xx]:
                    err += 1

        return err


    # ----------------------------------
    # Try replacement helper
    # ----------------------------------

    def try_replace(y, x, allowed_blocks):

        name_mid, pat_mid = blocks[y][x]

        tile = get_tile(y, x)

        old_err = tile_error(tile, pat_mid)

        best = None
        best_err = 10**9

        for name, pat in BLOCK_LIBRARY:

            if name not in allowed_blocks:
                continue

            err = tile_error(tile, pat)

            if err < best_err:
                best = (name, pat)
                best_err = err

        if best and best_err <= old_err + max_extra_error:
            blocks[y][x] = best


    # ==================================================
    # 1) Vertical smoothing
    # ==================================================

    for y in range(1, h - 1):
        for x in range(w):

            name_above, _ = blocks[y - 1][x]
            name_mid, _ = blocks[y][x]
            name_below, _ = blocks[y + 1][x]

            # A, stair, stair
            if (
                is_vertical_edge_block(name_above)
                and name_mid == "stair"
                and name_below == "stair"
            ):

                try_replace(
                    y, x,
                    ("full", "shelf", "trapdoor_open")
                )


    # ==================================================
    # 2) Horizontal smoothing
    # ==================================================

    for y in range(h):
        for x in range(1, w - 1):

            name_left, _ = blocks[y][x - 1]
            name_mid, _ = blocks[y][x]
            name_right, _ = blocks[y][x + 1]

            # A, stair, stair
            if (
                is_horizontal_edge_block(name_left)
                and name_mid == "stair"
                and name_right == "stair"
            ):

                try_replace(
                    y, x,
                    ("full", "slab", "trapdoor_closed")
                )


def get_block_orientation(name, pat):
    """
    Returns orientation string depending on block type.
    """

    H = len(pat)
    W = len(pat[0])

    top = sum(sum(r) for r in pat[H//2:]) > sum(sum(r) for r in pat[:H//2])
    left = sum(row[:W//2].count(1) for row in pat) > sum(row[W//2:].count(1) for row in pat)

    # Slabs / closed trapdoors
    if name in ("slab", "trapdoor_closed"):
        return "top" if top else "bottom"

    # Shelf / open trapdoors
    if name in ("shelf", "trapdoor_open"):
        return "left" if left else "right"

    # Stair (quadrants)
    if name == "stair":
        if top and left:
            return "tl"
        elif top and not left:
            return "tr"
        elif not top and left:
            return "bl"
        else:
            return "br"

    return "none"

# ======================================================
# ANSI Color Themes
# ======================================================

ANSI_RESET = "\033[0m"

ANSI_THEMES = {

    # For dark terminals (default)
    "dark": {
        "full": "\033[97m",            # bright white
        "slab": "\033[34m",            # blue
        "shelf": "\033[92m",           # bright green
        "stair": "\033[93m",           # yellow
        "trapdoor_open": "\033[31m",   # red
        "trapdoor_closed": "\033[31m", # red
        "empty": "",
    },

    # For light terminals
    "light": {
        "full": "\033[30m",            # black
        "slab": "\033[34m",            # blue
        "shelf": "\033[32m",           # dark green
        "stair": "\033[33m",           # brown/yellow
        "trapdoor_open": "\033[31m",   # red
        "trapdoor_closed": "\033[31m", # red
        "empty": "",
    }
}

UNICODE_MAP = {
    ("full", "none"): "█",

    ("slab", "bottom"): "▄",
    ("slab", "top"): "▀",

    ("trapdoor_closed", "bottom"): "▁",
    ("trapdoor_closed", "top"): "▔",

    ("shelf", "left"): "▍",
    ("shelf", "right"): "▐",

    ("trapdoor_open", "left"): "▏",
    ("trapdoor_open", "right"): "▕",

    ("stair", "bl"): "▙",
    ("stair", "br"): "▟",
    ("stair", "tl"): "▛",
    ("stair", "tr"): "▜",

    ("empty", "none"): " ",
}

ASCII_MAP = {
    ("full", "none"): "F",

    ("slab", "bottom"): "S",
    ("slab", "top"): "S",

    ("trapdoor_closed", "bottom"): "C",
    ("trapdoor_closed", "top"): "C",

    ("shelf", "left"): "H",
    ("shelf", "right"): "H",

    ("trapdoor_open", "left"): "O",
    ("trapdoor_open", "right"): "O",

    ("stair", "bl"): "T",
    ("stair", "br"): "T",
    ("stair", "tl"): "T",
    ("stair", "tr"): "T",

    ("empty", "none"): " ",
}

def block_to_char(name, pat, char_map):
    """
    Return RAW character only (no color).
    """

    orient = get_block_orientation(name, pat)

    return char_map.get((name, orient), " ")

def colorize(ch, block_name, use_color, theme_colors):

    if not use_color or ch == " ":
        return ch

    color = theme_colors.get(block_name, "")
    reset = ANSI_RESET if color else ""

    return f"{color}{ch}{reset}"

def print_blocks(blocks, offset, char_map, theme_colors,
                 use_color=False, grid=5):

    print("\nMinecraft Blocks:\n")

    # Flip vertically (Y up)
    rows = list(reversed(blocks))

    height = len(rows)
    width = len(rows[0]) if rows else 0

    max_y = height - 1

    visual_width = width * 2 - 1 if width > 0 else 0

    rendered = []

    # --------------------------------------------------
    # Helper: composite layers
    # --------------------------------------------------

    def composite(grid, block, knockout):

        out = []

        for g, b, k in zip(grid, block, knockout):

            if b != " ":
                out.append(b)

            elif k:
                out.append(" ")

            else:
                out.append(g)

        return "".join(out)


    # --------------------------------------------------
    # Build rows
    # --------------------------------------------------

    for row_i, row in enumerate(rows):

        # World Y coordinate
        y = max_y - row_i + offset[1] // 6

        # ==================================================
        # 1) BLOCK LAYER
        # ==================================================

        block_chars = []

        block_names = []

        for (name, pat) in row:

            ch = block_to_char(name, pat, char_map)

            block_chars.append(ch)
            block_names.append(name)

        block_line = " ".join(block_chars)

        # Pad
        block = list(block_line.ljust(visual_width))


        # ==================================================
        # 2) GRID LAYER
        # ==================================================

        grid_line = [" "] * visual_width

        if grid > 0:

            # Vertical lines
            for x in range(0, width, grid):

                pos = x * 2

                if pos < visual_width:
                    grid_line[pos] = "│"

            # Horizontal lines
            if y % grid == 0:

                for i in range(visual_width):
                    grid_line[i] = "═"

                # End caps
                if visual_width > 0:
                    grid_line[0] = "╪"
                    grid_line[-1] = "╪"


            # Intersections
            for x in range(0, width, grid):

                pos = x * 2

                if pos < visual_width and y % grid == 0:
                    grid_line[pos] = "╪"


        grid_chars = grid_line


        # ==================================================
        # 3) KNOCKOUT MASK
        # ==================================================

        knockout = [False] * visual_width

        # Find block positions
        block_pos = set(
            i for i, ch in enumerate(block)
            if ch != " "
        )

        # Knock out grid within distance 1
        for i in range(visual_width):

            for j in (i - 1, i, i + 1):

                if j in block_pos:
                    knockout[i] = True
                    break


        # ==================================================
        # 4) COMPOSITE
        # ==================================================

        raw_line = composite(grid_chars, block, knockout)

        # Apply color AFTER layout
        colored = []

        bi = 0  # block index

        for i, ch in enumerate(raw_line):

            # Block positions are at even indices: 0,2,4,...
            if i % 2 == 0 and bi < len(block_names):

                name = block_names[bi]
                colored.append(colorize(ch, name, use_color, theme_colors))

                bi += 1

            else:
                colored.append(ch)

        final_line = "".join(colored)

        # ==================================================
        # 5) Y LABEL
        # ==================================================

        if final_line.strip() == "":
            rendered.append(f"{y:>4}")
        else:
            rendered.append(f"{y:>4}   {final_line}")


    # --------------------------------------------------
    # Print rows
    # --------------------------------------------------

    for r in rendered:
        print(r)


    # --------------------------------------------------
    # X AXIS
    # --------------------------------------------------

    marker = ["═"] * visual_width
    label = [" "] * visual_width

    if grid > 0:

        for x in range(0, width, grid):

            pos = x * 2

            if pos < visual_width:

                marker[pos] = "╪"

                world_x = x + offset[0] // 6

                s = str(world_x)

                for i, c in enumerate(s):

                    if pos + i < visual_width:
                        label[pos + i] = c

    # End caps
    if visual_width > 0:
        marker[0] = "╪"
        marker[-1] = "╪"


    pad = " " * 7

    print(pad + "".join(marker))
    print(pad + "".join(label))

    print_legend(char_map, theme_colors, use_color)


def print_legend(char_map, theme_colors, use_color):
    """
    Print legend based on active character map and colors.
    """

    # Order we want to show blocks
    order = [
        "full",
        "slab",
        "shelf",
        "stair",
        "trapdoor_open",
        "trapdoor_closed",
    ]

    # Friendly names
    names = {
        "full": "Full block",
        "slab": "Slab",
        "shelf": "Shelf",
        "stair": "Stair",
        "trapdoor_open": "Open trapdoor",
        "trapdoor_closed": "Closed trapdoor",
    }

    print("\nLegend:")

    for block in order:

        chars = []

        # Collect all variants for this block
        for (name, orient), ch in char_map.items():

            if name != block:
                continue

            if ch == " ":
                continue

            # Colorize if needed
            if use_color:

                color = theme_colors.get(name, "")
                reset = ANSI_RESET if color else ""

                ch = f"{color}{ch}{reset}"

            chars.append(ch)

        if not chars:
            continue

        sym = " ".join(sorted(set(chars)))

        print(f"{sym:8} {names[block]}")

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
    p.add_argument("--grid", type=int, default=5)
    p.add_argument(
        "--style",
        choices=["unicode", "ascii"],
        default="unicode",
        help="Output style (unicode or ascii)"
    )

    p.add_argument(
        "--color",
        action="store_true",
        help="Enable ANSI color output"
    )

    p.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="Color theme (dark or light background)"
    )
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

    pixels, pts = rasterize(
        p0, p1, p2, p3,
        args.samples,
        args.width
    )

    bmp, offx, offy = pixels_to_bitmap(pixels)

    blocks = bitmap_to_blocks(bmp, pts)

    # Smooth ugly stair edges
    smooth_stair_edges(blocks, bmp, pts)

    # Store offsets for printing
    offset = (offx, offy)

    # Select style
    if args.style == "unicode":
        char_map = UNICODE_MAP
    else:
        char_map = ASCII_MAP

    # Select color theme
    theme_colors = ANSI_THEMES[args.theme]

    print_blocks(
        blocks,
        offset,
        char_map,
        theme_colors,
        use_color=args.color,
        grid=args.grid
    )


if __name__ == "__main__":
    main()
