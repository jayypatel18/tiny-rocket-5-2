"""The panda's hand drawn star map: an old sheet of paper pinned to a stick,
inked by paw. Used small in the wide shots and filling the frame in shot 9c,
where the paw paints one new star at the end of the dotted flight path.

    lay = starmap.sheet()                  # premultiplied RGBA, 1500 x 1050, upright, cached
    lay = starmap.fluttered(t, wind=1.0)   # the same sheet moving in the breeze (pinned along the top)
    starmap.NEW_STAR                       # (x, y) on the sheet where the new star is painted
    a, col = starmap.new_star(t)           # ink coverage (sheet sized) of the new star, t seconds after the brush lands
"""
import math
import os

import numpy as np
from scipy import ndimage

from . import core, title

SW, SH = 1500, 1050
NEW_STAR = (1118.0, 300.0)
INK = np.array([0.16, 0.13, 0.22], np.float32)
INK_WARM = np.array([0.45, 0.20, 0.13], np.float32)
GOLD = np.array([0.93, 0.60, 0.13], np.float32)
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "cache", "starmap.npy")
_mem = {}


def _paper():
    """Aged paper: mottled cream, fibres, stains, fold creases, a deckled edge."""
    yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
    base = np.array([0.95, 0.89, 0.74], np.float32)
    mott = core.fbm(SH, SW, 260, 5, seed=11)
    fibre = core.fbm(SH, SW, 5, 2, seed=12, aspect=9.0)
    tooth = core.fbm(SH, SW, 2.2, 2, seed=13)
    col = base[None, None, :] * (0.90 + 0.16 * mott[..., None]) + (fibre[..., None] - 0.5) * 0.05 + (tooth[..., None] - 0.5) * 0.035
    col[..., 2] -= (mott - 0.5) * 0.10                       # warmer where darker
    # browned edges
    ex = np.minimum(xx, SW - 1 - xx) / SW
    ey = np.minimum(yy, SH - 1 - yy) / SH
    edge = np.clip(1 - np.minimum(ex * 1.4, ey) / 0.10, 0, 1) ** 1.6
    edge *= 0.6 + 0.8 * core.fbm(SH, SW, 120, 3, seed=14)
    col *= 1 - 0.22 * edge[..., None] * np.array([0.7, 1.0, 1.5], np.float32)
    # tea stain ring and a few blotches
    for k, (cx, cy, r) in enumerate([(330, 760, 150), (1260, 880, 70), (980, 130, 46)]):
        d = np.hypot(xx - cx, yy - cy) / r + (core.fbm(SH, SW, 80, 3, seed=20 + k) - 0.5) * 0.25
        ring = np.exp(-((d - 1.0) / 0.07) ** 2) * 0.16 + np.clip(1 - d, 0, 1) * 0.05
        col *= 1 - ring[..., None] * np.array([0.6, 1.0, 1.6], np.float32)
    # fold creases: a light side and a dark side
    for x in (SW * 0.34, SW * 0.67):
        w = (xx - x + (core.fbm(SH, SW, 300, 2, seed=31) - 0.5) * 10)
        col += (np.exp(-(w / 2.2) ** 2) * 0.06 - np.exp(-((w - 4) / 3.0) ** 2) * 0.07)[..., None]
    w = (yy - SH * 0.5 + (core.fbm(SH, SW, 300, 2, seed=32) - 0.5) * 10)
    col += (np.exp(-(w / 2.2) ** 2) * 0.05 - np.exp(-((w - 4) / 3.0) ** 2) * 0.07)[..., None]
    # deckled, slightly torn outline
    n = core.fbm(SH, SW, 34, 4, seed=40)
    n2 = core.fbm(SH, SW, 300, 2, seed=41)
    inset = 14 + 26 * n + 22 * (n2 - 0.5)
    d = np.minimum(np.minimum(xx, SW - 1 - xx), np.minimum(yy, SH - 1 - yy)) - inset
    # one torn corner (lower right) and a nick on the left edge
    d = np.minimum(d, ((SW - xx) + (SH - yy)) * 0.7 - 95 - 40 * n)
    d = np.minimum(d, np.hypot(xx - 0, yy - 610) - 46 - 30 * n)
    alpha = np.clip(d / 1.6, 0, 1)
    return np.clip(col, 0, 1), alpha


def _star_pts(cx, cy, r, rot, wob, seed, points=5):
    """A star drawn in one line the way a child does it (a pentagram)."""
    r_ = core.rng("mapstar", seed)
    out = []
    for k in range(points + 1):
        a = rot + k * (2 * math.pi * 2 / points)
        rr = r * (1 + r_.normal(0, wob))
        out.append((cx + math.sin(a) * rr, cy - math.cos(a) * rr))
    dense = []
    for a, b in zip(out[:-1], out[1:]):
        for u in np.linspace(0, 1, 5, endpoint=False):
            dense.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
    dense.append(out[-1])
    return dense


def _ink_layer():
    """Everything the panda drew: constellations, a moon, the hill and rocket
    doodle, the dotted flight path, tick marks, a paw print signature."""
    c = core.Cel(SW, SH, ss=2)
    r = core.rng("starmap-ink")
    thin = INK * 1.0

    def line(pts, w=3.2, seed=0, col=thin, taper=(0.1, 0.15)):
        p = core.wobble(core.catmull(pts, 10, closed=False) if len(pts) > 2 else core.resample(pts, 12), 1.6, 300 + seed, 0.02)
        c.stroke(p, w, col, seed=seed, taper=taper)

    # faint chart grid in diluted ink, hand ruled (not straight)
    grid = INK * 0.55 + np.array([0.95, 0.89, 0.74], np.float32) * 0.45
    for k, x in enumerate(np.linspace(150, SW - 150, 6)):
        line([(x + r.normal(0, 5), 90), (x + r.normal(0, 7), SH * 0.5), (x + r.normal(0, 5), SH - 110)], 1.6, 50 + k, grid, (0.02, 0.02))
    for k, y in enumerate(np.linspace(140, SH - 150, 4)):
        line([(110, y + r.normal(0, 5)), (SW * 0.5, y + r.normal(0, 7)), (SW - 120, y + r.normal(0, 5))], 1.6, 60 + k, grid, (0.02, 0.02))
    # border tick marks
    for k in range(34):
        x = 130 + k * (SW - 260) / 33 + r.normal(0, 3)
        L = 20 if k % 4 == 0 else 10
        line([(x, 62), (x + r.normal(0, 1.5), 62 + L)], 2.2, 80 + k, thin, (0.05, 0.3))
    # constellations: each a set of stars joined by wobbly lines
    cons = [
        [(250, 250), (360, 190), (470, 240), (520, 350), (420, 410)],
        [(640, 170), (740, 250), (850, 210), (930, 300)],
        [(260, 520), (380, 560), (470, 500), (590, 590), (690, 520)],
        [(820, 470), (900, 560), (1010, 520), (1040, 640), (930, 700)],
        [(1230, 520), (1320, 600), (1270, 720), (1370, 780)],
        [(560, 760), (680, 800), (760, 730)],
    ]
    k = 0
    for ci, pts in enumerate(cons):
        pts = [(x + r.normal(0, 8), y + r.normal(0, 8)) for x, y in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            # stop the line short of each star, like joining dots
            ax, ay = a
            bx, by = b
            d = math.hypot(bx - ax, by - ay)
            ux, uy = (bx - ax) / d, (by - ay) / d
            line([(ax + ux * 22, ay + uy * 22), ((ax + bx) / 2 + r.normal(0, 4), (ay + by) / 2 + r.normal(0, 4)),
                  (bx - ux * 22, by - uy * 22)], 2.6, 100 + k)
            k += 1
        for (x, y) in pts:
            kind = r.integers(0, 4)
            if kind == 0:
                c.stroke(_star_pts(x, y, r.uniform(13, 19), r.uniform(0, 1), 0.10, k), 2.8, thin, seed=k, taper=(0.05, 0.1))
            elif kind == 1:
                s = r.uniform(11, 17)
                line([(x - s, y + r.normal(0, 1)), (x + s, y + r.normal(0, 1))], 3.0, 200 + k)
                line([(x + r.normal(0, 1), y - s), (x + r.normal(0, 1), y + s)], 3.0, 230 + k)
            else:
                c.fill(core.wobble(core.ellipse(x, y, r.uniform(5, 9), r.uniform(5, 9), 16), 1.0, k), thin)
            k += 1
    # scattered small stars
    for j in range(46):
        x, y = r.uniform(130, SW - 130), r.uniform(110, SH - 330)
        c.fill(core.wobble(core.ellipse(x, y, r.uniform(1.8, 4.2), r.uniform(1.8, 4.2), 10), 0.6, 500 + j), thin)
    # crescent moon, upper left
    mo = core.ellipse(200, 150, 46, 46, 40)
    m = c.mask(core.wobble(mo, 1.5, 7))
    c.crescent(core.wobble(mo, 1.5, 7), thin, (20, -8), clip=m)
    c.stroke(core.wobble(mo, 1.5, 7), 2.6, thin, closed=True, seed=7)
    # the hill, the tree and the tiny rocket (lower left), in warm ink
    warm = INK_WARM
    line([(90, 905), (210, 850), (340, 835), (470, 870), (600, 930)], 4.2, 700, warm)
    line([(170, 866), (176, 800), (200, 760)], 3.6, 701, warm)
    c.stroke(core.wobble(core.catmull([(150, 760), (170, 715), (225, 705), (262, 742), (240, 780), (190, 790)], 8), 2.0, 702), 3.0, warm, closed=True, seed=702)
    rk = [(352, 835), (352, 772), (366, 742), (380, 772), (380, 835)]
    line(rk, 3.4, 703, warm)
    line([(352, 835), (338, 852)], 3.0, 704, warm)
    line([(380, 835), (394, 852)], 3.0, 705, warm)
    line([(352, 800), (380, 800)], 2.4, 706, warm)
    # the dotted flight path: up from the rocket in a big arc to where the new star will be
    p0 = np.array([366.0, 730.0])
    p3 = np.array(NEW_STAR) + (-34, 30)
    p1 = np.array([420.0, 420.0])
    p2 = np.array([820.0, 300.0])
    arc = []
    for u in np.linspace(0, 1, 60):
        arc.append((1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3)
    arc = core.resample(np.array(arc), 150)
    for j in range(0, 144, 6):
        seg = arc[j:j + 4] + r.normal(0, 1.2, (4, 2))
        c.stroke(seg, 3.4, warm, seed=800 + j, taper=(0.2, 0.3))
    # a small arrow head near the end
    e = arc[-1]
    line([(e[0] - 26, e[1] + 2), (e[0] + 2, e[1] - 2), (e[0] - 10, e[1] + 26)], 3.0, 860, warm)
    # paw print signature, lower right
    px, py = 1290, 930
    c.fill(core.wobble(core.catmull([(px - 26, py + 8), (px, py - 14), (px + 26, py + 8), (px + 16, py + 34), (px - 16, py + 34)], 8), 1.5, 900), warm)
    for j, (dx, dy) in enumerate([(-34, -22), (-12, -40), (13, -40), (35, -22)]):
        c.fill(core.wobble(core.ellipse(px + dx, py + dy, 9, 11.5, 14, rot=dx * 0.012), 0.8, 910 + j), warm)
    return c.rgba()


def sheet():
    """The finished map (without the new star): premultiplied RGBA, SH x SW."""
    if "sheet" in _mem:
        return _mem["sheet"]
    if os.path.exists(_CACHE):
        _mem["sheet"] = np.load(_CACHE)
        return _mem["sheet"]
    col, alpha = _paper()
    ink = _ink_layer()
    # ink soaks into paper: it multiplies, with a little bleed and uneven take up
    a = ink[..., 3]
    take = 0.80 + 0.20 * core.fbm(SH, SW, 6, 2, seed=77)
    a = np.clip(0.80 * core.gblur(a, 0.8) + 0.25 * core.gblur(a, 2.5), 0, 1) * take
    rgb = ink[..., :3] / np.maximum(ink[..., 3:4], 1e-4)
    col = col * (1 - a[..., None]) + col * rgb * 1.1 * a[..., None]
    out = core.premul(np.clip(col, 0, 1), alpha)
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    np.save(_CACHE, out)
    _mem["sheet"] = out
    return out


def new_star(t, size=30.0):
    """Coverage of the freshly painted star, t seconds after the brush touches
    the paper (it takes about 0.55 s to draw). Returns (alpha SH x SW, colour)."""
    if "ns" not in _mem:
        pts = _star_pts(NEW_STAR[0], NEW_STAR[1], size, 0.05, 0.06, 4242)
        _mem["ns"] = title.brush_stroke(pts, [0.9, 1.0, 0.85, 0.8, 0.9, 0.7], size * 0.34, 4243, dry=0.35, speed=560.0)
    x0, y0, a, tm, dur = _mem["ns"]
    out = np.zeros((SH, SW), np.float32)
    if t > 0:
        out[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a * np.clip((t - tm * (0.55 / dur)) / 0.03, 0, 1)
    return out, GOLD


def with_star(t):
    """The sheet with the new star at time t (wet paint is a touch brighter)."""
    s = sheet()
    if t <= 0:
        return s
    a, col = new_star(t)
    wet = 1.0 + 0.15 * math.exp(-max(t - 0.55, 0) / 0.8)
    out = s.copy()
    k = (a * 0.95 * s[..., 3])[..., None]
    out[..., :3] = out[..., :3] * (1 - k) + np.clip(col * wet, 0, 1) * k
    return out


def fluttered(lay, t, wind=1.0, seed=0, rot=0.0):
    """The sheet moving in the breeze. It is pinned along its top edge, so the
    lower part swings and ripples; a travelling wave crosses the paper.
    rot (radians) tilts the whole sheet about the middle of its top edge."""
    yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
    if rot != 0.0:
        c, s = math.cos(-rot), math.sin(-rot)
        px, py = SW / 2, 40.0
        xr = (xx - px) * c - (yy - py) * s + px
        yr = (xx - px) * s + (yy - py) * c + py
        xx, yy = xr.astype(np.float32), yr.astype(np.float32)
    v = np.clip(yy / SH, 0, 1.2)
    u = xx / SW
    g = wind * (0.6 + 0.4 * core.noise1(t * 0.7, 5 + seed))
    sway = 16.0 * g * (v ** 1.6) * math.sin(t * 2.3 + 0.4) + 9.0 * g * (v ** 2) * np.sin(t * 5.1 + u * 5.0 + v * 3.0)
    lift = 7.0 * g * (v ** 2) * np.sin(t * 4.3 + u * 7.0 + 1.0)
    mx = xx - sway
    my = yy - lift
    out = np.empty_like(lay)
    for ch in range(4):
        out[..., ch] = ndimage.map_coordinates(lay[..., ch], [my, mx], order=1, mode="constant", cval=0.0)
    # the ripple catches light: a soft shading band that follows the wave
    shade = 1.0 + 0.06 * g * np.cos(t * 5.1 + u * 5.0 + v * 3.0) * (v ** 1.2)
    out[..., :3] *= shade[..., None]
    return out
