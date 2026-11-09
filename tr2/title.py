"""The end card: "Tiny Rocket" brushed in ink, the red hanko seal, "Fable 5.2".

No fonts. Every letter is a few brush strokes: a centre line (a spline through
hand placed points) and a pressure profile. The brush is modelled per pixel:
each pixel near a stroke gets its position along the stroke (s) and across it
(b), then the ink is the brush footprint with a rough bleeding edge, bristle
streaks that run dry toward the end of the stroke, and pooling at the edges.
Each pixel also stores WHEN the brush reaches it, so the title writes on.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import core

# Glyphs: name -> (advance width, [stroke, ...]); stroke = (points, pressure profile)
# Units: 1.0 = cap height. y = 0 is the cap line, y = 1 the baseline.
P_DOWN = [0.95, 1.0, 0.8, 0.7, 0.75, 0.5]       # loaded start, lift at the end
P_BAR = [0.9, 0.75, 0.7, 0.85, 0.55]
P_CURVE = [0.7, 0.95, 0.85, 0.7, 0.85, 0.55]
P_FLICK = [0.9, 1.0, 0.8, 0.55, 0.3, 0.12]
P_DAB = [0.9, 1.0, 0.6]

GLYPHS = {
    "T": (0.92, [([(0.00, 0.13), (0.26, 0.06), (0.58, 0.08), (0.90, 0.01)], [0.8, 1.0, 0.7, 0.75, 0.9, 0.6]),
                 ([(0.46, 0.07), (0.45, 0.42), (0.43, 0.82), (0.35, 1.01)], P_FLICK)]),
    "i": (0.34, [([(0.13, 0.31), (0.19, 0.39)], P_DAB),
                 ([(0.13, 0.52), (0.14, 0.76), (0.17, 0.97), (0.27, 0.93)], P_DOWN)]),
    "n": (0.64, [([(0.09, 0.50), (0.11, 0.74), (0.10, 1.00)], P_DOWN),
                 ([(0.11, 0.68), (0.23, 0.52), (0.37, 0.50), (0.46, 0.62), (0.47, 0.84), (0.52, 1.00)], P_CURVE)]),
    "y": (0.60, [([(0.05, 0.50), (0.13, 0.72), (0.27, 0.92)], P_DOWN),
                 ([(0.52, 0.48), (0.40, 0.80), (0.25, 1.14), (0.09, 1.36), (-0.08, 1.42)], P_FLICK)]),
    " ": (0.34, []),
    "R": (0.80, [([(0.13, 0.07), (0.14, 0.50), (0.11, 1.00)], P_DOWN),
                 ([(0.04, 0.12), (0.30, 0.03), (0.54, 0.11), (0.57, 0.31), (0.38, 0.47), (0.19, 0.52),
                   (0.36, 0.62), (0.55, 0.86), (0.76, 1.03)], [0.7, 0.9, 1.0, 0.85, 0.6, 0.55, 0.8, 1.0, 0.75, 0.3])]),
    "o": (0.58, [([(0.33, 0.51), (0.16, 0.54), (0.07, 0.73), (0.15, 0.95), (0.35, 0.98), (0.48, 0.77),
                   (0.41, 0.56), (0.27, 0.49)], P_CURVE)]),
    "c": (0.52, [([(0.45, 0.60), (0.31, 0.50), (0.15, 0.57), (0.08, 0.76), (0.18, 0.95), (0.36, 0.98),
                   (0.47, 0.89)], P_CURVE)]),
    "k": (0.58, [([(0.10, 0.03), (0.12, 0.50), (0.10, 1.00)], P_DOWN),
                 ([(0.45, 0.50), (0.29, 0.67), (0.13, 0.77)], [0.5, 0.8, 0.9]),
                 ([(0.20, 0.72), (0.35, 0.86), (0.53, 1.01)], [0.6, 0.95, 0.5])]),
    "e": (0.56, [([(0.10, 0.77), (0.30, 0.75), (0.45, 0.66), (0.39, 0.53), (0.25, 0.50), (0.11, 0.63),
                   (0.09, 0.84), (0.22, 0.97), (0.40, 0.97), (0.50, 0.89)], P_CURVE)]),
    "t": (0.50, [([(0.21, 0.20), (0.20, 0.60), (0.23, 0.90), (0.33, 0.99), (0.44, 0.93)], P_DOWN),
                 ([(0.01, 0.50), (0.22, 0.47), (0.46, 0.43)], P_BAR)]),
    "F": (0.66, [([(0.13, 0.07), (0.14, 0.50), (0.11, 1.00)], P_DOWN),
                 ([(0.06, 0.09), (0.34, 0.05), (0.62, 0.03)], P_BAR),
                 ([(0.14, 0.50), (0.30, 0.47), (0.47, 0.47)], P_BAR)]),
    "a": (0.62, [([(0.43, 0.58), (0.27, 0.50), (0.11, 0.61), (0.08, 0.84), (0.20, 0.98), (0.36, 0.90),
                   (0.45, 0.64)], P_CURVE),
                 ([(0.45, 0.50), (0.45, 0.80), (0.49, 0.97), (0.58, 0.94)], P_DOWN)]),
    "b": (0.58, [([(0.10, 0.03), (0.12, 0.50), (0.10, 0.98)], P_DOWN),
                 ([(0.11, 0.68), (0.25, 0.52), (0.42, 0.56), (0.48, 0.77), (0.36, 0.96), (0.20, 0.98),
                   (0.11, 0.90)], P_CURVE)]),
    "l": (0.30, [([(0.12, 0.03), (0.12, 0.50), (0.14, 0.91), (0.25, 0.98)], P_DOWN)]),
    "5": (0.62, [([(0.20, 0.08), (0.17, 0.28), (0.14, 0.47)], P_DOWN),
                 ([(0.13, 0.47), (0.31, 0.41), (0.48, 0.52), (0.51, 0.75), (0.35, 0.96), (0.15, 0.98),
                   (0.04, 0.87)], P_CURVE),
                 ([(0.19, 0.09), (0.36, 0.08), (0.54, 0.05)], P_BAR)]),
    ".": (0.26, [([(0.10, 0.90), (0.15, 0.97)], P_DAB)]),
    "2": (0.64, [([(0.08, 0.28), (0.19, 0.09), (0.39, 0.06), (0.50, 0.23), (0.39, 0.52), (0.17, 0.82),
                   (0.06, 0.99)], P_CURVE),
                 ([(0.06, 0.96), (0.30, 0.92), (0.58, 0.97)], P_BAR)]),
}


def layout(text, x, y, size, tracking=0.06, slant=0.10, seed=0, bounce=0.03):
    """Strokes for a line of text. (x, y) = left end of the baseline, size = cap
    height in px. Returns [(points Nx2 in px, pressure, letter index), ...]."""
    r = core.rng("title-layout", text, seed)
    out = []
    pen = x
    for k, ch in enumerate(text):
        adv, strokes = GLYPHS[ch]
        dy = (r.random() - 0.5) * 2 * bounce * size
        rot = (r.random() - 0.5) * 0.05
        for pts, pres in strokes:
            p = np.asarray(pts, np.float64).copy()
            p[:, 0] += (1.0 - p[:, 1]) * slant            # italic lean
            c, s = np.cos(rot), np.sin(rot)
            q = np.c_[p[:, 0] * c - (p[:, 1] - 1) * s, p[:, 0] * s + (p[:, 1] - 1) * c + 1]
            q = q * size + (pen, y - size + dy)
            out.append((q, pres, k))
        pen += (adv + tracking) * size
    return out, pen


def brush_stroke(pts, pressure, width, seed, dry=0.55, speed=900.0):
    """Render one stroke. Returns (x0, y0, alpha, tmap, duration): alpha is the
    ink coverage in a small box at (x0, y0); tmap is the time (seconds from the
    stroke start) at which the brush reaches each pixel."""
    path = core.catmull(pts, 24, closed=False) if len(pts) > 2 else np.asarray(pts, np.float64)
    seg = np.hypot(*np.diff(path, axis=0).T).sum()
    n = max(16, int(seg / 0.5))
    path = core.resample(path, n)
    s = np.linspace(0, 1, n)
    pr = np.interp(s, np.linspace(0, 1, len(pressure)), pressure)
    w = np.maximum(pr ** 1.7 * width * 1.18, 0.6)           # full width per sample: strong thick and thin
    pad = int(width) + 6
    x0 = int(np.floor(path[:, 0].min())) - pad
    y0 = int(np.floor(path[:, 1].min())) - pad
    x1 = int(np.ceil(path[:, 0].max())) + pad
    y1 = int(np.ceil(path[:, 1].max())) + pad
    yy, xx = np.mgrid[y0:y1, x0:x1]
    q = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]
    dist, idx = cKDTree(path).query(q)
    tan = np.gradient(path, axis=0)
    tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
    d = q - path[idx]
    side = d[:, 0] * -tan[idx, 1] + d[:, 1] * tan[idx, 0]
    half = w[idx] * 0.5
    ss = s[idx]
    b = side / half                                         # -1..1 across the brush
    # round caps: past the ends the footprint is a half disc that shrinks
    along = d[:, 0] * tan[idx, 0] + d[:, 1] * tan[idx, 1]
    r_eff = np.hypot(side, along * 1.25)
    # rough bleeding edge
    edge_n = core.fbm_at(q[:, 0], q[:, 1], 16.0, 2, seed + 3) - 0.5
    tooth = core.fbm_at(q[:, 0], q[:, 1], 2.2, 2, seed + 5) - 0.5
    edge = (half * (1.0 + 0.12 * edge_n) + 0.5 + 1.6 * tooth - r_eff)
    cover = np.clip(edge / 1.2, 0, 1)
    # bristle streaks: long along the stroke, fine across, running dry toward the end
    L = ss * seg
    streak = core.fbm_at(L / 70.0, b * 5.5 + 13.0, 1.0, 3, seed + 9)
    fine = core.fbm_at(L / 18.0, b * 14.0, 1.0, 2, seed + 21)
    dryness = dry * (0.10 + 0.90 * ss ** 1.5) * (1.25 - 0.6 * pr[idx]) + 0.16 * (np.abs(b) ** 3)
    wet = np.clip((0.60 * streak + 0.40 * fine - dryness * 0.9 + 0.16) / 0.07, 0, 1)
    body = 0.90 + 0.10 * np.clip(np.abs(b), 0, 1) ** 2       # pooling at the edges
    alpha = (cover * wet * body).reshape(yy.shape).astype(np.float32)
    dur = seg / speed + 0.06
    tmap = (ss * dur).reshape(yy.shape).astype(np.float32)
    return x0, y0, alpha, tmap, dur


class Lettering:
    """A set of strokes with start times. alpha(t) gives the ink coverage layer."""

    def __init__(self):
        self.items = []     # (x0, y0, alpha, tmap, t_start)
        self.t_end = 0.0

    def add_text(self, text, x, y, size, width, t0, t1, seed=0, dry=0.55, tracking=0.06):
        """Write `text` between times t0 and t1 (seconds)."""
        strokes, pen = layout(text, x, y, size, tracking=tracking, seed=seed)
        rendered = [brush_stroke(p, pr, width, core.seed_of("stroke", text, i, seed), dry=dry)
                    for i, (p, pr, k) in enumerate(strokes)]
        total = sum(r[4] for r in rendered) + 0.05 * len(rendered)
        scale = (t1 - t0) / total
        t = t0
        for x0, y0, a, tm, dur in rendered:
            self.items.append((x0, y0, a, tm * scale, t))
            t += (dur + 0.05) * scale
        self.t_end = max(self.t_end, t1)
        return pen

    def alpha(self, t, shape=(core.H, core.W)):
        out = np.zeros(shape, np.float32)
        for x0, y0, a, tm, ts in self.items:
            if t <= ts:
                continue
            h, w = a.shape
            X0, Y0 = max(x0, 0), max(y0, 0)
            X1, Y1 = min(x0 + w, shape[1]), min(y0 + h, shape[0])
            if X1 <= X0 or Y1 <= Y0:
                continue
            sa = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            st = tm[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            vis = sa * np.clip((t - ts - st) / 0.03, 0, 1)
            o = out[Y0:Y1, X0:X1]
            np.maximum(o, 1 - (1 - o) * (1 - vis), out=o)
        return out


# ------------------------------------------------------------------- hanko
# The character 星 (star) as carved seal strokes in a unit box (x, y in 0..1).
HOSHI = [
    # 日 on top
    [(0.26, 0.10), (0.26, 0.42)],
    [(0.26, 0.10), (0.74, 0.10), (0.74, 0.42)],
    [(0.26, 0.26), (0.74, 0.26)],
    [(0.26, 0.42), (0.74, 0.42)],
    # 生 below
    [(0.34, 0.50), (0.24, 0.64)],
    [(0.27, 0.58), (0.76, 0.58)],
    [(0.50, 0.46), (0.50, 0.90)],
    [(0.30, 0.73), (0.70, 0.73)],
    [(0.16, 0.90), (0.84, 0.90)],
]


def hanko(size=120, seed=5):
    """The red seal as a premultiplied RGBA layer (size x size, plus a margin).
    White carved strokes show the sky through, like a real stamp on paper."""
    m = int(size * 0.12)
    n = size + 2 * m
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    u = (xx - m) / size
    v = (yy - m) / size
    # rounded square with a hand cut wobble
    wob = (core.fbm(n, n, size * 0.35, 3, seed) - 0.5) * 0.06
    rr = 0.16
    dx = np.maximum(np.abs(u - 0.5) - (0.5 - rr), 0)
    dy = np.maximum(np.abs(v - 0.5) - (0.5 - rr), 0)
    d = np.hypot(dx, dy) - rr + wob
    body = np.clip(-d * size / 1.5, 0, 1)
    # carved strokes
    carve = np.zeros((n, n))
    q = np.c_[u.ravel(), v.ravel()]
    for k, st in enumerate(HOSHI):
        p = core.resample(np.asarray(st, np.float64), 60)
        dist, _ = cKDTree(p).query(q)
        wv = 0.040 + 0.012 * (core.fbm(n, n, size * 0.2, 2, seed + 11 + k) - 0.5)
        carve = np.maximum(carve, np.clip((wv.ravel() - dist).reshape(n, n) * size / 1.3, 0, 1))
    # uneven inking: missing specks, lighter patches, a darker rim where ink pools
    speck = core.fbm(n, n, 3.2, 2, seed + 40)
    patch = core.fbm(n, n, size * 0.5, 3, seed + 41)
    ink = np.clip((speck - 0.27 - 0.20 * (patch - 0.5)) / 0.10, 0, 1)
    a = body * (1 - carve) * (0.80 + 0.20 * ink) * np.clip(ink + 0.55, 0, 1)
    rim = np.clip(1 + d * size / 5.0, 0, 1)
    col = np.array([0.80, 0.13, 0.10])[None, None, :] * (1.0 - 0.18 * rim[..., None]) \
        + (patch[..., None] - 0.5) * 0.10
    out = np.zeros((n, n, 4), np.float32)
    out[..., :3] = np.clip(col, 0, 1) * a[..., None]
    out[..., 3] = a
    return out


# ------------------------------------------------------------- the end card
INK = np.array([0.085, 0.075, 0.12], np.float32)

_cache = {}


def _build(t_title, t_credit, x, y, size):
    key = (t_title, t_credit, x, y, size)
    if key not in _cache:
        L = Lettering()
        pen = L.add_text("Tiny Rocket", x, y, size, size * 0.165, t_title[0], t_title[1], seed=2, dry=0.62)
        cs = size * 0.30
        cx = x + size * 0.10
        cy = y + size * 0.80
        L.add_text("Fable 5.2", cx + size * 0.78, cy, cs, cs * 0.17, t_credit[0], t_credit[1], seed=4,
                   dry=0.45, tracking=0.09)
        hk = hanko(int(size * 0.50))
        _cache[key] = (L, hk, (cx, cy - cs * 1.42))
    return _cache[key]


def draw(pic, t, t_title=(1.5, 3.1), t_hanko=3.4, t_credit=(3.8, 4.5), x=150.0, y=330.0, size=210.0):
    """Composite the end card onto `pic` in place. t = seconds into the shot."""
    L, hk, (hx, hy) = _build(t_title, t_credit, x, y, size)
    a = L.alpha(t)
    if a.max() > 0:
        # ink soaks in: slightly transparent, multiplies the paper under it
        k = (a * 0.94)[..., None]
        pic *= (1 - k)
        pic += INK * k
    if t >= t_hanko:
        u = core.clamp((t - t_hanko) / 0.14)
        s = 1.0 + 0.22 * (1 - core.ease_out(u, 3.0))      # presses down onto the paper
        op = core.clamp((t - t_hanko) / 0.07)
        n = hk.shape[0]
        lay = core.warp(hk, s, hx - (s - 1) * n / 2, hy - (s - 1) * n / 2, pic.shape[:2])
        core.over(pic, lay, 0, 0, op)
    return pic
