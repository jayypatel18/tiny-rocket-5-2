"""Composes and paints the hilltop set layers (helper of hillset.py).

Every layer is composed as a reference (values and light first), run through the
stroke painter, then given crisp accents (blades, leaves, roofs, flowers).
All geometry is in SET coordinates; a layer has a resolution factor k
(layer px per set px) and an origin (x0, y0).
"""
import math

import numpy as np
from scipy import interpolate, ndimage

from tr2 import core, paint, palette
from tr2.paint import smoothstep

# name: (x0, y0, x1, y1, k, parallax)
SPEC = dict(
    far=(240, 520, 2960, 1000, 1.0, 0.07),
    near=(240, 640, 2960, 1060, 1.0, 0.14),
    valley=(240, 850, 2960, 1500, 1.5, 0.25),
    hill=(100, 960, 3100, 1620, 1.5, 1.0),
    tree=(430, 220, 1790, 1250, 1.5, 1.0),
)
HORIZON = 880.0
RIVER = [(1560, 886), (1850, 902), (2180, 930), (2300, 968), (2140, 1010), (1990, 1060),
         (2060, 1130), (2330, 1210), (2640, 1330), (2900, 1480), (3000, 1560)]
_river = interpolate.PchipInterpolator([p[1] for p in RIVER], [p[0] for p in RIVER])


# Per time of day: how much direct (shadow casting) light there is, and how much
# colour the eye still sees. Night and the blue hour are flatter and cooler.
LOOK = dict(dusk=(1.0, 1.0), night=(0.30, 0.42), predawn=(0.25, 0.46), launch=(0.36, 0.56),
            sunrise=(1.0, 1.0), morning=(1.0, 1.0))
# value gain on vegetation, and (tint, strength) of the rim light on crest and foliage tops
GAIN = dict(night=0.80, predawn=0.84, launch=0.88)
RIM = dict(dusk=((1.0, 0.98, 0.60), 1.0), night=((0.86, 0.94, 1.0), 0.50), predawn=((1.0, 1.05, 1.40), 0.50),
           launch=((1.0, 1.0, 1.15), 0.60), morning=((1.0, 0.98, 0.70), 1.0), sunrise=((1.0, 0.98, 0.62), 1.0))


def look(tod):
    return LOOK.get(tod, (1.0, 1.0)) if isinstance(tod, str) else (1.0, 1.0)


def gain(tod):
    return GAIN.get(tod, 1.0) if isinstance(tod, str) else 1.0


def rim_col(tod, T):
    """Colour of the thin rim light (crest grass, foliage tops, blade edges)."""
    tint, k = RIM.get(tod, ((1.0, 0.98, 0.7), 1.0)) if isinstance(tod, str) else ((1.0, 0.98, 0.7), 1.0)
    return np.clip(np.asarray(T["rim"], np.float32) * np.float32(tint) * k, 0, 1)


def desat(c, s):
    c = np.asarray(c, np.float32)
    g = c.mean(axis=-1, keepdims=True)
    return g + (c - g) * s


def ground_y(X):
    """Hill crest (the line characters stand on), set coords."""
    X = np.asarray(X, np.float64)
    dx = X - 1500.0
    y = 1050.0 + np.where(dx < 0, 1.555e-4, 2.49e-4) * dx * dx
    y = y + 3.0 * np.sin(X / 173.0 + 0.5) + 2.0 * np.sin(X / 67.0 + 2.1) - 1.1
    return y if y.ndim else float(y)


def grid(name):
    x0, y0, x1, y1, k, _ = SPEC[name]
    w = int(round((x1 - x0) * k))
    h = int(round((y1 - y0) * k))
    xs = x0 + (np.arange(w, dtype=np.float32) + 0.5) / k
    ys = y0 + (np.arange(h, dtype=np.float32) + 0.5) / k
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def mixc(a, b, t):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def lay(ref, alpha, col, a):
    """Composite a colour (3,) or H x W x 3 with coverage a onto ref and alpha."""
    ref *= (1 - a)[..., None]
    ref += np.asarray(col, np.float32) * a[..., None]
    if alpha is not None:
        alpha *= (1 - a)
        alpha += a


def hashf(a, b=0.0):
    v = np.abs(np.sin(np.asarray(a, np.float64) * 127.1 + np.asarray(b, np.float64) * 311.7 + 1.3)) * 43758.5453
    return (v - np.floor(v)).astype(np.float32)


# ------------------------------------------------------------------ mountains
def _ridge(xs, y0, amp, scale, seed):
    a = core.fbm_at(xs, np.full_like(xs, 3.3), scale, 5, seed)
    b = core.fbm_at(xs, np.full_like(xs, 9.1), scale * 0.45, 4, seed + 5)
    h = 0.62 * a + 0.38 * (1 - np.abs(2 * b - 1))
    return (y0 - amp * (h - 0.35) * 1.7), h


def paint_mountains(name, tod):
    T = palette.TOD[tod]
    k = SPEC[name][4]
    X, Y = grid(name)
    h, w = X.shape
    xs = X[0].astype(np.float64)
    key, fog = T["key"], T["fog"]
    sx, sy = T["sun"]
    ref = np.zeros((h, w, 3), np.float32)
    ref[:] = fog
    alpha = np.zeros((h, w), np.float32)
    if name == "far":
        ridges = [(748, 125, 900, 11, 0.86, True), (792, 88, 620, 12, 0.72, True), (828, 62, 420, 13, 0.58, False)]
        loc_l, loc_s = (0.60, 0.46, 0.50), (0.42, 0.42, 0.68)
    else:
        ridges = [(846, 64, 520, 21, 0.50, False), (872, 46, 330, 22, 0.36, False)]
        loc_l, loc_s = (0.50, 0.46, 0.34), (0.30, 0.36, 0.56)
    for (yb, amp, sc, seed, haze, snow) in ridges:
        ry, hh = _ridge(xs, yb, amp, sc, seed)
        if name == "near":
            ry = ry - 9 * core.fbm_at(xs, xs * 0 + 1.7, 15, 3, seed + 3) - 5 * core.fbm_at(xs, xs * 0 + 4.1, 5, 2, seed + 4)
        below = (Y - ry[None, :]).astype(np.float32)
        a = np.clip(below * k / 1.5 + 0.5, 0, 1)
        rel = core.fbm(h, w, (230 if name == "far" else 90) * k, 4, seed=seed + 40, aspect=1.3, gain=0.42)
        gx = np.gradient(rel, axis=1)
        gyy = np.gradient(rel, axis=0)
        l = smoothstep(-0.0008, 0.0016, -(gx * sx + gyy * sy))
        dr = np.gradient(ndimage.gaussian_filter1d(ry, 5 * k)) * k
        l2 = smoothstep(0.22, -0.22, dr * (1.0 if sx < 0 else -1.0)).astype(np.float32)
        wgt = np.exp(-np.maximum(below, 0) / (0.9 * amp)).astype(np.float32)
        # spurs running down from the ridge, away from the sun
        dirx = 1.0 if sx < 0 else -1.0
        rib = core.fbm_at((X + dirx * 1.25 * below).ravel(), (below * 0.22).ravel(), 95.0 if name == "far" else 60.0, 4, seed + 60).reshape(h, w)
        lf = smoothstep(0.44, 0.56, rib + 0.5 * (l - 0.5) * 0.4)
        l = (lf * (1 - 0.55 * wgt) + l2[None, :] * 0.55 * wgt) * look(tod)[0]
        hz = haze * T["fog_k"] / 0.75
        hz = np.clip(hz + (1 - hz) * 0.8 * smoothstep(0.3 * amp, 1.5 * amp + 40, below), 0, 0.97)
        hz = np.maximum(hz, 0.98 * smoothstep(790 if name == "far" else 822, 884, Y))
        col = mixc(palette.shade(loc_s, T), np.asarray(loc_l, np.float32) * key, l)
        if snow:
            sn = smoothstep(0.50, 0.70, hh).astype(np.float32)[None, :] * smoothstep(1.0, 0.25, below / (amp * 0.5) + 0.9 * (rel - 0.5))
            scol = mixc(palette.shade((0.86, 0.86, 1.0), T), np.clip(np.float32((1.0, 0.95, 0.92)) * key, 0, 1), l)
            col = mixc(col, scol, sn * 0.9)
        col = mixc(col, fog, hz)
        lay(ref, alpha, col, a)
    out = paint.paint(ref, alpha, seed=paint_seed(name, tod), passes=paint.LOOSE, scale=0.5 * k, spill=2.5, grain=0.035)
    return out


def paint_seed(name, tod):
    return core.seed_of("hillset", name) % 100000  # same brushwork in every light


# --------------------------------------------------------------------- valley
def river_x(Y):
    return _river(np.clip(Y, RIVER[0][1], RIVER[-1][1]))


def river_field(name="valley"):
    """(coverage 0..1, depth d 0..1) of the river on the valley layer grid."""
    x0, y0, x1, y1, k, _ = SPEC[name]
    X, Y = grid(name)
    h, w = X.shape
    Yd = np.linspace(RIVER[0][1], min(RIVER[-1][1], y1 - 1), 6000)
    Xd = river_x(Yd)
    px = np.clip(((Xd - x0) * k).astype(int), 0, w - 1)
    py = np.clip(((Yd - y0) * k).astype(int), 0, h - 1)
    centre = np.ones((h, w), bool)
    centre[py, px] = False
    dist, idx = ndimage.distance_transform_edt(centre, sampling=(3.4, 1.0), return_indices=True)
    yn = y0 + (idx[0] + 0.5) / k
    dn = np.clip((yn - HORIZON) / 620.0, 0, 1)
    hw = (5.0 + 120.0 * dn ** 1.15) * k
    hw = hw * (0.85 + 0.3 * core.fbm(h, w, 90 * k, 3, seed=77))
    cov = smoothstep(hw + 1.5, hw - 1.5, dist)
    cov *= smoothstep(RIVER[0][1] - 2, RIVER[0][1] + 10, Y)
    return cov.astype(np.float32), hw.astype(np.float32), dist.astype(np.float32)


def _blob(ref, alpha, cx, cy, rx, ry, col_l, col_s, sx, sy, seed, bump=0.25, a_max=1.0):
    """A rounded bumpy mass (tree clump, bush, stone) lit from the sun side."""
    h, w = ref.shape[:2]
    m = int(max(rx, ry) * 1.4) + 3
    i0, i1 = max(int(cx) - m, 0), min(int(cx) + m, w)
    j0, j1 = max(int(cy) - m, 0), min(int(cy) + m, h)
    if i1 <= i0 or j1 <= j0:
        return
    yy, xx = np.mgrid[j0:j1, i0:i1].astype(np.float32)
    u = (xx - cx) / rx
    v = (yy - cy) / ry
    ang = np.arctan2(v, u)
    r = np.sqrt(u * u + v * v)
    ph = hashf(seed, 3.0) * 6.28
    edge = 1.0 + bump * (0.6 * np.sin(ang * 3 + ph) + 0.4 * np.sin(ang * 7 + ph * 2.3))
    a = np.clip((edge - r) * min(rx, ry) / 1.5, 0, 1) * a_max
    lam = np.clip(0.5 - 0.75 * (u * sx + v * sy) / np.maximum(edge, 0.3), 0, 1)
    lam = smoothstep(0.35, 0.62, lam + 0.12 * np.sin(ang * 5 + ph))
    col = mixc(col_s, col_l, lam)
    sub = ref[j0:j1, i0:i1]
    sub *= (1 - a)[..., None]
    sub += col * a[..., None]
    if alpha is not None:
        al = alpha[j0:j1, i0:i1]
        al *= (1 - a)
        al += a


def _village(tod, x0, y0, k, river_cov):
    """Draws the houses on a cel in layer px. Returns (rgba layer, bbox origin, windows).
    windows: list of (X, Y, w, h) in SET coords."""
    T = palette.TOD[tod]
    key, amb, fog = T["key"], T["amb"], T["fog"]
    sx = T["sun"][0]
    bx0, by0, bx1, by1 = 1780, 880, 2600, 1130
    cw, ch = int((bx1 - bx0) * k), int((by1 - by0) * k)
    ox, oy = (bx0 - x0) * k, (by0 - y0) * k
    c = core.Cel(cw, ch, ss=3, x0=ox, y0=oy)
    r = core.rng("village", 5)
    L = lambda X, Y: ((X - x0) * k, (Y - y0) * k)

    def river_free(hx, hy, wd):
        jx = int((hx - x0) * k)
        jy = int((hy - y0) * k)
        rad = int(wd * 0.8 * k)
        sub = river_cov[max(jy - int(rad * 0.5), 0):jy + int(rad * 0.3), max(jx - rad, 0):jx + rad]
        return sub.size > 0 and sub.max() < 0.2

    houses = []
    # fixed: the temple and the mill
    houses.append(dict(x=2395.0, y=992.0, wd=74.0, kind="temple"))
    tries = 0
    while len(houses) < 26 and tries < 4000:
        tries += 1
        hy = 932 + 158 * r.random() ** 1.25
        hx = r.uniform(1850, 2500)
        dn = (hy - 930) / 160.0
        wd = (26 + 44 * dn) * r.uniform(0.8, 1.25)
        # keep near the river: the village follows it
        if abs(hx - float(river_x(hy))) > 330 * (0.5 + 0.5 * r.random()):
            continue
        if hy > float(ground_y(hx)) - 16:
            continue
        if not river_free(hx, hy, wd):
            continue
        if any(abs(hx - q["x"]) < 0.62 * (wd + q["wd"]) and abs(hy - q["y"]) < 0.42 * (wd + q["wd"]) * 0.6 for q in houses):
            continue
        houses.append(dict(x=hx, y=hy, wd=wd, kind=r.choice(["gable", "gable", "front", "front", "thatch", "thatch", "hip"])))
    houses.sort(key=lambda q: q["y"])
    windows = []
    walls = [(0.93, 0.87, 0.76), (0.86, 0.78, 0.66), (0.62, 0.46, 0.36), (0.96, 0.93, 0.86), (0.74, 0.62, 0.50)]
    roofs = dict(
        tile=[(0.40, 0.46, 0.58), (0.66, 0.36, 0.28), (0.34, 0.38, 0.44), (0.56, 0.40, 0.38)],
        thatch=[(0.74, 0.60, 0.36), (0.66, 0.54, 0.36), (0.80, 0.68, 0.42)],
    )
    for q in houses:
        hx, hy, wd = q["x"], q["y"], q["wd"]
        dn = np.clip((hy - 930) / 160.0, 0, 1)
        hz = T["fog_k"] * (0.34 - 0.24 * dn)

        def C(local, lightness):
            col = mixc(palette.shade(local, T), desat(np.clip(np.asarray(local, np.float32) * key, 0, 1), 0.5 + 0.5 * look(tod)[1]),
                       np.float32(lightness * (0.30 + 0.70 * look(tod)[0]))) * (0.5 + 0.5 * gain(tod))
            return mixc(col, fog, np.float32(hz))

        bx, by = L(hx, hy)
        W_ = wd * k
        kind = q["kind"]
        side = -1 if r.random() < 0.6 else 1
        lit_side = 0.95 if side * sx > 0 else 0.05
        wloc = walls[r.integers(len(walls))]
        wh = W_ * r.uniform(0.36, 0.5)
        sw = W_ * r.uniform(0.30, 0.48)
        sk = W_ * r.uniform(0.07, 0.12)
        o = W_ * 0.09
        xe = bx + side * W_ / 2
        front = [(bx - W_ / 2, by), (bx + W_ / 2, by), (bx + W_ / 2, by - wh), (bx - W_ / 2, by - wh)]
        sidew = [(xe, by), (xe + side * sw, by - sk), (xe + side * sw, by - sk - wh), (xe, by - wh)]
        if kind == "temple":
            # two tier roof with upturned eaves, red pillars
            wloc = (0.80, 0.34, 0.26)
            rl = (0.30, 0.36, 0.40)
            c.fill(front, C(wloc, 0.5))
            for t in np.linspace(-0.42, 0.42, 5):
                c.fill([(bx + t * W_ - 1.6, by), (bx + t * W_ + 1.6, by), (bx + t * W_ + 1.6, by - wh), (bx + t * W_ - 1.6, by - wh)], C((0.55, 0.2, 0.16), 0.3))
            for tier, (ty, tw, th) in enumerate([(by - wh, W_ * 0.80, W_ * 0.30), (by - wh - W_ * 0.42, W_ * 0.52, W_ * 0.34)]):
                if tier == 1:
                    c.fill([(bx - W_ * 0.3, ty + W_ * 0.14), (bx + W_ * 0.3, ty + W_ * 0.14), (bx + W_ * 0.3, ty), (bx - W_ * 0.3, ty)], C(wloc, 0.45))
                eave = [(bx - tw, ty - th * 0.16), (bx - tw * 0.82, ty + th * 0.10), (bx, ty + th * 0.16), (bx + tw * 0.82, ty + th * 0.10), (bx + tw, ty - th * 0.16)]
                top = [(bx + tw * 0.36, ty - th), (bx - tw * 0.36, ty - th)]
                up = core.catmull(eave, 6, closed=False)
                lf = core.catmull([eave[0], (bx - tw * 0.52, ty - th * 0.45), top[1]], 6, closed=False)
                rt = core.catmull([top[0], (bx + tw * 0.52, ty - th * 0.45), eave[-1]], 6, closed=False)
                poly = np.vstack([up, rt[::-1], lf[::-1]])
                c.fill(poly, C(rl, 0.55))
                c.fill(np.vstack([lf, [(bx - tw * 0.1, ty - th), (bx - tw * 0.3, ty + th * 0.12)]]), C(rl, 0.95 if sx < 0 else 0.1))
                c.stroke(up, 2.2, C((0.20, 0.22, 0.28), 0.2), taper=(0.05, 0.05))
                c.stroke([top[1], top[0]], 2.4, C((0.86, 0.70, 0.36), 0.9), taper=(0.1, 0.1))
            c.dot(bx, by - wh - W_ * 0.42 - W_ * 0.34 - 3, 2.2, C((0.95, 0.78, 0.36), 1.0))
            for t in (-0.3, 0.0, 0.3):
                windows.append((hx + t * wd, hy - wh / k * 0.5, wd * 0.11, wh / k * 0.5))
            continue
        c.fill(front, C(wloc, 0.52))
        c.fill(sidew, C(wloc, lit_side))
        rh = W_ * r.uniform(0.30, 0.50)
        if kind in ("gable", "thatch", "hip"):
            if kind == "thatch":
                rh = W_ * r.uniform(0.55, 0.78)
                rl = roofs["thatch"][r.integers(3)]
            else:
                rl = roofs["tile"][r.integers(4)]
            shift = side * sw * 0.5
            rise = sk * 0.5
            inset = W_ * (0.30 if kind == "hip" else 0.04 if kind == "gable" else 0.12)
            p0 = (bx - W_ / 2 - o, by - wh + o * 0.5)
            p1 = (bx + W_ / 2 + o, by - wh + o * 0.5)
            p2 = (bx + W_ / 2 + shift - inset * (1 if side < 0 or kind != "gable" else 0.2), by - wh - rh - rise)
            p3 = (bx - W_ / 2 + shift + inset * (1 if side > 0 or kind != "gable" else 0.2), by - wh - rh - rise)
            # gable end on the side wall
            ge = [(xe, by - wh), (xe + side * sw, by - wh - sk), (xe + side * sw * 0.5, by - wh - rh - rise)]
            if kind == "thatch":
                slope = core.catmull([p0, p1, (p1[0] - W_ * 0.02, p1[1] - rh * 0.5), p2, p3, (p0[0] + W_ * 0.02, p0[1] - rh * 0.5)], 6)
                c.fill(ge, C(rl, lit_side * 0.9))
                c.fill(slope, C(rl, 0.62))
                c.stroke([p3, p2], 3.0, C((0.46, 0.36, 0.26), 0.4), taper=(0.1, 0.1))
            else:
                c.fill(ge, C(wloc if kind == "gable" else rl, lit_side))
                c.fill([p0, p1, p2, p3], C(rl, 0.60))
                # side slope sliver catching or losing the sun
                c.fill([p1 if side > 0 else p0, p2 if side > 0 else p3, (xe + side * sw + side * o, by - wh - sk + o * 0.4)], C(rl, lit_side))
                c.stroke([p3, p2], 1.8, C(rl, 1.0), taper=(0.1, 0.1))
                c.stroke([p0, p1], 1.6, C(rl, 0.05), taper=(0.05, 0.05))
        else:  # gable end faces us
            rl = roofs["tile"][r.integers(4)] if r.random() < 0.7 else roofs["thatch"][r.integers(3)]
            apex = (bx, by - wh - rh)
            c.fill([(bx - W_ / 2, by - wh), (bx + W_ / 2, by - wh), apex], C(wloc, 0.52))
            far_apex = (bx + side * sw, by - wh - rh - sk)
            c.fill([apex, (bx + side * (W_ / 2 + o), by - wh + o * 0.4), (xe + side * sw + side * o, by - wh - sk + o * 0.4), far_apex], C(rl, lit_side))
            c.stroke([(bx - W_ / 2 - o, by - wh + o * 0.5), apex], 2.6, C(rl, 0.9 if sx < 0 else 0.2), taper=(0.05, 0.05))
            c.stroke([apex, (bx + W_ / 2 + o, by - wh + o * 0.5)], 2.6, C(rl, 0.2 if sx < 0 else 0.9), taper=(0.05, 0.05))
            if r.random() < 0.5:
                windows.append((hx, hy - (wh + rh * 0.3) / k, wd * 0.10, wd * 0.10))
        # door and windows on the front wall
        nwin = int(r.integers(1, 4))
        slots = np.linspace(-0.32, 0.32, nwin + 1)[:-1] + 0.32 / max(nwin, 1) * r.uniform(0.6, 1.4)
        for t in slots:
            ww, wh2 = wd * r.uniform(0.10, 0.15), (wh / k) * r.uniform(0.30, 0.42)
            windows.append((hx + t * wd, hy - wh / k * r.uniform(0.50, 0.62), ww, wh2))
        if r.random() < 0.35:  # chimney
            cx_ = bx + r.uniform(-0.25, 0.25) * W_
            c.fill([(cx_ - 2.2, by - wh - rh * 0.5), (cx_ + 2.2, by - wh - rh * 0.5), (cx_ + 2.2, by - wh - rh * 1.15), (cx_ - 2.2, by - wh - rh * 1.15)], C((0.5, 0.36, 0.32), 0.5))
    # windows, unlit: dark glass with a hint of sky
    for (wx, wy, ww, wh2) in windows:
        dn = np.clip((wy - 930) / 160.0, 0, 1)
        col = mixc(palette.shade((0.30, 0.26, 0.36), T), fog, np.float32(T["fog_k"] * (0.42 - 0.3 * dn)))
        a, b = L(wx - ww / 2, wy - wh2 / 2)
        a2, b2 = L(wx + ww / 2, wy + wh2 / 2)
        c.fill([(a, b), (a2, b), (a2, b2), (a, b2)], col)
    # bridge across the river at set y 1012
    Yb = 1014.0
    row = river_cov[int((Yb - y0) * k)]
    xs_ = np.nonzero(row > 0.5)[0]
    cxr = (float(river_x(Yb)) - x0) * k
    xs_ = xs_[np.abs(xs_ - cxr) < 120 * k]
    if len(xs_):
        xa, xb = xs_.min() - 12 * k, xs_.max() + 12 * k
        yb = (Yb - y0) * k
        arch = [(xa + (xb - xa) * t, yb - 9 * k * math.sin(math.pi * t) - 2) for t in np.linspace(0, 1, 14)]
        deck = arch + [(x_, y_ + 3.2 * k) for x_, y_ in arch[::-1]]
        hzb = T["fog_k"] * 0.3
        c.fill(deck, mixc(mixc(palette.shade((0.6, 0.42, 0.34), T), np.float32((0.6, 0.42, 0.34)) * key, np.float32(0.5)), fog, np.float32(hzb)))
        c.stroke(arch, 1.6, mixc(np.clip(np.float32((0.85, 0.62, 0.46)) * key, 0, 1), fog, np.float32(hzb)), taper=(0.02, 0.02))
        for t in (0.25, 0.5, 0.75):
            xp = xa + (xb - xa) * t
            yp = yb - 9 * k * math.sin(math.pi * t) + 2
            c.fill([(xp - 1.5, yp), (xp + 1.5, yp), (xp + 1.5, yp + 9 * k), (xp - 1.5, yp + 9 * k)], mixc(palette.shade((0.34, 0.26, 0.28), T), fog, np.float32(hzb)))
    # water wheel by the river at set y 1052
    Yw = 1052.0
    row = river_cov[int((Yw - y0) * k)]
    xs_ = np.nonzero(row > 0.5)[0]
    cxr = (float(river_x(Yw)) - x0) * k
    xs_ = xs_[np.abs(xs_ - cxr) < 120 * k]
    if len(xs_):
        xw, yw, rw = xs_.min() + 2.0, (Yw - y0) * k - 9 * k, 11 * k
        wc = mixc(palette.shade((0.40, 0.28, 0.24), T), fog, np.float32(T["fog_k"] * 0.22))
        wl = mixc(np.clip(np.float32((0.70, 0.50, 0.36)) * key, 0, 1), fog, np.float32(T["fog_k"] * 0.22))
        hut = [(xw - rw * 2.6, yw + rw), (xw - rw * 0.5, yw + rw), (xw - rw * 0.5, yw - rw * 0.6), (xw - rw * 2.6, yw - rw * 0.6)]
        c.fill(hut, mixc(mixc(palette.shade(walls[2], T), np.float32(walls[2]) * key, np.float32(0.5)), fog, np.float32(T["fog_k"] * 0.22)))
        c.fill([(xw - rw * 2.9, yw - rw * 0.5), (xw - rw * 0.2, yw - rw * 0.5), (xw - rw * 1.0, yw - rw * 1.7), (xw - rw * 2.2, yw - rw * 1.7)],
               mixc(mixc(palette.shade(roofs["thatch"][0], T), np.float32(roofs["thatch"][0]) * key, np.float32(0.6)), fog, np.float32(T["fog_k"] * 0.22)))
        ring = core.ellipse(xw, yw, rw * 0.62, rw, 28)
        c.stroke(ring, 2.4, wc, closed=True)
        for a_ in np.linspace(0, math.pi, 5)[:-1] + 0.2:
            c.stroke([(xw - math.cos(a_) * rw * 0.62, yw - math.sin(a_) * rw), (xw + math.cos(a_) * rw * 0.62, yw + math.sin(a_) * rw)], 1.5, wl, taper=(0.02, 0.02))
        windows.append(((xw - rw * 1.5) / k + x0, (yw + rw * 0.2) / k + y0, 5.0, 6.0))
    return c.rgba(), (int(ox), int(oy)), windows


def paint_valley(tod):
    T = palette.TOD[tod]
    x0, y0, x1, y1, k, _ = SPEC["valley"]
    X, Y = grid("valley")
    h, w = X.shape
    key, amb, fog = T["key"], T["amb"], T["fog"]
    sx, sy = T["sun"]
    d = np.clip((Y - HORIZON) / 620.0, 0, 1)
    ps = 0.10 + 0.90 * d
    wx = ((X - 1900.0) / ps).ravel()
    wz = (1.0 / (d + 0.06)).ravel()
    zz = wz * 2.3 + 0.7 * (core.fbm_at(wx, wz * 300, 1300.0, 3, 31) - 0.5)
    rid = np.floor(zz)
    fr = zz - rid
    cc = wx / 420.0 + 13.7 * hashf(rid, 1.0) + 0.9 * (core.fbm_at(wx + 500, zz * 400, 800.0, 3, 32) - 0.5)
    cid = np.floor(cc)
    fc = cc - cid
    hv = hashf(rid, cid).reshape(h, w)
    hv2 = hashf(rid + 7.7, cid + 3.1).reshape(h, w)
    edge = np.minimum(np.minimum(fr, 1 - fr) / 0.07, np.minimum(fc, 1 - fc) / 0.045).reshape(h, w)
    fc = fc.reshape(h, w)
    pal = np.array([(0.50, 0.64, 0.30), (0.62, 0.72, 0.34), (0.78, 0.68, 0.40), (0.36, 0.52, 0.32),
                    (0.58, 0.44, 0.38), (0.82, 0.76, 0.38), (0.44, 0.60, 0.34), (0.68, 0.70, 0.42)], np.float32)
    local = pal[np.minimum((hv * len(pal)).astype(int), len(pal) - 1)]
    local = local * (0.90 + 0.2 * hv2)[..., None]
    local = local * (1.0 + 0.05 * np.sin(fc * 6.283 * (4 + 6 * hv2)))[..., None]
    big = core.fbm(h, w, 420 * k, 4, seed=41, aspect=2.6)
    lv = np.clip(0.50 + 0.9 * (big - 0.5), 0, 1) * look(tod)[0]
    local = desat(local, look(tod)[1])
    shade_loc = local * np.float32((0.80, 1.0, 1.10))
    col = mixc(shade_loc * amb * 1.1, local * key, lv)
    hedge = smoothstep(1.0, 0.35, edge)
    col = mixc(col, palette.shade((0.22, 0.34, 0.30), T), hedge * 0.85)
    ref = np.ascontiguousarray(col, np.float32)
    # haze with distance
    mean = np.float32((0.50, 0.60, 0.32)) * (0.45 * key + 0.5 * amb)
    ref = mixc(ref, mean, smoothstep(0.26, 0.02, d) * 0.32)      # far fields simplify
    ref *= 0.86 * gain(tod)
    hz = T["fog_k"] * (0.05 + 0.50 * np.clip(1 - d / 0.50, 0, 1) ** 2.0)
    ref = mixc(ref, fog, hz)
    # river: a ribbon of sky
    rc, rhw, rdist = river_field()
    sky_far = np.clip(palette.sky_color(T, 1.0) * 1.03, 0, 1)
    sky_mid = np.clip(palette.sky_color(T, 0.86) * 1.06, 0, 1)
    sky_near = np.clip(0.5 * palette.sky_color(T, 0.2) * 1.30 + 0.5 * palette.sky_color(T, 0.9) + 0.02, 0, 1)
    sky = mixc(np.broadcast_to(sky_far, (h, w, 3)), sky_mid, smoothstep(0.04, 0.20, d))
    sky = mixc(sky, sky_near, smoothstep(0.30, 0.62, d))
    rip = core.fbm(h, w, 5 * k, 3, seed=43, aspect=9.0)
    rip2 = core.fbm(h, w, 24 * k, 3, seed=48, aspect=7.0)
    rcol = sky * (0.93 + 0.20 * (rip - 0.5) + 0.20 * (rip2 - 0.5))[..., None]
    rcol = mixc(rcol, np.clip(T["rim"] * 1.02 + 0.05, 0, 1), smoothstep(0.56, 0.68, rip) * 0.65 * smoothstep(0.75, 0.1, d))
    rcol = rcol * (1.0 if look(tod)[0] >= 1 else 1.30)
    inner = smoothstep(rhw * 0.35, rhw, rdist) * smoothstep(0.38, 0.62, rip2 + 0.25 * (rip - 0.5) + 0.12)
    rcol = np.clip(mixc(rcol, mixc(palette.shade((0.24, 0.36, 0.36), T), fog, hz * 0.6), inner * 0.6), 0, 1)
    bank = smoothstep(rhw + 11 * k * ps + 3, rhw, rdist) * (1 - rc)
    ref = mixc(ref, mixc(palette.shade((0.18, 0.28, 0.28), T), fog, hz * 0.6), bank * 0.88)
    ref = mixc(ref, rcol, rc)
    # tree clumps
    r = core.rng("valleytrees", 3)
    n = 380
    ty = HORIZON + 14 + 600 * r.random(n) ** 1.9
    tx = r.uniform(x0, x1, n)
    near_river = r.random(n) < 0.45
    tx = np.where(near_river, river_x(ty) + r.normal(0, 1, n) * (60 + 200 * (ty - HORIZON) / 620), tx)
    order = np.argsort(ty)
    for i in order:
        dd = (ty[i] - HORIZON) / 620.0
        pp = 0.10 + 0.9 * dd
        jx, jy = (tx[i] - x0) * k, (ty[i] - y0) * k
        if not (0 <= jx < w and 0 <= jy < h) or rc[int(jy), int(jx)] > 0.3:
            continue
        rx = (26 + 60 * r.random()) * pp * k
        ry_ = rx * r.uniform(0.55, 0.9)
        hzz = float(T["fog_k"] * (0.05 + 0.50 * max(0.0, 1 - dd / 0.50) ** 2.0)) * 0.85
        cs0 = palette.shade((0.16, 0.30, 0.32), T) * gain(tod)
        cl0 = cs0 + (desat(np.clip(np.float32((0.40, 0.50, 0.24)) * key, 0, 1), look(tod)[1]) * gain(tod) - cs0) * look(tod)[0]
        cl = mixc(cl0, fog, np.float32(hzz))
        cs = mixc(cs0, fog, np.float32(hzz))
        # cast shadow on the ground, away from the sun
        _blob(ref, None, jx - sx * rx * 2.5, jy + ry_ * 0.25, rx * 3.1, ry_ * 0.32, cs, cs, sx, sy, i, bump=0.1, a_max=0.52 * (0.3 + 0.7 * look(tod)[0]))
        for q in range(int(r.integers(1, 4))):
            _blob(ref, None, jx + r.normal(0, rx * 0.5), jy - ry_ * (0.6 + 0.3 * q) + r.normal(0, ry_ * 0.2), rx * r.uniform(0.6, 1.0), ry_ * r.uniform(0.7, 1.0), cl, cs, sx, sy, i * 7 + q)
    # mist pooled low and far
    mist = smoothstep(0.50, 0.74, core.fbm(h, w, 150 * k, 4, seed=44, aspect=9.0)) * smoothstep(0.40, 0.05, d) * 0.50
    mist = np.maximum(mist, np.exp(-rdist / (rhw * 2.5 + 30 * k)) * smoothstep(0.46, 0.66, core.fbm(h, w, 110 * k, 3, seed=51, aspect=5.0)) * smoothstep(0.5, 0.1, d) * 0.42)
    mcol = np.clip(fog * 1.04 + 0.02, 0, 1)
    ref = mixc(ref, mcol, mist)
    mn = core.fbm(h, w, 240 * k, 4, seed=49, aspect=6.0)
    ref = mixc(ref, fog, smoothstep(0.085, 0.0, d + 0.05 * (mn - 0.5)) * 0.85)   # the far edge dissolves
    # the village
    vil, (vx, vy), windows = _village(tod, x0, y0, k, rc)
    core.over(ref, vil, vx, vy)
    veil = smoothstep(0.5, 0.85, core.fbm(h, w, 220 * k, 3, seed=45, aspect=6.0)) * smoothstep(0.5, 0.08, d) * 0.10
    ref = mixc(ref, mcol, veil)
    edge_n = core.fbm(h, w, 70 * k, 3, seed=46, aspect=4.0)
    edge_n = 0.6 * edge_n + 0.4 * core.fbm(h, w, 320 * k, 3, seed=50, aspect=5.0)
    alpha = smoothstep(HORIZON - 30, HORIZON + 24, Y + 46 * (edge_n - 0.5)).astype(np.float32)
    ang = (0.12 * (core.fbm(h, w, 200 * k, 2, seed=47) - 0.5)).astype(np.float32)  # near horizontal drags
    size = (0.35 + 0.9 * d).astype(np.float32)
    out = paint.paint(ref, None, direction=ang, seed=paint_seed("valley", tod), passes=paint.PASSES[1:], scale=0.6 * k, size=size, grain=0.045)
    # crisp village back on top of the brushwork
    core.over(out, vil, vx, vy, opacity=0.78)
    out = core.premul(out, alpha)
    return out, np.asarray(windows, np.float32), rc


# ----------------------------------------------------------------------- hill
GRASS = [
    dict(w=44, l=120, dens=1.4, op=0.92, keep=0.50, detail=0.0, jh=0.035, jv=0.08, dry=0.7, pool=0.05),
    dict(w=20, l=66, dens=1.6, op=0.88, keep=0.30, detail=0.0, jh=0.035, jv=0.08, dry=0.8, pool=0.06),
    dict(w=9, l=38, dens=1.6, op=0.88, keep=0.20, detail=0.25, jh=0.030, jv=0.08, dry=0.8, pool=0.06),
    dict(w=4, l=24, dens=1.5, op=0.90, keep=0.14, detail=0.45, jh=0.025, jv=0.07, dry=0.7, pool=0.04),
]
PATH = [(1292, 0, 9), (1272, 22, 12), (1215, 56, 16), (1120, 96, 22), (1030, 160, 32), (1010, 260, 48), (930, 380, 66), (780, 520, 92), (640, 640, 120)]


def hill_light(tod, X, Y, t):
    """0..1 direct light on the hill surface (t = set px below the crest), with
    the cast shadows of the tree and the rocket."""
    T = palette.TOD[tod]
    sx = T["sun"][0]
    h, w = X.shape
    k = SPEC["hill"][4]
    n1 = core.fbm(h, w, 260 * k, 4, seed=51, aspect=2.6)
    n2 = core.fbm(h, w, 60 * k, 3, seed=52, aspect=2.0)
    if sx < 0:   # sun on the left: the left flank is lit, the right falls away
        base = smoothstep(2650, 1750, X + 1.25 * t)
    else:        # sun on the right, behind the valley: the right flank and crest are lit
        base = smoothstep(560, 1350, X - 1.1 * t)
    l = smoothstep(0.30, 0.62, base * 0.85 + 0.55 * (n1 - 0.5) + 0.18 * (n2 - 0.5) + 0.05)
    l *= 1 - 0.72 * smoothstep(150, 430, t + 120 * (n1 - 0.5))
    # tree shadow: long, dappled, thrown away from the sun
    dirx = 1.0 if sx < 0 else -1.0
    cx, cy = 1180 + dirx * 330, 1210.0
    ca, sa = math.cos(0.17 * dirx), math.sin(0.17 * dirx)
    u = ((X - cx) * ca + (Y - cy) * sa) / 560.0
    v = (-(X - cx) * sa + (Y - cy) * ca) / 78.0
    e = np.clip(1 - (u * u + v * v), 0, 1)
    dap = smoothstep(0.30, 0.52, core.fbm(h, w, 38 * k, 3, seed=53, aspect=2.2))
    sh = smoothstep(0.0, 0.35, e) * (0.25 + 0.75 * dap)
    # trunk shadow joining the tree to it
    gx = ground_y(880.0)
    tu = np.clip((X - 880) * dirx / 330.0, 0, 1)
    tv = (Y - (gx + 20 + 85 * tu)) / (16 + 10 * tu)
    sh = np.maximum(sh, np.exp(-tv * tv) * ((X - 880) * dirx > -10) * (tu < 1))
    # rocket shadow
    rx, ry = 1720.0, ground_y(1720.0)
    ru = (X - rx) * dirx / 640.0
    rv = (Y - (ry + 8 + 150 * np.clip(ru, 0, 1.3))) / (13 + 26 * np.clip(ru, 0, 1))
    rs = np.exp(-rv * rv * 1.2) * smoothstep(-0.02, 0.03, ru) * smoothstep(1.05, 0.75, ru)
    l = l * (1 - 0.88 * sh)
    # the rocket shadow is returned apart (the rocket is not always there)
    return l.astype(np.float32), n1, n2, (rs * 0.9 * l).astype(np.float32)


def paint_hill(tod):
    T = palette.TOD[tod]
    x0, y0, x1, y1, k, _ = SPEC["hill"]
    X, Y = grid("hill")
    h, w = X.shape
    key, amb, fog = T["key"], T["amb"], T["fog"]
    sx, sy = T["sun"]
    gy = ground_y(X[0].astype(np.float64)).astype(np.float32)
    xs = X[0].astype(np.float64)
    tuft = 7 * core.fbm_at(xs, xs * 0 + 2.2, 9, 3, 61) + 6 * core.fbm_at(xs, xs * 0 + 5.2, 36, 2, 62) - 4
    t = Y - gy[None, :]
    alpha = np.clip((t + tuft[None, :].astype(np.float32)) * k / 1.5 + 0.5, 0, 1).astype(np.float32)
    l0, n1, n2, rshadow = hill_light(tod, X, Y, t)
    direct, sat = look(tod)
    l = l0 * direct
    rshadow = rshadow * direct * alpha
    n3 = core.fbm(h, w, 130 * k, 4, seed=63, aspect=1.8)
    n4 = core.fbm(h, w, 22 * k, 3, seed=64, aspect=0.6)
    g1 = np.float32((0.50, 0.64, 0.22))
    g2 = np.float32((0.70, 0.68, 0.26))
    g3 = np.float32((0.32, 0.52, 0.26))
    local = mixc(g1, g2, smoothstep(0.45, 0.70, n3))
    local = mixc(local, g3, smoothstep(0.52, 0.72, n1) * 0.8)
    local = desat(local * (0.88 + 0.24 * n4)[..., None], sat)
    sh_loc = local * np.float32((0.62, 0.98, 1.22))
    violet = smoothstep(0.50, 0.75, n2) * 0.45
    sh_col = mixc(sh_loc * amb, palette.shade((0.36, 0.34, 0.60), T), violet)
    lit_col = local * key * 1.04
    col = mixc(sh_col, lit_col, l)
    # worn path
    pm = np.zeros((h, w), np.float32)
    pts = np.array(PATH, np.float64)
    ss = np.linspace(0, len(pts) - 1, 90)
    pxs = np.interp(ss, np.arange(len(pts)), pts[:, 0])
    pdt = np.interp(ss, np.arange(len(pts)), pts[:, 1])
    pw = np.interp(ss, np.arange(len(pts)), pts[:, 2])
    for a_, b_, c_ in zip(pxs, pdt, pw):
        cy_ = ground_y(a_) + 6 + b_
        j0, j1 = int((cy_ - c_ * 0.5 - y0) * k), int((cy_ + c_ * 0.5 - y0) * k) + 1
        i0, i1 = int((a_ - c_ * 1.6 - x0) * k), int((a_ + c_ * 1.6 - x0) * k) + 1
        j0, i0 = max(j0, 0), max(i0, 0)
        if j1 <= j0 or i1 <= i0:
            continue
        uu = (X[j0:j1, i0:i1] - a_) / (c_ * 1.25)
        vv = (Y[j0:j1, i0:i1] - cy_) / (c_ * 0.36)
        pm[j0:j1, i0:i1] = np.maximum(pm[j0:j1, i0:i1], np.clip(1.25 - (uu * uu + vv * vv), 0, 1))
    pm = smoothstep(0.25, 0.8, pm * (0.55 + 0.9 * core.fbm(h, w, 26 * k, 3, seed=65, aspect=2.5)))
    earth = np.float32((0.70, 0.62, 0.44))
    pcol = mixc(palette.shade(earth * (0.8, 0.9, 1.15), T), np.clip(earth * key, 0, 1), l)
    col = mixc(col, pcol, pm * 0.62) * gain(tod)
    # sun (or moon) lit grass tips along the crest
    rim_w = smoothstep(1000, 1800, X) if tod in ("predawn", "launch") else 1.0
    rimv = np.exp(-np.maximum(t, 0) / 11.0) * l0 * rim_w * (0.4 + 0.6 * smoothstep(0.35, 0.65, core.fbm(h, w, 12 * k, 2, seed=66, aspect=0.5)))
    col = mixc(col, rim_col(tod, T), np.clip(rimv * 0.85, 0, 1))
    ref = np.ascontiguousarray(np.clip(col, 0, 1), np.float32)
    # stones
    r = core.rng("stones", 2)
    stone = np.float32((0.74, 0.70, 0.66))
    for (sxx, dt, rr) in [(1232, 30, 9), (1010, 150, 15), (1950, 60, 10), (2210, 120, 14), (760, 90, 12), (1560, 300, 20), (1265, 46, 5), (2420, 40, 9)]:
        jx, jy = (sxx - x0) * k, (ground_y(float(sxx)) + dt - y0) * k
        li = float(l[min(int(jy), h - 1), min(int(jx), w - 1)])
        cl = mixc(palette.shade(stone, T), np.clip(stone * key, 0, 1), np.float32(0.25 + 0.75 * li))
        cs = palette.shade(stone * np.float32((0.72, 0.74, 0.95)), T)
        _blob(ref, None, jx + rr * k * 0.5 * (1 if sx < 0 else -1), jy + rr * k * 0.42, rr * k * 1.3, rr * k * 0.28, cs * 0.7, cs * 0.7, sx, sy, sxx, bump=0.1, a_max=0.5)
        _blob(ref, None, jx, jy, rr * k, rr * k * 0.62, cl, cs, sx, sy, sxx + 1, bump=0.12)
    # grass grows up, leaning with the slope and in windy clumps
    lean = 1.1 * (core.fbm(h, w, 150 * k, 3, seed=67) - 0.5) + 0.18 * np.sign(X - 1500) * np.clip(np.abs(X - 1500) / 900, 0, 1)
    ang = (-np.pi / 2 + lean).astype(np.float32)
    size = (0.42 + 0.95 * smoothstep(0, 520, t)).astype(np.float32)
    out = paint.paint(ref, alpha, direction=ang, seed=paint_seed("hill", tod), passes=GRASS, scale=1.0, size=size, spill=0.0, grain=0.05)
    rgb = out[..., :3] / np.maximum(out[..., 3:4], 1e-4)
    # ---- blades: crisp grass over the brushwork
    r = core.rng("blades", 4)
    n = 52000
    bx = r.uniform(0, w, n)
    by = r.uniform(0, h, n) ** 1.0
    bt = t[np.minimum(by.astype(int), h - 1), np.minimum(bx.astype(int), w - 1)]
    okb = bt > 2
    keep_p = 0.30 + 0.70 * np.exp(-np.maximum(bt, 0) / 170.0)
    okb &= r.random(n) < keep_p
    bx, by, bt = bx[okb], by[okb], bt[okb]
    m = len(bx)
    yi, xi = np.minimum(by.astype(int), h - 1), np.minimum(bx.astype(int), w - 1)
    base = rgb[yi, xi]
    li = l[yi, xi]
    sz = size[yi, xi]
    up = r.random(m) < (0.35 + 0.4 * li)
    tint_l = base * np.float32((1.30, 1.22, 0.95)) + np.float32((0.06, 0.05, 0.0)) * li[:, None]
    tint_d = base * np.float32((0.66, 0.78, 0.88))
    cols = np.where(up[:, None], tint_l, tint_d)
    order = np.argsort(by)
    a_b = ang[yi, xi] + r.normal(0, 0.28, m)
    paint.strokes(out, bx[order], by[order], a_b[order], (2.2 + 2.6 * r.random(m)) * (0.7 + 0.6 * sz[order]),
                  (20 + 46 * r.random(m)) * sz[order] * 1.25, np.clip(cols[order], 0, 1), op=0.9,
                  bend=r.normal(0, 0.55, m), seed=71, dry=0.25, taper=0.88, streak=0.05, paper_k=0.2)
    # ---- crest silhouette blades: rim lit on the sun side
    m = 5200
    cxs = r.uniform(x0, x1, m)
    cys = ground_y(cxs) + r.uniform(-2, 9, m)
    jx, jy = (cxs - x0) * k, (cys - y0) * k
    lc = l0[np.clip(jy.astype(int) + 4, 0, h - 1), np.clip(jx.astype(int), 0, w - 1)]
    dark = palette.shade((0.22, 0.42, 0.34), T) * gain(tod)
    midg = desat(np.clip(np.float32((0.52, 0.64, 0.24)) * key, 0, 1), sat) * gain(tod)
    midg = dark + (midg - dark) * (0.35 + 0.65 * direct)
    if direct < 1.0:
        midg = midg * np.float32((0.80, 1.0, 1.20))
    gold = midg + (rim_col(tod, T) - midg) * (0.55 + 0.45 * direct)
    if tod in ("predawn", "launch"):
        lc = lc * smoothstep(1000, 1800, cxs)
    cols = np.where((lc > 0.45)[:, None], mixc(midg, gold, r.random(m) ** 0.7), mixc(dark, dark * 1.5 + 0.03, r.random(m)))
    paint.strokes(out, jx, jy, -np.pi / 2 + r.normal(0, 0.38, m) + 0.2 * np.sign(cxs - 1500) * 0.5, 2.0 + 2.2 * r.random(m),
                  (12 + 28 * r.random(m) ** 1.5) * k * 0.8, np.clip(cols, 0, 1), op=0.95, bend=r.normal(0, 0.6, m),
                  seed=72, dry=0.2, taper=0.9, streak=0.04, paper_k=0.1)
    # ---- wild flowers in drifts
    r = core.rng("flowers", 6)
    fl = [np.float32((1.0, 0.98, 0.92)), np.float32((1.0, 0.86, 0.40)), np.float32((0.96, 0.62, 0.70)), np.float32((0.80, 0.72, 1.0)), np.float32((1.0, 0.55, 0.32))]
    fx, fy, fcol, fw = [], [], [], []
    for q in range(64):
        cxf = r.uniform(x0 + 100, x1 - 100)
        dtf = 12 + 420 * r.random() ** 1.6
        cyf = ground_y(cxf) + dtf
        kind = fl[r.integers(len(fl))]
        cnt = int(r.integers(6, 34))
        spread = 30 + 90 * r.random()
        szf = 0.42 + 0.95 * min(dtf / 520.0, 1.0)
        for _ in range(cnt):
            px_ = cxf + r.normal(0, spread)
            py_ = cyf + r.normal(0, spread * 0.28)
            jx_, jy_ = int((px_ - x0) * k), int((py_ - y0) * k)
            if not (0 <= jx_ < w and 0 <= jy_ < h) or t[jy_, jx_] < 6:
                continue
            lf = float(l[jy_, jx_])
            cf = mixc(palette.shade(kind * np.float32((0.8, 0.85, 1.1)), T) * 1.15, np.clip(kind * key, 0, 1), np.float32(lf))
            fx.append((px_ - x0) * k)
            fy.append((py_ - y0) * k)
            fcol.append(cf)
            fw.append((3.0 + 3.2 * r.random()) * szf * 1.2)
    fx, fy, fw = np.array(fx), np.array(fy), np.array(fw)
    paint.strokes(out, fx, fy, r.uniform(0, 6.28, len(fx)), fw, fw * 0.8, np.clip(np.array(fcol), 0, 1), op=0.96,
                  seed=73, dry=0.0, taper=0.0, streak=0.0, paper_k=0.0)
    return out, rshadow


# ----------------------------------------------------------------------- tree
TRUNK = [(872, 1118, 104), (884, 1040, 82), (900, 950, 72), (932, 860, 64), (982, 772, 56), (1036, 694, 48)]
LIMBS = [
    [(982, 772, 40), (930, 700, 32), (858, 620, 25), (792, 548, 17), (742, 486, 9)],
    [(1036, 694, 44), (1078, 610, 34), (1106, 520, 24), (1122, 440, 13), (1128, 392, 7)],
    [(1036, 694, 38), (1130, 652, 30), (1250, 618, 23), (1370, 596, 16), (1470, 588, 10), (1540, 600, 5)],
    [(932, 866, 26), (868, 818, 19), (800, 790, 12), (742, 780, 6)],
    [(1250, 618, 16), (1290, 560, 11), (1320, 500, 6)],
    [(858, 620, 16), (900, 540, 11), (930, 470, 6)],
]
ROOTS = [
    [(860, 1080, 52), (836, 1106, 34), (808, 1122, 16), (790, 1130, 6)],
    [(892, 1084, 52), (924, 1106, 32), (952, 1120, 15), (970, 1127, 6)],
    [(874, 1100, 40), (872, 1128, 26), (866, 1146, 8)],
]
MASSES = [
    # cx, cy, r, sublayer (0 back, 1 mid, 2 front)
    (800, 545, 158, 1), (985, 492, 162, 1), (1165, 505, 150, 1), (1335, 560, 126, 1), (1478, 566, 88, 1),
    (705, 665, 104, 1), (930, 640, 132, 2), (1135, 655, 112, 2), (862, 770, 78, 2), (1262, 716, 74, 2),
    (1400, 640, 70, 2), (760, 450, 90, 0), (1060, 400, 110, 0), (1280, 470, 110, 0), (900, 600, 150, 0),
    (1180, 600, 140, 0), (1420, 520, 90, 0), (690, 580, 90, 0), (1010, 720, 90, 0),
]


def _limb_mask(shape, x0, y0, k, pts, seed):
    """Coverage of a tapered limb through (x, y, width) control points (set coords)."""
    p = np.asarray(pts, np.float64)
    cl = core.catmull(p[:, :2], 10, closed=False)
    wd = np.interp(np.linspace(0, 1, len(cl)), np.linspace(0, 1, len(p)), p[:, 2])
    cl = core.wobble(cl, 3.0, seed, 0.02)
    n = max(8, int(np.hypot(*np.diff(cl, axis=0).T).sum() / 2.0))
    s = np.linspace(0, 1, len(cl))
    sn = np.linspace(0, 1, n)
    cx = np.interp(sn, s, cl[:, 0])
    cy = np.interp(sn, s, cl[:, 1])
    cw = np.interp(sn, s, wd) * (0.92 + 0.16 * core.fbm_at(sn * 9, sn * 0 + 0.3, 1.0, 2, seed))
    m = np.zeros(shape, np.float32)
    h, w = shape
    for a, b, c in zip(cx, cy, cw):
        rr = c * 0.5 * k
        jx, jy = (a - x0) * k, (b - y0) * k
        i0, i1 = max(int(jx - rr - 2), 0), min(int(jx + rr + 3), w)
        j0, j1 = max(int(jy - rr - 2), 0), min(int(jy + rr + 3), h)
        if i1 <= i0 or j1 <= j0:
            continue
        yy, xx = np.ogrid[j0:j1, i0:i1]
        dd = np.sqrt((xx + 0.5 - jx) ** 2 + (yy + 0.5 - jy) ** 2)
        m[j0:j1, i0:i1] = np.maximum(m[j0:j1, i0:i1], np.clip(rr - dd + 0.5, 0, 1))
    return m


def paint_tree(tod):
    """Returns [back foliage, trunk, mid foliage, front foliage] layers."""
    T = palette.TOD[tod]
    x0, y0, x1, y1, k, _ = SPEC["tree"]
    X, Y = grid("tree")
    h, w = X.shape
    key, amb, fog = T["key"], T["amb"], T["fog"]
    sx, sy = T["sun"]
    # ---- trunk
    tm = _limb_mask((h, w), x0, y0, k, TRUNK, 80)
    for i, lb in enumerate(LIMBS + ROOTS):
        tm = np.maximum(tm, _limb_mask((h, w), x0, y0, k, lb, 81 + i))
    soft = core.gblur(tm, 9 * k)
    gx = np.gradient(soft, axis=1)
    gyy = np.gradient(soft, axis=0)
    side = -(gx * sx + gyy * sy)            # > 0 on the edge facing the sun
    bark_n = core.fbm(h, w, 40 * k, 4, seed=83, aspect=0.22)
    bark_f = core.fbm(h, w, 9 * k, 3, seed=84, aspect=0.3)
    lt = smoothstep(0.002, 0.012, side + 0.012 * (bark_n - 0.5))
    under = smoothstep(900, 640, Y)          # shaded under the canopy
    lt = lt * (1 - 0.7 * under) * (0.2 + 0.8 * look(tod)[0])
    bark = np.float32((0.50, 0.36, 0.28)) * (0.78 + 0.44 * bark_n)[..., None]
    bark = mixc(bark, np.float32((0.40, 0.46, 0.26)), smoothstep(0.62, 0.8, core.fbm(h, w, 70 * k, 3, seed=85)) * 0.5)
    b_sh = mixc(bark * np.float32((0.70, 0.74, 1.10)) * amb * 0.80, palette.shade((0.24, 0.20, 0.36), T), 0.35 + 0.3 * under)
    b_lit = np.clip(bark * key * 1.12, 0, 1)
    tref = mixc(b_sh, b_lit, lt)
    tref = tref * (0.86 + 0.28 * bark_f)[..., None]
    # bounce light on the shadow edge (cool sky)
    back = smoothstep(0.004, 0.014, -side) * (1 - under) * 0.35
    tref = mixc(tref, np.clip(palette.sky_color(T, 0.25) * 0.75, 0, 1), back) * gain(tod)
    tdir = paint.flow_from(np.repeat(soft[..., None], 3, 2), 4.0)
    trunk = paint.paint(np.clip(tref, 0, 1), tm, direction=tdir, seed=paint_seed("trunk", tod), passes=paint.FINE, scale=1.0, spill=0.0, grain=0.05)
    # bark strokes: dark furrows and a lit edge
    r = core.rng("bark", 3)
    ys_, xs_ = np.nonzero(tm > 0.9)
    pick = r.integers(0, len(xs_), 2600)
    bxs, bys = xs_[pick].astype(np.float64), ys_[pick].astype(np.float64)
    ltp = lt[ys_[pick], xs_[pick]]
    trgb = trunk[..., :3] / np.maximum(trunk[..., 3:4], 1e-4)
    bc = trgb[ys_[pick], xs_[pick]] * np.where(r.random(len(pick)) < 0.6, 0.62, 1.32)[:, None]
    paint.strokes(trunk, bxs, bys, 0.0, 2.0 + 2.5 * r.random(len(pick)), 14 + 40 * r.random(len(pick)), np.clip(bc, 0, 1),
                  op=0.75, seed=86, dry=0.5, taper=0.6, field=tdir, paper_k=0.6)
    trunk[..., :3] = np.where(tm[..., None] > 0.02, trunk[..., :3], 0) * 1.0
    trunk *= (tm / np.maximum(trunk[..., 3], 1e-4))[..., None].clip(0, 1)
    # ---- foliage sub layers
    r = core.rng("foliage", 9)
    Lz = 0.42
    ln = math.sqrt(sx * sx + sy * sy + Lz * Lz)
    lx, ly, lz = sx / ln, sy / ln, Lz / ln
    leaf_l = np.float32((0.45, 0.63, 0.23))
    leaf_m = np.float32((0.34, 0.54, 0.24))
    leaf_d = np.float32((0.22, 0.42, 0.34))
    c_hi = np.clip(leaf_l * key * 1.16 + np.float32((0.06, 0.05, 0.0)), 0, 1)
    c_li = np.clip(leaf_l * key, 0, 1)
    c_mid = np.clip(leaf_m * (0.55 * key + 0.5 * amb), 0, 1)
    c_dk = palette.shade(leaf_d, T)
    c_deep = palette.shade((0.15, 0.24, 0.36), T) * 0.86
    direct, sat = look(tod)
    gn = gain(tod)
    c_dk, c_deep = c_dk * gn, c_deep * gn
    c_mid = desat(c_dk + (c_mid * gn - c_dk) * (0.45 + 0.55 * direct), sat)
    c_li = desat(c_mid + (c_li * gn - c_mid) * direct, sat)
    c_hi = desat(c_mid + (c_hi * gn - c_mid) * direct, sat)
    hi_a, hi_b, hi_dab = 0.70, 0.82, 0.44
    if direct < 1.0:   # night and the blue hour: cooler, and only a thin rim on the tops of the clumps
        cool = np.float32((0.80, 1.0, 1.20))
        c_mid, c_li = c_mid * cool, c_li * cool
        c_hi = c_li + (rim_col(tod, T) - c_li) * 0.45
        hi_a, hi_b, hi_dab = 0.90, 0.97, 0.92
    layers = []
    # how much each point would catch a fire burning low on the right (launch)
    fmap = np.zeros((h, w), np.float32)
    tfire = smoothstep(0.002, 0.012, -(gx * 0.6 + gyy * 0.75)) * (0.35 + 0.65 * smoothstep(640, 900, Y))
    for z in (0, 1, 2):
        if z == 1:
            fmap = fmap * (1 - tm) + tfire * tm
        Hf = np.zeros((h, w), np.float32)
        al = np.zeros((h, w), np.float32)
        for mi, (mx, my, mr, mz) in enumerate(MASSES):
            if mz != z:
                continue
            ncl = 7 if z > 0 else 4
            for q in range(ncl):
                if q == 0:
                    cx, cy, cr = mx, my, mr * 0.72
                else:
                    a_ = r.uniform(0, 6.283)
                    dd = mr * r.uniform(0.35, 0.72)
                    cx, cy = mx + math.cos(a_) * dd, my + math.sin(a_) * dd * 0.8
                    cr = mr * r.uniform(0.34, 0.56)
                jx, jy, rr = (cx - x0) * k, (cy - y0) * k, cr * k
                mgn = int(rr * 1.35) + 3
                i0, i1 = max(int(jx) - mgn, 0), min(int(jx) + mgn, w)
                j0, j1 = max(int(jy) - mgn, 0), min(int(jy) + mgn, h)
                if i1 <= i0 or j1 <= j0:
                    continue
                yy, xx = np.mgrid[j0:j1, i0:i1].astype(np.float32)
                u, v = (xx - jx) / rr, (yy - jy) / (rr * 0.86)
                an = np.arctan2(v, u)
                ph = r.uniform(0, 6.28)
                edge = 1.0 + 0.15 * np.sin(an * 3 + ph) + 0.09 * np.sin(an * 7 + ph * 1.7) + 0.03 * np.sin(an * 13 + ph * 0.6)
                rad = np.sqrt(u * u + v * v) / edge
                dome = np.sqrt(np.clip(1 - rad * rad, 0, 1)) * rr * 0.9 + (q == 0) * rr * 0.1
                Hf[j0:j1, i0:i1] = np.maximum(Hf[j0:j1, i0:i1], np.where(rad < 1, dome + 6.0 * z, 0))
                al[j0:j1, i0:i1] = np.maximum(al[j0:j1, i0:i1], np.clip((1 - rad) * rr / 2.0, 0, 1))
        # leaf scale relief and sky holes
        leafn = core.fbm(h, w, 38 * k, 3, seed=90 + z)
        leaff = core.fbm(h, w, 8 * k, 3, seed=93 + z)
        holes = core.fbm(h, w, 60 * k, 4, seed=96 + z)
        depth = Hf / (np.maximum(core.gblur(Hf, 40 * k), 1.0) + 40.0)
        hole = smoothstep(0.47, 0.41, holes + 0.22 * np.clip(Hf / (90.0 * k), 0, 1)) if z > 0 else smoothstep(0.44, 0.38, holes)
        al = al * (1 - hole)
        Hs = core.gblur(Hf + 20.0 * k * (leafn - 0.5) + 4.0 * k * (leaff - 0.5), 3.0 * k)
        nx = -np.gradient(Hs, axis=1)
        ny = -np.gradient(Hs, axis=0)
        nz = 1.0
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        lam = (nx * lx + ny * ly + nz * lz) / nn
        lam = lam + 0.10 * (leaff - 0.5)
        # undersides: canopy with open air just below it
        below_al = core.gblur(np.roll(al, -int(34 * k), axis=0), 12 * k)
        und = np.clip(1.0 - below_al, 0, 1) ** 1.5 * (0.55 + 0.9 * leaff)
        fmap = fmap * (1 - al) + np.clip(und, 0, 1) * (0.5 if z == 0 else 1.0) * al
        # occlusion: undersides and the interior away from the sun
        occ = core.gblur(al, 34 * k)
        occ_dir = np.clip(core.gblur(np.roll(np.roll(al, int(-sx * 40 * k), 1), int(-sy * 40 * k), 0), 22 * k), 0, 1)
        lam = lam - 0.35 * occ_dir * occ + 0.12
        lam = lam - (0.36, 0.10, 0.0)[z]
        t_hi = smoothstep(hi_a, hi_b, lam)
        t_li = smoothstep(0.44, 0.58, lam)
        t_mid = smoothstep(-0.10, 0.12, lam)
        t_dk = smoothstep(-0.46, -0.26, lam)
        col = mixc(c_deep, c_dk, t_dk)
        col = mixc(col, c_mid, t_mid)
        col = mixc(col, c_li, t_li)
        col = mixc(col, c_hi, t_hi)
        if z == 0:
            col = mixc(col, fog, np.full((h, w), 0.10, np.float32))
        # translucent glow where the sun comes through thin edges
        thin = smoothstep(0.75, 0.35, occ) * smoothstep(0.0, 0.4, -(np.gradient(core.gblur(al, 12 * k), axis=1) * sx)) * 0.0
        ref = np.clip(col, 0, 1).astype(np.float32)
        fdir = paint.flow_from(np.repeat(Hs[..., None] / (Hs.max() + 1e-6), 3, 2), 5.0)
        passes = [dict(p) for p in paint.PASSES[1:]]
        passes[0]["keep"] = 0.45
        fo = paint.paint(ref, al, direction=fdir, seed=paint_seed("fol%d" % z, tod), passes=passes, scale=0.9, spill=7 * k, grain=0.05)
        # leaf dabs: clusters of single leaves on the lit tops and along the silhouette
        ys_, xs_ = np.nonzero(al > 0.5)
        if len(xs_):
            nd = 9000 if z > 0 else 3500
            pick = r.integers(0, len(xs_), nd)
            lx_, ly_ = xs_[pick].astype(np.float64), ys_[pick].astype(np.float64)
            lam_p = lam[ys_[pick], xs_[pick]]
            occ_p = occ[ys_[pick], xs_[pick]]
            u_ = r.random(nd)
            colp = np.where((lam_p > hi_dab)[:, None], mixc(c_li, c_hi, u_),
                            np.where((lam_p > 0.15)[:, None], mixc(c_mid, c_li, u_ * 0.55), mixc(c_deep, c_dk * 1.25, u_)))
            if z == 0:
                colp = mixc(colp, np.broadcast_to(c_dk, colp.shape), np.full(nd, 0.5, np.float32))
            edge_p = occ_p < 0.62
            keepd = (r.random(nd) < 0.30) | edge_p
            paint.strokes(fo, lx_[keepd] + r.normal(0, 3, keepd.sum()), ly_[keepd] + r.normal(0, 3, keepd.sum()), r.uniform(0, 6.283, keepd.sum()),
                          (4.0 + 3.5 * r.random(keepd.sum())) * k * 0.8, (4.5 + 5.5 * r.random(keepd.sum())) * k * 0.8, np.clip(colp[keepd], 0, 1),
                          op=0.90, bend=r.normal(0, 0.4, keepd.sum()), seed=97 + z, dry=0.1, taper=0.25, streak=0.06, paper_k=0.25)
        layers.append(fo)
    return [layers[0], trunk, layers[1], layers[2]], fmap.astype(np.float32)
