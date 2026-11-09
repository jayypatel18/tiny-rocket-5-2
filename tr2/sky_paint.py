"""Brush and cloud helpers shared by sky.py, cloudset.py and vista.py.

  stroke_field   a field of overlapping brush strokes with bristle streaks
  repaint        repaint a picture with strokes that sample its own colours
  cloud_fields   one painted cloud mass as fields (alpha, light, underside),
                 lit from the upper RIGHT; flip it for a sun on the left
  cloud_sprite   the cached version of cloud_fields
  tones, colourise   turn cloud fields into a premultiplied RGBA layer in a light
"""
import math
import os

import numpy as np
from numba import njit, prange
from scipy import ndimage

from tr2 import core, palette

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "build", "cache", "sky")
os.makedirs(CACHE, exist_ok=True)
VERSION = 4


def cached(name, make):
    """Load build/cache/sky/<name>.npy or build it with make() and store it."""
    p = os.path.join(CACHE, name + ".npy")
    if os.path.exists(p):
        return np.load(p)
    a = make()
    np.save(p, a)
    return a


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------ brush strokes
@njit(cache=True, fastmath=True)
def _strokes(out, xs, ys, ang, ln, wd, val, brs, opacity):
    h, w = out.shape
    nb = brs.shape[0]
    for k in range(xs.shape[0]):
        c = math.cos(ang[k])
        s = math.sin(ang[k])
        L = ln[k] * 0.5
        Wd = wd[k] * 0.5
        ex = abs(c) * L + abs(s) * Wd + 2
        ey = abs(s) * L + abs(c) * Wd + 2
        x0 = max(0, int(xs[k] - ex))
        x1 = min(w, int(xs[k] + ex) + 1)
        y0 = max(0, int(ys[k] - ey))
        y1 = min(h, int(ys[k] + ey) + 1)
        for j in range(y0, y1):
            for i in range(x0, x1):
                dx = i - xs[k]
                dy = j - ys[k]
                u = dx * c + dy * s
                v = -dx * s + dy * c
                if abs(u) > L or abs(v) > Wd:
                    continue
                b = brs[(k * 131 + int((v + Wd) / 1.7)) % nb]
                # ragged ends: every bristle lifts at a different place
                en = (1.0 - abs(u) / L) * (2.5 + 5.0 * b)
                if en > 1.0:
                    en = 1.0
                ed = (1.0 - abs(v) / Wd) * 3.0
                if ed > 1.0:
                    ed = 1.0
                al = en * ed * (0.55 + 0.45 * b) * opacity
                out[j, i] = out[j, i] * (1 - al) + val[k] * al


def stroke_field(h, w, n, length, width, angle=0.0, spread=0.15, seed=0, opacity=0.9, base=0.0):
    """h x w field of n overlapping strokes, each carrying a value in -1..1.
    length and width are (lo, hi) in px. Returns float32 roughly -1..1."""
    r = core.rng("strokes", seed, h, w, n)
    out = np.full((h, w), base, np.float32)
    xs = r.uniform(-50, w + 50, n)
    ys = r.uniform(-20, h + 20, n)
    ang = angle + r.normal(0, spread, n)
    ln = r.uniform(length[0], length[1], n)
    wd = r.uniform(width[0], width[1], n)
    val = r.uniform(-1, 1, n)
    brs = r.uniform(0, 1, 4099) ** 0.7
    _strokes(out, xs, ys, ang, ln, wd, val, brs, opacity)
    return out


@njit(cache=True, fastmath=True)
def _paint(dst, src, xs, ys, ang, ln, wd, jit, brs, opacity):
    h, w, ch = dst.shape
    nb = brs.shape[0]
    for k in range(xs.shape[0]):
        cx = int(xs[k])
        cy = int(ys[k])
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        c0 = src[cy, cx, 0] + jit[k, 0]
        c1 = src[cy, cx, 1] + jit[k, 1]
        c2 = src[cy, cx, 2] + jit[k, 2]
        c = math.cos(ang[k])
        s = math.sin(ang[k])
        L = ln[k] * 0.5
        Wd = wd[k] * 0.5
        ex = abs(c) * L + abs(s) * Wd + 2
        ey = abs(s) * L + abs(c) * Wd + 2
        x0 = max(0, int(xs[k] - ex))
        x1 = min(w, int(xs[k] + ex) + 1)
        y0 = max(0, int(ys[k] - ey))
        y1 = min(h, int(ys[k] + ey) + 1)
        for j in range(y0, y1):
            for i in range(x0, x1):
                dx = i - xs[k]
                dy = j - ys[k]
                u = dx * c + dy * s
                v = -dx * s + dy * c
                if abs(u) > L or abs(v) > Wd:
                    continue
                b = brs[(k * 131 + int((v + Wd) / 1.7)) % nb]
                en = (1.0 - abs(u) / L) * (2.5 + 5.0 * b)
                if en > 1.0:
                    en = 1.0
                ed = (1.0 - abs(v) / Wd) * 3.0
                if ed > 1.0:
                    ed = 1.0
                al = en * ed * (0.6 + 0.4 * b) * opacity
                dst[j, i, 0] = dst[j, i, 0] * (1 - al) + c0 * al
                dst[j, i, 1] = dst[j, i, 1] * (1 - al) + c1 * al
                dst[j, i, 2] = dst[j, i, 2] * (1 - al) + c2 * al


def repaint(img, n, length, width, angle=0.0, spread=0.3, seed=0, opacity=0.85, jitter=0.02,
            angle_field=None, mask=None):
    """Lay n brush strokes over a copy of img; each stroke takes the colour
    under its centre (plus a little jitter). angle_field (h x w, radians) makes
    strokes follow the form. mask (h x w, 0..1) is where strokes may start."""
    h, w = img.shape[:2]
    r = core.rng("repaint", seed, h, w, n)
    xs = r.uniform(0, w, n)
    ys = r.uniform(0, h, n)
    if mask is not None:
        keep = r.uniform(0, 1, n) < mask[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)]
        xs, ys = xs[keep], ys[keep]
        n = len(xs)
    if angle_field is not None:
        ang = angle_field[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)].astype(np.float64)
        ang = ang + r.normal(0, spread, n)
    else:
        ang = angle + r.normal(0, spread, n)
    ln = r.uniform(length[0], length[1], n)
    wd = r.uniform(width[0], width[1], n)
    jit = (r.normal(0, jitter, (n, 1)) + r.normal(0, jitter * 0.5, (n, 3))).astype(np.float32)
    brs = r.uniform(0, 1, 4099) ** 0.7
    dst = np.ascontiguousarray(img, np.float32).copy()
    _paint(dst, np.ascontiguousarray(img, np.float32), xs, ys, ang, ln, wd, jit, brs, opacity)
    return dst


# ------------------------------------------------------------------- clouds
def _shift(a, dx, dy):
    """a moved by integer (dx, dy), zero filled."""
    out = np.zeros_like(a)
    h, w = a.shape
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def _puffs(r, w, h, kind):
    """Cloud anatomy: big lumps on a flat base, smaller lumps budding from
    their faces and upper sides, smaller again on those.
    Returns rows (x, y, radius, xstretch, z offset toward the viewer)."""
    puffs = []
    if kind == "cumulus":
        nm = int(r.integers(4, 8))
        xs = np.sort(r.uniform(0.0, 1.0, nm))
        peak = r.uniform(0.3, 0.7)
        for x in xs:
            env = max(0.0, 1 - ((x - peak) / 0.6) ** 2)
            R = (0.38 + 0.95 * env) * r.uniform(0.8, 1.15)
            puffs.append((x * 5.2, -R * 0.72, R, 1.0, 0.0))
        counts, stretch = (int(r.integers(5, 8)), int(r.integers(4, 6))), 1.0
    elif kind == "tower":
        nm = int(r.integers(3, 5))
        for k in range(nm):
            R = r.uniform(0.7, 1.15)
            puffs.append((k * 1.25 + r.uniform(-0.2, 0.2), -R * 0.72, R, 1.0, 0.0))
        top = puffs[int(r.integers(0, nm))]
        y = top[1]
        x = top[0]
        for k in range(int(r.integers(2, 4))):
            R = r.uniform(0.75, 1.0) * (0.88 ** k)
            y -= R * 0.95
            x += r.uniform(-0.4, 0.4)
            puffs.append((x, y, R, 1.0, 0.0))
        counts, stretch = (int(r.integers(5, 8)), 5), 1.0
    else:  # bar: a long flat bank
        nm = int(r.integers(9, 15))
        x = 0.0
        for k in range(nm):
            env = math.sin(math.pi * (k + 0.5) / nm) ** 0.6
            R = (0.35 + 0.5 * env) * r.uniform(0.7, 1.2)
            puffs.append((x, -R * 0.6, R, 2.3, 0.0))
            x += R * r.uniform(1.5, 2.6)
        counts, stretch = (int(r.integers(3, 5)), 3), 1.8
    lvl = list(puffs)
    for depth, cnt in enumerate(counts):
        new = []
        for (x, y, R, st, z0) in lvl:
            for q in range(cnt):
                if q % 3 != 2:      # on the upper outline
                    a = r.uniform(-2.9, -0.25)
                    d = r.uniform(0.62, 0.95)
                else:               # on the face, toward the viewer
                    a = r.uniform(-math.pi, math.pi * 0.25)
                    d = r.uniform(0.15, 0.7)
                rr = R * r.uniform(0.30, 0.56)
                zc = z0 + R * math.sqrt(max(0.0, 1 - d * d)) * 0.9 - rr * 0.45
                new.append((x + math.cos(a) * d * R * st, y + math.sin(a) * d * R, rr, stretch, zc))
        puffs += new
        lvl = new
    p = np.array(puffs, np.float64)
    # fit into the sprite with a margin; base line at 0.84 h
    x0 = (p[:, 0] - p[:, 2] * p[:, 3]).min()
    x1 = (p[:, 0] + p[:, 2] * p[:, 3]).max()
    y0 = (p[:, 1] - p[:, 2]).min()
    m = 0.06 * w
    base = 0.84 * h
    sc = min((w - 2 * m) / (x1 - x0), (base - 0.07 * h) / (0 - y0))
    p[:, 0] = (p[:, 0] - (x0 + x1) / 2) * sc + w / 2
    p[:, 1] = p[:, 1] * sc + base
    p[:, 2] *= sc
    p[:, 4] *= sc
    return p, base


SIZES = {"cumulus": (960, 540), "tower": (720, 760), "bar": (1400, 380), "wisp": (1300, 300)}


def cloud_fields(kind, seed, z=0.0):
    """Returns float32 h x w x 3: alpha, light (0 deep shadow .. 1 sunlit top),
    underside (0..1). Light comes from the upper right. z shifts the edge
    noise a little (two z values give two drawings of the same cloud)."""
    w, h = SIZES[kind]
    r = core.rng("cloud", kind, seed)
    sd = core.seed_of("cloudn", kind, seed) % 100000
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "wisp":
        # thin high cloud: combed streaks
        n1 = core.fbm(h, w, 70, 5, sd + 1, aspect=7.0, z=z)
        n2 = core.fbm(h, w, 22, 3, sd + 2, aspect=9.0, z=z)
        rid = 1 - np.abs(2 * n1 - 1)
        env = np.exp(-((yy - h * 0.5) / (h * 0.24)) ** 2) * ss(0.0, 0.25, xx / w) * ss(0.0, 0.25, 1 - xx / w)
        tilt = (xx / w - 0.5) * r.uniform(-0.5, 0.5) * h * 0.4
        env = np.exp(-((yy - h * 0.5 - tilt) / (h * 0.2)) ** 2) * ss(0.0, 0.3, xx / w) * ss(0.0, 0.3, 1 - xx / w)
        a = ss(0.42, 0.9, rid * (0.6 + 0.6 * n2) * (0.35 + env)) * env
        a = np.clip(a * 1.25, 0, 1) ** 0.9
        light = 0.48 + 0.42 * ss(0.2, 0.9, a) + 0.25 * (n2 - 0.5) - 0.25 * (yy / h - 0.5)
        under = ss(0.45, 0.8, yy / h) * 0.6
        return np.stack([a, np.clip(light, 0, 1), under], -1).astype(np.float32)

    p, base = _puffs(r, w, h, kind)
    # domain warp: billows on the outline at three sizes
    S = h / 540.0 if kind != "bar" else 1.0
    wx = (core.fbm(h, w, 130 * S, 3, sd + 3, z=z) - 0.5) * 44 * S
    wy = (core.fbm(h, w, 130 * S, 3, sd + 4, z=z) - 0.5) * 44 * S
    wx += (core.fbm(h, w, 34 * S, 3, sd + 5, z=z) - 0.5) * 20 * S
    wy += (core.fbm(h, w, 34 * S, 3, sd + 6, z=z) - 0.5) * 20 * S
    wx += (core.fbm(h, w, 9 * S, 2, sd + 7, z=z) - 0.5) * 5 * S
    wy += (core.fbm(h, w, 9 * S, 2, sd + 8, z=z) - 0.5) * 5 * S
    X = xx + wx
    Y = yy + wy
    zf = np.zeros((h, w), np.float32)
    for (cx, cy, R, st, z0) in p:
        x0 = max(0, int(cx - R * st - 50))
        x1 = min(w, int(cx + R * st + 50))
        y0 = max(0, int(cy - R - 50))
        y1 = min(h, int(cy + R + 50))
        if x1 <= x0 or y1 <= y0:
            continue
        d2 = ((X[y0:y1, x0:x1] - cx) / st) ** 2 + (Y[y0:y1, x0:x1] - cy) ** 2
        zz = np.sqrt(np.maximum(R * R - d2, 0.0))
        zz = np.where(zz > 0, zz + z0, 0.0)
        np.maximum(zf[y0:y1, x0:x1], zz, out=zf[y0:y1, x0:x1])
    # flat, slightly ragged base
    bw = (core.fbm(h, w, 150, 4, sd + 9, aspect=2.5) - 0.5) * 70 * S
    cut = 1 - ss(base - 30 * S + bw, base + 12 * S + bw, yy)
    zf *= cut
    a = ss(0.5, 5.0 * S, zf)
    # softer, dry brushed edge low on the shadow side
    low = ss(base - 70 * S, base, yy)
    fine = core.fbm(h, w, 7, 2, sd + 10)
    a = a * (1 - low * 0.5 * ss(0.35, 0.75, fine) * (1 - ss(6 * S, 26 * S, zf)))
    # form light from the union of spheres
    # two scales: the big masses carry the light, the small lumps only break it
    lx, ly, lz = 0.60, -0.62, 0.50
    lam = 0.0
    for sg, wgt, k in ((9.0 * S, 0.62, 0.55), (2.6 * S, 0.38, 1.0)):
        zs = ndimage.gaussian_filter(zf, sg)
        gy, gx = np.gradient(zs)
        nrm = np.sqrt(gx * gx + gy * gy + k * k)
        lam = lam + wgt * (-gx * lx - gy * ly + k * lz) / nrm
    lam = np.clip(lam, -0.2, 1.0)
    # how much cloud lies between this point and the sun
    ab = ndimage.gaussian_filter(a, 3.0 * S)
    mass = np.zeros_like(a)
    step = 9.0 * S
    for s in range(1, 15):
        mass += _shift(ab, -int(round(lx * step * s)), -int(round(ly * step * s))) * (1.0 - s / 18.0)
    trans = np.exp(-mass * 0.11)
    under = ss(base - (230 if kind != "bar" else 110) * S, base - 4 * S, yy) ** 1.6
    light = 0.12 + 0.66 * np.clip(lam, 0, 1) * (0.5 + 0.5 * trans) + 0.24 * trans - 0.20 * under * (1 - 0.5 * trans)
    # a thin bounce of light on the very bottom lumps
    light += 0.10 * ss(base - 26 * S, base - 4 * S, yy) * (1 - ss(0, 30 * S, zf))
    # brush: tonal planes get stroke shaped borders
    sf = stroke_field(h, w, int(w * h / 900), (40 * S, 130 * S), (10 * S, 30 * S), angle=-0.5, spread=0.7,
                      seed=sd + 11, opacity=0.8)
    sf = ndimage.gaussian_filter(sf, 0.8)
    light += 0.085 * sf + 0.05 * (fine - 0.5)
    return np.stack([a, np.clip(light, 0, 1), under], -1).astype(np.float32)


def cloud_sprite(kind, seed, z=0.0):
    name = "cloud_v%d_%s_%d_%d" % (VERSION, kind, seed, int(round(z * 100)))
    return cached(name, lambda: cloud_fields(kind, seed, z).astype(np.float16)).astype(np.float32)


def tones(tod, sky_c=None, haze=0.0, strength=1.0):
    """The five cloud tones (deep, shadow, turn, lit, highlight) in this light.
    sky_c: sky colour behind the cloud; haze 0..1 pulls every tone toward it."""
    t = palette.TOD[tod] if isinstance(tod, str) else tod
    key, amb, rimc = t["key"], t["amb"], t["rim"]
    white = np.array([1.0, 0.99, 0.97], np.float32)
    lum = float(key.mean())
    lit = np.clip(white * key / max(1.0, float(key.max())) * min(1.0, 0.55 + 0.6 * lum), 0, 1)
    hi = np.clip(palette.mix(lit, rimc, 0.35) * 1.10 + 0.05 * lum, 0, 1)
    sh = white * amb * (0.86 + 0.5 * lum * 0.3)
    if sky_c is not None:
        sh = palette.mix(sh, sky_c, 0.42)
    g = sh.mean()
    sh = np.clip(g + (sh - g) * 1.25, 0, 1)
    deep = np.clip(sh * np.array([0.83, 0.82, 0.92], np.float32), 0, 1)
    turn = palette.mix(lit, sh, 0.52) * np.array([1.10, 0.93, 0.95], np.float32)
    cols = [deep, sh, np.clip(turn, 0, 1), lit, hi]
    if sky_c is not None and haze > 0:
        cols = [palette.mix(c, sky_c, haze * (0.9 if i < 2 else 0.6)) for i, c in enumerate(cols)]
    if strength < 1.0:
        mid = cols[2]
        cols = [palette.mix(mid, c, strength) for c in cols]
    return [np.asarray(c, np.float32) for c in cols]


def colourise(F, cols, alpha=1.0):
    """Cloud fields (h x w x 3) to a premultiplied RGBA layer in five tones."""
    L = F[..., 1]
    deep, sh, turn, lit, hi = cols
    c = deep + (sh - deep) * ss(0.10, 0.24, L)[..., None]
    c += (turn - c) * ss(0.36, 0.45, L)[..., None]
    c += (lit - c) * ss(0.50, 0.60, L)[..., None]
    c += (hi - c) * ss(0.76, 0.86, L)[..., None]
    return core.premul(c.astype(np.float32), np.clip(F[..., 0] * alpha, 0, 1))
