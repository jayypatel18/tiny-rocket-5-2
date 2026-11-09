"""Shared pieces for the panda close up cuts (s1, s4b, s5a, s5c, s7b): cached
defocused backgrounds, the rack focus on a cel layer, the bamboo leg with its
lashing, string strands, and small acting overlays (blink, flick)."""
import math
import os

import numpy as np

from . import core, palette

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "build", "cache", "panda")
_mem = {}


def cached(name, fn):
    """A static painted array: memory, then disk (atomic write), then built by fn()."""
    if name in _mem:
        return _mem[name]
    path = os.path.join(CACHE, name + ".npy")
    if os.path.exists(path):
        try:
            _mem[name] = np.load(path).astype(np.float32)
            return _mem[name]
        except Exception:  # noqa: a half written file from another worker: rebuild
            pass
    arr = fn()
    os.makedirs(CACHE, exist_ok=True)
    tmp = path + ".%d.tmp.npy" % os.getpid()
    np.save(tmp, arr.astype(np.float16))
    os.replace(tmp, path)
    _mem[name] = arr.astype(np.float32)
    return _mem[name]


_grid = None


def grid():
    global _grid
    if _grid is None:
        yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
        _grid = (xx, yy)
    return _grid


def radial(cx, cy, r_in, r_out):
    """1 inside r_in, 0 outside r_out, smooth between."""
    xx, yy = grid()
    d = np.hypot(xx - cx, yy - cy)
    u = np.clip((r_out - d) / max(r_out - r_in, 1e-3), 0, 1)
    return u * u * (3 - 2 * u)


def rack(layer, centre, r, sig_far, sig_near):
    """Depth of field on one cel layer: the disc around `centre` (the near
    plane: paws, knot) gets sig_near, everything else sig_far."""
    if sig_far < 0.35 and sig_near < 0.35:
        return layer
    m = radial(centre[0], centre[1], r * 0.55, r * 1.25)[..., None]
    a = core.gblur(layer, sig_far) if sig_far >= 0.35 else layer
    b = core.gblur(layer, sig_near) if sig_near >= 0.35 else layer
    return a * (1 - m) + b * m


def glow(pic, cx, cy, r, color, amount, squash=1.0):
    xx, yy = grid()
    d2 = ((xx - cx) / r) ** 2 + ((yy - cy) / (r * squash)) ** 2
    core.add_light(pic, np.exp(-d2).astype(np.float32), color, amount)


def bump(t, t0, up=0.06, down=0.14):
    """A quick out and back: 0 before t0, 1 at t0 + up, 0 again at t0 + up + down."""
    if t <= t0 or t >= t0 + up + down:
        return 0.0
    if t < t0 + up:
        return core.ease_out((t - t0) / up, 2.0)
    return 1.0 - core.smooth((t - t0 - up) / down)


def blink(t, times, hold=0.09):
    """Eye openness multiplier: closes fast, holds, opens a little slower."""
    k = 1.0
    for t0 in times:
        if t0 - 0.045 < t < t0 + hold + 0.09:
            if t < t0:
                k = min(k, 1 - core.smooth((t - (t0 - 0.045)) / 0.045))
            elif t < t0 + hold:
                k = 0.0
            else:
                k = min(k, core.smooth((t - t0 - hold) / 0.09))
    return k


# ----------------------------------------------------------------- the props
def draw_strut(cel, knot, d, s, tod, drawing=0, loops=5, spread=2.4, wide=4.7, back=120.0, fwd=110.0):
    """A bamboo leg of the rocket through screen point `knot`, running along the
    unit vector d, with the string lashing wound round it at the knot."""
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    ink = T["ink"]
    sun = np.asarray(T["sun"], np.float64)
    bam_l = palette.mix(palette.lit(palette.ROCKET["bamboo"], T), palette.lit((0.80, 0.70, 0.46), T), 0.45)
    bam_d = palette.mix(palette.shade(palette.ROCKET["bamboo"], T), palette.shade((0.62, 0.50, 0.36), T), 0.4)
    knot = np.asarray(knot, np.float64)
    d = np.asarray(d, np.float64) / np.hypot(*d)
    nv = np.array([-d[1], d[0]])
    a, b = knot - d * back * s, knot + d * fwd * s
    sd = core.seed_of("strut", int(drawing)) % 9000
    # the cane swells a little at each node
    us = np.linspace(0, 1, 60)
    nodes = (0.13, 0.42, 0.70, 0.93)
    wdt = wide * s * (1 + 0.10 * sum(np.exp(-((us - q) / 0.012) ** 2) for q in nodes))
    cl = a[None, :] + (b - a)[None, :] * us[:, None]
    L = core.wobble(cl + nv * wdt[:, None], 0.35 * s ** 0.5, sd, 0.02)
    R = core.wobble(cl - nv * wdt[:, None], 0.35 * s ** 0.5, sd, 0.02)
    pole = np.vstack([L, R[::-1]])
    m = cel.mask(pole)
    cel.fill(pole, bam_l)
    side = 1.0 if (nv @ sun) < 0 else -1.0          # the side away from the light
    cel.fill(np.vstack([cl + nv * wdt[:, None] * side * 0.25, (cl + nv * wdt[:, None] * side * 1.2)[::-1]]), bam_d, clip=m)
    hl = palette.rim(palette.ROCKET["bamboo"], T)
    cel.fill(np.vstack([cl - nv * wdt[:, None] * side * 0.62, (cl - nv * wdt[:, None] * side * 0.84)[::-1]]), hl, clip=m)
    lw = 1.5 * s ** 0.55
    cel.stroke(L, lw * (2.0 if side > 0 else 1.2), ink, seed=sd + 1, taper=(0.0, 0.0))
    cel.stroke(R, lw * (2.0 if side < 0 else 1.2), ink, seed=sd + 2, taper=(0.0, 0.0))
    for q in nodes:
        c = a + (b - a) * q
        w_ = wide * s * 1.1
        ring = np.array([c + nv * w_, c + d * 1.4 * s + nv * w_ * 0.3, c + d * 1.4 * s - nv * w_ * 0.3, c - nv * w_])
        cel.stroke(core.catmull(ring, 6, closed=False), lw * 0.9, ink, seed=sd + 3)
    # the lashing: loops of string wound round the cane
    strc = palette.lit(palette.ROCKET["string"], T)
    strd = palette.shade(palette.ROCKET["string"], T)
    for j in range(loops):
        c = knot + d * (j - (loops - 1) / 2) * spread * s
        seg = np.array([c + nv * wide * s * 1.10 + d * 0.5 * s, c + d * 1.0 * s, c - nv * wide * s * 1.10 + d * 0.3 * s])
        seg = core.catmull(seg, 6, closed=False)
        cel.stroke(seg, 0.26 * s * 6.5 ** 0.0 * 3.2, ink, taper=(0.0, 0.0))
        cel.stroke(seg, 0.26 * s * 2.0, strc, taper=(0.0, 0.0))
        cel.stroke(seg + nv * 0.0 + d * 0.22 * s, 0.26 * s * 0.6, strd, taper=(0.3, 0.3))
    # the knot itself: a small bow of two loops on the near face
    for sg in (-1, 1):
        lp = core.catmull([knot, knot + d * sg * 2.2 * s + nv * 1.6 * s, knot + d * sg * 3.4 * s - nv * 0.4 * s,
                           knot + d * sg * 1.4 * s - nv * 1.2 * s], 7, closed=True)
        cel.stroke(lp, 0.26 * s * 3.0, ink, closed=True)
        cel.stroke(lp, 0.26 * s * 1.8, strc, closed=True)


def draw_string(cel, pts, s, tod, curve=True):
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    p = np.asarray(pts, np.float64)
    if curve and len(p) > 2:
        p = core.catmull(p, 10, closed=False)
    cel.stroke(p, 0.26 * s * 3.2, T["ink"], taper=(0.0, 0.03))
    cel.stroke(p, 0.26 * s * 2.0, palette.lit(palette.ROCKET["string"], T), taper=(0.0, 0.03))


def pendulum(knot, v0, tau, damp=3.0, freq=6.5, lag=0.045, stiff=0.5):
    """A strand let go at tau = 0: it hung from `knot` along v0, now it swings
    down under gravity and settles. Returns three points for draw_string."""
    L = float(np.hypot(*v0))
    th0 = math.atan2(v0[0], v0[1])                 # angle from straight down

    def th(x):
        x = max(x, 0.0)
        return th0 * math.exp(-damp * x) * math.cos(freq * x)

    a1, a2 = th(tau), th(tau - lag)
    mid = knot + stiff * L * np.array([math.sin(a1), math.cos(a1)])
    end = mid + (1 - stiff) * L * np.array([math.sin(a2), math.cos(a2)])
    return np.array([knot, mid, end])
