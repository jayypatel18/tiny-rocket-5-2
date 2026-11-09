"""Fireflies and drifting pollen. Each one has its own slow wander, its own
blink rhythm and its own depth (so they shift against each other when the
camera moves). In focus they are a tiny hot core with a soft halo; out of
focus they open into bokeh discs with a brighter rim.

    fireflies.draw(pic, cam, t, n=40, region=(x0, x1, y0, y1), seed=0, ...)   # adds light in place
"""
import math

import numpy as np

from . import core

_cat = {}


def _catalogue(n, region, seed, p_range):
    key = (n, region, seed, p_range)
    if key not in _cat:
        r = core.rng("fireflies", seed, n)
        x0, x1, y0, y1 = region
        _cat[key] = dict(
            X=r.uniform(x0, x1, n), Y=r.uniform(y0, y1, n) , p=r.uniform(p_range[0], p_range[1], n),
            period=r.uniform(1.8, 4.2, n), phase=r.uniform(0, 1, n), size=r.uniform(0.7, 1.4, n),
            wx=r.uniform(50, 130, n), wy=r.uniform(25, 70, n), sp=r.uniform(0.12, 0.3, n),
            hue=r.uniform(0, 1, n), duty=r.uniform(0.35, 0.6, n))
    return _cat[key]


def _splat(buf, x, y, rad, col, amp, disc):
    R = int(rad * (1.3 if disc else 3.0)) + 2
    x0, x1 = int(x) - R, int(x) + R + 1
    y0, y1 = int(y) - R, int(y) + R + 1
    X0, X1, Y0, Y1 = max(x0, 0), min(x1, core.W), max(y0, 0), min(y1, core.H)
    if X1 <= X0 or Y1 <= Y0:
        return
    yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
    d = np.hypot(xx - x, yy - y)
    if disc:
        a = np.clip((rad - d) / 1.5, 0, 1) * (0.55 + 0.45 * np.clip(d / rad, 0, 1) ** 3)
    else:
        a = np.exp(-(d / (rad * 0.45)) ** 2) + 0.35 * np.exp(-(d / (rad * 1.4)) ** 2)
    buf[Y0:Y1, X0:X1] += a[..., None] * (col * amp)


def draw(pic, cam, t, n=40, region=(900, 2300, 930, 1260), seed=0, color=(0.80, 1.0, 0.42), amount=1.0,
         size=1.0, p_range=(0.85, 1.7), bokeh=0.0, focus_p=None, blink=True, rise=0.0, twinkle=0.0):
    """Add fireflies (or pollen with blink=False) to `pic` in place.
    region is in SET coordinates. bokeh = disc radius in px for every mote, or
    pass focus_p to defocus by depth (motes far from focus_p open up).
    rise: px per second of upward drift (pollen floats up and wraps)."""
    c = _catalogue(n, region, seed, p_range)
    col0 = np.asarray(color, np.float32)
    buf = np.zeros((core.H, core.W, 3), np.float32)
    y0, y1 = region[2], region[3]
    for i in range(n):
        ph = t * c["sp"][i]
        X = c["X"][i] + core.noise1(ph + i * 3.1, 11 + seed) * c["wx"][i] + core.noise1(ph * 2.7 + i, 12 + seed) * 18
        Y = c["Y"][i] + core.noise1(ph * 1.1 + i * 1.7, 13 + seed) * c["wy"][i]
        if rise:
            Y = y0 + (Y - y0 - rise * t * (0.6 + 0.8 * c["hue"][i])) % (y1 - y0)
        p = c["p"][i]
        sx, sy = cam.pt(X, Y, p)
        if sx < -60 or sx > core.W + 60 or sy < -60 or sy > core.H + 60:
            continue
        if blink:
            k = ((t / c["period"][i] + c["phase"][i]) % 1.0) / c["duty"][i]
            b = math.sin(math.pi * k) ** 1.5 if k < 1.0 else 0.0
        else:
            b = 0.55 + 0.45 * math.sin(t * 1.3 * (0.5 + c["hue"][i]) + i) * twinkle + 0.0
        if b <= 0.02:
            continue
        sc = cam.xf(p)[0]
        col = col0 * (0.85 + 0.3 * c["hue"][i]) + np.array([0.12, 0.0, -0.05], np.float32) * (c["hue"][i] - 0.5)
        rad_b = bokeh
        if focus_p is not None:
            rad_b = max(bokeh, 26.0 * abs(1.0 / p - 1.0 / focus_p) * sc)
        if rad_b > 2.5:
            area = max(rad_b / 6.0, 1.0)
            _splat(buf, sx, sy, rad_b * c["size"][i], col, amount * b * 0.5 / area ** 0.6, True)
        else:
            _splat(buf, sx, sy, 7.0 * size * c["size"][i] * max(sc, 0.6), col, amount * b, False)
    core.add_light(pic, np.clip(buf, 0, 1))
    return pic
