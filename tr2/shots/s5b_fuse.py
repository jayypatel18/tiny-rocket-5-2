"""Shot 5b: predawn, ground level, macro. The spark head fizzes along the fuse
through the grass toward the rocket's bamboo legs, in hard held steps (it holds
two or three frames, then jumps), spraying sparks, trailing a thread of smoke,
leaving the fuse black behind it. Its warm light flickers on the nearby blades.
The camera tracks on ones, lagging then catching up, and the focus racks from
the spark to the rocket as it nears the legs.
"""
import math

import numpy as np

from .. import core, palette
from ..timeline import FPS
from . import common

NAME = "s5b_fuse"
TOD = "predawn"
X0, X1 = 1436.0, 1664.0          # set x of the spark at the start and the end of the cut
FUSE_A, FUSE_UP, FUSE_B = 1150.0, 1690.0, 1717.0
DUR = 2.5
WARM = np.float32((1.0, 0.70, 0.36))


def fuse_y(X):
    """The fuse lies in the grass just under the crest, then climbs to the nozzle."""
    X = np.asarray(X, np.float64)
    gy = common.ground_y(X)
    y = gy + 5.0 + 2.5 * np.sin(X / 31.0) + 1.5 * np.sin(X / 11.0 + 1.0)
    up = np.clip((X - FUSE_UP) / (FUSE_B - FUSE_UP), 0, 1)
    return y - 61.0 * up * up * (3 - 2 * up)


_starts = None


def _step(fl):
    """(index of the held step, first frame of that step). Holds are 2 or 3 frames."""
    global _starts
    if _starts is None:
        holds = core.rng("fuse steps", 2).integers(2, 4, 80)
        _starts = np.r_[0, np.cumsum(holds)] - 40
    k = int(np.searchsorted(_starts, fl, side="right") - 1)
    k = min(max(k, 0), len(_starts) - 2)
    return k, int(_starts[k])


def _spark_x(k):
    f0 = int(_starts[k])
    return X0 + (X1 - X0) * (f0 / (DUR * FPS)) + 2.0 * (core.rng("fuse jit", k).random() - 0.5)


def camera(t):
    xs = lambda q: X0 + (X1 - X0) * q / DUR
    dx, dy = common.drift(t + 3.0, 1.6, 4)
    cx = core.keys(t, [(-0.5, X0 - 6.0), (0.0, X0 + 30.0), (0.8, xs(0.8) - 10.0), (1.6, xs(1.6) + 30.0), (2.5, xs(2.5) + 52.0), (3.0, xs(2.5) + 60.0)], core.ease_io)
    zoom = core.keys(t, [(0.0, 3.55), (2.5, 3.3)], core.ease_io)
    cy = float(common.ground_y(cx)) - 42.0
    return core.Camera(cx0=1600, cy0=900, cx=cx + dx / zoom, cy=cy + dy / zoom, zoom=zoom)


def _grass_cols(T):
    from .. import hillset_paint as hp
    direct, sat = hp.look(TOD)
    gn = hp.gain(TOD)
    dark = palette.shade((0.22, 0.42, 0.34), T) * gn
    mid = hp.desat(np.clip(np.float32((0.50, 0.64, 0.24)) * T["key"], 0, 1), sat) * gn * np.float32((0.80, 1.0, 1.20))
    mid = dark + (mid - dark) * (0.45 + 0.55 * direct)
    return dark, mid


def _mid_blades(pic, cam, tt, spark, flick, front):
    """Grass on the plane of the fuse: one row behind it, one in front of it."""
    from .. import hillset_paint as hp
    from .. import paint
    T = palette.TOD[TOD]
    s, tx, ty = cam.xf(1.0)
    step = 7.5 if front else 3.0
    salt = 7.0 if front else 3.0
    idx = np.arange(int((-80 - tx) / s // step), int((core.W + 80 - tx) / s // step) + 1)
    h1, h2, h3, h4 = (hp.hashf(idx, salt + q) for q in range(4))
    X = idx * step + (h1 - 0.5) * step
    gy = common.ground_y(X)
    Y = gy + (15.0 + 9.0 * h2 if front else -1.0 + 4.0 * h2)
    length = (20.0 + 34.0 * h3 ** 1.5) if front else (38.0 + 72.0 * h3 ** 1.4)
    wave = tt * 1.6 + X / 90.0 + h1 * 6.283
    ang = -np.pi / 2 + (h4 - 0.5) * 0.8 + 0.07 * np.sin(wave)
    bend = (h2 - 0.5) * 1.3 + 0.12 * np.sin(wave - 0.9)
    dark, mid = _grass_cols(T)
    cols = dark + (mid - dark) * h4[:, None]
    warm = np.clip(np.exp(-((X - spark[0]) / 74.0) ** 2) * flick * 0.9, 0, 1)[:, None]
    cols = cols + (WARM * (0.45 + 0.55 * h3[:, None]) - cols) * warm
    lay = np.zeros((core.H, core.W, 4), np.float32)
    o = np.argsort(Y)
    paint.strokes(lay, X[o] * s + tx, Y[o] * s + ty, ang[o], (2.2 + 3.0 * h1[o]) * s, length[o] * s, np.clip(cols[o], 0, 1), op=0.97,
                  bend=bend[o], seed=21 + int(front), dry=0.1, taper=0.9, streak=0.06, paper_k=0.0)
    core.over(pic, lay)


def _fore(pic, cam, tt, p, blur, spark_sx, flick, seed):
    """Big soft out of focus blades between the lens and the fuse."""
    from .. import hillset_paint as hp
    from .. import paint
    T = palette.TOD[TOD]
    s, tx, ty = cam.xf(p)
    near = p > 2.0
    step = 34.0 if near else 22.0
    idx = np.arange(int((-320 - tx) / s // step), int((core.W + 320 - tx) / s // step) + 1)
    h1, h2, h3, h4, h5 = (hp.hashf(idx, seed + q) for q in range(5))
    clump = core.fbm_at(idx * step, idx * 0.0 + seed, 250.0, 2, seed)
    ok = (clump > 0.45) & (h5 < (0.55 if near else 0.75))
    idx, h1, h2, h3, h4 = idx[ok], h1[ok], h2[ok], h3[ok], h4[ok]
    if len(idx) == 0:
        return
    xs = (idx * step + (h1 - 0.5) * step) * s + tx
    ys = core.H + 40.0 + 110.0 * h2
    length = (480.0 + 560.0 * h3) if near else (300.0 + 380.0 * h3)
    width = (90.0 + 110.0 * h4) if near else (44.0 + 66.0 * h4)
    wave = tt * 1.4 + idx * 0.7
    ang = -np.pi / 2 + (h1 - 0.5) * 0.7 + 0.05 * np.sin(wave)
    bend = (h3 - 0.5) * 1.0 + 0.10 * np.sin(wave - 0.8)
    dark, mid = _grass_cols(T)
    dark = dark * (0.50 if near else 0.66)
    cols = dark + (mid * 0.7 - dark) * (h4[:, None] * 0.6)
    warm = np.clip(np.exp(-((xs - spark_sx) / 430.0) ** 2) * flick * 0.55, 0, 1)[:, None]
    lay = np.zeros((core.H, core.W, 4), np.float32)
    paint.strokes(lay, xs, ys, ang, width, length, np.clip(cols, 0, 1), op=1.0, bend=bend, seed=seed, dry=0.05, taper=0.92,
                  streak=0.08, paper_k=0.0)
    rim = cols * 1.5 + 0.02 + (WARM * 0.62 - cols) * warm
    paint.strokes(lay, xs - width * 0.26, ys, ang, width * 0.28, length * 0.97, np.clip(rim, 0, 1), op=0.85, bend=bend, seed=seed + 1,
                  dry=0.3, taper=0.92, streak=0.05, paper_k=0.0)
    core.over(pic, core.gblur(lay, blur))


def _fuse(pic, cam, k, fl, hx):
    """The string: pale and twisted ahead of the spark, charred behind it, a few embers."""
    T = palette.TOD[TOD]
    s, tx, ty = cam.xf(1.0)
    xa = max(FUSE_A, (-40 - tx) / s)
    xb = min(FUSE_B, (core.W + 40 - tx) / s)
    if xb <= xa:
        return
    sc = lambda X: np.c_[np.asarray(X) * s + tx, fuse_y(X) * s + ty]
    allx = np.linspace(xa, xb, 60)
    p = sc(allx)
    y0 = int(max(p[:, 1].min() - 30, 0))
    y1 = int(min(p[:, 1].max() + 30, core.H))
    if y1 - y0 < 4:
        return
    cel = core.Cel(core.W, y1 - y0, ss=2, x0=0, y0=y0)
    ink = T["ink"]
    string_l = palette.lit(palette.ROCKET["string"], T)
    string_s = palette.shade(palette.ROCKET["string"], T)
    if hx < xb:                                     # unburnt
        X = np.linspace(max(hx, xa), xb, 50)
        q = sc(X)
        cel.stroke(q, 5.6 * s, ink, taper=(0.0, 0.0))
        cel.stroke(q, 4.2 * s, string_s, taper=(0.0, 0.0))
        cel.stroke(q - (0, 0.8 * s), 2.3 * s, string_l, taper=(0.0, 0.0))
        for xt in np.arange(math.ceil(max(hx, xa) / 5.0) * 5.0 + 3.0, xb, 5.0):     # the twist of the string
            c = sc([xt])[0]
            cel.stroke(np.array([c + (-1.6 * s, 2.0 * s), c + (1.6 * s, -2.0 * s)]), 0.9 * max(s, 1), ink, taper=(0.1, 0.1))
    if hx > xa:                                     # burnt: thin, black, crumbling
        X = np.linspace(xa, min(hx, xb), 50)
        q = sc(X)
        pr = 0.55 + 0.45 * core.fbm_at(X, X * 0 + 0.3, 9.0, 2, 5)
        cel.stroke(q + (0, 0.5 * s), 3.8 * s, np.float32((0.085, 0.08, 0.10)), taper=(0.0, 0.0), pressure=pr)
        r = core.rng("ash", 1)
        for xt in np.arange(math.ceil(xa / 7.0) * 7.0, min(hx, xb), 7.0):
            if r.random() < 0.45:
                c = sc([xt + r.uniform(-2, 2)])[0]
                cel.dot(c[0], c[1] + r.uniform(-1, 2) * s, (0.5 + 0.5 * r.random()) * s, np.float32((0.42, 0.42, 0.50)))
        # embers where it has just burnt: the last three steps, dying
        for back, amt in ((1, 0.9), (2, 0.55), (3, 0.28)):
            if k - back < 0:
                continue
            ex = _spark_x(k - back)
            if ex < xa:
                continue
            c = sc([ex])[0]
            fr = core.rng("ember", k - back, fl).random()
            col = palette.mix(np.float32((0.45, 0.10, 0.04)), np.float32((1.0, 0.62, 0.20)), amt * (0.6 + 0.4 * fr))
            cel.dot(c[0], c[1], (0.9 + 1.0 * amt) * s, col)
    core.over(pic, cel.rgba(), 0, y0)


def _head(pic, hx_s, hy_s, s, fl):
    """The fizzing head: a ragged star, redrawn every frame."""
    r = core.rng("spark head", fl)
    box = int(52 * s / 3.5) + 8
    x0, y0 = int(hx_s - box), int(hy_s - box)
    cel = core.Cel(2 * box, 2 * box, ss=3, x0=x0, y0=y0)
    for scale, col in ((1.0, (1.0, 0.56, 0.14)), (0.66, (1.0, 0.83, 0.40)), (0.36, (1.0, 0.98, 0.86))):
        n = int(r.integers(6, 9))
        a0 = r.uniform(0, 6.283)
        pts = []
        for i in range(n * 2):
            a = a0 + i * math.pi / n + r.normal(0, 0.10)
            rad = (r.uniform(8.5, 13.5) if i % 2 == 0 else r.uniform(2.8, 4.4)) * s * scale
            pts.append((hx_s + math.cos(a) * rad, hy_s + math.sin(a) * rad * 0.92))
        cel.fill(np.array(pts), np.float32(col))
    core.over(pic, cel.rgba(), x0, y0)


def _smoke(pic, cam, hx, hy, tt, k):
    T = palette.TOD[TOD]
    s, tx, ty = cam.xf(1.0)
    u = np.linspace(0, 1, 18)
    x = hx - 62.0 * u ** 1.3 - 7.0 * u * np.sin(5.0 * u - 3.0 * tt) + 3.0 * np.sin(9.0 * u + k)
    y = hy - 5.0 - 128.0 * u
    p = np.c_[x * s + tx, y * s + ty]
    x0 = int(max(p[:, 0].min() - 40, 0))
    y0 = int(max(p[:, 1].min() - 40, 0))
    x1 = int(min(p[:, 0].max() + 40, core.W))
    y1 = int(min(p[:, 1].max() + 40, core.H))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    cel = core.Cel(x1 - x0, y1 - y0, ss=2, x0=x0, y0=y0)
    col = palette.mix(palette.lit((0.74, 0.76, 0.86), T), WARM, 0.12)
    cel.stroke(p, 7.5 * s, col, taper=(0.06, 0.5), pressure=[0.25, 0.6, 1.0, 0.9, 0.55, 0.25], seed=int(tt * 12))
    core.over(pic, cel.rgba(), x0, y0, opacity=0.46)


def render(fl, n, f):
    from .. import fx, rocket
    from .. import hillset as hs
    t = fl / FPS
    tt = common.twos(fl) / FPS
    cam = camera(t)
    s, tx, ty = cam.xf(1.0)
    k, kf0 = _step(fl)
    hx = _spark_x(k)
    hy = float(fuse_y(hx)) - 1.5
    hx_s, hy_s = hx * s + tx, hy * s + ty
    flick = 0.72 + 0.5 * core.rng("fuse flick", fl).random()
    rack = core.smooth(core.span(t, 1.55, 2.2))

    pic = common.sky_pic(TOD, cam, f, t, stars=0.3)
    hs.render_back(pic, cam, TOD, f, lights=0.12, defocus=common.dof(1.0, 4.0))
    common.draw_rocket(pic, cam, TOD, common.twos(fl), state=rocket.State())
    sig = 7.5 * (1 - rack) + 0.5
    if sig > 0.6:                                   # the rocket and the grass behind are out of focus until the rack
        pic[:] = core.gblur(pic, sig)
    fx.ground_light(pic, hx_s, hy_s + 5 * s, 38 * s * flick, (1.0, 0.55, 0.20), 0.34 * flick, squash=0.42)
    _mid_blades(pic, cam, tt, (hx, hy), flick, front=False)
    _fuse(pic, cam, k, fl, hx)
    _smoke(pic, cam, hx, hy, tt, k)
    fx.glow_at(pic, hx_s, hy_s, 30 * s * flick, (1.0, 0.72, 0.36), 0.55)
    _head(pic, hx_s, hy_s, s, fl)
    gy = lambda X: float(common.ground_y(X)) + 9.0
    for back in (2, 1, 0):                          # the spray of this step and the two before it, still flying
        if k - back < 0:
            continue
        kx = _spark_x(k - back)
        age = (fl - int(_starts[k - back])) / FPS + 0.02
        core.over(pic, fx.sparks_layer(age, (kx, float(fuse_y(kx)) - 3.0), seed=100 + k - back, n=20, ground_y=gy, spread=(14.0, 166.0),
                                       speed=(70.0, 340.0), burst=0.05, shutter=1.0 / 24.0, size=0.62, frame=fl, xf=(s, tx, ty),
                                       gravity=560.0, drag=2.4, dur=0.10))
    xa, xb = (-60 - tx) / s, (core.W + 60 - tx) / s
    core.over(pic, hs.grass_band(cam, TOD, f, xa, xb), opacity=0.6)
    _mid_blades(pic, cam, tt, (hx, hy), flick, front=True)
    fx.glow_at(pic, hx_s, hy_s, 9 * s, (1.0, 0.95, 0.80), 0.8 * flick)
    _fore(pic, cam, tt, 1.7, 5.0, hx_s, flick, 31)
    _fore(pic, cam, tt, 2.6, 12.0, hx_s, flick, 47)
    return pic, dict(bloom_amt=0.62)
