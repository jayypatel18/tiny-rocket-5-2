"""Shot 8: the climax vista. Very far and very high at sunrise: the curve of
the world, a sea of clouds in perspective, mountains as islands, rivers and a
lake through the gaps, the sun just risen at lower right.

    pic = vista.render(cam, f, t)      # float RGB 1080 x 1920, no shooting star

How it is built. The world is two height fields in real perspective: the
ground (valley floor, mountains, water) and a floating slab of cloud with
billowed tops and holes. Both are baked once on a (depth slice, screen column)
grid, lit by a low sun with long marched shadows, coloured in gouache tones,
repainted with brush strokes, then drawn front to back each frame. Every depth
slice has its own parallax, so a push in or drift moves hundreds of layers.
Default camera: core.Camera() (centre 960, 540). cam.dolly pushes in,
cam.cx / cam.cy drift, cam.zoom scales.
"""
import math

import numpy as np
from numba import njit, prange

from tr2 import core, palette
from tr2 import sky_paint as sp

W, H = core.W, core.H
VERSION = 5
F = 1150.0            # focal length in px
HZ = 590.0            # horizon line at screen centre
CURVE = 44.0          # how far the horizon drops at the frame edges (px)
XC = 1010.0           # where the curve of the world crests
NA, ND = 2304, 760    # baked grid: columns (with pan margin), depth slices
Z0, Z1 = 1.9, 230.0   # nearest and farthest slice (camera height = 1)
CAMH = 1.0
TH = 0.75             # thickness of the cloud slab
GD = 2.7              # the valley floor lies this far under the cloud tops
SUN_X = 1395.0        # sun on screen
A_SUN = (SUN_X - 960.0) / F
LSUN = np.array([0.36, 0.17, 0.92])
LSUN = LSUN / np.linalg.norm(LSUN)
T = palette.TOD["sunrise"]


def _grid():
    k = np.arange(ND)
    z = Z0 * (Z1 / Z0) ** (k / (ND - 1.0))
    a = (np.arange(NA) - NA / 2 + 0.5) / F
    return z, a


def _n(X, Z, scale, octv, seed):
    return core.fbm_at(X, Z, scale, octv, seed).reshape(X.shape).astype(np.float32)


# mountains: (screen x at the cloud tops, depth, half width, half depth, height above the cloud tops, seed)
MOUNTAINS = [
    (470.0, 10.5, 2.6, 1.5, 1.35, 1),     # the big island, left of centre
    (250.0, 14.0, 2.2, 1.6, 1.0, 2),
    (760.0, 15.0, 1.8, 1.4, 0.8, 3),
    (1700.0, 5.6, 1.15, 0.8, 0.95, 4),    # near peak, lower right, against the sun
    (1040.0, 46.0, 10.0, 4.0, 2.7, 5),    # far range left of the sun
    (1700.0, 60.0, 9.0, 5.0, 2.6, 6),     # far range right
    (120.0, 34.0, 7.0, 4.0, 2.4, 7),
    (640.0, 90.0, 22.0, 8.0, 3.6, 8),     # the farthest, pale
    (1500.0, 120.0, 30.0, 10.0, 3.8, 9),
]
# holes in the cloud: (screen x, depth of the hole in the cloud, half width, half depth)
GAPS = [(600.0, 6.3, 2.1, 1.25), (1180.0, 12.5, 2.6, 1.9), (150.0, 5.0, 1.0, 0.7), (1500.0, 24.0, 5.0, 3.0)]


def _heights(X, Z):
    """World height fields at points: cloud top, cloud presence, ground."""
    wx = (_n(X, Z, 4.0, 3, 11) - 0.5) * 1.6
    wz = (_n(X, Z, 4.0, 3, 12) - 0.5) * 1.6
    Xw, Zw = X + wx, Z + wz
    cover = _n(Xw, Zw * 1.25, 5.0, 4, 1)
    cover = cover + 0.30 * sp.ss(4.2, 2.6, Z) + 0.10 * sp.ss(30.0, 80.0, Z)
    gap = np.zeros_like(X)
    for (sx, zc, rx, rz) in GAPS:
        xc = (sx - 960.0) / F * zc
        gap = np.maximum(gap, np.exp(-(((Xw - xc) / rx) ** 2 + ((Zw - zc) / rz) ** 2) ** 1.5))
    cover = cover - 0.75 * gap
    e = sp.ss(0.34, 0.50, cover)
    b1 = np.abs(2 * _n(Xw, Zw, 1.25, 5, 2) - 1)                 # small cauliflower tops
    b2 = np.abs(2 * _n(Xw * 0.8, Zw, 3.6, 4, 3) - 1)            # the big banks
    tower = sp.ss(0.60, 0.80, _n(X, Z, 8.0, 3, 9))
    top = 0.95 * b2 + 0.42 * b1 * (0.5 + b2) + tower * (0.7 + 1.1 * b1) - 0.30
    hc = -TH + (TH + top) * e ** 0.55
    # ground
    g = -GD + 0.35 * (_n(X, Z, 6.0, 4, 21) - 0.5)
    for (sx, zc, rx, rz, hh, sd) in MOUNTAINS:
        xc = (sx - 960.0) / F * zc
        d2 = ((X - xc) / rx) ** 2 + ((Z - zc) / rz) ** 2
        env = np.exp(-d2 * 1.2)
        sc = max(0.5, rx * 0.55)
        wxm = (_n(X, Z, sc * 0.8, 3, 40 + sd) - 0.5) * sc * 0.5
        r1 = 1 - np.abs(2 * _n(X + wxm, Z * 1.3, sc, 5, 30 + sd) - 1)
        r2 = 1 - np.abs(2 * _n(X * 1.3 - wxm, Z * 1.6, sc * 0.32, 4, 50 + sd) - 1)
        m = (GD + hh) * env ** 0.8 * (0.30 + 0.55 * r1 ** 1.6 + 0.15 * r2)
        g = np.maximum(g, -GD + m)
    return hc, e, g


@njit(cache=True, fastmath=True, parallel=True)
def _shadow(hs, zs, avals, lx, ly, lz, out):
    nd, na = hs.shape
    lz0 = math.log(zs[0])
    lzr = (nd - 1) / (math.log(zs[nd - 1]) - lz0)
    a0 = avals[0]
    da = avals[1] - avals[0]
    for k in prange(nd):
        z = zs[k]
        for i in range(na):
            X = avals[i] * z
            y = hs[k, i]
            sh = 0.0
            d = 0.04 * z ** 0.5
            for s in range(46):
                d *= 1.11
                Zp = z + lz * d
                Xp = X + lx * d
                Yp = y + ly * d
                kk = int((math.log(Zp) - lz0) * lzr + 0.5)
                ii = int((Xp / Zp - a0) / da + 0.5)
                if kk >= nd or ii < 0 or ii >= na:
                    break
                over = hs[kk, ii] - Yp
                if over > 0:
                    q = over / (0.06 + 0.05 * d)
                    if q > sh:
                        sh = q
                    if sh >= 1.0:
                        break
            out[k, i] = min(sh, 1.0)


def _bake():
    z, a = _grid()
    Zg = np.repeat(z[:, None], NA, 1)
    Xg = a[None, :] * z[:, None]
    hc, e, g = _heights(Xg, Zg)
    # normals by finite differences in the world
    eps = np.maximum(0.05, Zg / F * 1.5)
    hcx, _, gx = _heights(Xg + eps, Zg)
    hcz, _, gz = _heights(Xg, Zg + eps)
    lx, ly, lz = LSUN

    def lamb(h0, hx, hz, k=1.0):
        dx = (hx - h0) / eps
        dz = (hz - h0) / eps
        n = np.sqrt(dx * dx + dz * dz + k * k)
        return (-dx * lx + k * ly - dz * lz) / n, dx, dz, n

    hs = np.maximum(np.where(e > 0.02, hc, -99.0), g).astype(np.float32)
    shd = np.zeros((ND, NA), np.float32)
    _shadow(hs, z, a, lx, ly, lz, shd)
    shd = core.gblur(shd, 1.2)

    sunward = np.exp(-((a[None, :] - A_SUN) / 0.42) ** 2).astype(np.float32) * np.ones((ND, 1), np.float32)
    fogz = (1 - np.exp(-np.maximum(Zg - 2.5, 0) / 42.0)).astype(np.float32)
    fog_far = palette.mix(np.array([0.95, 0.70, 0.68]), np.array([1.0, 0.88, 0.64]), 1.0)
    fogc = (np.array([0.90, 0.68, 0.72], np.float32)[None, None, :] * (1 - sunward[..., None])
            + np.array([1.0, 0.86, 0.62], np.float32)[None, None, :] * sunward[..., None])

    # ---- cloud tone
    lam, dx, dz, nn = lamb(hc, hcx, hcz, 0.55)
    lam = np.clip(lam * 1.5 + 0.12, 0, 1)
    crest = np.clip(core.gblur(hc, 6.0) * -1 + hc, -0.3, 0.3) / 0.3      # above the local mean = a top
    strokes = sp.stroke_field(ND, NA, 26000, (30, 120), (5, 16), 0.0, 0.35, seed=5, opacity=0.8)
    Lc = 0.16 + 0.62 * lam * (1 - 0.85 * shd) + 0.16 * np.clip(crest, 0, 1) * (1 - 0.6 * shd) \
        - 0.12 * np.clip(-crest, 0, 1) + 0.07 * strokes
    Lc += 0.14 * sunward * (1 - shd) * lam
    Lc = np.clip(Lc, 0, 1)
    skyc = np.array([0.80, 0.62, 0.74], np.float32)
    cols = sp.tones("sunrise", skyc, 0.0)
    deep, sh, turn, lit, hi = cols
    deep = np.array([0.30, 0.31, 0.56], np.float32)
    sh = np.array([0.50, 0.46, 0.70], np.float32)
    turn = np.array([0.90, 0.60, 0.62], np.float32)
    lit = np.array([1.0, 0.84, 0.62], np.float32)
    hi = np.array([1.0, 0.96, 0.82], np.float32)
    c = deep + (sh - deep) * sp.ss(0.10, 0.26, Lc)[..., None]
    c += (turn - c) * sp.ss(0.34, 0.44, Lc)[..., None]
    c += (lit - c) * sp.ss(0.50, 0.60, Lc)[..., None]
    c += (hi - c) * sp.ss(0.74, 0.86, Lc)[..., None]
    ccol = c * (1 - fogz[..., None]) + fogc * fogz[..., None]
    ccol = sp.repaint(ccol, 30000, (26, 90), (4, 12), 0.0, 0.3, seed=7, opacity=0.8, jitter=0.012)
    wall = palette.mix(deep, sh, 0.35)
    cwall = wall[None, None, :] * (1 - fogz[..., None]) + fogc * fogz[..., None]

    # ---- ground: rock, snow, valley, water
    lamg, dxg, dzg, ng = lamb(g, gx, gz, 1.0)
    slope = np.sqrt(dxg * dxg + dzg * dzg)
    lamg = np.clip(lamg * 1.4 + 0.1, 0, 1)
    rockn = _n(Xg * 3.0, Zg * 3.0 + g * 8.0, 1.0, 4, 61)
    snow_line = 0.25 + 0.5 * (rockn - 0.5)
    snow = sp.ss(snow_line, snow_line + 0.22, g) * sp.ss(2.6, 1.2, slope + 0.8 * (rockn - 0.5))
    litg = lamg * (1 - 0.9 * shd)
    rock_l = np.array([0.74, 0.47, 0.40], np.float32)
    rock_s = np.array([0.24, 0.20, 0.38], np.float32)
    rock_d = np.array([0.15, 0.13, 0.28], np.float32)
    snow_l = np.array([1.0, 0.86, 0.78], np.float32)
    snow_s = np.array([0.46, 0.48, 0.76], np.float32)
    lt = sp.ss(0.30, 0.42, litg + 0.10 * (strokes))[..., None]
    rock = rock_d + (rock_s - rock_d) * sp.ss(0.05, 0.2, lamg + 0.3 * rockn)[..., None]
    rock = rock + (rock_l - rock) * lt
    sn = snow_s + (snow_l - snow_s) * lt
    gcol = rock + (sn - rock) * snow[..., None]
    # valley floor far below: fields, forest, in blue air
    low = sp.ss(-GD + 0.9, -GD + 0.25, g)
    fld = _n(Xg, Zg, 1.6, 4, 71)
    patch = _n(Xg * 2.0, Zg * 2.0, 0.5, 2, 72)
    land = np.array([0.20, 0.27, 0.40], np.float32)[None, None, :] * (0.8 + 0.5 * fld[..., None]) \
        + np.array([0.10, 0.06, -0.02], np.float32) * sp.ss(0.5, 0.6, patch)[..., None]
    land = land * (0.75 + 0.5 * (1 - shd[..., None] * 0.6))
    gcol = gcol + (land - gcol) * low[..., None]
    # rivers and a lake: they mirror the sky, and blaze toward the sun
    ang = Xg / Zg
    r1 = (GAPS[0][0] - 960.0) / F + 0.085 * np.sin(Zg * 0.31 + 0.6) + 0.03 * np.sin(Zg * 0.83)
    w1 = (0.11 + 0.05 * np.sin(Zg * 0.5)) / Zg
    riv = sp.ss(w1 * 1.25, w1 * 0.75, np.abs(ang - r1))
    lake = sp.ss(1.15, 0.85, np.sqrt(((Xg - (r1 * Zg + 0.2)) / 1.5) ** 2 + ((Zg - 21.0) / 2.6) ** 2)
                 + 0.35 * (fld - 0.5))
    r2 = (GAPS[1][0] - 960.0) / F - 0.02 + 0.05 * np.sin(Zg * 0.17 + 2.0) + 0.02 * np.sin(Zg * 0.5)
    w2 = (0.16 + 0.06 * np.sin(Zg * 0.3)) / Zg
    riv2 = sp.ss(w2 * 1.25, w2 * 0.75, np.abs(ang - r2))
    r3 = (GAPS[3][0] - 960.0) / F + 0.04 * np.sin(Zg * 0.09 + 1.0)
    riv3 = sp.ss(0.26 / Zg * 1.25, 0.26 / Zg * 0.75, np.abs(ang - r3))
    water = np.maximum(np.maximum(riv, lake), np.maximum(riv2, riv3)) * low
    wcol = (np.array([0.66, 0.70, 0.92], np.float32)[None, None, :] * (1 - sunward[..., None])
            + np.array([1.0, 0.93, 0.70], np.float32)[None, None, :] * sunward[..., None])
    wcol = wcol * (0.92 + 0.16 * patch[..., None])
    gcol = gcol + (wcol - gcol) * water[..., None]
    # air: the valley sits in deep blue haze, the far ranges fade to rose
    depth_air = sp.ss(0.0, -GD, g)[..., None] * 0.42
    gcol = gcol * (1 - depth_air) + np.array([0.34, 0.40, 0.66], np.float32) * depth_air * (1 - 0.6 * water[..., None]) \
        + gcol * depth_air * 0.6 * water[..., None]
    gcol = gcol * (1 - fogz[..., None]) + fogc * fogz[..., None]
    gcol = sp.repaint(gcol.astype(np.float32), 26000, (16, 60), (3, 9), 0.5, 0.6, seed=9, opacity=0.75, jitter=0.012)
    gcol = gcol + (wcol * (1 - fogz[..., None] * 0.5) - gcol) * (water[..., None] * 0.85)

    out = np.concatenate([np.where(e > 0.02, hc, -99.0)[..., None], g[..., None], ccol, cwall, gcol], -1)
    return out.astype(np.float32)


@njit(cache=True, fastmath=True, parallel=True)
def _march(out, alpha, B, zs, hzrow, f, camh, th, sk, pxk, pyk):
    """Front to back. B channels: 0 cloud top, 1 ground, 2:5 cloud colour,
    5:8 cloud wall colour, 8:11 ground colour."""
    h, w, _ = out.shape
    nd, na, _ = B.shape
    for i in prange(w):
        ygr = h          # everything from here down is ground or filled
        for k in range(nd):
            z = zs[k]
            s = sk[k]
            ai = int(na * 0.5 + (i + 0.5 - w * 0.5) / s + pxk[k])
            if ai < 0 or ai >= na:
                continue
            hz = hzrow[ai]
            base = (hz - h * 0.5) * s + h * 0.5 - pyk[k]
            sc = f * s / z
            # ground
            yg = int(base + (camh - B[k, ai, 1]) * sc)
            if yg < ygr:
                y0 = max(yg, 0)
                for y in range(y0, ygr):
                    if alpha[y, i] == 0.0:
                        alpha[y, i] = 1.0
                        out[y, i, 0] = B[k, ai, 8]
                        out[y, i, 1] = B[k, ai, 9]
                        out[y, i, 2] = B[k, ai, 10]
                ygr = y0
            if ygr <= 0:
                break
            # cloud slab
            ct = B[k, ai, 0]
            if ct > -50.0:
                yt = int(base + (camh - ct) * sc)
                yb = int(base + (camh + th) * sc) + 1
                if yb > ygr:
                    yb = ygr
                cap = 0.16 * sc + 2.0
                for y in range(max(yt, 0), yb):
                    if alpha[y, i] == 0.0:
                        alpha[y, i] = 1.0
                        q = (y - yt) / cap
                        if q > 1.0:
                            q = 1.0
                        q = q * q * (3 - 2 * q)
                        out[y, i, 0] = B[k, ai, 2] * (1 - q) + B[k, ai, 5] * q
                        out[y, i, 1] = B[k, ai, 3] * (1 - q) + B[k, ai, 6] * q
                        out[y, i, 2] = B[k, ai, 4] * (1 - q) + B[k, ai, 7] * q


_B = None
_SKY = None


def _bakeload():
    global _B
    if _B is None:
        _B = sp.cached("vista_bake_v%d" % VERSION, lambda: _bake().astype(np.float16)).astype(np.float32)
    return _B


def _hz_at(x):
    return HZ + CURVE * ((x - XC) / 960.0) ** 2


def _sky():
    """The static painted sky: gradient that follows the curve of the world,
    the sun, the thin line of atmosphere. Larger than the screen for drift."""
    global _SKY
    if _SKY is not None:
        return _SKY

    def make():
        m = 160
        hh, ww = H + 2 * m, W + 2 * m
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        xx -= m
        yy -= m
        hz = _hz_at(xx)
        st = sp.stroke_field(hh, ww, 2400, (300, 900), (30, 90), 0.0, 0.06, seed=3, opacity=0.9)
        st2 = sp.stroke_field(hh, ww, 5000, (100, 380), (12, 36), 0.0, 0.12, seed=4, opacity=0.75)
        v = np.clip((yy + m) / (hz + m), 0, 1.2)
        v = v ** 1.9 + (0.030 * st + 0.018 * st2) * (0.3 + 0.7 * v)
        pic = palette.sky_color("sunrise", np.clip(v, 0, 1))
        top = sp.ss(0.42, 0.0, (yy + m) / (hz + m))[..., None]
        pic = pic * (1 - 0.42 * top)                                  # deeper toward space
        pic *= 1.0 + 0.022 * st2[..., None]
        # sun
        sx, sy = SUN_X, _hz_at(SUN_X) - 16.0
        d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        dw = np.sqrt(((xx - sx) * 0.40) ** 2 + (yy - sy) ** 2)
        g = 0.55 * np.exp(-d / 420.0) + 0.6 * np.exp(-(d / 150.0) ** 2) + 0.55 * np.exp(-dw / 120.0)
        core.add_light(pic, g.astype(np.float32), (1.0, 0.86, 0.58), 0.85)
        core.add_light(pic, np.exp(-(d / 64.0) ** 2).astype(np.float32), (1.0, 0.96, 0.84), 0.95)
        disc = sp.ss(34.0, 30.0, d)
        pic = pic * (1 - disc[..., None]) + np.array([1.0, 0.99, 0.93], np.float32) * disc[..., None]
        return pic.astype(np.float16)
    _SKY = sp.cached("vista_sky_v%d" % VERSION, make).astype(np.float32)
    return _SKY


def render(cam, f, t):
    B = _bakeload()
    z, a = _grid()
    # sky, a little parallax
    s0 = cam.zoom * cam.dolly ** 0.03
    px0 = (cam.cx - cam.cx0) * 0.03
    py0 = (cam.cy - cam.cy0) * 0.03
    sky = _sky()
    m = 160
    pic = core.warp(sky, s0, W / 2 - (W / 2 + px0 + m) * s0, H / 2 - (H / 2 + py0 + m) * s0)
    # stars high up
    from tr2 import sky as skymod
    st = skymod._stars(core.Camera(1600, 900, cy=900 - 2600), t, 0.8, 0.0)
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    pic = 1 - (1 - np.clip(pic, 0, 1)) * (1 - np.clip(st * sp.ss(300.0, 20.0, yy) * 0.9, 0, 1))

    p = np.clip((7.0 / z) ** 0.5, 0.05, 2.2)
    sk = (cam.zoom * cam.dolly ** p).astype(np.float64)
    pxk = ((cam.cx - cam.cx0) * p * 0.6).astype(np.float64)
    pyk = ((cam.cy0 - cam.cy) * p * 0.6).astype(np.float64)
    hzrow = _hz_at((np.arange(NA) - NA / 2 + 960.0).astype(np.float64))
    land = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    _march(land, alpha, B, z, hzrow, F, CAMH, TH, sk, pxk, pyk)
    # soften the stair steps of the march a little, keep the paint
    a3 = core.gblur(alpha, 0.7)[..., None]
    land = core.gblur(land, 0.6)
    pic = pic * (1 - a3) + land * a3

    # the air on top: thin bright line of atmosphere, haze on the horizon, sun bloom over the sea
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    hz = (_hz_at((xs - W / 2) / s0 + W / 2 + px0) - H / 2 - py0) * s0 + H / 2
    sx = (SUN_X - W / 2 - px0) * s0 + W / 2
    sy = (_hz_at(SUN_X) - 16.0 - H / 2 - py0) * s0 + H / 2
    near_sun = np.exp(-np.abs(xs - sx) / 900.0)
    haze = np.exp(-((ys - hz - 4) / 26.0) ** 2) * (0.35 + 0.65 * near_sun)
    core.add_light(pic, haze.astype(np.float32), (1.0, 0.84, 0.66), 0.55)
    line = np.exp(-((ys - hz + 7) / 2.6) ** 2) * (0.35 + 0.65 * near_sun) + 0.35 * np.exp(-((ys - hz + 12) / 9.0) ** 2)
    core.add_light(pic, line.astype(np.float32), (0.80, 0.93, 1.0), 0.5)
    d = np.sqrt(((xs - sx) * 0.55) ** 2 + (ys - sy) ** 2)
    bloom = 0.5 * np.exp(-d / 330.0) + 0.5 * np.exp(-(d / 110.0) ** 2)
    core.add_light(pic, bloom.astype(np.float32), (1.0, 0.82, 0.52), 0.6)
    return np.clip(pic, 0, 1).astype(np.float32)
