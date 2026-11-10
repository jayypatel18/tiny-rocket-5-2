"""Shot 8: the climax vista. Very far and very high at sunrise: the curve of
the world, a sea of clouds in perspective, mountains as islands, a river and a
lake through gaps in the floor, the sun just risen at lower right.

    pic = vista.render(cam, f, t)      # float RGB 1080 x 1920, no shooting star

Built back to front, all painted once and cached, then placed every frame
through the camera, each layer with its own parallax (about 30 layers):
  sky (curved gradient, sun, stars)  ->  cloud floor in perspective, with the
  two gaps (dark land, river)  ->  rows of cumulus tops from the horizon to
  the lens, each row larger, mountains slotted between the rows by depth  ->
  air: haze on the horizon, the thin line of atmosphere, sun bloom.
Use the default core.Camera() (centre 960, 540): cam.dolly pushes in,
cam.cx / cam.cy drift (px on the near cloud plane), cam.zoom scales.
"""
import math

import numpy as np
from numba import njit
from scipy import ndimage

from tr2 import core, palette
from tr2 import sky_paint as sp

W, H = core.W, core.H
VERSION = 19
F = 1150.0            # focal length in px
HZ = 598.0            # horizon at the crest of the curve
CURVE = 42.0          # how far the horizon drops at the frame edges (px)
XC = 1000.0           # where the curve of the world crests
M = 192               # margin of the painted canvas, for drift
CW = W + 2 * M
SUN_X = 1395.0
S0 = 1.55
SQUASH = 0.62         # cloud lumps are flattened, seen from above             # cloud sprite scale at depth 1

# cloud tones for this shot: the floor is light, the shadows mid lavender
DEEP = np.array([0.55, 0.54, 0.80], np.float32)
SHAD = np.array([0.69, 0.64, 0.85], np.float32)
TURN = np.array([0.95, 0.72, 0.72], np.float32)
LIT = np.array([1.00, 0.87, 0.68], np.float32)
HI = np.array([1.00, 0.96, 0.84], np.float32)

# gaps in the floor: (screen x, screen y, half width, half height)
GAPS = [(600.0, 770.0, 215.0, 46.0), (1250.0, 704.0, 150.0, 30.0)]
# mountains: (screen x of the middle, y of the cloud line at its foot, width px, height px, seed)
MOUNTAINS = [(585.0, 690.0, 430.0, 122.0, 1), (960.0, 652.0, 210.0, 54.0, 2), (1640.0, 668.0, 270.0, 74.0, 3),
             (40.0, 630.0, 260.0, 32.0, 4)]


def hz_at(x):
    """The horizon line: the curve of the world."""
    return HZ + CURVE * ((np.asarray(x, np.float32) - XC) / 960.0) ** 2


def _sunward(x, wid=560.0):
    return np.exp(-((np.asarray(x, np.float32) - SUN_X) / wid) ** 2)


def _fog(x):
    s = _sunward(x)[..., None]
    return np.array([0.94, 0.75, 0.79], np.float32) * (1 - s) + np.array([1.0, 0.91, 0.72], np.float32) * s


def _fz(z):
    return 1 - np.exp(-np.maximum(np.asarray(z, np.float32) - 2.5, 0) / 20.0)


# ------------------------------------------------------------------ the sky
def _sky():
    def make():
        hh, ww = H + 2 * M, CW
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        xx -= M
        yy -= M
        hz = hz_at(xx)
        st = sp.stroke_field(hh, ww, 2400, (300, 900), (30, 90), 0.0, 0.06, seed=3, opacity=0.9)
        st2 = sp.stroke_field(hh, ww, 5000, (100, 380), (12, 36), 0.0, 0.12, seed=4, opacity=0.75)
        v = np.clip((yy + M) / (hz + M), 0, 1.2)
        v = v ** 2.3 + (0.030 * st + 0.018 * st2) * (0.3 + 0.7 * v)
        pic = palette.sky_color("sunrise", np.clip(v, 0, 1))
        top = sp.ss(0.78, 0.05, (yy + M) / (hz + M))[..., None]
        pic = pic * (1 - 0.60 * top)                                  # much deeper toward space
        pic[..., 2] *= 1 + 0.10 * top[..., 0]
        pic *= 1.0 + 0.022 * st2[..., None]
        sx, sy = SUN_X, float(hz_at(SUN_X)) - 16.0
        d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        dw = np.sqrt(((xx - sx) * 0.40) ** 2 + (yy - sy) ** 2)
        g = 0.55 * np.exp(-d / 420.0) + 0.6 * np.exp(-(d / 150.0) ** 2) + 0.55 * np.exp(-dw / 120.0)
        core.add_light(pic, g.astype(np.float32), (1.0, 0.86, 0.58), 0.85)
        core.add_light(pic, np.exp(-(d / 64.0) ** 2).astype(np.float32), (1.0, 0.96, 0.84), 0.95)
        disc = sp.ss(34.0, 30.0, d)
        pic = pic * (1 - disc[..., None]) + np.array([1.0, 0.99, 0.93], np.float32) * disc[..., None]
        return pic.astype(np.float16)
    return sp.cached("vista_sky_v%d" % VERSION, make).astype(np.float32)


def mass_at(X, Z):
    """The big rolling structure of the cloud sea, 0..1, in world coordinates
    (X to the right, Z away, camera height 1): banks and the valleys between."""
    X = np.asarray(X, np.float64)
    Z = np.asarray(Z, np.float64)
    m = core.fbm_at(X * 0.55 + 31.0, Z * 0.9 + 7.0, 2.6, 3, 201).reshape(X.shape)
    m = np.clip((m - 0.5) * 2.4 + 0.5, 0, 1)
    # a valley in front of each gap, so we can see down into it
    for (gx, gy, rx, ry) in GAPS:
        zg = F / (gy - HZ)
        xg = (gx - 960.0) / F * zg
        rw = rx / F * zg * 1.25
        along = sp.ss(zg * 0.50, zg * 0.62, Z) * sp.ss(zg * 1.22, zg * 1.08, Z)
        m = m * (1 - 0.95 * np.exp(-((X - xg * Z / zg) / rw) ** 4) * along)
    return m.astype(np.float32)


def shade_at(X, Z):
    """How much a point lies in the long shadow of a bank on its sun side (right)."""
    m0 = mass_at(X, Z)
    sh = np.zeros_like(m0)
    for d, w in ((0.5, 1.0), (1.1, 0.8), (1.9, 0.6), (3.0, 0.4)):
        sh = np.maximum(sh, (mass_at(X + d, Z + d * 0.25) - m0 - 0.05 * d) * w)
    return np.clip(sh * 3.2, 0, 1)


# ------------------------------------------------------------ the cloud floor
def _floor():
    """Canvas (H - 560) x CW RGBA from y = 560 down: the cloud floor between
    the lumps, in perspective, with long shadows and the two gaps."""
    y0 = 560
    hh = H + M - y0
    yy, xx = np.mgrid[0:hh, 0:CW].astype(np.float32)
    xx -= M
    yy += y0
    hz = hz_at(xx)
    dy = np.maximum(yy - hz, 0.6)
    z = F / dy
    X = (xx - 960.0) / F * z
    def n(a, b, sc, o, sd):
        return core.fbm_at(a, b, sc, o, sd).reshape(a.shape).astype(np.float32)
    zc = np.minimum(z, 400.0)
    mass = mass_at(X, zc)
    shd = shade_at(X, zc)
    streak = n(X * 0.16 + z * 0.10, z, 0.55, 4, 102)       # long shadows lying to the left
    fine = n(X * 0.5, z, 0.22, 3, 103)
    calm = sp.ss(14.0, 40.0, z)                             # the far field is quiet
    lightv = 0.50 + 0.30 * (mass - 0.5) - 0.42 * shd * (1 - 0.6 * calm) + 0.45 * (streak - 0.5) * (1 - 0.5 * calm) \
        + 0.14 * (fine - 0.5) + 0.12 * calm
    c = DEEP + (SHAD - DEEP) * sp.ss(0.22, 0.38, lightv)[..., None]
    c = c + (TURN - c) * sp.ss(0.50, 0.60, lightv)[..., None] * 0.6
    c = c + (LIT - c) * sp.ss(0.62, 0.72, lightv)[..., None]
    fz = _fz(z)[..., None]
    c = c * (1 - 0.9 * fz) + _fog(xx) * 0.9 * fz
    c = sp.repaint(c.astype(np.float32), 9000, (40, 150), (5, 14), 0.0, 0.08, seed=21, opacity=0.7, jitter=0.01)
    a = sp.ss(-1.0, 3.0, yy - hz)
    glint_m = np.zeros(xx.shape, np.float32)
    # gaps: the far wall of the hole in shadow, then dark blue land far below
    for gi, (gx, gy, rx, ry) in enumerate(GAPS):
        wob = 1.0 * (n(xx, yy * 4.5, 170.0, 4, 110 + gi) - 0.5)
        e1 = np.sqrt(((xx - gx) / rx) ** 2 + ((yy - gy) / ry) ** 2) + wob
        hole = sp.ss(1.06, 0.80, e1)
        wall_t = ry * 0.42
        e2 = np.sqrt(((xx - gx) / rx) ** 2 + ((yy - gy - wall_t) / ry) ** 2) + wob
        landm = sp.ss(1.04, 0.84, e2) * hole
        fzg = float(_fz(F / (gy - HZ)))
        wallc = palette.mix(np.array([0.52, 0.52, 0.80], np.float32), _fog(np.float32(gx)), fzg * 0.5)
        c = c + (wallc - c) * hole[..., None]
        ln = n(xx, yy * 4.0, 46.0, 4, 120 + gi)
        pt = n(xx, yy * 4.0, 13.0, 2, 130 + gi)
        land = np.array([0.20, 0.29, 0.50], np.float32) * (0.78 + 0.5 * ln[..., None]) \
            + np.array([0.05, 0.05, -0.02], np.float32) * sp.ss(0.5, 0.58, pt)[..., None]
        land = palette.mix(land, np.array([0.50, 0.52, 0.78], np.float32), 0.22 + 0.25 * fzg)
        # the river: a thread that winds across the valley and glints gold
        u = np.linspace(-1.15, 1.15, 260)
        px = gx + u * rx
        mnd = core.fbm_at(u * 1.7 + 5.0 * gi, u * 0 + 0.3, 1.0, 2, 140 + gi) - 0.5
        py = gy + wall_t * 0.15 + ry * (1.7 * mnd - 0.22 * u)
        dmin = np.full(xx.shape, 1e9, np.float32)
        sel = (np.abs(xx - gx) < rx * 1.3) & (np.abs(yy - gy) < ry * 2.6)
        xs_, ys_ = xx[sel], yy[sel]
        dm = np.full(xs_.shape, 1e9, np.float32)
        for q in range(len(u)):
            dm = np.minimum(dm, (xs_ - px[q]) ** 2 + ((ys_ - py[q]) * 2.6) ** 2)
        dmin[sel] = np.sqrt(dm)
        wr = 1.5 + 0.9 * np.sin(xx * 0.043 + 1.0)
        riv = sp.ss(wr + 1.0, wr - 0.6, dmin)
        if gi == 0:   # a small lake where the river slows
            lq = int(np.argmin(np.abs(u + 0.42)))
            lk = np.sqrt(((xx - px[lq]) / (rx * 0.17)) ** 2 + ((yy - py[lq]) / (ry * 0.20)) ** 2) \
                + 1.1 * (n(xx, yy * 3.0, 40.0, 3, 150) - 0.5)
            riv = np.maximum(riv, sp.ss(1.0, 0.86, lk))
        glint = np.array([1.0, 0.92, 0.66], np.float32)
        land = land + (glint - land) * riv[..., None]
        glint_m = np.maximum(glint_m, riv * landm)
        land = land + 0.25 * np.exp(-(dmin / 7.0) ** 2)[..., None] * glint * (1 - riv[..., None])
        c = c + (land - c) * landm[..., None]
    return core.premul(np.clip(c, 0, 1).astype(np.float32), a.astype(np.float32)), y0, glint_m[..., None]


# ---------------------------------------------------------------- mountains
@njit(cache=True, fastmath=True)
def _mrender(hf, col, out, yb0, sz, fade):
    nz, nx = hf.shape
    hh = out.shape[0]
    for i in range(nx):
        ybuf = hh
        for k in range(nz - 1, -1, -1):          # front to back
            y = int(yb0 + k * sz - hf[k, i])
            if y < 0:
                y = 0
            if y < ybuf:
                a = hf[k, i] / fade
                if a > 1.0:
                    a = 1.0
                a = a * a * (3 - 2 * a)
                for yy in range(y, ybuf):
                    out[yy, i, 0] = col[k, i, 0] * a
                    out[yy, i, 1] = col[k, i, 1] * a
                    out[yy, i, 2] = col[k, i, 2] * a
                    out[yy, i, 3] = a
                ybuf = y


def _mountain(wpx, hpx, seed, haze, fogc):
    """An alpine massif as a small height field, lit by the low sun on the
    right and behind, drawn in steep side view. Returns RGBA, foot at bottom."""
    ssx = 2                               # paint at double size
    nx, nz = int(wpx * ssx), 520
    r = core.rng("mountain", seed)
    X = np.linspace(-1, 1, nx)[None, :] * np.ones((nz, 1))
    Zc = np.linspace(-1, 1, nz)[:, None] * np.ones((1, nx))
    def n(a, b, sc, o, sd):
        return core.fbm_at(a, b, sc, o, seed * 97 + sd).reshape(a.shape)
    wx = (n(X, Zc, 0.7, 3, 1) - 0.5) * 0.5
    wz = (n(X, Zc, 0.7, 3, 2) - 0.5) * 0.5
    env = np.zeros_like(X)
    for q in range(int(r.integers(2, 5))):
        px, ph, pw = r.uniform(-0.5, 0.5), r.uniform(0.55, 1.0), r.uniform(0.28, 0.5)
        env = np.maximum(env, ph * np.exp(-(((X - px) / pw) ** 2 + (Zc / 0.62) ** 2)))
    env = np.maximum(env, 0.55 * np.exp(-((X / 0.75) ** 2 + (Zc / 0.7) ** 2) ** 1.5))
    r1 = 1 - np.abs(2 * n(X * 1.0 + wx, Zc * 0.8 + wz, 0.62, 5, 3) - 1)
    r2 = 1 - np.abs(2 * n(X + wx * 0.5, Zc * 0.8, 0.17, 4, 4) - 1)
    hgt = env * (0.34 + 0.52 * r1 ** 1.4 + 0.14 * r2)
    hgt = hgt / hgt.max()
    hf = (hgt * hpx * ssx).astype(np.float64)
    # light
    dxw = 1.0
    dzw = 0.55 * nx / nz
    gz, gx = np.gradient(hf)
    gx = gx / dxw
    gzw = -gz / dzw                         # derivative toward the back
    nn = np.sqrt(gx * gx + gzw * gzw + 1)
    lam = (-gx * 0.74 + 0.40 - gzw * 0.54) / nn
    tex = n(X * 3.0, Zc * 2.0 + hgt * 3.0, 0.12, 3, 5)
    litf = sp.ss(0.30, 0.37, lam + 0.10 * (tex - 0.5))[..., None]
    slope = np.sqrt(gx * gx + gzw * gzw)
    snow = (sp.ss(0.30, 0.46, hgt + 0.25 * (tex - 0.5)) * sp.ss(2.9, 1.5, slope + 1.2 * (tex - 0.5)))[..., None]
    rock = np.array([0.37, 0.34, 0.58]) + (np.array([0.86, 0.60, 0.54]) - np.array([0.37, 0.34, 0.58])) * litf
    rock = rock * (0.86 + 0.28 * tex[..., None])
    sn = np.array([0.60, 0.62, 0.90]) + (np.array([1.0, 0.87, 0.80]) - np.array([0.60, 0.62, 0.90])) * litf
    col = rock + (sn - rock) * snow
    col = col * (1 - haze) + np.asarray(fogc, np.float64) * haze
    oh = int(hpx * ssx * 1.35) + 8
    out = np.zeros((oh, nx, 4), np.float32)
    depth_px = hpx * ssx * 0.30
    _mrender(hf, np.ascontiguousarray(col, np.float64), out, oh - depth_px - 2, depth_px / nz, hpx * ssx * 0.20)
    # reduce to 1x
    out = out[:oh // 2 * 2, :nx // 2 * 2]
    out = out.reshape(oh // 2, 2, nx // 2, 2, 4).mean(axis=(1, 3))
    return out.astype(np.float32)


# ------------------------------------------------------------ rows of cloud
_MIP = {}


def _mips(kind, seed):
    key = (kind, seed)
    if key not in _MIP:
        Fm = sp.cloud_sprite(kind, seed)
        if kind != "band":
            # seen from above the lumps are low soft domes: squash, and calm the light
            Fm = ndimage.zoom(Fm, (SQUASH, 1, 1), order=1)
            side = np.linspace(-1, 1, Fm.shape[1], dtype=np.float32)[None, :]
            Fm[..., 1] = np.clip(ndimage.gaussian_filter(Fm[..., 1], 3.2) * 1.02 - 0.08 + 0.10 * side, 0, 1)
            Fm[..., 0] = ndimage.gaussian_filter(Fm[..., 0], 1.2)
        lv = [Fm]
        while lv[-1].shape[1] > 24:
            a = lv[-1]
            a = a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2]
            lv.append(a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, 3).mean(axis=(1, 3)))
        _MIP[key] = lv
    return _MIP[key]


def _blit(dst, oy, kind, seed, cx, base_y, sc, cols, op=1.0, dl=0.0):
    """Place a cloud sprite on an RGBA strip whose top row is canvas y = oy and
    whose column 0 is canvas x = -M. The sprite base line sits at base_y."""
    lv = _mips(kind, seed)
    li = 0
    s = sc
    while s < 0.5 and li < len(lv) - 1:
        s *= 2
        li += 1
    Fm = lv[li]
    h, w = Fm.shape[:2]
    x0 = cx + M - w * s / 2
    y0 = base_y - oy - 0.84 * h * s
    X0, Y0 = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
    X1, Y1 = min(int(x0 + w * s) + 2, dst.shape[1]), min(int(y0 + h * s) + 2, dst.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    Fw = core.warp(Fm, s, x0 - X0, y0 - Y0, out_hw=(Y1 - Y0, X1 - X0))
    if dl != 0.0:
        Fw[..., 1] = np.clip(Fw[..., 1] + dl, 0, 1)
    core.over_rgba(dst[Y0:Y1, X0:X1], sp.colourise(Fw, cols, op))


def _rows():
    """Every row of cumulus tops, far to near: list of (z, y0, RGBA strip).
    The lumps gather into big rolling banks (mass_at) with valleys between;
    lumps in the shadow of a bank go lavender; the far field stays calm."""
    rows = []
    zs = []
    z = 150.0
    while z > 1.75:
        zs.append(z)
        z /= 1.2 if z < 40 else 1.3
    for ri, z in enumerate(zs):
        r = core.rng("vrow", ri)
        nearb = float(sp.ss(3.6, 2.0, z))
        sc = S0 / z * (1.0 + 0.5 * nearb)
        far = z > 34
        kind_w = 1600 if far else 960
        kind_h = 170 if far else 760 * SQUASH
        sh = int(kind_h * max(sc, 0.02) * 1.9) + 40
        base0 = HZ + F / z
        oy = int(base0 - sh + 12)
        strip = np.zeros((sh + int(CURVE) + 40, CW, 4), np.float32)
        step = kind_w * sc * (0.36 if not far else 0.42)
        x = -M - r.uniform(0, step)
        fz = float(_fz(z))
        calm = float(sp.ss(14.0, 40.0, z))
        while x < W + M + step:
            jx = x + r.uniform(-0.25, 0.25) * step
            x += step * r.uniform(0.7, 1.3)
            Xw = (jx - 960.0) / F * z
            mass = float(mass_at(Xw, z))
            shd = float(shade_at(Xw, z)) * (1 - 0.6 * calm)
            lift = F / z * (0.36 * (mass - 0.35)) * (1 - 0.7 * calm)        # banks stand higher
            by = float(hz_at(jx)) + F / z - lift + r.uniform(-0.1, 0.1) * F / z * 0.17
            skip = False
            for (gx, gy, rx, ry) in GAPS:
                if ((jx - gx) / (rx * 0.80)) ** 2 + ((by - gy - ry * 0.9) / (ry * 1.7)) ** 2 < 1.0:
                    skip = True
            u1, u2, u3 = r.uniform(), r.uniform(), r.uniform()
            if far:
                kind, seed = "band", int(r.integers(0, 7))
            else:
                kind = "cumulus" if u1 < 0.74 else ("tower" if (u1 < 0.84 and mass > 0.6) else "bar")
                seed = int(r.integers(0, {"cumulus": 7, "tower": 4, "bar": 6}[kind]))
            # valleys are nearly empty, banks are crowded with bigger lumps
            keep = 0.24 + 1.2 * sp.ss(0.30, 0.62, mass) if not far else 0.9
            if skip or u2 > keep or (mass < 0.07 and not far):
                continue
            scj = sc * (0.80 + 0.42 * mass if not far else 1.0) * r.uniform(0.85, 1.2) * (0.8 if kind == "tower" else 1.0)
            scj = min(scj, 1.25)
            fogc = _fog(np.float32(jx))
            hz_ = 0.86 * fz
            sw = float(_sunward(jx, 700.0))
            cols = [palette.mix(c, fogc, hz_ * (0.95 if i < 2 else 0.75)) for i, c in enumerate((DEEP, SHAD, TURN, LIT, HI))]
            cols[3] = palette.mix(cols[3], np.array([1.0, 0.93, 0.76], np.float32), 0.5 * sw)
            if calm > 0:
                mid = palette.mix(cols[2], cols[3], 0.6)
                cols = [palette.mix(c, mid, 0.45 * calm) for c in cols]
            if z < 3.2:      # the nearest tops are quiet masses, not patterns
                mid = palette.mix(cols[1], cols[3], 0.55)
                cols = [palette.mix(c, mid, 0.42 * float(sp.ss(3.2, 2.2, z))) for c in cols]
            _blit(strip, oy, kind, seed, jx, by, scj, cols, dl=-0.26 * shd + 0.06 * (mass - 0.5) + 0.5 * (mass - float(mass_at(Xw - 0.5, z))))
        if z < 2.7:      # nearest tops: large, soft, out of focus
            strip = core.gblur(strip, 3.0 + (2.7 - z) * 9.0)
        rows.append((z, oy, strip))
    return rows


_L = None
_GLINT = None


def _layers():
    """All baked layers, back to front: (depth z, x0, y0, rgba). Cached on disk."""
    global _L, _GLINT
    if _L is not None:
        return _L
    import os
    path = os.path.join(sp.CACHE, "vista_layers_v%d.npz" % VERSION)
    if os.path.exists(path):
        d = np.load(path)
        meta = d["meta"]
        _L = [(float(m[0]), float(m[1]), float(m[2]), d["L%03d" % i].astype(np.float32)) for i, m in enumerate(meta)]
        _GLINT = (float(d["glint_y"]), d["glint"].astype(np.float32))
        return _L
    layers = []
    fl, fy, gm = _floor()
    _GLINT = (fy, gm)
    layers.append((40.0, -M, fy, fl))
    items = [(z, "row", oy, strip) for (z, oy, strip) in _rows()]
    for (mx, my, mw, mh, sd) in MOUNTAINS:
        z = F / (my - HZ)
        hz_ = float(_fz(z)) * 0.8
        img = _mountain(mw, mh, sd, 0.12 + hz_, _fog(np.float32(mx)))
        items.append((z * 0.98, "mtn", (mx - img.shape[1] / 2, my - img.shape[0] + mh * 0.16), img))
    items.sort(key=lambda it: -it[0])
    for it in items:
        if it[1] == "row":
            layers.append((it[0], -M, it[2], it[3]))
        else:
            layers.append((it[0], it[2][0], it[2][1], it[3]))
    _L = layers
    arrs = {"L%03d" % i: l[3].astype(np.float16) for i, l in enumerate(layers)}
    tmp = path + ".tmp%d.npz" % os.getpid()
    np.savez(tmp, meta=np.array([(l[0], l[1], l[2]) for l in layers], np.float64), glint=gm.astype(np.float16),
             glint_y=np.float64(fy), **arrs)
    os.replace(tmp, path)
    return _L


def _par(z):
    return float(np.clip((7.0 / z) ** 0.5, 0.06, 2.2))


def _place(pic, cam, p, x0, y0, layer):
    s, tx, ty = cam.xf(p)
    h, w = layer.shape[:2]
    fx0, fy0 = x0 * s + tx, y0 * s + ty
    X0, Y0 = max(int(math.floor(fx0)), 0), max(int(math.floor(fy0)), 0)
    X1, Y1 = min(int(fx0 + w * s) + 2, W), min(int(fy0 + h * s) + 2, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    out = core.warp(layer, s, fx0 - X0, fy0 - Y0, out_hw=(Y1 - Y0, X1 - X0))
    core.over(pic[Y0:Y1, X0:X1], out)


_SKY = None


def render(cam, f, t, star=None):
    """star: optional (x, y, power) of the shooting star on screen: its warm
    light slides over the cloud tops below it and glints in the rivers."""
    global _SKY
    if _SKY is None:
        _SKY = _sky()
    p0 = 0.03
    s0, tx0, ty0 = cam.xf(p0)
    pic = core.warp(_SKY, s0, tx0 - M * s0, ty0 - M * s0)
    # stars high up, twinkling
    from tr2 import sky as skymod
    st = skymod._stars(core.Camera(1600, 900, tilt=2300.0 + (ty0 - (H / 2 - H / 2 * s0)), pan=-(tx0 - (W / 2 - W / 2 * s0))),
                       t, 0.9, 0.0)
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    lum = pic.mean(-1, keepdims=True)
    pic = 1 - (1 - np.clip(pic, 0, 1)) * (1 - np.clip(st * np.clip(1.15 - 2.0 * lum, 0, 1), 0, 1))

    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    hz = hz_at((xs - tx0) / s0) * s0 + ty0
    below = sp.ss(-4.0, 30.0, ys - hz)
    layers = _layers()
    for li, (z, x0, y0, layer) in enumerate(layers):
        # the cloud rows drift slowly left on the wind, nearer ones faster (mountains and floor stay)
        dx = -t * (1.0 + 2.4 * _par(z)) if (li > 0 and x0 == -M) else 0.0
        _place(pic, cam, _par(z), x0 + dx, y0, layer)
        if li == 0:
            # rivers (on the floor, under the cloud rows): a faint shimmer always,
            # a blaze when the star passes over
            gy0, gm = _GLINT
            s40, tx40, ty40 = cam.xf(_par(40.0))
            glw = core.warp(gm, s40, tx40 - M * s40, ty40 + gy0 * s40)[..., 0]
            shim = 0.10 + 0.08 * math.sin(t * 5.0) * math.sin(t * 1.7 + 1.0)
            core.add_light(pic, glw, (1.0, 0.93, 0.72), shim)
            if star is not None:
                near = np.exp(-((xs - star[0]) / 230.0) ** 2)
                core.add_light(pic, (glw * near).astype(np.float32), (1.0, 0.95, 0.75), min(1.0, 0.9 * star[2]))
                core.add_light(pic, core.gblur((glw * near).astype(np.float32), 5.0), (1.0, 0.85, 0.55), min(1.0, 0.7 * star[2]))
    if star is not None:
        sx_, sy_, pw = star
        d2 = ((xs - sx_) / 400.0) ** 2 + ((ys - (hz + 105.0)) / 120.0) ** 2
        pool = np.exp(-d2) * below * pw
        lumc = pic.mean(-1)
        core.add_light(pic, (pool * (0.35 + 0.65 * lumc)).astype(np.float32), (1.0, 0.84, 0.56), 0.80)

    # the air on top: haze on the horizon, the thin line of atmosphere, sun bloom over the sea
    sx = SUN_X * s0 + tx0
    sy = (float(hz_at(SUN_X)) - 16.0) * s0 + ty0
    near_sun = np.exp(-np.abs(xs - sx) / 900.0)
    haze = np.exp(-((ys - hz - 3) / 20.0) ** 2) * (0.35 + 0.65 * near_sun)
    core.add_light(pic, haze.astype(np.float32), (1.0, 0.86, 0.68), 0.6)
    line = np.exp(-((ys - hz + 6) / 2.4) ** 2) * (0.3 + 0.7 * near_sun) + 0.3 * np.exp(-((ys - hz + 11) / 9.0) ** 2)
    core.add_light(pic, line.astype(np.float32), (0.80, 0.93, 1.0), 0.5)
    d = np.sqrt(((xs - sx) * 0.55) ** 2 + (ys - sy) ** 2)
    bloom = 0.5 * np.exp(-d / 330.0) + 0.5 * np.exp(-(d / 110.0) ** 2)
    core.add_light(pic, bloom.astype(np.float32), (1.0, 0.82, 0.52), 0.55)
    return np.clip(pic, 0, 1).astype(np.float32)
