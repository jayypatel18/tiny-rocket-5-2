"""The living sky behind every hilltop shot.

    pic = sky.render(tod, cam, f, t, stars=None, clouds=1.0, star_turn=0.0,
                     shooting=None, sun=True, fade_star=None)

SKY coordinates: the default screen is SX 0..1920, SY 0..1080 with the horizon
at SY = 520. The sky is tall: it goes on up to SY = -2160 (three screens).
Dolly and crane moves (cam.cx, cam.cy) move it by 0.02 and 0.05 of the move.
cam.tilt and cam.pan (the camera rotating) scroll it 1:1 in screen px, so a
tilt of 2500 px looks at the high sky near SY = -2500.
"""
import math

import numpy as np
from numba import njit

from tr2 import core, palette
from tr2 import sky_paint as sp

W, H = core.W, core.H
HZ = 520.0            # horizon line on the default screen
TOP = -2800.0         # top of the tall sky
TILT = 0.05           # sky scroll per px of camera crane rise (cam.cy); cam.tilt scrolls 1:1
PAN = 0.02
POLE = (1500.0, -200.0)   # the point the stars turn around (sky coords)
CX0, CX1 = -192, 2112     # extent of the painted texture canvas
CY0, CY1 = -2900, 1140


# ------------------------------------------------------------- sky mapping
def view(cam):
    """(s, ox, oy): sky point (SX, SY) lands on screen at (SX*s + ox, SY*s + oy)."""
    s = cam.zoom * (cam.dolly ** PAN)
    pan = (cam.cx - cam.cx0) * PAN
    rise = max((cam.cy0 - cam.cy) * TILT, -300.0)
    # cam.tilt / cam.pan: the camera rotates, so the sky moves 1:1 in screen px
    ox = W / 2 - (W / 2 + pan) * s + cam.shake[0] * 0.3 - getattr(cam, "pan", 0.0)
    oy = H / 2 - (H / 2 - rise) * s + cam.shake[1] * 0.3 + getattr(cam, "tilt", 0.0)
    return s, ox, oy


def to_screen(cam, SX, SY):
    s, ox, oy = view(cam)
    return SX * s + ox, SY * s + oy


def _v_of(SY):
    """Gradient height v (0 zenith side .. 1 horizon) and a zenith deepening."""
    SY = np.asarray(SY, np.float32)
    lo = 1.0 - 0.9 * (540.0 - SY) / 540.0
    hi = 0.1 * np.exp(np.minimum(SY, 0) / 900.0)
    v = np.where(SY >= 0, lo, hi)
    deep = 1.0 - 0.22 * (1 - np.exp(np.minimum(SY, 0) / 1300.0))
    return np.clip(v, 0, 1), deep


def tod_name(tod):
    if isinstance(tod, str):
        return tod
    if "name" in tod:
        return tod["name"]
    z = tod["sky"][0][1]
    best, bd = "dusk", 1e9
    for n, T in palette.TOD.items():
        d = float(((T["sky"][0][1] - z) ** 2).sum() + 0.5 * ((T["key"] - tod["key"]) ** 2).sum())
        if d < bd:
            best, bd = n, d
    return best


def sun_pos(tod):
    """Where the sun or moon sits in sky coords, from the palette direction."""
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    sx, sy = T["sun"]
    x = 960 + sx * 1080
    y = HZ - float(np.clip((-sy - 0.42) * 1500, -70, 2000))
    return x, y


# --------------------------------------------------------- painted texture
def _canvas():
    def make():
        h, w = CY1 - CY0, CX1 - CX0
        f1 = sp.stroke_field(h, w, 3100, (320, 1000), (36, 110), 0.0, 0.07, seed=11, opacity=0.9)
        f2 = sp.stroke_field(h, w, 6200, (120, 420), (14, 44), 0.0, 0.14, seed=12, opacity=0.75)
        big = core.fbm(h, w, 600, 3, 21, aspect=2.0) - 0.5
        dv = 0.55 * f1 + 0.3 * f2 + 1.6 * big
        grain = 0.6 * (core.fbm(h, w, 5, 2, 31) - 0.5) + 0.5 * (core.fbm(h, w, 46, 3, 32, aspect=4.0) - 0.5) + 0.2 * f2
        return np.stack([dv, grain], -1).astype(np.float16)
    return sp.cached("canvas_v3", make).astype(np.float32)


_CANVAS = None


def _tex(cam):
    global _CANVAS
    if _CANVAS is None:
        _CANVAS = _canvas()
    s, ox, oy = view(cam)
    return core.warp(_CANVAS, s, ox + CX0 * s, oy + CY0 * s)


# -------------------------------------------------------------------- stars
def _catalogue():
    def make():
        r = core.rng("stars", 5)
        n = 66000
        R = 3350.0
        rad = R * np.sqrt(r.uniform(0, 1, n))
        th = r.uniform(0, 2 * np.pi, n)
        # the milky way: a band of extra faint stars, offset from the pole
        m = 40000
        u = r.uniform(-R, R, m)
        vv = r.normal(0, 210, m) + 60 * np.sin(u / 500.0)
        bx, by = _band_xy(u, vv)
        rad = np.r_[rad, np.hypot(bx, by)]
        th = np.r_[th, np.arctan2(by, bx)]
        N = n + m
        b = r.uniform(0, 1, N) ** 9.0
        b[n:] *= 0.55
        b[r.integers(0, n, 100)] = r.uniform(0.75, 1.0, 100)      # the few bright ones
        kind = r.uniform(0, 1, N)
        col = np.ones((N, 3))
        col[kind < 0.30] = (0.72, 0.82, 1.0)
        col[(kind >= 0.30) & (kind < 0.55)] = (0.90, 0.94, 1.0)
        col[(kind >= 0.72) & (kind < 0.88)] = (1.0, 0.93, 0.74)
        col[(kind >= 0.88) & (kind < 0.96)] = (1.0, 0.80, 0.58)
        col[kind >= 0.96] = (1.0, 0.66, 0.55)
        col += r.normal(0, 0.03, (N, 3))
        ph = r.uniform(0, 2 * np.pi, N)
        fr = r.uniform(1.2, 5.0, N)
        return np.c_[rad, th, b, col, ph, fr].astype(np.float32)
    return sp.cached("stars_v4", make)


BAND_ANG = 0.285
BAND_OFF = 575.0


def _band_xy(u, v):
    """Milky way band coords (u along, v across) to pole centred x, y."""
    c, s = math.cos(BAND_ANG), math.sin(BAND_ANG)
    return u * c - (v + BAND_OFF) * s, u * s + (v + BAND_OFF) * c


@njit(cache=True, fastmath=True)
def _splat(out, xs, ys, rad, col, amp):
    h, w, _ = out.shape
    for k in range(xs.shape[0]):
        r = rad[k]
        R = int(r * 3.0) + 1
        x0 = max(0, int(xs[k]) - R)
        x1 = min(w, int(xs[k]) + R + 2)
        y0 = max(0, int(ys[k]) - R)
        y1 = min(h, int(ys[k]) + R + 2)
        inv = 1.0 / (2 * r * r)
        for j in range(y0, y1):
            for i in range(x0, x1):
                d2 = (i - xs[k]) ** 2 + (j - ys[k]) ** 2
                g = math.exp(-d2 * inv) * amp[k]
                out[j, i, 0] += g * col[k, 0]
                out[j, i, 1] += g * col[k, 1]
                out[j, i, 2] += g * col[k, 2]


@njit(cache=True, fastmath=True)
def _glint(out, x, y, ln, amp, c0, c1, c2, ang):
    """A four point glint: two thin crossed strokes, a little uneven."""
    h, w, _ = out.shape
    for arm in range(4):
        a = ang + arm * math.pi / 2
        L = ln * (1.0 if arm % 2 == 0 else 0.72)
        n = int(L * 2)
        for q in range(n):
            d = q * 0.5
            px = x + math.cos(a) * d
            py = y + math.sin(a) * d
            g = amp * (1 - d / L) ** 2.2
            i = int(px)
            j = int(py)
            fx = px - i
            fy = py - j
            if 0 <= i < w - 1 and 0 <= j < h - 1:
                for (jj, ii, wt) in ((j, i, (1 - fx) * (1 - fy)), (j, i + 1, fx * (1 - fy)),
                                     (j + 1, i, (1 - fx) * fy), (j + 1, i + 1, fx * fy)):
                    out[jj, ii, 0] += g * wt * c0 * 0.5
                    out[jj, ii, 1] += g * wt * c1 * 0.5
                    out[jj, ii, 2] += g * wt * c2 * 0.5


_CAT = None


def _stars(cam, t, amount, turn):
    """Additive star light, H x W x 3."""
    global _CAT
    if _CAT is None:
        _CAT = _catalogue()
    c = _CAT
    s, ox, oy = view(cam)
    out = np.zeros((H, W, 3), np.float32)
    # milky way glow, at quarter size
    q = 4
    ys, xs = np.mgrid[0:H // q, 0:W // q].astype(np.float32)
    SX = (xs * q + q / 2 - ox) / s - POLE[0]
    SY = (ys * q + q / 2 - oy) / s - POLE[1]
    ct, st = math.cos(-turn), math.sin(-turn)
    px = SX * ct - SY * st
    py = SX * st + SY * ct
    cb, sb = math.cos(-BAND_ANG), math.sin(-BAND_ANG)
    u = px * cb - py * sb
    v = px * sb + py * cb - BAND_OFF - 60 * np.sin(u / 500.0)
    n1 = core.fbm_at(u, v * 1.6, 230.0, 5, 77).reshape(u.shape)
    n2 = core.fbm_at(u + 900, v * 2.2, 90.0, 4, 78).reshape(u.shape)
    band = np.exp(-(v / 250.0) ** 2)
    core_b = np.exp(-(v / 110.0) ** 2)
    dust = sp.ss(0.50, 0.66, n2) * core_b                       # dark lanes
    glow = band * (0.25 + 1.1 * n1 ** 1.6) * (1 - 0.75 * dust) + 0.5 * core_b * n1 * (1 - dust)
    glow = glow.astype(np.float32)
    mw = np.stack([glow * 0.50, glow * 0.56, glow * 0.80], -1) * 0.17
    mw[..., 0] += glow * core_b * 0.035 * n2                    # a warmer heart
    mw = core.gblur(mw, 1.2)
    mw = np.repeat(np.repeat(mw, q, 0), q, 1)
    out += core.gblur(mw, 2.5) * amount
    # the stars
    a = c[:, 1] + turn
    X = (POLE[0] + c[:, 0] * np.cos(a)) * s + ox
    Y = (POLE[1] + c[:, 0] * np.sin(a)) * s + oy
    b = c[:, 2]
    # at low amounts only the bright ones are out
    thr = (1 - amount) ** 2 * 0.5
    keep = (X > -8) & (X < W + 8) & (Y > -8) & (Y < H + 8) & (b > thr)
    X, Y, b, cc, ph, fr = X[keep], Y[keep], b[keep], c[keep, 3:6], c[keep, 6], c[keep, 7]
    tw = 1.0 + (0.10 + 0.28 * b) * np.sin(t * fr + ph) * np.sin(t * fr * 0.37 + ph * 1.7)
    amp = (0.13 + 0.22 * np.abs(np.sin(ph * 7.0)) + 1.2 * b ** 0.8) * tw * min(1.0, 0.35 + amount * 1.5)
    rad = (0.55 + 1.55 * b ** 1.4) * max(0.8, s ** 0.5)
    _splat(out, X.astype(np.float64), Y.astype(np.float64), rad.astype(np.float64),
           np.ascontiguousarray(cc, np.float64), amp.astype(np.float64))
    big = np.nonzero(b > 0.8)[0]
    for k in big:
        halo_r = 5.0 + 9.0 * b[k]
        _splat(out, X[k:k + 1].astype(np.float64), Y[k:k + 1].astype(np.float64), np.array([halo_r]),
               np.ascontiguousarray(cc[k:k + 1], np.float64), np.array([0.10 * tw[k] * amount]))
        if b[k] > 0.955:
            _glint(out, float(X[k]), float(Y[k]), 5.0 + 13.0 * b[k], 0.8 * tw[k] * amount,
                   float(cc[k, 0]), float(cc[k, 1]), float(cc[k, 2]), 0.12 + 0.5 * math.sin(ph[k]))
    return out


# ------------------------------------------------------------------- clouds
# One layout per look. Each layer: kinds, count, SY range, scale range,
# drift px/s, haze toward the sky, opacity, tilt gain, pan parallax.
LOOKS = {
    "dusk": [
        dict(kinds=("band",), n=16, y=(455, 600), sc=(0.20, 0.38), v=1.4, haze=0.50, op=0.9, tg=1.0, pp=0.02, tint=(1.0, 0.92, 0.82)),
        dict(kinds=("band", "band", "bar"), n=12, y=(350, 470), sc=(0.36, 0.62), v=2.4, haze=0.36, op=0.95, tg=1.0, pp=0.025, tint=(1.0, 0.96, 0.90)),
        dict(kinds=("cumulus", "bar", "cumulus", "band"), n=6, y=(180, 340), sc=(0.42, 0.66), v=5.0, haze=0.20, op=1.0, tg=1.06, pp=0.035),
        dict(kinds=("cumulus", "tower"), n=2, y=(10, 150), sc=(0.66, 0.86), v=10.0, haze=0.05, op=1.0, tg=1.15, pp=0.06),
        dict(kinds=("cumulus", "wisp", "bar"), n=12, y=(-2700, -350), sc=(0.6, 1.0), v=8.0, haze=0.12, op=1.0, tg=1.1, pp=0.04),
    ],
    "night": [
        dict(kinds=("band", "wisp", "band"), n=9, y=(330, 480), sc=(0.35, 0.6), v=2.0, haze=0.45, op=0.6, tg=1.0, pp=0.02),
        dict(kinds=("wisp",), n=4, y=(60, 300), sc=(0.7, 1.1), v=5.0, haze=0.25, op=0.42, tg=1.06, pp=0.035),
        dict(kinds=("wisp",), n=8, y=(-2700, -300), sc=(0.8, 1.4), v=7.0, haze=0.2, op=0.4, tg=1.1, pp=0.04),
    ],
    "dawn": [
        dict(kinds=("band", "wisp", "band"), n=12, y=(380, 520), sc=(0.26, 0.5), v=2.0, haze=0.40, op=0.95, tg=1.0, pp=0.02, tint=(1.0, 0.84, 0.84)),
        dict(kinds=("wisp", "band", "bar"), n=6, y=(150, 360), sc=(0.5, 0.85), v=5.0, haze=0.2, op=0.85, tg=1.06, pp=0.035, tint=(1.0, 0.88, 0.89)),
        dict(kinds=("cumulus",), n=2, y=(-60, 110), sc=(0.6, 0.85), v=10.0, haze=0.05, op=1.0, tg=1.15, pp=0.06, tint=(1.0, 0.92, 0.93)),
        dict(kinds=("wisp", "wisp", "cumulus", "wisp", "bar"), n=22, y=(-2750, -330), sc=(0.7, 1.35), v=8.0, haze=0.12, op=0.8, tg=1.1, pp=0.04, tint=(1.0, 0.84, 0.88)),
    ],
    "morning": [
        dict(kinds=("cumulus", "bar"), n=10, y=(350, 470), sc=(0.22, 0.4), v=2.5, haze=0.5, op=0.95, tg=1.0, pp=0.02),
        dict(kinds=("cumulus",), n=6, y=(150, 330), sc=(0.45, 0.75), v=6.0, haze=0.2, op=1.0, tg=1.06, pp=0.035),
        dict(kinds=("cumulus", "tower"), n=3, y=(-140, 100), sc=(1.0, 1.4), v=12.0, haze=0.03, op=1.0, tg=1.15, pp=0.06),
        dict(kinds=("cumulus", "wisp"), n=12, y=(-2700, -350), sc=(0.7, 1.4), v=8.0, haze=0.1, op=1.0, tg=1.1, pp=0.04),
    ],
}
LOOK_OF = dict(dusk="dusk", night="night", predawn="dawn", launch="dawn", sunrise="dawn", morning="morning")
NSPR = dict(cumulus=7, tower=4, bar=6, wisp=6, band=7)
PERIOD = 3400.0


def _layout(look):
    out = []
    for li, lay in enumerate(LOOKS[look]):
        r = core.rng("layout", look, li)
        n = lay["n"]
        xs = (np.arange(n) + r.uniform(0.1, 0.9, n)) / n * PERIOD
        r.shuffle(xs)
        items = []
        for k in range(n):
            kind = lay["kinds"][int(r.integers(0, len(lay["kinds"])))]
            items.append(dict(kind=kind, seed=int(r.integers(0, NSPR[kind])), x=float(xs[k]),
                              y=float(r.uniform(*lay["y"])), sc=float(r.uniform(*lay["sc"])),
                              rank=float((k + 0.5) / n), flip=bool(r.integers(0, 2))))
        out.append((lay, items))
    return out


_LAY = {}


def _clouds(tod, look, cam, t, amount, skyc_fn):
    """Premultiplied RGBA layer of all clouds, back to front."""
    if look not in _LAY:
        _LAY[look] = _layout(look)
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    s, ox, oy = view(cam)
    rise = (cam.cy0 - cam.cy) * TILT
    pan = (cam.cx - cam.cx0)
    sun_left = T["sun"][0] < 0
    out = np.zeros((H, W, 4), np.float32)
    for lay, items in _LAY[look]:
        for it in items:
            vis = core.clamp((amount * 1.15 - it["rank"]) * 5.0)
            if vis <= 0:
                continue
            w, h = sp.SIZES[it["kind"]]
            sc = it["sc"] * s
            SX = ((it["x"] + lay["v"] * t) % PERIOD) - 700.0 - (lay["pp"] - PAN) * pan
            SY = it["y"] + (lay["tg"] - 1.0) * rise
            x0 = SX * s + ox - w * sc / 2
            y0 = SY * s + oy - h * sc / 2
            bx0, by0 = int(math.floor(x0)), int(math.floor(y0))
            bw, bh = int(w * sc) + 2, int(h * sc) + 2
            X0, Y0 = max(bx0, 0), max(by0, 0)
            X1, Y1 = min(bx0 + bw, W), min(by0 + bh, H)
            if X1 <= X0 or Y1 <= Y0:
                continue
            F = sp.cloud_sprite(it["kind"], it["seed"])
            # sprites are lit from the right; mirror them for a sun on the left
            if sun_left:
                F = F[:, ::-1]
            Fw = core.warp(F, sc, x0 - X0, y0 - Y0, out_hw=(Y1 - Y0, X1 - X0))
            cols = sp.tones(T, skyc_fn(SY), lay["haze"], tint=lay.get("tint"))
            rgba = sp.colourise(Fw, cols, lay["op"] * vis)
            core.over_rgba(out[Y0:Y1, X0:X1], rgba)
    return out


# -------------------------------------------------------------- the render
GLOW = dict(dusk=0.80, night=0.20, predawn=0.34, launch=0.46, sunrise=0.8, morning=0.5)
STARS = dict(dusk=0.15, night=1.0, predawn=0.3, launch=0.12, sunrise=0.0, morning=0.0)


def render(tod, cam, f, t, stars=None, clouds=1.0, star_turn=0.0, shooting=None, sun=True, fade_star=None):
    """The sky for this frame, float RGB 1080 x 1920.
    tod: a palette name or a palette.blend_tod dict. stars: 0..1 (None = the
    usual amount for this time of day). clouds: 0..1 amount. star_turn: radians.
    shooting: (t0, x0, y0, x1, y1[, duration]) in screen px. fade_star: (x, y, amount)."""
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    name = tod_name(tod)
    s, ox, oy = view(cam)
    tex = _tex(cam)
    ys = np.arange(H, dtype=np.float32)
    SYrow = (ys + 0.5 - oy) / s
    v, deep = _v_of(SYrow)
    # brushed gradient: strokes push the gradient up and down, so the bands of
    # colour have brush shaped borders
    vv = np.clip(v[:, None] + tex[..., 0] * 0.045 * (0.35 + 0.65 * sp.ss(0.0, 0.3, v))[:, None], 0, 1)
    pic = palette.sky_color(T, vv)
    pic *= (deep[:, None, None] * (1.0 + 0.030 * tex[..., 1:2]))

    # sun or moon glow, and a band of light along the horizon on its side
    gx, gy = sun_pos(T)
    px, py = gx * s + ox, gy * s + oy
    hz_y = HZ * s + oy
    xs = np.arange(W, dtype=np.float32)[None, :]
    yy = ys[:, None]
    gl = GLOW.get(name, 0.4) if sun else 0.0
    if gl > 0:
        d = np.sqrt(((xs - px) * 0.52) ** 2 + (yy - py) ** 2) / s
        d = d * (1.0 + 0.10 * tex[..., 0])
        g = 0.55 * np.exp(-d / 760.0) + 0.55 * np.exp(-(d / 300.0) ** 2) + 0.9 * np.exp(-(d / 80.0) ** 2)
        band = np.exp(-((yy - hz_y) / (120.0 * s)) ** 2) * np.exp(-np.abs(xs - px) / (1500.0 * s))
        gcol = palette.mix(T["rim"], T["key"] / max(1.0, float(T["key"].max())), 0.3)
        core.add_light(pic, (g + 0.5 * band).astype(np.float32), gcol, gl)
        if name == "night":
            _moon(pic, px, py, 30.0 * s, T)
        elif name == "morning":
            dd = np.sqrt((xs - px) ** 2 + (yy - py) ** 2) / s
            core.add_light(pic, np.exp(-(dd / 46.0) ** 4).astype(np.float32), (1.0, 0.98, 0.9), 0.95)

    def skyc(SY):
        vq, dq = _v_of(np.float32(SY))
        return palette.sky_color(T, vq) * dq

    cl = None
    if clouds > 0:
        cl = _clouds(T, LOOK_OF.get(name, "dawn"), cam, t, clouds, skyc)

    amt = STARS.get(name, 0.0) if stars is None else stars
    if amt > 0.003:
        st = _stars(cam, t, amt, star_turn)
        # the air near the horizon and any glow swallow the faint ones
        ext = sp.ss(hz_y + 10 * s, hz_y - 330 * s, yy)
        lum = pic.mean(-1, keepdims=True)
        st *= ext[..., None] * np.clip(1.25 - 1.7 * lum, 0.0, 1.0)
        if cl is not None:
            st *= (1 - cl[..., 3:4]) ** 2
        pic = 1 - (1 - np.clip(pic, 0, 1)) * (1 - np.clip(st, 0, 1))
    if cl is not None:
        core.over(pic, cl)
    if shooting is not None:
        _shooting(pic, t, shooting)
    if fade_star is not None and fade_star[2] > 0.003:
        _fade_star(pic, fade_star, t)
    return np.clip(pic, 0, 1).astype(np.float32)


def _moon(pic, px, py, r, T):
    R = int(r * 2) + 2
    x0, y0 = int(px) - R, int(py) - R
    if x0 > W or y0 > H or x0 + 2 * R < 0 or y0 + 2 * R < 0:
        return
    yy, xx = np.mgrid[y0:y0 + 2 * R, x0:x0 + 2 * R].astype(np.float32)
    d = np.hypot(xx - px, yy - py)
    disc = sp.ss(r + 0.9, r - 0.9, d)
    seas = core.fbm(2 * R, 2 * R, r * 0.9, 3, 5)
    col = np.array([0.93, 0.95, 1.0], np.float32) * (0.90 + 0.10 * (1 - sp.ss(0.45, 0.6, seas)))[..., None]
    lay = core.premul(col.astype(np.float32), disc.astype(np.float32))
    core.over(pic, lay, x0, y0)


def _shooting(pic, t, sh):
    t0, x0, y0, x1, y1 = sh[:5]
    dur = sh[5] if len(sh) > 5 else 0.85
    u = (t - t0) / dur
    if u < 0 or u > 1.35:
        return
    head = core.ease_out(min(u, 1.0), 1.6)
    fade = 1.0 if u < 0.85 else max(0.0, 1 - (u - 0.85) / 0.5)
    tail0 = max(0.0, head - 0.6 * (0.4 + 0.6 * math.sin(math.pi * min(u, 1.0)) + 0.3))
    n = 160
    q = np.linspace(tail0, head, n)
    bend = -0.035 * math.hypot(x1 - x0, y1 - y0)
    nx, ny = -(y1 - y0), (x1 - x0)
    nl = math.hypot(nx, ny) + 1e-6
    X = x0 + (x1 - x0) * q + nx / nl * bend * np.sin(np.pi * q)
    Y = y0 + (y1 - y0) * q + ny / nl * bend * np.sin(np.pi * q)
    w = np.linspace(0, 1, n)
    amp = (w ** 2.0) * 0.42 * fade
    rad = 0.8 + 1.5 * w
    col = np.empty((n, 3))
    col[:] = (0.78, 0.88, 1.0)
    col[:, 0] += 0.22 * w
    col[:, 1] += 0.10 * w
    glow = np.zeros_like(pic)
    _splat(glow, X, Y, rad, col, amp)
    hx, hy = X[-1:], Y[-1:]
    _splat(glow, hx, hy, np.array([2.4]), np.array([[1.0, 0.98, 0.92]]), np.array([1.2 * fade]))
    _splat(glow, hx, hy, np.array([9.0]), np.array([[0.7, 0.82, 1.0]]), np.array([0.28 * fade]))
    _glint(glow, float(hx[0]), float(hy[0]), 16.0, 0.8 * fade, 1.0, 0.97, 0.9, math.atan2(y1 - y0, x1 - x0))
    pic[:] = 1 - (1 - np.clip(pic, 0, 1)) * (1 - np.clip(glow, 0, 1))


def _fade_star(pic, fs, t):
    x, y, a = fs
    tw = 1.0 + 0.12 * math.sin(t * 5.3) * math.sin(t * 2.1 + 1.0)
    glow = np.zeros_like(pic)
    X, Y = np.array([float(x)]), np.array([float(y)])
    c = np.array([[1.0, 0.97, 0.88]])
    _splat(glow, X, Y, np.array([2.0]), c, np.array([1.3 * a * tw]))
    _splat(glow, X, Y, np.array([10.0]), c, np.array([0.22 * a * tw]))
    _glint(glow, float(x), float(y), 10.0 + 14.0 * a, 0.9 * a * tw, 1.0, 0.97, 0.88, 0.1)
    pic[:] = 1 - (1 - np.clip(pic, 0, 1)) * (1 - np.clip(glow, 0, 1))
