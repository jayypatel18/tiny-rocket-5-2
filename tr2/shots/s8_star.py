"""Shot 8, The Shooting Star (the climax). The vista from very far and very
high. A thin white trail climbs out of the cloud sea on the left; at
`star_ignites` it flashes and becomes a blazing star that arcs across the whole
sky, fast then gliding, trailing light and falling glitter. Its light slides
over the cloud tops and glints in the rivers. Slow push in, slight pan right.
The cut dissolves out, so it renders past n."""
import math

import numpy as np

from .. import core, vista
from .. import sky as skymod
from ..timeline import FPS
from . import common

NAME = "s8_star"
W, H = core.W, core.H
P_STAR = 0.06                      # parallax of the star (it lives in the sky)

# the climb of the rocket before ignition, then the arc (screen px at the default camera)
CLIMB = np.array([(250.0, 668.0), (300.0, 520.0), (436.0, 372.0)])
ARC = np.array([(436.0, 372.0), (548.0, 150.0), (1060.0, 28.0), (1650.0, 262.0)])
KDEC = 2.4                         # how hard the star decelerates
XMAX = 1.32


def camera(t, T):
    u = core.ease_io(core.span(t, -0.4, T + 0.8), 1.5)
    dx, dy = common.drift(t, 1.6, 8)
    return core.Camera(cx=W / 2 + 120.0 * u + dx, cy=H / 2 - 8.0 * u + dy, dolly=core.lerp(1.0, 1.12, u), pan=34.0 * core.smooth(core.span(t, 1.0, T + 0.8)))


def _bez2(p, q):
    q = np.asarray(q, np.float64)[:, None]
    return (1 - q) ** 2 * p[0] + 2 * (1 - q) * q * p[1] + q * q * p[2]


def _bez3(p, q):
    q = np.asarray(q, np.float64)[:, None]
    return (1 - q) ** 3 * p[0] + 3 * (1 - q) ** 2 * q * p[1] + 3 * (1 - q) * q * q * p[2] + q ** 3 * p[3]


_C = 1 - math.exp(-KDEC * XMAX)


def _u_of_x(x):
    return (1 - np.exp(-KDEC * np.clip(x, 0, XMAX))) / _C


def _x_of_u(u):
    return -np.log(1 - np.clip(u, 0, 0.99999) * _C) / KDEC


def _screen(cam, pts):
    s, tx, ty = cam.xf(P_STAR)
    return pts * s + (tx, ty)


def _streak(glow, x, y, half, thick, amp, col):
    """A long soft horizontal streak of light through a point."""
    x0, x1 = int(max(0, x - half * 2.5)), int(min(W, x + half * 2.5))
    y0, y1 = int(max(0, y - thick * 4)), int(min(H, y + thick * 4))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-np.abs(xx - x) / half) * np.exp(-((yy - y) / thick) ** 2) * amp
    glow[y0:y1, x0:x1] += g[..., None] * np.asarray(col, np.float32)


def _star(pic, cam, t, fl, t_ign, t_end):
    """Everything luminous: returns (x, y, power) of the head for the vista light."""
    glow = np.zeros_like(pic)
    span = t_end - t_ign
    # ---- the climb: a thin white thread out of the clouds
    pre = core.clamp((t + 0.42) / (t_ign + 0.42)) ** 1.25
    n = 240
    q = np.linspace(0, pre, n)
    age_c = (pre - q) * (t_ign + 0.42) + max(0.0, t - t_ign)
    P = _bez2(CLIMB, q)
    drift = core.fbm_at(q * 9.0, age_c * 0.0 + 3.1, 1.0, 3, 61) - 0.5
    P[:, 0] += drift * 16.0 * np.minimum(age_c, 3.0) + 5.0 * age_c
    P[:, 1] += 2.0 * age_c
    S = _screen(cam, P)
    fade = np.exp(-np.maximum(t - t_ign, 0) / 2.4)
    amp = 0.34 * fade * (0.45 + 0.55 * np.exp(-age_c / 1.2)) * np.clip(q * 14.0, 0, 1)
    rad = 0.9 + 0.9 * np.minimum(age_c, 3.0) / 3.0
    col = np.tile(np.array([[1.0, 0.97, 0.94]]), (n, 1))
    skymod._splat(glow, S[:, 0], S[:, 1], rad, col, amp)
    if t < t_ign:
        hx, hy = S[-1]
        fk = 0.8 + 0.2 * math.sin(fl * 2.3)
        skymod._splat(glow, np.array([hx]), np.array([hy]), np.array([2.2]), np.array([[1.0, 0.9, 0.7]]), np.array([1.3 * fk]))
        skymod._splat(glow, np.array([hx]), np.array([hy]), np.array([9.0]), np.array([[1.0, 0.7, 0.4]]), np.array([0.3 * fk]))
        pic[:] = 1 - (1 - np.clip(pic, 0, 1)) * (1 - np.clip(glow, 0, 1))
        return float(hx), float(hy), 0.05

    # ---- the star
    x = (t - t_ign) / span
    u = float(_u_of_x(x))
    # the trail, older toward the tail
    n = 1100
    q = np.linspace(0, u, n)
    age = (x - _x_of_u(q)) * span
    P = _bez3(ARC, q)
    wob = core.fbm_at(q * 14.0, q * 0 + 0.7, 1.0, 3, 62) - 0.5
    P[:, 1] += 5.0 * np.minimum(age, 5.0) + wob * 2.2 * np.minimum(age, 3.0)
    P[:, 0] += wob * 1.5 * np.minimum(age, 3.0)
    S = _screen(cam, P)
    seg = np.r_[np.hypot(*np.diff(S, axis=0).T), 1.0]
    back = u - q                                    # how far behind the head, in fractions of the arc
    taper = np.clip(1 - back / 0.6, 0, 1) ** 1.4    # the bright body tapers over 60 percent of the arc
    k = 0.30 + 0.70 * np.exp(-age / 3.5)
    hot = np.exp(-age / 0.30)
    fadein = np.clip(q * 30.0, 0, 1)
    # the edge: gold near the head, rose further back, violet at the old end
    rad = 1.2 + 8.5 * taper + 3.0 * hot
    a2 = np.clip(back / 0.35, 0, 1)[:, None]
    a3 = np.clip((back - 0.45) / 0.4, 0, 1)[:, None]
    c = np.array([1.0, 0.82, 0.46]) * (1 - a2) + np.array([1.0, 0.58, 0.60]) * a2
    c = c * (1 - a3) + np.array([0.66, 0.58, 1.0]) * a3
    col = np.ascontiguousarray(c)
    amp = seg / (2.5 * rad) * (0.30 * k + 0.55 * taper + 0.5 * hot) * fadein
    skymod._splat(glow, S[:, 0], S[:, 1], rad, col, amp)
    # the white core
    rad2 = 0.6 + 3.2 * taper + 1.5 * hot
    amp2 = seg / (2.5 * rad2) * (0.15 * k + 1.3 * taper ** 1.2 + 1.0 * hot) * fadein
    skymod._splat(glow, S[:, 0], S[:, 1], rad2, np.tile(np.array([[1.0, 0.99, 0.93]]), (n, 1)), amp2)
    wide = np.zeros_like(pic)
    skymod._splat(wide, S[::6, 0], S[::6, 1], 16.0 + 0 * rad[::6], col[::6],
                  seg[::6] * 6 / 40.0 * (0.10 * k[::6] + 0.30 * taper[::6]) * fadein[::6])
    glow += core.gblur(wide, 5.0)

    # glitter shed along the way: it falls, twinkles and dies
    r = core.rng("glitter", 8)
    m = 2800
    gq = r.uniform(0, 1, m) ** 0.85
    life = r.uniform(0.9, 3.4, m)
    vx = r.normal(0, 9.0, m)
    vy = r.normal(6.0, 9.0, m)
    off = r.normal(0, 2.5, m)
    ph = r.uniform(0, 6.28, m)
    fr = r.uniform(7.0, 19.0, m)
    gb = r.uniform(0, 1, m) ** 2.5
    gk = r.uniform(0, 1, m)
    ga = (x - _x_of_u(gq)) * span
    on = (gq <= u) & (ga < life) & (ga > 0)
    if on.any():
        ga_, lf = ga[on], life[on]
        G = _bez3(ARC, gq[on])
        dr = 1 - np.exp(-ga_ * 1.4)
        G[:, 0] += vx[on] * dr / 1.4 + off[on]
        G[:, 1] += vy[on] * dr / 1.4 + 11.0 * ga_ + 6.5 * ga_ ** 2
        GS = _screen(cam, G)
        tw = 0.55 + 0.45 * np.sin(ga_ * fr[on] + ph[on])
        e = (1 - ga_ / lf) ** 1.4
        gamp = (0.45 + 1.8 * gb[on]) * e * tw
        grad = 0.75 + 1.0 * gb[on]
        gc = np.empty((len(ga_), 3))
        gc[:] = (1.0, 0.93, 0.72)
        gc[gk[on] < 0.25] = (1.0, 0.99, 0.94)
        gc[gk[on] > 0.9] = (0.80, 0.88, 1.0)
        skymod._splat(glow, GS[:, 0], GS[:, 1], grad, gc, gamp)
        big = np.nonzero((gb[on] > 0.82) & (e > 0.35))[0]
        for j in big[:60]:
            skymod._glint(glow, float(GS[j, 0]), float(GS[j, 1]), 5.0 + 8.0 * gb[on][j], 0.7 * e[j] * tw[j],
                          1.0, 0.95, 0.8, 0.2 + ph[on][j])

    # the head
    hx, hy = S[-1]
    since = t - t_ign
    flash = math.exp(-since / 0.16)
    surge = 1.0 + 1.6 * math.exp(-since / 0.5)
    fk = 1.0 + 0.07 * math.sin(fl * 1.9) * math.sin(fl * 0.7 + 1.0)
    pw = surge * fk
    X1, Y1 = np.array([float(hx)]), np.array([float(hy)])
    for rad_, a_, c_ in ((9.5, 4.5, (1.0, 1.0, 0.98)), (24.0, 1.2, (1.0, 0.96, 0.82)), (66.0, 0.55, (1.0, 0.84, 0.52)),
                         (190.0, 0.24, (1.0, 0.72, 0.42)), (430.0, 0.085, (0.9, 0.7, 0.9))):
        skymod._splat(glow, X1, Y1, np.array([rad_ * (0.9 + 0.1 * pw)]), np.array([c_]), np.array([a_ * pw]))
    # four fine diffraction spikes that breathe
    breathe = 1.0 + 0.16 * math.sin(t * 2.3) + 0.06 * math.sin(t * 5.1 + 1.0)
    rot = 0.16 + 0.04 * math.sin(t * 0.9)
    for o in (-0.45, 0.45):
        skymod._glint(glow, float(hx) + o, float(hy) + o, 185.0 * breathe * pw ** 0.5, 2.6 * pw, 1.0, 0.98, 0.88, rot)
    _streak(glow, hx, hy, 230.0 * pw ** 0.5, 3.4, 0.6 * pw, (1.0, 0.92, 0.78))
    _streak(glow, hx, hy, 420.0, 1.3, 0.16 * pw, (0.75, 0.85, 1.0))
    # a soft lens flare: ghosts on the line through the middle of the frame
    cx, cy = W / 2, H / 2
    for kq, rr, aa, cc in ((0.38, 22.0, 0.05, (0.6, 0.8, 1.0)), (0.72, 12.0, 0.07, (1.0, 0.72, 0.5)),
                           (1.22, 40.0, 0.035, (0.7, 1.0, 0.85)), (1.6, 70.0, 0.025, (1.0, 0.8, 0.9))):
        gx_, gy_ = cx + (cx - hx) * kq, cy + (cy - hy) * kq
        skymod._splat(glow, np.array([gx_]), np.array([gy_]), np.array([rr]), np.array([cc]), np.array([aa * pw]))
    pic[:] = 1 - (1 - np.clip(pic, 0, 1)) * (1 - np.clip(glow, 0, 1))
    kf = fl - int(round(t_ign * FPS))
    if kf == 0:        # the ignition: one white frame
        core.add_light(pic, np.ones((H, W), np.float32), (1.0, 0.99, 0.95), 0.94)
    elif kf in (1, 2):  # then two frames of heavy bloom ringing down
        d = np.hypot(np.arange(W, dtype=np.float32)[None, :] - hx, np.arange(H, dtype=np.float32)[:, None] - hy)
        core.add_light(pic, (0.30 + 0.7 * np.exp(-d / 620.0)).astype(np.float32), (1.0, 0.97, 0.88), 0.62 if kf == 1 else 0.36)
    elif flash > 0.02:
        d = np.hypot(np.arange(W, dtype=np.float32)[None, :] - hx, np.arange(H, dtype=np.float32)[:, None] - hy)
        core.add_light(pic, np.exp(-d / 520.0).astype(np.float32), (1.0, 0.97, 0.88), 0.5 * flash)
    return float(hx), float(hy), 0.55 * pw / 2.6 + 0.45


def render(fl, n, f):
    t = fl / FPS
    T = n / FPS
    cam = camera(t, T)
    t_ign, t_end = common.cue("star_arc", NAME)
    # where the head is now, for the light on the clouds (computed before the vista is drawn)
    if t >= t_ign:
        u = float(_u_of_x((t - t_ign) / (t_end - t_ign)))
        hp = _screen(cam, _bez3(ARC, [u]))[0]
        since = t - t_ign
        power = 0.55 + 0.75 * math.exp(-since / 0.5)
        star = (float(hp[0]), float(hp[1]), power)
    else:
        star = None
    pic = vista.render(cam, f, t, star=star)
    _star(pic, cam, t, fl, t_ign, t_end)
    kf = fl - int(round(t_ign * FPS))
    heavy = kf in (1, 2)
    return pic, dict(bloom_amt=0.95 if heavy else 0.42, bloom_thr=0.60 if heavy else 0.86, vig=0.26)
