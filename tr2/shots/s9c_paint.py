"""Shot 9c: close on the star map. A paw with a dab of gold paint on one toe
comes in from the lower right and paints one tiny new star at the end of the
dotted flight path. The paper flutters in the morning breeze; leaf shadows
drift across it; the hill behind is far out of focus."""
import math

import numpy as np

from .. import core, palette, starmap
from ..timeline import FPS
from . import common

NAME = "s9c_paint"
ROT = -0.045                    # the sheet hangs a little crooked
SX, SY = 205.0, 28.0            # where the sheet's top left sits on screen at scale 1
_bg = {}


def _background(f, t):
    """The morning hilltop far behind the map, heavily defocused. Rendered once
    every 4 frames at most (it barely moves) and held."""
    key = f // 4
    if key not in _bg:
        _bg.clear()
        cam = core.Camera(cx0=1600, cy0=900, cx=1210, cy=985, zoom=2.6)
        pic = common.sky_pic("morning", cam, key * 4, t)
        try:
            from .. import hillset
            hillset.render_back(pic, cam, "morning", key * 4)
        except Exception as e:  # noqa
            if not isinstance(e, ImportError):
                print("s9c hill fallback:", repr(e)[:160])
        _bg[key] = core.gblur(pic, 26.0)
    return _bg[key].copy()


def _map_xf(t, n_t):
    """Slow push in toward the place where the star will be painted."""
    u = core.smooth(t / n_t)
    s = 0.93 + 0.085 * u
    ax, ay = starmap.NEW_STAR
    # keep the star's place drifting toward a point right of centre
    tx = SX + (1 - s) * ax - 70 * u
    ty = SY + (1 - s) * ay + 55 * u
    return s, tx, ty


def star_screen_pos(fl, n):
    """Screen position of the new star (for the paw's acting)."""
    t = fl / FPS
    s, tx, ty = _map_xf(t, n / FPS)
    x, y = starmap.NEW_STAR
    c, sn = math.cos(ROT), math.sin(ROT)
    px, py = starmap.SW / 2, 40.0
    xr = (x - px) * c - (y - py) * sn + px
    yr = (x - px) * sn + (y - py) * c + py
    return xr * s + tx, yr * s + ty


def _paw_path(t, tb, star):
    """Where the painting toe tip is, and how far the arm is in (0 out, 1 on the paper)."""
    sx, sy = star
    t_in = tb - 0.95
    if t < t_in:
        return None
    off = np.array([760.0, 520.0])
    if t < tb:
        u = core.ease_out(core.span(t, t_in, tb - 0.12), 2.2)
        hov = np.array([26.0, -34.0]) * (1 - core.smooth(core.span(t, tb - 0.2, tb)))   # hovers, then touches
        p = np.array([sx, sy]) + off * (1 - u) + hov * u
        return p, u
    if t < tb + 0.58:
        # painting: the toe follows the brush stroke (a small pentagram), on ones
        k = (t - tb) / 0.55
        a = 0.05 + k * 2 * math.pi * 2
        r = 26.0 * (0.5 + 0.5 * min(k * 4, 1.0))
        return np.array([sx + math.sin(a) * r * 0.55, sy - math.cos(a) * r * 0.55]), 1.0
    u = core.ease_in(core.span(t, tb + 0.70, tb + 1.25), 2.0)
    lift = np.array([30.0, -46.0]) * core.ease_out(core.span(t, tb + 0.58, tb + 0.8))
    return np.array([sx, sy]) + lift + off * u, 1.0 - u


def render(fl, n, f):
    t = fl / FPS
    n_t = n / FPS
    tb = common.cue("brush_stroke", NAME)
    pic = _background(f, t)
    dx, dy = common.drift(t, 2.5, 9)

    # the stick behind the sheet, and its top
    s, tx, ty = _map_xf(t, n_t)
    tx += dx
    ty += dy
    cel = core.Cel(ss=2)
    wood = palette.lit((0.50, 0.36, 0.22), "morning")
    wood_dk = palette.shade((0.50, 0.36, 0.22), "morning")
    ink = palette.TOD["morning"]["ink"]
    cx = tx + (starmap.SW / 2 + 16) * s
    stick = core.wobble(np.array([(cx - 30 * s, -30), (cx + 26 * s, -30), (cx + 34 * s, core.H + 30), (cx - 24 * s, core.H + 30)]), 1.2, 3)
    m = cel.mask(stick)
    cel.fill(stick, wood)
    cel.crescent(stick, wood_dk, (16 * s, 0), clip=m)
    cel.stroke(stick[[0, 3]], 3.0, ink, seed=1)
    cel.stroke(stick[[1, 2]], 3.6, ink, seed=2)
    core.over(pic, core.gblur(cel.rgba(), 1.2))

    # the sheet, fluttering, with the new star once the brush has landed
    sheet = starmap.with_star(t - tb)
    wind = 0.75 + 0.5 * core.smooth(core.span(t, 0.0, 0.5)) * (1 - 0.7 * core.smooth(core.span(t, tb - 0.5, tb)))
    lay = starmap.fluttered(sheet, t + 3.0, wind=wind, rot=ROT)
    placed = core.warp(lay, s, tx, ty)
    # a soft shadow of the sheet on what is behind it
    sh = core.gblur(placed[..., 3], 14.0)
    sh = np.roll(np.roll(sh, 16, axis=0), -12, axis=1)
    pic *= (1 - 0.30 * sh[..., None] * np.array([1.0, 0.95, 0.80], np.float32))
    # morning light on the paper: warm key, and drifting leaf shadows from the camphor tree
    yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
    leaf = core.fbm(core.H, core.W, 210, 4, seed=71, ox=t * 26.0, oy=t * 9.0 + 40 * math.sin(t * 1.3))
    leaf = np.clip((leaf - 0.56) / 0.10, 0, 1)
    leaf = core.gblur(leaf.astype(np.float32), 7.0)
    shade_c = np.array([0.74, 0.76, 0.95], np.float32)
    light = (1.0 - leaf[..., None] * 0.55 * (1 - shade_c)) * np.array([1.03, 1.0, 0.95], np.float32)
    light = light * (1 - leaf[..., None] * 0.07)
    placed[..., :3] *= light
    core.over(pic, placed)

    # two brass pins along the top edge
    cel = core.Cel(ss=2)
    brass = palette.lit((0.86, 0.66, 0.28), "morning")
    for px in (starmap.SW / 2 - 40, starmap.SW / 2 + 60):
        c0, s0 = math.cos(ROT), math.sin(ROT)
        qx = (px - starmap.SW / 2) * c0 - (58 - 40.0) * s0 + starmap.SW / 2
        qy = (px - starmap.SW / 2) * s0 + (58 - 40.0) * c0 + 40.0
        X, Y = qx * s + tx, qy * s + ty
        cel.dot(X + 3, Y + 4, 10 * s, (0.25, 0.2, 0.3))
        cel.dot(X, Y, 10 * s, brass)
        cel.dot(X - 3 * s, Y - 3 * s, 3.5 * s, (1.0, 0.95, 0.8))
        cel.stroke(core.ellipse(X, Y, 10 * s, 10 * s, 24), 2.0, ink, closed=True)
    core.over(pic, cel.rgba())

    # the paw
    star = star_screen_pos(fl, n)
    star = (star[0] + dx, star[1] + dy)
    pp = _paw_path(t, tb, star)
    if pp is not None:
        tip, reach = pp
        try:
            from .. import panda
            cel = core.Cel(ss=2)
            on_ones = tb - 0.3 < t < tb + 0.9
            panda.draw_paw_closeup(cel, float(tip[0]), float(tip[1]), s=1.0, tod="morning",
                                   drawing=(fl if on_ones else common.twos(fl)), paint=GOLD_DAB if t < tb + 0.6 else 0.35,
                                   press=1.0 if tb <= t < tb + 0.58 else 0.0)
            arm = cel.rgba()
            sh = core.gblur(arm[..., 3], 10.0)
            sh = np.roll(np.roll(sh, 14, axis=0), -10, axis=1) * placed[..., 3]
            pic *= (1 - 0.32 * sh[..., None] * np.array([1.0, 0.95, 0.8], np.float32))
            core.over(pic, arm)
        except Exception as e:  # noqa
            if not isinstance(e, (ImportError, AttributeError)):
                print("s9c paw fallback:", repr(e)[:200])
    return pic, dict(bloom_amt=0.35, vig=0.26)


GOLD_DAB = 1.0
