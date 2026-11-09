"""Helpers shared by every cut. A cut is a module in this folder named after
its shot (see tr2/timeline.py) with one function:

    def render(fl, n, f):
        ...
        return pic                 # float32 H x W x 3, before the finishing pass
        # or: return pic, dict(tod=..., bloom_amt=..., bloom_thr=..., vig=..., grain_amt=..., fade=(colour, amount))

fl = frame inside the cut (0 based; it may run a few frames past n when the cut
dissolves into the next one, so never index past the end of a list), n = length
of the cut in frames, f = global frame number (use it for seeds and grain).
The finishing pass (bloom, grade for the cut's time of day, vignette, grain) is
applied by tr2/film.py, so do not grade or add grain yourself.
"""
import numpy as np

from .. import core, palette
from ..timeline import CUES, FPS, SHOTS


def shot_start(name):
    for n, a, b, tod in SHOTS:
        if n == name:
            return a
    raise KeyError(name)


def cue(name, shot):
    """A cue time (or list of times) from the timeline, in seconds LOCAL to the cut."""
    v = CUES[name]
    a = shot_start(shot)
    if isinstance(v, (list, tuple)):
        return [x - a for x in v]
    return v - a


def twos(fl, start=0):
    """Drawing number when animating on twos."""
    return core.on_n(fl, 2, start)


def gradient_sky(tod):
    ys = np.linspace(0, 1, core.H, dtype=np.float32)
    return np.repeat(palette.sky_color(tod, ys)[:, None, :], core.W, 1).copy()


def sky_pic(tod, cam, f, t, **kw):
    """The sky for this frame (agent C's module), or the plain gradient if it is not there yet."""
    try:
        from .. import sky
        return sky.render(tod, cam, f, t, **kw)
    except Exception as e:  # noqa
        if not isinstance(e, ImportError):
            print("sky fallback:", repr(e)[:200])
        return gradient_sky(tod)


def hilltop(cam, tod, f, t, mid=None, sky_kw=None, lights=0.0, wind=None, defocus=None, flatten=None,
            band=(1100, 2000), front=True, sky=None):
    """The standard layer stack of a hilltop cut, back to front:
    sky -> mountains, valley, village, hill, tree (hillset.render_back)
        -> mid(pic): the owner draws cels and effects (rocket, panda, props) in place
        -> the band of grass tufts that overlaps feet -> foreground grass (p 1.6).
    Returns the picture. Falls back gracefully while modules are still being built."""
    pic = sky if sky is not None else sky_pic(tod, cam, f, t, **(sky_kw or {}))
    hs = None
    try:
        from .. import hillset as hs
    except ImportError:
        hs = None
    if hs is not None:
        try:
            hs.render_back(pic, cam, tod, f, lights=lights, wind=wind, defocus=defocus, flatten=flatten)
        except Exception as e:  # noqa
            print("hillset.render_back failed:", repr(e)[:200])
    if mid is not None:
        mid(pic)
    if hs is not None:
        try:
            if band is not None:
                core.over(pic, hs.grass_band(cam, tod, f, band[0], band[1], flatten=flatten))
            if front:
                core.over(pic, hs.render_front(cam, tod, f, wind=wind, defocus=defocus, flatten=flatten))
        except Exception as e:  # noqa
            print("hillset front failed:", repr(e)[:200])
    return pic


def ground_y(X):
    """Hill crest height in set coordinates (hillset.ground_y, or the brief's layout while it is missing)."""
    try:
        from .. import hillset
        return hillset.ground_y(X)
    except Exception:
        return float(np.interp(X, [640, 1500, 2560], [1165, 1050, 1330]))


def contact_shadow(pic, cam, X, Y, w, tod, strength=0.35, squash=0.22):
    """A soft shadow on the ground under a character or prop at set point (X, Y), w wide (set px)."""
    s, tx, ty = cam.xf(1.0)
    cx, cy = X * s + tx, Y * s + ty
    rx, ry = w * s * 0.5, max(w * s * 0.5 * squash, 2.0)
    x0, x1 = int(max(cx - rx * 2.2, 0)), int(min(cx + rx * 2.2, core.W))
    y0, y1 = int(max(cy - ry * 3, 0)), int(min(cy + ry * 3, core.H))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    a = np.exp(-d * 1.1) * strength
    t = palette.TOD[tod] if isinstance(tod, str) else tod
    col = np.asarray(t["amb"], np.float32) * 0.35
    reg = pic[y0:y1, x0:x1]
    reg *= (1 - a[..., None])
    reg += col * a[..., None]


ROCKET_X = 1720.0
MAP_X = 1180.0


def draw_rocket(pic, cam, tod, drawing, state=None, view="side", X=ROCKET_X, dY=0.0, rot=0.0, shadow=True,
                behind=None, scale=1.0):
    """The rocket standing on the hill (or wherever X, dY put it), as a cel over `pic`.
    behind(cel, info): optional hook to draw something inside the window before the glass.
    Returns the info dict from rocket.draw (window centre, nozzle, tip) or None."""
    try:
        from .. import rocket
    except ImportError:
        return None
    s, tx, ty = cam.xf(1.0)
    gy = ground_y(X)
    if shadow:
        contact_shadow(pic, cam, X, gy + 6, 190 * scale, tod, 0.42)
    cel = core.Cel(ss=2 if s * scale > 1.6 else 3)
    info = rocket.draw(cel, X * s + tx, (gy + dY) * s + ty, s=s * scale, tod=tod, drawing=drawing, rot=rot,
                       view=view, state=state)
    if behind is not None:
        behind(cel, info)
    try:
        rocket.draw_glass(cel, info, tod, drawing)
    except Exception as e:  # noqa
        print("draw_glass:", repr(e)[:120])
    core.over(pic, cel.rgba())
    return info


def draw_panda(pic, cam, tod, act, light=None, shadow=True):
    """The panda from an acting dict (tr2/acting.py): pose, X (set x of the feet),
    dY (height above the ground, set px, negative is up), s (scale), drawing."""
    if act is None or not act.get("visible", True):
        return
    from .. import panda
    s, tx, ty = cam.xf(1.0)
    X = act["X"]
    gy = ground_y(X)
    sc = s * act.get("s", 1.0)
    px, py = X * s + tx, (gy + act.get("dY", 0.0)) * s + ty
    # nothing to draw when the panda is out of frame (the camera has tilted away)
    if px < -260 * sc or px > core.W + 260 * sc or py < -60 * sc or py > core.H + 230 * sc:
        return
    if shadow and act.get("shadow", True):
        contact_shadow(pic, cam, X, gy + 4, 120 * act.get("s", 1.0), tod, 0.40)
    # the panda gets its own cel, big enough for every limb and the tail, so no part ever
    # falls off the edge of the drawing; it is then laid on the picture at its place
    m = int(250 * sc)
    bx0, by0 = int(px - m), int(py - 1.25 * m)
    cel = core.Cel(w=2 * m, h=int(1.55 * m), ss=2 if sc > 1.6 else 3, x0=bx0, y0=by0)
    panda.draw(cel, act["pose"], px, py, s=sc, tod=tod, drawing=act.get("drawing", 0), light=light)
    core.over(pic, cel.rgba(), bx0, by0)


def draw_map_prop(pic, cam, tod, t, wind=1.0, X=MAP_X, star_t=-1.0):
    """The star map on its stick, small, as it appears in the wide cuts. The
    sheet is the same painting as the close up (tr2/starmap.py), fluttering."""
    from .. import starmap
    s, tx, ty = cam.xf(1.0)
    gy = ground_y(X)
    k = 0.085                                    # sheet scale in set px per sheet px: about 128 x 89
    tt = palette.TOD[tod] if isinstance(tod, str) else tod
    # stick
    cel = core.Cel(ss=3)
    wood = palette.lit((0.50, 0.36, 0.22), tt)
    wood_dk = palette.shade((0.50, 0.36, 0.22), tt)
    x0, y0 = X * s + tx, gy * s + ty
    top = y0 - 150 * s
    stick = np.array([(x0 - 3.2 * s, y0 + 6 * s), (x0 - 2.6 * s, top), (x0 + 2.6 * s, top - 2 * s), (x0 + 3.6 * s, y0 + 6 * s)])
    m = cel.mask(stick)
    cel.fill(stick, wood)
    cel.crescent(stick, wood_dk, (-2.2 * s * np.sign(tt["sun"][0]), 0), clip=m)
    cel.stroke(stick[[0, 1]], 1.3 * max(s, 0.8), tt["ink"])
    cel.stroke(stick[[3, 2]], 1.5 * max(s, 0.8), tt["ink"])
    core.over(pic, cel.rgba())
    # sheet
    lay = starmap.fluttered(starmap.with_star(star_t), t, wind=wind, rot=-0.05)
    lay = lay.copy()
    lay[..., :3] *= np.clip(0.5 * (np.asarray(tt["key"]) + np.asarray(tt["amb"])) * 1.08, 0, 1.15)
    sc = k * s
    px = x0 - (starmap.SW / 2) * sc
    py = top + 6 * s
    placed = core.warp(lay, sc, px, py)
    core.over(pic, placed)


def shake(f, amp, seed=0, hz=24.0):
    """Camera shake offset (dx, dy) in px. Changes every frame (hz=24) or slower."""
    k = int(f * hz / FPS)
    r = core.rng("shake", seed, k)
    a = r.normal(0, 1, 2)
    return float(a[0] * amp), float(a[1] * amp * 0.8)


def drift(t, amp=3.0, seed=0):
    """Slow hand operated camera breathing (dx, dy) in px, for held shots."""
    return core.noise1(t * 0.35, 40 + seed) * amp, core.noise1(t * 0.31 + 7.0, 41 + seed) * amp * 0.7


def dof(focus_p=1.0, strength=6.0, near_boost=1.6):
    """A defocus function for rack focus: parallax factor -> blur sigma (px).
    Layers at focus_p are sharp; far layers (small p) and near layers (p > focus) blur."""
    def fn(p):
        p = max(p, 0.02)
        d = abs(1.0 / p - 1.0 / focus_p)
        s = strength * min(d, 3.0) / 3.0 * 3.0
        if p > focus_p:
            s = strength * near_boost * (p - focus_p)
        return float(s)
    return fn
