"""The hilltop location: mountains, valley and village, hill, camphor tree, grass.

    from tr2 import hillset
    hillset.ground_y(X)                     hill crest in set coords
    hillset.render_back(pic, cam, tod, f, lights=0.0, wind=None, defocus=None, flatten=None)
    hillset.render_front(cam, tod, f, wind=None, defocus=None, flatten=None)  -> RGBA
    hillset.grass_band(cam, tod, f, x_from, x_to, flatten=None)               -> RGBA
    hillset.tree_layer(cam, tod, f)                                           -> RGBA
    hillset.sky_plain(tod, cam)             plain palette sky for test stills
    hillset.build(tod, force=False)         paint and cache every layer of one time of day

Painted layers are cached in ROOT/build/cache/hill/ as float16 .npy, one set per
time of day. Run `python3 -m tr2.hillset build dusk` (or `all`) to paint them,
`python3 -m tr2.hillset still dusk` to save the review stills.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tr2 import core, palette  # noqa: E402
from tr2 import hillset_paint as hp  # noqa: E402
from tr2.hillset_paint import SPEC, ground_y  # noqa: E402,F401

CACHE = os.path.join(ROOT, "build", "cache", "hill")
VERSION = 1
NAMES = ["far", "near", "valley", "hill", "rshadow", "tree0", "tree1", "tree2", "tree3", "treefire", "windows", "river"]
_mem = {}


def _file(tod, name):
    return os.path.join(CACHE, "v%d_%s_%s.npy" % (VERSION, tod, name))


def build(tod, force=False, log=True):
    """Paint every layer of one time of day into the cache."""
    os.makedirs(CACHE, exist_ok=True)
    if not force and all(os.path.exists(_file(tod, n)) for n in NAMES):
        return
    t0 = time.time()

    def put(name, arr, raw=False):
        tmp = _file(tod, name) + ".tmp.npy"
        np.save(tmp, arr if raw else arr.astype(np.float16))
        os.replace(tmp, _file(tod, name))     # atomic: a render never reads half a file
        if log:
            print("  %s %s %s %.1fs" % (tod, name, arr.shape, time.time() - t0), flush=True)

    put("far", hp.paint_mountains("far", tod))
    put("near", hp.paint_mountains("near", tod))
    val, windows, river = hp.paint_valley(tod)
    put("valley", val)
    put("windows", windows, raw=True)
    put("river", river)
    hill, rshadow = hp.paint_hill(tod)
    put("hill", hill)
    put("rshadow", rshadow)
    tree, treefire = hp.paint_tree(tod)
    for i, lay in enumerate(tree):
        put("tree%d" % i, lay)
    put("treefire", treefire)
    _mem.pop(tod, None)


def layers(tod):
    """The painted layers of a time of day (builds them on first use)."""
    if tod not in _mem:
        build(tod)
        if len(_mem) >= 2:
            _mem.pop(next(iter(_mem)))
        d = {}
        for n in NAMES:
            a = np.load(_file(tod, n))
            d[n] = a if n == "windows" else np.ascontiguousarray(a, np.float32)
        _mem[tod] = d
    return _mem[tod]


def _place(cam, layer, name, blur=0.0, off=(0.0, 0.0)):
    x0, y0, _, _, k, p = SPEC[name]
    s, tx, ty = cam.xf(p)
    out = core.warp(layer, s / k, tx + (x0 + off[0]) * s, ty + (y0 + off[1]) * s)
    return core.gblur(out, blur) if blur > 0.3 else out


def sky_plain(tod, cam=None):
    """The plain palette gradient, horizon at set y 880 (test stills only)."""
    cam = cam or Camera()
    _, hy = cam.pt(1600.0, hp.HORIZON + 30, 0.02)
    v = (np.arange(core.H, dtype=np.float32) + 0.5) / max(hy, 1.0)
    col = palette.sky_color(tod, v)
    return np.ascontiguousarray(np.repeat(col[:, None, :], core.W, 1), np.float32)


def Camera(**kw):
    """core.Camera with the set defaults (centre 1600, 900)."""
    return core.Camera(cx0=1600.0, cy0=900.0, **kw)


_order = None


def _lights(pic, cam, L, tod, f, lights):
    """Village windows coming on one by one, with glow and river reflections."""
    global _order
    win = L["windows"]
    n = len(win)
    if n == 0 or lights <= 0:
        return
    if _order is None or len(_order) != n:
        _order = core.rng("window order").permutation(n)
    s, tx, ty = cam.xf(SPEC["valley"][5])
    buf = np.zeros((core.H, core.W), np.float32)
    level = lights * (n + 1.5)
    for rank, i in enumerate(_order):
        e = min(max(level - rank, 0.0), 1.0)
        if e <= 0:
            continue
        e = e * e * (3 - 2 * e)
        fl = 0.90 + 0.10 * core.noise1(core.on_n(f, 2) * 0.11 + i * 3.7, seed=int(i))
        X, Y, w, h = win[i]
        xa, xb = (X - w / 2) * s + tx, (X + w / 2) * s + tx
        ya, yb = (Y - h / 2) * s + ty, (Y + h / 2) * s + ty
        i0, i1 = int(np.floor(xa)), int(np.ceil(xb))
        j0, j1 = int(np.floor(ya)), int(np.ceil(yb))
        if i1 <= 0 or j1 <= 0 or i0 >= core.W or j0 >= core.H:
            continue
        xs = np.arange(max(i0, 0), min(i1, core.W))
        ys = np.arange(max(j0, 0), min(j1, core.H))
        cx = np.clip(np.minimum(xs + 1, xb) - np.maximum(xs, xa), 0, 1)
        cy = np.clip(np.minimum(ys + 1, yb) - np.maximum(ys, ya), 0, 1)
        buf[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] = np.maximum(buf[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1], np.outer(cy, cx) * e * fl)
    warm = np.float32((1.0, 0.72, 0.34))
    glow = core.gblur(buf, 9.0 * s) * 5.0 + core.gblur(buf, 2.6 * s) * 1.2
    river = _place(cam, L["river"][..., None], "valley")[..., 0]
    sh = int(round(10 * s))
    refl = np.zeros_like(buf)
    refl[sh:] = buf[:-sh] if sh > 0 else buf
    from scipy import ndimage
    refl = ndimage.gaussian_filter(refl, (16.0 * s, 2.2 * s)) * 9.0 * river
    core.add_light(pic, np.clip(glow, 0, 1), warm, 0.85)
    core.add_light(pic, np.clip(refl, 0, 1), np.float32((1.0, 0.78, 0.42)), 0.9)
    pic *= (1 - buf)[..., None]
    pic += buf[..., None] * np.float32((1.0, 0.90, 0.62))


def _sway(f, i, wind):
    """Offset in set px of foliage sub layer i (0 back, 2 mid, 3 front)."""
    amp = 1.0 if wind is None else float(wind)
    t = core.on_n(f, 2) / core.FPS
    a = (2.2 + 1.3 * i) * amp
    dx = a * (0.7 * np.sin(t * 1.05 + i * 1.7) + 0.3 * np.sin(t * 2.3 + i * 0.6)) + a * 0.5 * core.noise1(t * 0.6 + i * 9.1, seed=3)
    dy = 0.35 * a * np.sin(t * 1.4 + i * 2.2 + 0.8)
    return float(dx), float(dy)


_grid = None


def _screen_grid():
    global _grid
    if _grid is None:
        yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
        _grid = (xx + 0.5, yy + 0.5)
    return _grid


def _flat_blades(pic, cam, tod, f, flatten):
    """Hill grass pressed outward around (X_centre, ground): blades redrawn lying
    away from the centre, strongest at the centre, fading out at `radius`."""
    from tr2 import paint
    xc, rad, amt = flatten
    if amt <= 0.01 or rad <= 1:
        return
    r = core.rng("flat ring", 1)
    n = 2600
    a = r.uniform(0, 2 * np.pi, n)
    rr = np.sqrt(r.random(n)) * 1.6            # in units of radius
    h1, h2, ph = r.random(n), r.random(n), r.uniform(0, 6.283, n)
    X = xc + np.cos(a) * rr * rad
    Y = ground_y(xc) + 16.0 + np.sin(a) * rr * rad * 0.34
    fall = amt * np.exp(-rr * rr)
    ok = (Y > ground_y(X) + 3.0) & (fall > 0.03)
    if not ok.any():
        return
    a, rr, h1, h2, ph, X, Y, fall = a[ok], rr[ok], h1[ok], h2[ok], ph[ok], X[ok], Y[ok], fall[ok]
    s, tx, ty = cam.xf(1.0)
    xs, ys = X * s + tx, Y * s + ty
    vis = (xs > 2) & (xs < core.W - 2) & (ys > 2) & (ys < core.H - 2)
    if not vis.any():
        return
    a, h1, h2, ph, xs, ys, fall = a[vis], h1[vis], h2[vis], ph[vis], xs[vis], ys[vis], fall[vis]
    out_ang = np.arctan2(np.sin(a) * 0.34 - 0.10, np.cos(a))          # outward on the ground, a little up
    up = -np.pi / 2 + (h1 - 0.5) * 0.7
    turn = (out_ang - up + np.pi) % (2 * np.pi) - np.pi
    lay = np.clip(fall * 1.35, 0, 1)
    ang = up + turn * lay + 0.10 * fall * np.sin(f * 2.3 + ph)        # shudder, on ones
    base = pic[ys.astype(int), xs.astype(int)]
    cols = np.clip(base * np.where(h2 < 0.55, 1.34, 0.74)[:, None] + 0.035 * lay[:, None], 0, 1)
    o = np.argsort(ys)
    layer = np.zeros((core.H, core.W, 4), np.float32)
    paint.strokes(layer, xs[o], ys[o], ang[o], (2.2 + 2.4 * h1[o]) * s, (18 + 30 * h2[o]) * (1 + 0.5 * lay[o]) * s, cols[o],
                  op=np.clip(0.25 + 1.2 * lay[o], 0, 0.95), bend=(h1[o] - 0.5) * 0.8 * (1 - lay[o]), seed=13, dry=0.15, taper=0.9,
                  streak=0.05, paper_k=0.0)
    core.over(pic, layer)


def _fire_falloff(cam, fire, wide, squash):
    X, Y, rad, amt = fire
    s, tx, ty = cam.xf(1.0)
    xx, yy = _screen_grid()
    r = max(rad * s * wide, 1.0)
    d2 = ((xx - (X * s + tx)) / r) ** 2 + ((yy - (Y * s + ty)) / (r * squash)) ** 2
    return amt * np.exp(-d2)


def render_back(pic, cam, tod, f, lights=0.0, wind=None, defocus=None, flatten=None, rocket_shadow=1.0, fire=None):
    """Composite mountains, valley, village, hill and tree onto pic (in place).
    rocket_shadow 0..1: the long shadow of the rocket on the grass (set it to 0
    when the rocket is not standing at x 1720, fade it as the rocket lifts).
    flatten (X_centre, radius, amount): hill grass pressed outward around X_centre.
    fire (X, Y, radius, amount): a warm light pool on the grass around set point
    (X, Y) and on the underside of the tree canopy and the trunk side facing it."""
    L = layers(tod)
    T = palette.TOD[tod]
    blur = (lambda p: 0.0) if defocus is None else defocus
    core.over(pic, _place(cam, L["far"], "far", blur(SPEC["far"][5])))
    core.over(pic, _place(cam, L["near"], "near", blur(SPEC["near"][5])))
    core.over(pic, _place(cam, L["valley"], "valley", blur(SPEC["valley"][5])))
    _lights(pic, cam, L, tod, f, lights)
    hill = _place(cam, L["hill"], "hill", blur(1.0))
    core.over(pic, hill)
    if rocket_shadow > 0:
        m = _place(cam, L["rshadow"][..., None], "hill", blur(1.0))
        tint = np.clip(T["amb"] / np.maximum(T["key"], 1e-3) * np.float32((0.86, 1.0, 1.12)), 0, 1)
        pic *= 1 - (m * (0.80 * rocket_shadow)) * (1 - tint)
    if flatten is not None:
        _flat_blades(pic, cam, tod, f, flatten)
    firec = np.asarray(T.get("fire", (1.0, 0.72, 0.30)), np.float32)
    if fire is not None and fire[3] > 0:
        g = (_fire_falloff(cam, fire, 1.0, 0.40) + 0.30 * _fire_falloff(cam, fire, 2.6, 0.45)) * hill[..., 3]
        lum = pic.mean(axis=2, keepdims=True)
        warm = np.clip(lum * 3.4 * firec + 0.05 * firec + 0.35 * pic, 0, 1)
        gg = np.clip(g, 0, 1)[..., None] * 0.9
        pic *= 1 - gg
        pic += warm * gg
    for i in range(4):
        off = (0.0, 0.0) if i == 1 else _sway(f, i, wind)
        core.over(pic, _place(cam, L["tree%d" % i], "tree", blur(1.0), off))
    if fire is not None and fire[3] > 0:
        tm = _place(cam, L["treefire"][..., None], "tree", blur(1.0), _sway(f, 2, wind))[..., 0]
        g = tm * np.minimum(_fire_falloff(cam, fire, 4.0, 1.0) * 1.1, 1.0)
        pic *= 1 + g[..., None] * 1.5 * firec
        core.add_light(pic, g, firec, 0.16)
    return pic


def tree_layer(cam, tod, f, wind=None):
    L = layers(tod)
    out = np.zeros((core.H, core.W, 4), np.float32)
    for i in range(4):
        off = (0.0, 0.0) if i == 1 else _sway(f, i, wind)
        core.over_rgba(out, _place(cam, L["tree%d" % i], "tree", 0.0, off))
    return out


P_FRONT = 1.6
_blades = None


def smoothstep_(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _front_blades():
    """Fixed description of every foreground blade (set coords on the p 1.6 plane)."""
    global _blades
    if _blades is None:
        r = core.rng("front grass", 2)
        n = 900
        X = r.uniform(100, 3100, n)
        u = np.clip(np.abs(X - 1600) / 950.0, 0, 1)
        clump = core.fbm_at(X, X * 0 + 0.7, 210.0, 3, 19)
        ok = r.random(n) < (0.16 + 0.84 * u ** 1.4) * smoothstep_(0.36, 0.60, clump + 0.25 * u)
        X, u = X[ok], u[ok]
        n = len(X)
        stalk = r.random(n) < 0.16
        _blades = dict(
            X=X, Y=1478 + r.uniform(0, 70, n), u=u, stalk=stalk,
            length=r.uniform(120, 300, n) * (0.55 + 0.6 * u) * np.where(stalk, 1.25, 1.0),
            ang=-np.pi / 2 + r.normal(0, 0.26, n), bend=r.normal(0, 0.55, n),
            width=np.where(stalk, r.uniform(4, 6, n), r.uniform(14, 34, n)),
            phase=r.uniform(0, 6.283, n), stiff=r.uniform(0.6, 1.4, n), tone=r.random(n),
            order=np.argsort(r.random(n)))
    return _blades


def render_front(cam, tod, f, wind=None, defocus=None, flatten=None):
    """Foreground grass (p 1.6): big soft blades and seed heads at the bottom
    edge, redrawn every frame so they sway (each blade its own phase, the tip
    lagging the root). Returns a premultiplied RGBA screen layer."""
    from tr2 import paint
    B = _front_blades()
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    amp = 1.0 if wind is None else float(wind)
    t = core.on_n(f, 2) / core.FPS
    s, tx, ty = cam.xf(P_FRONT)
    X, Y = B["X"], B["Y"]
    gust = np.sin(t * 1.5 + X / 270.0 + B["phase"] * 0.25)
    ang = B["ang"] + amp * B["stiff"] * (0.10 * gust + 0.035 * np.sin(t * 3.3 + B["phase"]))
    bend = B["bend"] + amp * B["stiff"] * 0.30 * np.sin(t * 1.5 + X / 270.0 + B["phase"] * 0.25 - 1.0)
    length = B["length"].copy()
    if flatten is not None:
        xc, rad, amt = flatten
        fall = np.exp(-((X - xc) / max(rad, 1.0)) ** 2) * amt
        ang = ang + 1.15 * fall * np.sign(X - xc)
        bend = bend + 0.6 * fall * np.sign(X - xc)
        length = length * (1 - 0.25 * fall)
    xs, ys = X * s + tx, Y * s + ty
    vis = (xs > -260) & (xs < core.W + 260) & (ys - length * s < core.H)
    o = B["order"]
    o = o[vis[o]]
    out = np.zeros((core.H, core.W, 4), np.float32)
    if len(o) == 0:
        return out
    gn = hp.gain(tod)
    dark = palette.shade((0.20, 0.42, 0.32), T) * 0.72 * gn
    mid = dark + (np.clip(np.float32((0.46, 0.60, 0.24)) * T["key"], 0, 1) * gn - dark) * 0.38
    direct, sat = hp.look(tod)
    mid = hp.desat(dark + (mid - dark) * (0.4 + 0.6 * direct), sat) * (np.float32((0.80, 1.0, 1.20)) if direct < 1.0 else 1.0)
    rimc = mid + (hp.rim_col(tod, T) * 0.92 - mid) * (0.22 + 0.78 * direct)
    tone = B["tone"][o][:, None]
    cols = dark + (mid - dark) * tone
    paint.strokes(out, xs[o], ys[o], ang[o], B["width"][o] * s, length[o] * s, cols, op=1.0, bend=bend[o],
                  seed=5, dry=0.05, taper=0.93, streak=0.10, paper_k=0.0)
    # rim light on the sun side of some blades
    sx = T["sun"][0]
    lit = o[B["tone"][o] > 0.35]
    if len(lit):
        paint.strokes(out, xs[lit] + np.sign(sx) * B["width"][lit] * s * 0.27, ys[lit], ang[lit], B["width"][lit] * s * 0.30,
                      length[lit] * s * 0.97, mid + (rimc - mid) * (0.35 + 0.65 * B["tone"][lit][:, None]), op=0.9, bend=bend[lit],
                      seed=6, dry=0.3, taper=0.93, streak=0.05, paper_k=0.0)
    # seed heads on the stalks
    st = o[B["stalk"][o]]
    if len(st):
        hx, hy = xs[st].copy(), ys[st].copy()
        step = length[st] * s / 4.0
        a_end = ang[st]
        for i in range(4):
            a_end = ang[st] + bend[st] * ((i + 0.5) / 4.0)
            hx += np.cos(a_end) * step * (0.93 if i == 3 else 1.0)
            hy += np.sin(a_end) * step * (0.93 if i == 3 else 1.0)
        r = core.rng("seed heads", 1)
        for q in range(6):
            back = q * 7.5 * s
            jit = r.normal(0, 1.0, len(st))
            sd = (q % 2) * 2 - 1
            paint.strokes(out, hx - np.cos(a_end) * back, hy - np.sin(a_end) * back, a_end + sd * (0.75 + 0.2 * jit),
                          6.5 * s, 15.0 * s, mid + (rimc - mid) * (0.25 + 0.5 * B["tone"][st][:, None]), op=0.95,
                          bend=0.3 * sd, seed=7 + q, dry=0.0, taper=0.6, streak=0.05, paper_k=0.0)
    blur = 1.5 * s + (defocus(P_FRONT) if defocus is not None else 0.0)
    return core.gblur(out, blur)


def grass_band(cam, tod, f, x_from, x_to, flatten=None, wind=None):
    """Short grass tufts along the ground line between set x_from and x_to, to
    lay over feet and rocket legs (p 1.0). Returns a premultiplied RGBA screen layer."""
    from tr2 import paint
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    direct, sat = hp.look(tod)
    amp = 1.0 if wind is None else float(wind)
    t = core.on_n(f, 2) / core.FPS
    s, tx, ty = cam.xf(1.0)
    out = np.zeros((core.H, core.W, 4), np.float32)
    n = int(max(x_to - x_from, 0) * 0.55)
    if n == 0:
        return out
    # fixed blades: hash from an absolute lattice so a tuft never changes with the range
    i0 = int(np.floor(x_from / 1.8))
    idx = np.arange(i0, i0 + int((x_to - x_from) / 1.8) + 1)
    h1, h2, h3, h4 = hp.hashf(idx, 1.0), hp.hashf(idx, 2.0), hp.hashf(idx, 3.0), hp.hashf(idx, 4.0)
    X = idx * 1.8 + (h1 - 0.5) * 1.8
    Y = ground_y(X) + 3.0 + 9.0 * h2
    length = 9.0 + 20.0 * h3 ** 1.6
    ang = -np.pi / 2 + (h4 - 0.5) * 0.9 + amp * 0.10 * np.sin(t * 1.7 + X / 120.0 + h1 * 6.283)
    bend = (h2 - 0.5) * 1.2 + amp * 0.2 * np.sin(t * 1.7 + X / 120.0 + h1 * 6.283 - 0.9)
    if flatten is not None:
        xc, rad, amt = flatten
        fall = np.exp(-((X - xc) / max(rad, 1.0)) ** 2) * amt
        ang = ang + 1.25 * fall * np.sign(X - xc)
        length = length * (1 - 0.3 * fall)
    sx = T["sun"][0]
    litx = hp.smoothstep(2650, 1750, X) if sx < 0 else hp.smoothstep(560, 1350, X)
    gn = hp.gain(tod)
    dark = palette.shade((0.22, 0.42, 0.34), T) * gn
    mid = hp.desat(np.clip(np.float32((0.50, 0.64, 0.24)) * T["key"], 0, 1), sat) * gn
    mid = (dark + (mid - dark) * (0.35 + 0.65 * direct)) * (np.float32((0.80, 1.0, 1.20)) if direct < 1.0 else 1.0)
    gold = hp.rim_col(tod, T)
    lit_c = mid + (gold - mid) * (h3[:, None] * 0.7) * (0.55 + 0.45 * direct)
    base = dark + (mid - dark) * (0.3 + 0.7 * h4[:, None])
    cols = np.where((litx[:, None] > 0.25 + 0.5 * h2[:, None]), lit_c, base)
    o = np.argsort(Y)
    paint.strokes(out, X[o] * s + tx, Y[o] * s + ty, ang[o], (2.0 + 2.4 * h1[o]) * s, length[o] * s, np.clip(cols[o], 0, 1), op=0.96,
                  bend=bend[o], seed=11, dry=0.1, taper=0.9, streak=0.04, paper_k=0.0)
    return out


def still(tod, path, lights=0.0, cam=None, f=0):
    cam = cam or Camera()
    pic = sky_plain(tod, cam)
    render_back(pic, cam, tod, f, lights=lights)
    core.over(pic, render_front(cam, tod, f))
    core.save(path, pic)
    return pic


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "still"
    tod = sys.argv[2] if len(sys.argv) > 2 else "dusk"
    review = os.path.join(ROOT, "build", "review")
    if cmd == "build":
        for t in (list(palette.TOD) if tod == "all" else [tod]):
            if t == "sunrise":
                continue
            t0 = time.time()
            build(t, force=True)
            print("built", t, "%.1fs" % (time.time() - t0), flush=True)
    else:
        if "force" in sys.argv:
            build(tod, force=True)
        t0 = time.time()
        pic = still(tod, os.path.join(review, "hill-%s.png" % tod), lights=0.45 if tod in ("dusk", "night") else 0.0)
        print("frame %.2fs" % (time.time() - t0))
        # 1:1 crops: village and tree foliage side by side
        cam = Camera()
        vx, vy = cam.pt(2190, 1000, 0.25)
        fx, fy = cam.pt(1010, 560, 1.0)
        vy, fy = min(max(int(vy), 270), core.H - 270), min(max(int(fy), 270), core.H - 270)
        vx, fx = min(max(int(vx), 480), core.W - 480), min(max(int(fx), 480), core.W - 480)
        a = pic[vy - 270:vy + 270, vx - 480:vx + 480]
        b = pic[fy - 270:fy + 270, fx - 480:fx + 480]
        core.save(os.path.join(review, "hill-%s-crop.png" % tod), np.concatenate([a, b], 1))
