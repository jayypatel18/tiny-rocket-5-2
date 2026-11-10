"""Shot 7a: the climb through the clouds at sunrise. A vertical journey.

    back  = cloudset.render(cam, f, t, climb)          # float RGB picture behind the rocket
    front = cloudset.render_front(cam, f, t, climb)    # premultiplied RGBA: near cloud and wisps over the rocket
    cloudset.sun_xy(cam, climb), cloudset.sun_amount(climb), cloudset.last_alpha()   for flares and god rays

climb 0..1 is the journey (the director animates it). The camera rides with the
rocket: the world streams DOWN. Altitude A = climb * TRAVEL px (at parallax 1).
Six depths of cloud (parallax 0.10 to 2.6) in three decks plus strays: dark
blue grey undersides low down, then rose, then blazing gold tops. At
climb = DECK the camera passes the top of the last deck; after that the frame is
open sky above a floor of cloud, with the sun low on the right.
cam: use the default core.Camera() and put shake and zoom on it.
"""
import math

import numpy as np
from scipy import ndimage

from tr2 import core, palette
from tr2 import sky_paint as sp

W, H = core.W, core.H
TRAVEL = 3600.0
DECK = 0.90
A_DECK = DECK * TRAVEL
RATE = 1.0 / 3.5                      # climb per second, for motion blur
SUN_X = 1570.0
# parallax, sprite scale range, haze, in front of the rocket?
LAYERS = [
    dict(p=0.10, sc=(0.22, 0.36), haze=0.62, front=False, dens=6.0),
    dict(p=0.22, sc=(0.38, 0.58), haze=0.44, front=False, dens=4.5),
    dict(p=0.45, sc=(0.60, 0.85), haze=0.26, front=False, dens=3.0),
    dict(p=0.80, sc=(0.85, 1.15), haze=0.10, front=False, dens=1.7),
    dict(p=1.50, sc=(1.6, 2.2), haze=0.0, front=True, dens=0.8),
    dict(p=2.60, sc=(2.6, 3.6), haze=0.0, front=True, dens=0.55),
]
DECKS = (0.20, 0.52, DECK)            # where the cloud is thick, in climb units

# cloud tones by altitude: (deep, shadow, turn, lit, highlight)
BLUE = [(0.20, 0.25, 0.40), (0.30, 0.36, 0.52), (0.42, 0.46, 0.62), (0.55, 0.60, 0.74), (0.70, 0.73, 0.85)]
ROSE = [(0.36, 0.33, 0.54), (0.52, 0.45, 0.64), (0.82, 0.57, 0.63), (0.99, 0.75, 0.70), (1.00, 0.87, 0.79)]
GOLD = [(0.50, 0.47, 0.74), (0.67, 0.60, 0.82), (0.96, 0.71, 0.66), (1.00, 0.87, 0.63), (1.00, 0.97, 0.84)]
BLUE, ROSE, GOLD = [np.array(v, np.float32) for v in (BLUE, ROSE, GOLD)]


def _tones(g):
    """Five tones at light level g: 0 deep in the cloud .. 1 at the gold tops."""
    g = core.clamp(g)
    if g < 0.55:
        u = core.smooth(g / 0.55)
        return BLUE * (1 - u) + ROSE * u
    u = core.smooth((g - 0.55) / 0.45)
    return ROSE * (1 - u) + GOLD * u


def _in_cloud(g):
    """Colour of the air inside the cloud at light level g (array ok)."""
    g = np.clip(np.asarray(g, np.float32), 0, 1)
    stops = np.array([0.0, 0.35, 0.62, 0.85, 1.0], np.float32)
    cols = np.array([(0.21, 0.27, 0.43), (0.34, 0.39, 0.57), (0.66, 0.54, 0.68), (0.97, 0.76, 0.69), (1.0, 0.89, 0.70)],
                    np.float32)
    return np.stack([np.interp(g, stops, cols[:, k]) for k in range(3)], -1).astype(np.float32)


def _open_sky(h):
    """Open sky above the deck: h = px above the far cloud horizon (array ok)."""
    h = np.asarray(h, np.float32)
    stops = np.array([-200.0, 0.0, 60.0, 180.0, 420.0, 900.0], np.float32)
    cols = np.array([(1.0, 0.86, 0.66), (1.0, 0.90, 0.70), (0.99, 0.80, 0.70), (0.72, 0.72, 0.90), (0.36, 0.50, 0.84),
                     (0.17, 0.28, 0.66)], np.float32)
    return np.stack([np.interp(h, stops, cols[:, k]) for k in range(3)], -1).astype(np.float32)


_ITEMS = None


def _items():
    """Every cloud of the journey: per layer a list of dicts (x, altitude a, ...)."""
    global _ITEMS
    if _ITEMS is not None:
        return _ITEMS
    out = []
    for li, L in enumerate(LAYERS):
        r = core.rng("cloudset", li)
        p = L["p"]
        items = []
        win = H / p                               # altitude span of one screen at this depth
        lo, hi = -0.6 * win, A_DECK
        n = int(L["dens"] * (hi - lo) / win)
        for k in range(n):
            # gather toward the decks, a few strays between
            if r.uniform() < 0.72:
                dk = DECKS[int(r.integers(0, len(DECKS)))] * TRAVEL
                a = dk - abs(r.normal(0, 0.16 * win + 110.0))
            else:
                a = r.uniform(lo, hi)
            kind = ("cumulus", "cumulus", "tower", "bar")[int(r.integers(0, 4))]
            if L["front"]:
                kind = ("wisp", "wisp", "bar", "wisp")[int(r.integers(0, 4))]
            items.append(dict(kind=kind, seed=int(r.integers(0, {"cumulus": 7, "tower": 4, "bar": 6, "wisp": 6}[kind])),
                              x=float(r.uniform(-250, W + 250)), a=float(a), sc=float(r.uniform(*L["sc"])),
                              deck=False))
        # the last deck: a closed floor of big cumulus right across the frame
        sc_d = 0.5 * (L["sc"][0] + L["sc"][1]) * 1.15
        step = 960 * sc_d * 0.42
        for row in range(0 if L["front"] else (3 if p < 0.6 else 2)):
            x = -300.0 - r.uniform(0, step)
            while x < W + 300:
                kind = "cumulus" if r.uniform() < 0.8 else "tower"
                items.append(dict(kind=kind, seed=int(r.integers(0, 7 if kind == "cumulus" else 4)), x=float(x),
                                  a=float(A_DECK - (row * 0.55 + r.uniform(0.05, 0.45)) * 300 * sc_d),
                                  sc=float(sc_d * r.uniform(0.85, 1.2)), deck=True))
                x += step * r.uniform(0.7, 1.3)
        items.sort(key=lambda it: it["a"])        # low clouds first: tops of nearer ones overlap
        out.append(items)
    _ITEMS = out
    return out


_DECKF = {}


def _deck_fields(li):
    """The last deck of one depth, pre composed as cloud FIELDS (alpha, light,
    underside) on one wide strip, so a frame colours it once. Returns
    (fields, x0, y_ref): strip pixel (i, j) sits at layer x = x0 + i and at
    j - y_ref px below the altitude A_DECK line (in layer px, parallax applied)."""
    if li in _DECKF:
        return _DECKF[li]
    L = LAYERS[li]
    p = L["p"]
    its = [it for it in _items()[li] if it["deck"]]
    if not its:
        _DECKF[li] = None
        return None

    def make():
        x0 = -420
        wc = W + 840
        top = max(sp.SIZES[it["kind"]][1] * it["sc"] for it in its) + 40
        bot = max((A_DECK - it["a"]) * p + sp.SIZES[it["kind"]][1] * it["sc"] * 0.5 for it in its) + 20
        hs = int(top + bot)
        Fb = np.zeros((hs, wc, 3), np.float32)
        for it in its:
            lv = sp.mips(it["kind"], it["seed"])
            sc, k = it["sc"], 0
            while sc < 0.5 and k < len(lv) - 1:
                sc *= 2
                k += 1
            Fm = lv[k]
            h, w = Fm.shape[:2]
            cx = it["x"] - x0
            cy = top + (A_DECK - it["a"]) * p
            fx0, fy0 = cx - w * sc / 2, cy - h * sc / 2
            X0, Y0 = max(int(math.floor(fx0)), 0), max(int(math.floor(fy0)), 0)
            X1, Y1 = min(int(fx0 + w * sc) + 2, wc), min(int(fy0 + h * sc) + 2, hs)
            if X1 <= X0 or Y1 <= Y0:
                continue
            Fw = core.warp(Fm, sc, fx0 - X0, fy0 - Y0, out_hw=(Y1 - Y0, X1 - X0))
            reg = Fb[Y0:Y1, X0:X1]
            a_s = Fw[..., 0:1]
            a_n = a_s + reg[..., 0:1] * (1 - a_s)
            reg[..., 1:] = (Fw[..., 1:] * a_s + reg[..., 1:] * reg[..., 0:1] * (1 - a_s)) / np.maximum(a_n, 1e-4)
            reg[..., 0:1] = a_n
        return np.concatenate([Fb.reshape(-1), np.array([x0, top], np.float32)]).astype(np.float16), (hs, wc)

    name = "cloudset_deck_v3_%d" % li
    import os
    pth = os.path.join(sp.CACHE, name + ".npz")
    if os.path.exists(pth):
        d = np.load(pth)
        Fb, x0, top = d["F"].astype(np.float32), float(d["x0"]), float(d["top"])
    else:
        flat, (hs, wc) = make()
        Fb = flat[:-2].astype(np.float32).reshape(hs, wc, 3)
        x0, top = float(flat[-2]), float(flat[-1])
        tmp = pth + ".tmp%d.npz" % os.getpid()
        np.savez(tmp, F=Fb.astype(np.float16), x0=x0, top=top)
        os.replace(tmp, pth)
    _DECKF[li] = (Fb, x0, top)
    return _DECKF[li]


def sun_xy(cam, climb):
    A = climb * TRAVEL
    s, tx, ty = cam.xf(0.06)
    return SUN_X * s + tx, (540.0 + (A - (A_DECK + 150.0)) * 0.06 - 40.0) * s + ty


def sun_amount(climb):
    """0 under the clouds, 1 once the camera is out in the open."""
    A = climb * TRAVEL
    return float(sp.ss(A_DECK - 1100.0, A_DECK + 60.0, A)) ** 1.5


def thickness(climb):
    """How deep inside a deck the camera is, 0..1 (whiteout as it punches through)."""
    A = climb * TRAVEL
    d = 0.0
    for dk in DECKS:
        a = dk * TRAVEL
        d = max(d, math.exp(-((A - (a - 130.0)) / 150.0) ** 2) if A < a - 30 else
                math.exp(-((A - (a - 130.0)) / 150.0) ** 2) * max(0.0, 1 - (A - (a - 30)) / 60.0))
    return d


_TEX = None


def _tex():
    global _TEX
    if _TEX is None:
        def make():
            f1 = sp.stroke_field(2 * H, W, 1800, (260, 800), (30, 90), 0.0, 0.10, seed=41, opacity=0.9)
            f2 = sp.stroke_field(2 * H, W, 4200, (90, 320), (12, 34), 0.0, 0.2, seed=42, opacity=0.75)
            return (0.6 * f1 + 0.4 * f2).astype(np.float16)
        _TEX = sp.cached("cloudset_tex_v1", make).astype(np.float32)
    return _TEX


def _layer(li, cam, A, light, blur_px=0.0, res=1, dst=None):
    """One depth of cloud as a premultiplied RGBA layer (at 1/res size; onto dst if given)."""
    L = LAYERS[li]
    p = L["p"]
    s, tx, ty = cam.xf(min(p, 1.6))
    lay = np.zeros((H // res, W // res, 4), np.float32) if dst is None else dst
    bgc = _in_cloud(light)
    dk = _deck_fields(li)
    if dk is not None:
        Fb, dx0, top = dk
        y0 = 540.0 - (A_DECK - A) * p - top                     # layer y of the strip's first row
        fx0, fy0 = (dx0 * s + tx) / res, (y0 * s + ty) / res
        sc = s / res
        Y0, Y1 = max(int(math.floor(fy0)), 0), min(int(fy0 + Fb.shape[0] * sc) + 2, lay.shape[0])
        if Y1 > Y0:
            Fw = core.warp(Fb, sc, fx0, fy0 - Y0, out_hw=(Y1 - Y0, lay.shape[1]))
            rel = max(-1.0, min(1.0, (A - A_DECK + 0.3 * top) * p / 420.0 + 0.3))
            cols = _tones(light * 0.85 + 0.30 * rel + 0.06)
            if L["haze"] > 0:
                cols = [palette.mix(c, bgc, L["haze"] * (0.95 if i < 2 else 0.7)) for i, c in enumerate(cols)]
            core.over_rgba(lay[Y0:Y1], sp.colourise(Fw, cols, 1.0))
    for it in _items()[li]:
        if it["deck"]:
            continue
        y = 540.0 - (it["a"] - A) * p
        hh = sp.SIZES[it["kind"]][1] * it["sc"]
        if y < -hh * 0.7 or y > H + hh * 0.7:
            continue
        # above the camera we see dark undersides, below it the lit tops
        rel = max(-1.0, min(1.0, (A - it["a"] + 0.25 * hh) * p / 420.0))
        g = light * 0.85 + 0.30 * rel + 0.06
        cols = _tones(g)
        if L["haze"] > 0:
            cols = [palette.mix(c, bgc, L["haze"] * (0.95 if i < 2 else 0.7)) for i, c in enumerate(cols)]
        op = 1.0 if not L["front"] else (0.8 if it["kind"] != "wisp" else 0.6)
        sp.place(lay, it["kind"], it["seed"], (it["x"] * s + tx) / res, (y * s + ty) / res, it["sc"] * s / res, cols,
                 op=op, dl=-0.10 * (1 - rel) * 0.5)
    if blur_px / res > 1.5:
        lay = ndimage.uniform_filter1d(lay, int(blur_px / res) | 1, axis=0, mode="nearest")
    return lay


_LAST_ALPHA = None


def last_alpha():
    """Cloud coverage (H x W) of the last back picture rendered, for god rays."""
    return _LAST_ALPHA


def render(cam, f, t, climb, rate=RATE):
    global _LAST_ALPHA
    A = climb * TRAVEL
    light = float(sp.ss(-0.05, 1.0, climb / DECK)) ** 1.15            # the light climbs with us
    opened = float(sp.ss(A_DECK - 420.0, A_DECK + 40.0, A))
    s0, tx0, ty0 = cam.xf(0.06)
    ys = (np.arange(H, dtype=np.float32)[:, None] - ty0) / s0
    xs = (np.arange(W, dtype=np.float32)[None, :] - tx0) / s0
    tex = _tex()
    sh = int(A * 0.06) % H
    tx_ = np.roll(tex, sh, axis=0)[:H]
    # inside and under the cloud: lighter above, darker below, brushed
    g_in = light + (540.0 - ys) / 1000.0 + 0.05 * tx_
    pic = _in_cloud(g_in + 0 * xs)
    # the open sky above the far horizon of cloud
    y_h = 540.0 + (A - A_DECK) * 0.06
    sky = _open_sky((y_h - ys) + 26.0 * tx_ + 0 * xs)
    k = opened * sp.ss(60.0, -80.0, ys - y_h + 0 * xs)
    pic = pic * (1 - k[..., None]) + sky * k[..., None]
    pic *= 1.0 + 0.02 * tx_[..., None]
    # the sun, low on the right
    sa = sun_amount(climb)
    sx, sy = sun_xy(cam, climb)
    if sa > 0.01:
        d = np.hypot((np.arange(W, dtype=np.float32)[None, :] - sx) * 0.7, np.arange(H, dtype=np.float32)[:, None] - sy)
        gl = 0.5 * np.exp(-d / 520.0) + 0.6 * np.exp(-(d / 190.0) ** 2) + 1.0 * np.exp(-(d / 46.0) ** 2)
        core.add_light(pic, gl.astype(np.float32), (1.0, 0.88, 0.60), 0.95 * sa)
        core.add_light(pic, np.exp(-(d / 30.0) ** 4).astype(np.float32), (1.0, 0.99, 0.92), sa * opened)
    speed = rate * TRAVEL / core.FPS                                  # px a frame at parallax 1
    far = np.zeros((H, W, 4), np.float32)
    for li in (0, 1):                                                 # far depths: slow and crisp
        _layer(li, cam, A, light, dst=far)
    core.over(pic, far)
    mid = _layer(2, cam, A, light, 0.45 * speed * LAYERS[2]["p"])     # nearer: smeared along y by its speed
    core.over(pic, mid)
    near = _layer(3, cam, A, light, 0.65 * speed * LAYERS[3]["p"])
    core.over(pic, near)
    al = far[..., 3] * (1 - mid[..., 3]) + mid[..., 3]
    _LAST_ALPHA = al * (1 - near[..., 3]) + near[..., 3]
    return np.clip(pic, 0, 1).astype(np.float32)


def render_front(cam, f, t, climb, rate=RATE):
    A = climb * TRAVEL
    light = float(sp.ss(-0.05, 1.0, climb / DECK)) ** 1.15
    speed = rate * TRAVEL / core.FPS
    half = np.zeros((H // 2, W // 2, 4), np.float32)
    for li, L in enumerate(LAYERS):
        if not L["front"]:
            continue
        lay = _layer(li, cam, A, light, min(0.75 * speed * L["p"], 150.0), res=2)
        lay = core.gblur(lay, 1.0 + 1.0 * (L["p"] - 1.5))
        core.over_rgba(half, lay)
    out = core.warp(half, 2.0, 0.5, 0.5)
    # punching through a deck: streaming mist right across the lens
    th = thickness(climb)
    if th > 0.01:
        n = core.fbm(H // 4, W // 4, 46, 4, 77, oy=-A * 0.9, aspect=0.35)
        n2 = core.fbm(H // 4, W // 4, 150, 3, 78, oy=-A * 0.5)
        m = np.clip(0.35 + 1.5 * (n - 0.5) + 0.8 * (n2 - 0.5), 0, 1)
        m = ndimage.zoom(m, 4, order=1)[:H, :W] * th * 0.72
        col = _in_cloud(np.float32(min(1.0, light + 0.12)))
        core.over_rgba(out, core.premul(np.broadcast_to(col, (H, W, 3)).astype(np.float32).copy(), m.astype(np.float32)))
    return out
