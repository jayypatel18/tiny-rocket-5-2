"""The handmade rocket: a cel prop, rebuilt from its pose on every drawing.

    from tr2 import rocket
    st = rocket.State(shake=(dx, dy, rot), squash=0.0, fin_rattle=0.0, flame=0.0,
                      soot=0.0, legs_fold=0.0, hatch_open=0.0, window_glow=0.0, light=0.0)
    info = rocket.draw(cel, x, y, s=1.0, tod="dusk", drawing=0, rot=0.0, view="side", state=st)
        (x, y)  screen point at the centre of the feet. s=1 is 500 px feet to tip.
        view    "side" (wide shots), "low34" (launch, low three quarter), "below" (climbing away)
        light   state.light 0..1 is the fire light from below (warm hard tone on undersides)
        returns dict(window=(cx, cy, r), window_poly=pts, mask=HxW float alpha of the window hole,
                     nozzle=(x, y), tip=(x, y), bbox=(x0, y0, x1, y1))
    ... draw the panda behind the glass, clipped by info["mask"] ...
    rocket.draw_glass(cel, info, tod, drawing)     # reflections and the brass rim on top

Layout is a body of revolution seen from slightly below, so every ring on it
bends like an arch. Heights h are in rocket units (0 at the feet, 500 at the
tip), radii r from the axis, angles th around the axis (0 faces the viewer at
yaw 0, +90 degrees is screen right).
"""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from . import core, palette

R = palette.ROCKET
D2R = math.pi / 180.0

VIEWS = dict(
    side=dict(e=0.03, e1=0.10, k=0.0, yaw=0.0),
    low34=dict(e=0.07, e1=0.40, k=0.20, yaw=0.50),
    below=dict(e=0.50, e1=0.22, k=0.85, yaw=0.50),
)


def mix_views(a, b, t):
    """A view part way between two named views (pass the result as `view`), for a
    camera that swings under the rocket during one continuous shot."""
    A, B = VIEWS[a], VIEWS[b]
    return {k: A[k] + (B[k] - A[k]) * float(t) for k in A}


@dataclass
class State:
    shake: tuple = (0.0, 0.0, 0.0)   # dx, dy px, rot radians
    squash: float = 0.0              # 0..1 anticipation before the leap (negative stretches)
    fin_rattle: float = 0.0          # 0..1
    flame: float = 0.0               # 0..1 nozzle interior glows
    soot: float = 0.0                # 0..1 after the flight
    legs_fold: float = 0.0           # 0..1
    hatch_open: float = 0.0          # 0..1 (kettle tips back)
    window_glow: float = 0.0         # 0..1 warm light inside the cup
    light: float = 0.0               # 0..1 fire light from below


PROF = dict(
    nozzle=[(56, 41), (62, 42.5), (70, 39), (102, 30)],
    skirt=[(98, 50), (104, 52.5), (128, 49), (152, 45.5), (156, 47)],
    body=[(154, 44), (200, 45.5), (262, 45), (332, 42)],
    saucer=[(323, 34), (326, 55), (330, 61), (333, 59)],
    cup=[(330, 31), (334, 36), (339, 35), (351, 50), (368, 60), (392, 66.5), (416, 70.5), (428, 75)],
    kettle=[(424, 66), (431, 63), (446, 60.5), (462, 51.5), (476, 38), (486, 23), (491, 11)],
    knob=[(490, 4.5), (494, 4), (498, 8), (503, 7.5), (506, 3)],
)
WIN_H, WIN_TH, WIN_R = 384.0, 10 * D2R, 25.0
SOOT = np.array([0.17, 0.14, 0.16], np.float32)


def _prof(name, n=5):
    p = np.asarray(PROF[name], np.float64)
    if name in ("cup", "kettle", "knob"):
        p = core.catmull(p, n=n, closed=False)
    return p[:, 0], p[:, 1]


class _Proj:
    def __init__(self, x, y, s, rot, view, st):
        v = VIEWS[view] if isinstance(view, str) else view      # a dict blends views (see mix_views)
        self.e0, self.e1, self.k, self.yaw = v["e"], v["e1"], v["k"], v["yaw"]
        self.e = self.e0 + 0.5 * self.e1
        self.s = s
        self.rot = rot + st.shake[2]
        self.sx = 1 + 0.11 * st.squash
        self.sy = 1 - 0.15 * st.squash
        self.ox, self.oy = x + st.shake[0], y + st.shake[1]
        self.pa, self.pv = 0.0, np.zeros(2)

    def sc(self, h):
        return 1.0 / (1.0 + self.k * np.asarray(h, np.float64) / 500.0)

    def P(self, h, r, a):
        """View relative angle a (0 = toward the viewer)."""
        h, r, a = np.broadcast_arrays(np.asarray(h, np.float64), np.asarray(r, np.float64),
                                      np.asarray(a, np.float64))
        sc = self.sc(h)
        lx = r * np.sin(a) * sc * self.sx
        e = self.e0 + self.e1 * np.clip(h, 0, 520) / 500.0      # rings arch more the higher above the eye
        ly = (-(h * sc * (1 + 0.45 * self.k)) - r * e * np.cos(a) * sc) * self.sy
        c, sn = math.cos(self.rot), math.sin(self.rot)
        out = np.c_[(self.ox + (lx * c - ly * sn) * self.s).ravel(),
                    (self.oy + (lx * sn + ly * c) * self.s).ravel()]
        if self.pa != 0.0:                      # a part swung about a hinge (the kettle lid)
            c, sn = math.cos(self.pa), math.sin(self.pa)
            q = out - self.pv
            out = np.c_[self.pv[0] + q[:, 0] * c - q[:, 1] * sn, self.pv[1] + q[:, 0] * sn + q[:, 1] * c]
        return out

    def S(self, h, r, th):
        """Body fixed angle th (turns with the yaw of the view)."""
        return self.P(h, r, np.asarray(th, np.float64) + self.yaw)

    def depth(self, th):
        return math.cos(th + self.yaw)


def _ribbon(path, w0, w1, n=5):
    """A band of varying width around a centre line (screen points)."""
    p = core.catmull(path, n=n, closed=False) if len(path) > 2 else np.asarray(path, np.float64)
    p = core.resample(p, max(6, len(p)))
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nr = np.c_[-t[:, 1], t[:, 0]]
    w = np.linspace(w0, w1, len(p))[:, None] * 0.5
    return np.vstack([p + nr * w, (p - nr * w)[::-1]]), p, nr


class _Draw:
    def __init__(self, cel, pr, tod, drawing, st):
        self.c, self.pr, self.st, self.dr = cel, pr, st, int(drawing)
        self.T = palette.TOD[tod] if isinstance(tod, str) else tod
        self.ink = self.T["ink"]
        self.fire = palette.TOD["launch"]["fire"]
        self.lw = 2.1 * max(0.75, pr.s ** 0.55)
        self.sun_r = self.T["sun"][0] >= 0
        self.n = 0

    # ---- colour
    def lit(self, c):
        return palette.lit(c, self.T)

    def shd(self, c):
        return palette.shade(c, self.T)

    def rimc(self, c):
        return palette.rim(c, self.T)

    def hot(self, c, k=1.0):
        """Local colour under the fire light from below."""
        c = np.asarray(c, np.float32)
        f = np.clip(c * np.array([1.34, 0.87, 0.47], np.float32) + self.fire * 0.05, 0, 1)
        return palette.mix(self.lit(c), f, min(1.0, 0.9 * k))

    def sooty(self, c, k):
        return palette.mix(c, SOOT * (0.6 + 0.6 * self.T["amb"]), k)

    # ---- primitives
    def wob(self, pts, amp=0.55):
        self.n += 1
        return core.wobble(pts, amp * self.pr.s ** 0.5, core.seed_of("rk", self.n, self.dr) % 9973, 0.035)

    def fill(self, pts, col, ink=0.0, wob=0.5, seed=None):
        p = self.wob(pts, wob) if wob > 0 else np.asarray(pts)
        self.c.fill(p, col)
        if ink > 0:
            self.c.stroke(p, self.lw * ink, self.ink, closed=True, seed=self.n if seed is None else seed)
        return p

    def line(self, pts, w=1.0, col=None, taper=(0.15, 0.2), wob=0.5, pressure=None):
        p = self.wob(pts, wob) if wob > 0 else np.asarray(pts)
        self.c.stroke(p, self.lw * w, self.ink if col is None else col, taper=taper, seed=self.n,
                      pressure=pressure)

    # ---- bodies of revolution
    def solid(self, name, col, shade_k=1.0, fire_frac=0.3, under=None, soot_frac=0.0):
        pr, st = self.pr, self.st
        hs, rs = _prof(name)
        hp = math.pi / 2
        arc = np.linspace(-hp, hp, 26)
        left = pr.P(hs, rs, -hp)
        right = pr.P(hs[::-1], rs[::-1], hp)
        top = pr.P(hs[-1], rs[-1], arc[1:-1])
        bot = pr.P(hs[0], rs[0], np.linspace(hp, 3 * hp, 26)[1:-1])
        lit, shd = self.lit(col), self.shd(col)
        if name == "body":                      # hand rolled paper is never a true cylinder
            left = core.wobble(left, 1.5 * pr.s, 11, 0.012)
            right = core.wobble(right, 1.5 * pr.s, 12, 0.012)
        outline = self.wob(np.vstack([left, top, right, bot]), 0.5)
        self.c.fill(outline, lit)

        def patch(h0, h1, a0, a1, m=14):
            sel = (hs >= h0 - 1e-6) & (hs <= h1 + 1e-6)
            if sel.sum() < 2:
                hh = np.array([h0, h1])
            else:
                hh = hs[sel]
            rr = np.interp(hh, hs, rs)
            return np.vstack([pr.P(hh, rr, a0), pr.P(hh[-1], rr[-1], np.linspace(a0, a1, m)[1:-1]),
                              pr.P(hh[::-1], rr[::-1], a1), pr.P(hh[0], rr[0], np.linspace(a1, a0, m)[1:-1])])

        self.patch = patch
        self.rfun = lambda h: np.interp(h, hs, rs)
        h0, h1 = hs[0], hs[-1]
        # the one hard shadow tone, on the side away from the sun
        a0, a1 = (-hp, -0.36) if self.sun_r else (0.36, hp)
        self.c.fill(self.wob(patch(h0, h1, a0, a1), 0.9), palette.mix(lit, shd, shade_k))
        if soot_frac > 0 and st.soot > 0.02:
            hsoot = h0 + (h1 - h0) * soot_frac * (0.5 + 0.5 * st.soot)
            self.c.fill(self.wob(patch(h0, hsoot, -hp, hp), 2.2), self.sooty(lit, 0.62 * st.soot))
            self.c.fill(self.wob(patch(h0, hsoot, a0, a1), 2.2), self.sooty(shd, 0.7 * st.soot))
        if st.light > 0.02 and fire_frac > 0:
            hf = h0 + (h1 - h0) * fire_frac * (0.45 + 0.55 * st.light)
            self.c.fill(self.wob(patch(h0, hf, -hp, hp), 1.2), self.hot(col, st.light))
            self.c.fill(self.wob(patch(h0, h0 + (hf - h0) * 0.42, -1.0, 1.0), 1.2),
                        np.clip(self.hot(col, st.light) * 1.12 + 0.06, 0, 1))
        # thin rim light inside the line on the sun side
        b0, b1 = (1.30, hp) if self.sun_r else (-hp, -1.30)
        self.c.fill(patch(h0, h1, b0, b1), self.rimc(col))
        # underside disc (seen when the view is low)
        if pr.e0 + pr.e1 * h0 / 500.0 > 0.11:
            uc = self.shd(col) * 0.8 if under is None else under
            disc = pr.P(h0, rs[0] * 0.96, np.linspace(0, 2 * math.pi, 40, endpoint=False))
            self.c.fill(disc, uc)
            self.line(pr.P(h0, rs[0], arc), 0.8, wob=0.3)
        self._outline = (left, right, top, bot, hs, rs)
        return outline

    def solid_ink(self):
        left, right, top, bot, hs, rs = self._outline
        wl, wr = (1.3, 0.9) if self.sun_r else (0.9, 1.3)
        self.line(left, wl, taper=(0.04, 0.04))
        self.line(right, wr, taper=(0.04, 0.04))
        self.line(np.vstack([left[-1:], top, right[:1]]), 0.9, taper=(0.03, 0.03))
        self.line(np.vstack([right[-1:], bot, left[:1]]), 1.15, taper=(0.03, 0.03))

    def spatch(self, h0, h1, th0, th1, m=8, pad=0.0):
        """A patch stuck on the surface of the current solid between body angles."""
        a0 = max(th0 + self.pr.yaw, -math.pi / 2)
        a1 = min(th1 + self.pr.yaw, math.pi / 2)
        if a1 - a0 < 0.03:
            return None
        hh = np.linspace(h0, h1, 6)
        rr = self.rfun(hh) + pad
        pr = self.pr
        return np.vstack([pr.P(hh, rr, a0), pr.P(hh[-1], rr[-1], np.linspace(a0, a1, m)[1:-1]),
                          pr.P(hh[::-1], rr[::-1], a1), pr.P(hh[0], rr[0], np.linspace(a1, a0, m)[1:-1])])

    def ring(self, h, th0=-math.pi, th1=math.pi, w=0.7, col=None, pad=0.0, m=20):
        a0 = max(th0 + self.pr.yaw, -math.pi / 2)
        a1 = min(th1 + self.pr.yaw, math.pi / 2)
        if a1 - a0 < 0.05:
            return
        self.line(self.pr.P(h, self.rfun(h) + pad, np.linspace(a0, a1, m)), w, col, taper=(0.08, 0.08), wob=0.4)

    def on_surface(self, h, th, pts_uv):
        """Local (u, v) offsets in rocket units mapped onto the surface at (h, th)."""
        uv = np.asarray(pts_uv, np.float64)
        r = self.rfun(h)
        a = np.clip(th + uv[:, 0] / max(r, 1.0) + self.pr.yaw, -math.pi / 2, math.pi / 2)
        return self.pr.P(h + uv[:, 1], self.rfun(h + uv[:, 1]) + 0.3, a)

    def visible(self, th, margin=0.25):
        return abs(((th + self.pr.yaw + math.pi) % (2 * math.pi)) - math.pi) < math.pi / 2 - margin


def _star(rg, size):
    k = 5
    a = -math.pi / 2 + rg.uniform(-0.3, 0.3) + np.arange(2 * k) * math.pi / k
    rad = np.where(np.arange(2 * k) % 2 == 0, size, size * rg.uniform(0.40, 0.52))
    rad = rad * rg.uniform(0.8, 1.18, 2 * k)
    return np.c_[np.cos(a) * rad, np.sin(a) * rad]


# ------------------------------------------------------------------ the parts
def _legs(d, behind):
    pr, st = d.pr, d.st
    fold = st.legs_fold
    mids = []
    for i, th in enumerate((-100 * D2R, 20 * D2R, 140 * D2R)):
        front = pr.depth(th) >= 0
        top = pr.S(124, 30, th)[0]
        foot = pr.S(0 + 30 * fold, 88 - 46 * fold, th)[0]
        mid = (top + foot) / 2 + np.array([math.sin(th + pr.yaw) * 3.5, 0]) * pr.s
        mids.append((front, top + (foot - top) * 0.52))
        if front == behind:
            continue
        poly, cl, nr = _ribbon([top, mid, foot], 10.5 * pr.s, 8.2 * pr.s)
        base = R["bamboo"] if front else R["bamboo"] * 0.86
        col = d.lit(base) if (math.sin(th + pr.yaw) > -0.2) == d.sun_r else d.shd(base)
        if st.light > 0.02:
            col = palette.mix(col, d.hot(base, st.light), 0.8)
        d.fill(poly, col, ink=0.95, wob=0.3)
        half = len(poly) // 2
        d.c.fill(np.vstack([poly[:half], (cl + nr * 1.2 * pr.s)[::-1]]) if not d.sun_r else
                 np.vstack([poly[half:], (cl - nr * 1.2 * pr.s)]), d.shd(base) if st.light < 0.3 else d.lit(base * 0.8))
        for t in (0.24, 0.5, 0.74, 0.93):          # bamboo nodes
            k = int(t * (len(cl) - 1))
            w = (10.9 - 2.3 * t) * pr.s * 0.5
            d.line(np.array([cl[k] + nr[k] * w * 1.15, cl[k] - nr[k] * w * 1.15]), 0.85, taper=(0.1, 0.1), wob=0)
        for j in range(4):                          # string lashing at the top
            k = 1 + j
            w = 6.6 * pr.s
            a = cl[k] + nr[k] * w + (0, j * 1.2 * pr.s)
            b = cl[k] - nr[k] * w + (0, (j * 1.2 - 2.5) * pr.s)
            d.line(np.array([a, b]), 1.5, d.ink, wob=0)
            d.line(np.array([a, b]), 0.8, d.lit(R["string"]), wob=0)
        d.fill(_ribbon([foot - (0, 5 * pr.s), foot + (0, 1.5 * pr.s)], 9.5 * pr.s, 10.5 * pr.s)[0],
               d.shd(R["bamboo"]) * 0.8, ink=0.8, wob=0.2)
    if not behind:                                   # sagging brace string between the legs
        pts = [m for _, m in mids]
        for a, b in ((pts[0], pts[1]), (pts[1], pts[2])):
            mid = (a + b) / 2 + (0, 7 * pr.s)
            path = core.catmull([a, mid, b], n=8, closed=False)
            d.line(path, 1.3, wob=0.2)
            d.line(path, 0.7, d.lit(R["string"]), wob=0)


def _big_fin(d):
    pr, st = d.pr, d.st
    th = -90 * D2R
    rg = core.rng("finrattle", d.dr)
    rat = st.fin_rattle

    def sp(h, r, flut=0.0):
        p = pr.S(h, r, th)[0]
        if rat > 0 and flut > 0:
            p = p + rg.normal(0, 1.0, 2) * flut * rat * 5.0 * pr.s
        return p

    cloth = [sp(292, 46), sp(250, 68), sp(196, 97), sp(150, 121), sp(101, 147),
             sp(96, 126, 1.0), sp(109, 99, 1.4), sp(113, 72, 1.0), sp(124, 47)]
    cloth = np.array(cloth)
    edge = core.catmull(cloth[4:], n=6, closed=False)
    poly = np.vstack([cloth[:4], edge])
    lit_side = d.depth_lit(th)
    base = R["sail"]
    col = d.lit(base) if lit_side else d.shd(base)
    d.fill(poly, col, ink=0, wob=0.4)
    # shadow tone: a hard wedge near the body and under the spar
    d.c.fill(np.array([sp(286, 47), sp(232, 62), sp(160, 60), sp(126, 62), sp(124, 47)]), d.shd(base))
    if st.light > 0.02:
        lo = np.vstack([edge, [sp(150, 121), sp(140, 96), sp(138, 70), sp(140, 47)]])
        d.c.fill(lo, d.hot(base, st.light))
    # patches, each sewn on by paw
    for (h, r, hw, rw, c, sd) in ((208, 66, 17, 13, R["patch_red"], 1), (150, 88, 13, 15, R["patch_blue"], 2),
                                  (258, 53, 9, 7, R["tape"], 3)):
        q = core.rng("finpatch", sd)
        cs = [(h + hw * q.uniform(0.8, 1.1), r - rw), (h + hw, r + rw * q.uniform(0.8, 1.1)),
              (h - hw * q.uniform(0.8, 1.1), r + rw), (h - hw, r - rw * q.uniform(0.8, 1.1))]
        pp = np.array([sp(a, b, 0.4) for a, b in cs])
        d.fill(pp, d.lit(c) if lit_side else d.shd(c), ink=0.55, wob=0.5)
        for k in range(4):                            # stitches across the edge
            a, b = pp[k], pp[(k + 1) % 4]
            nrm = np.array([-(b - a)[1], (b - a)[0]])
            nrm /= max(np.hypot(*nrm), 1e-6)
            for t in (0.25, 0.5, 0.75):
                m = a + (b - a) * t
                d.line(np.array([m - nrm * 2.6 * pr.s, m + nrm * 2.6 * pr.s]), 0.5, wob=0)
    # a seam and a fold in the cloth
    d.line(np.array([sp(270, 50), sp(190, 82, 0.5), sp(118, 118, 0.8)]), 0.55, wob=0.6)
    d.line(np.array([sp(170, 50), sp(150, 72, 0.5), sp(120, 86, 0.8)]), 0.5, wob=0.6)
    d.line(np.vstack([cloth[:1], poly[3:], cloth[-1:]]), 1.0, taper=(0.02, 0.02), wob=0.3)
    d.line(np.array([sp(124, 47), sp(292, 46)]), 0.8, wob=0.3)
    # the bamboo spar with its lashings
    a, b = sp(306, 39), sp(84, 158)
    sparp, cl, nr = _ribbon([a, (a + b) / 2 + (1.5 * pr.s, 1.5 * pr.s), b], 7.5 * pr.s, 6.0 * pr.s)
    d.fill(sparp, d.lit(R["bamboo"]) if st.light < 0.3 else d.hot(R["bamboo"], st.light), ink=0.9, wob=0.25)
    for t in (0.18, 0.42, 0.66, 0.88):
        k = int(t * (len(cl) - 1))
        d.line(np.array([cl[k] + nr[k] * 4.6 * pr.s, cl[k] - nr[k] * 4.6 * pr.s]), 0.8, wob=0)
    for t in (0.07, 0.3, 0.55, 0.8):
        k = int(t * (len(cl) - 1))
        for j in (-1, 0, 1):
            o = (cl[min(k + 1, len(cl) - 1)] - cl[k])
            o = o / max(np.hypot(*o), 1e-6) * j * 1.7 * pr.s
            seg = np.array([cl[k] + nr[k] * 5.2 * pr.s + o, cl[k] - nr[k] * 5.2 * pr.s + o * 2.2])
            d.line(seg, 1.35, wob=0)
            d.line(seg, 0.7, d.lit(R["string"]), wob=0)


def _small_fin(d, th, variant):
    pr, st = d.pr, d.st
    rg = core.rng("sfin", d.dr, variant)
    rat = st.fin_rattle

    def sp(h, r, fl=0.0):
        p = pr.S(h, r, th)[0]
        return p + rg.normal(0, 1, 2) * fl * rat * 4.0 * pr.s if rat > 0 and fl > 0 else p

    shape = [(212, 44), (198, 58), (150, 84, 1), (114, 96, 1.5), (108, 84, 1.2), (136, 45)] if variant == 0 else \
        [(206, 44), (188, 62), (146, 80, 1), (118, 90, 1.5), (110, 76, 1.2), (140, 45)]
    pts = np.array([sp(*q) for q in shape])
    base = R["paper_dk"] if variant == 0 else R["patch_blue"] * 0.5 + R["paper"] * 0.5
    lit_side = d.depth_lit(th)
    d.fill(pts, d.lit(base) if lit_side else d.shd(base), ink=0.9, wob=0.4)
    if st.light > 0.02:
        d.c.fill(np.array([pts[2] * 0.6 + pts[5] * 0.4, pts[2], pts[3], pts[4], pts[5]]), d.hot(base, st.light))
        d.line(np.vstack([pts[2:], pts[:1]]), 0.9, wob=0)
    d.line(np.array([sp(196, 47), sp(150, 68, 0.6), sp(116, 88, 1)]), 0.5, wob=0.5)     # fold
    tp = np.array([sp(182, 43), sp(186, 56), sp(174, 58), sp(170, 44)])                  # tape at the root
    d.fill(tp, d.lit(R["tape"]), ink=0.5, wob=0.3)


def _handle(d):
    pr = d.pr
    th = 100 * D2R
    cl = [pr.S(h, r, th)[0] for h, r in ((411, 68), (417, 90), (402, 106), (376, 104), (359, 86), (354, 58))]
    poly, c2, nr = _ribbon(cl, 10 * pr.s, 8.5 * pr.s, n=6)
    col = R["cup"]
    d.fill(poly, d.lit(col), ink=0.95, wob=0.3)
    half = len(poly) // 2
    d.c.fill(np.vstack([poly[half:], c2 - nr * 0.8 * pr.s]), d.shd(col))
    if d.st.light > 0.02:
        k = len(c2) // 2
        d.c.fill(np.vstack([(c2 + nr * 4.4 * pr.s)[k:], (c2 - nr * 1.0 * pr.s)[k:][::-1]]), d.hot(col, d.st.light))
    d.line(poly[:half], 0.9, wob=0)
    d.line(poly[half:], 1.1, wob=0)


def _spout(d):
    pr, st = d.pr, d.st
    th = 80 * D2R
    cl = [pr.S(h, r, th)[0] for h, r in ((447, 54), (457, 73), (476, 83), (498, 82), (515, 88))]
    poly, c2, nr = _ribbon(cl, 15 * pr.s, 6.5 * pr.s, n=6)
    col = R["kettle"]
    d.fill(poly, d.lit(col), ink=0.95, wob=0.3)
    half = len(poly) // 2
    d.c.fill(np.vstack([poly[half:], c2 - nr * 0.5 * pr.s]), d.shd(col))
    d.line(poly[:half], 0.9, wob=0)
    d.line(poly[half:], 1.1, wob=0)
    tip = c2[-1]
    d.fill(core.ellipse(tip[0], tip[1], 3.6 * pr.s, 1.8 * pr.s, 14, rot=-0.5), d.shd(col) * 0.5, ink=0.6, wob=0)
    # a paper pennant tied to the antenna
    rg = core.rng("pennant", d.dr)
    fl = st.fin_rattle * 4.0 * pr.s
    a = c2[-2]
    b = a + np.array([17, -9]) * pr.s + rg.normal(0, 1, 2) * fl
    c = a + np.array([15, 4]) * pr.s + rg.normal(0, 1, 2) * fl
    d.fill(np.array([a + (0, -4 * pr.s), b, c, a + (0, 5 * pr.s)]), d.lit(R["patch_red"]), ink=0.7, wob=0.3)


def _key(d):
    pr = d.pr
    th = 92 * D2R
    col = R["copper"]
    a, b = pr.S(262, 43, th)[0], pr.S(263, 60, th)[0]
    d.fill(_ribbon([a, b], 4.2 * pr.s, 3.6 * pr.s)[0], d.lit(col), ink=0.8, wob=0)
    for hh, sgn in ((274, 1), (251, -1)):
        q = np.linspace(0, 2 * math.pi, 22, endpoint=False)
        ring = np.array([pr.S(hh + 10.5 * math.sin(t), 68 + 8.5 * math.cos(t) + sgn * 1.0, th)[0] for t in q])
        hole = np.array([pr.S(hh + 4.8 * math.sin(t), 68.5 + 3.4 * math.cos(t), th)[0] for t in q])
        d.fill(ring, d.lit(col), ink=0.85, wob=0.25)
        d.c.fill(np.array([pr.S(hh - 2 + 8.4 * math.sin(t), 67 + 7 * math.cos(t), th)[0]
                           for t in np.linspace(math.pi * 0.9, math.pi * 2.1, 14)]), d.shd(col))
        d.fill(hole, d.shd(R["paper"]) * 0.75, ink=0.6, wob=0)


def _window(d, glass_only=False, info=None):
    pr, st = d.pr, d.st
    q = np.linspace(0, 2 * math.pi, 44, endpoint=False)
    rf = lambda h: np.interp(h, *_prof("cup"))

    def ell(rad):
        hh = WIN_H + rad * np.sin(q)
        return pr.S(hh, rf(hh) + 0.6, WIN_TH + rad * np.cos(q) / rf(WIN_H))

    inner, outer, mid = ell(WIN_R), ell(WIN_R + 6.2), ell(WIN_R + 3.1)
    if not glass_only:
        dark = d.shd(R["patch_blue"]) * 0.55
        glow = np.array([1.0, 0.78, 0.42], np.float32)
        d.c.fill(inner, palette.mix(dark, glow, st.window_glow))
    else:
        # two slanted reflections, clipped to the opening by construction
        c = inner.mean(0)
        u = (inner - c) @ np.array([0.72, 0.69]) / (WIN_R * pr.s)
        fine = inner
        for lo, hi, colk in ((-0.78, -0.42, 0.9), (-0.30, -0.20, 0.75)):
            sel = (u > lo) & (u < hi)
            if sel.sum() >= 3:
                pts = fine[sel]
                ang = np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean())
                d.c.fill(pts[np.argsort(ang)], palette.mix(d.lit(R["glass"]), (1, 1, 1), 0.35) * colk + (1 - colk) * 0.3)
    # brass rim
    col = R["copper"] * 0.55 + R["star"] * 0.45
    d.c.fill(np.vstack([outer, outer[:1], inner[:1], inner[::-1]]), d.lit(col))
    half = slice(8, 30) if d.sun_r else slice(30, 44)
    d.c.fill(np.vstack([outer[half], mid[half][::-1]]), d.shd(col))
    d.c.stroke(outer, d.lw * 0.95, d.ink, closed=True, seed=3)
    d.c.stroke(inner, d.lw * 0.8, d.ink, closed=True, seed=4)
    for k in range(0, 44, 7):
        d.c.dot(mid[k][0], mid[k][1], 1.25 * pr.s, d.ink)
    c = inner.mean(0)
    rad = float(np.hypot(*(inner - c).T).mean())
    return inner, (float(c[0]), float(c[1]), rad)


def _body_details(d):
    pr, st = d.pr, d.st
    # the overlap seam of the rolled paper, with tape across it
    seam_th = -34 * D2R
    if d.visible(seam_th, 0.05):
        hh = np.linspace(157, 329, 14)
        d.line(pr.S(hh, d.rfun(hh) + 0.3, seam_th + 0.012 * np.sin(hh * 0.11)), 0.6, taper=(0.02, 0.02), wob=0.7)
    for h, th0, sd in ((300, -50, 0), (236, -47, 1), (176, -52, 2)):
        q = core.rng("tape", sd)
        p = d.spatch(h, h + 10 + q.uniform(-2, 3), th0 * D2R, (th0 + 32 + q.uniform(-4, 6)) * D2R)
        if p is not None:
            d.fill(p, d.lit(R["tape"]) if d.sun_r else d.shd(R["tape"]), ink=0.5, wob=0.5)
    # sheet joins
    d.ring(214, w=0.5)
    d.ring(280, w=0.5)
    # paper patches
    for h, th0, th1, hh, c, sd in ((238, 22, 58, 26, R["patch_red"], 0), (166, -6, 22, 20, R["patch_blue"], 1),
                                   (292, -84, -58, 22, R["paper_dk"], 2)):
        p = d.spatch(h, h + hh, th0 * D2R, th1 * D2R)
        if p is not None:
            lit = (th0 + th1) * 0.5 * D2R + pr.yaw > -0.36 if d.sun_r else (th0 + th1) * 0.5 * D2R + pr.yaw < 0.36
            d.fill(p, d.lit(c) if lit else d.shd(c), ink=0.6, wob=0.7)
    # stars painted on by paw, with drips
    for h, th, size, sd in ((298, 12, 15.0, 0), (258, -16, 9.0, 1), (212, 40, 7.0, 2), (194, -24, 12.0, 3),
                            (176, 58, 6.0, 4), (310, 52, 7.0, 5), (238, -62, 7.5, 6), (286, -66, 5.5, 7),
                            (222, 96, 8.0, 8), (300, 128, 9.0, 9), (190, -110, 9.0, 10)):
        t = th * D2R
        if not d.visible(t, 0.12):
            continue
        rg = core.rng("star", sd)
        lit = t + pr.yaw > -0.36 if d.sun_r else t + pr.yaw < 0.36
        col = d.lit(R["star"]) if lit else d.shd(R["star"])
        d.c.fill(d.on_surface(h, t, _star(rg, size)), col)
        if sd % 3 == 0:
            dl = rg.uniform(8, 16)
            d.c.stroke(d.on_surface(h, t, [(1.0, -size * 0.3), (1.4, -size * 0.3 - dl)]), 1.7 * pr.s, col,
                       taper=(0.0, 0.5))


def _cup_details(d):
    pr = d.pr
    band = d.patch(403, 419, -math.pi / 2, math.pi / 2)
    d.c.fill(d.wob(band, 0.5), d.lit(R["cup_band"]))
    a0, a1 = (-math.pi / 2, -0.36) if d.sun_r else (0.36, math.pi / 2)
    d.c.fill(d.patch(403, 419, a0, a1), d.shd(R["cup_band"]))
    for th, ln in ((-52, 13), (-20, 7), (30, 16), (44, 6), (66, 10), (-80, 9), (100, 12), (140, 8), (-130, 10)):
        t = th * D2R
        if d.visible(t, 0.1):
            lit = t + pr.yaw > -0.36 if d.sun_r else t + pr.yaw < 0.36
            col = d.lit(R["cup_band"]) if lit else d.shd(R["cup_band"])
            hh = np.array([405.0, 403 - ln])
            d.c.stroke(pr.S(hh, d.rfun(hh) + 0.3, t), 2.6 * pr.s, col, taper=(0.0, 0.45))
    d.ring(403, w=0.45)
    d.ring(419, w=0.45)
    # a chip in the rim and a crack
    if d.visible(-40 * D2R):
        d.line(d.on_surface(424, -40 * D2R, [(0, 3), (2, -3), (-1, -8), (3, -15)]), 0.45, wob=0.3)


def _kettle_details(d):
    pr = d.pr
    d.ring(432, w=0.6)
    d.ring(446, w=0.45)
    for h, th, sz in ((462, -30, 7), (470, 26, 5), (452, 52, 6)):       # dents
        t = th * D2R
        if d.visible(t, 0.3):
            pts = d.on_surface(h, t, [(-sz, 1), (-sz * 0.3, -sz * 0.55), (sz * 0.6, -sz * 0.3), (sz, sz * 0.4)])
            d.line(core.catmull(pts, n=5, closed=False), 0.55, wob=0.2)
            d.c.fill(d.on_surface(h, t, core.ellipse(0, sz * 0.25, sz * 0.75, sz * 0.36, 12)), d.shd(R["kettle"]))
    # one highlight
    hl = d.spatch(452, 474, 30 * D2R - pr.yaw, 38 * D2R - pr.yaw) if d.sun_r else d.spatch(452, 474, -38 * D2R - pr.yaw, -30 * D2R - pr.yaw)
    if hl is not None:
        d.c.fill(d.wob(hl, 0.4), d.rimc(R["kettle"]))


def _skirt_details(d):
    pr = d.pr
    for h in (109, 147):
        for th in np.arange(-180, 180, 24.0) + (7 if h == 147 else 0):
            t = th * D2R
            if d.visible(t, 0.12):
                p = pr.S(h, d.rfun(h) + 0.4, t)[0]
                d.c.dot(p[0], p[1], 1.9 * pr.s, d.ink)
                d.c.dot(p[0] + 0.5 * pr.s, p[1] - 0.6 * pr.s, 0.8 * pr.s, d.rimc(R["tin"]))
    if d.visible(-12 * D2R):
        seam = np.linspace(100, 154, 6)
        d.line(pr.S(seam, d.rfun(seam) + 0.3, -12 * D2R), 0.55, wob=0.4)
    for h, th, sz in ((128, 30, 8), (120, -48, 6)):
        t = th * D2R
        if d.visible(t, 0.3):
            d.c.fill(d.on_surface(h, t, core.ellipse(0, 0, sz, sz * 0.5, 12, rot=0.4)), d.shd(R["tin"]))
            d.line(core.catmull(d.on_surface(h, t, [(-sz, 2), (-sz * 0.2, -sz * 0.55), (sz, -1)]), n=5, closed=False),
                   0.5, wob=0.2)


def draw(cel, x, y, s=1.0, tod="dusk", drawing=0, rot=0.0, view="side", state=None, light=None):
    st = state or State()
    if light is not None:
        st = State(**{**st.__dict__, "light": float(light)})
    pr = _Proj(x, y, s, rot, view, st)
    d = _Draw(cel, pr, tod, drawing, st)
    d.depth_lit = lambda th: (math.sin(th + pr.yaw) > -0.25) == d.sun_r or abs(math.sin(th + pr.yaw)) < 0.25
    parts = [(pr.depth(-90 * D2R) - 0.01, lambda: _big_fin(d)),
             (pr.depth(70 * D2R), lambda: _small_fin(d, 70 * D2R, 0)),
             (pr.depth(-12 * D2R), lambda: _small_fin(d, -12 * D2R, 1)),
             (pr.depth(100 * D2R), lambda: _handle(d)),
             (pr.depth(80 * D2R), lambda: _spout(d)),
             (pr.depth(92 * D2R), lambda: _key(d))]
    hatch = core.clamp(st.hatch_open)
    if hatch > 0.01:
        parts = [p for k_, p in enumerate(parts) if k_ != 4]       # the spout swings with the kettle
    parts.sort(key=lambda p: p[0])
    _legs(d, behind=True)
    for dep, fn in parts:
        if dep < -0.02:
            fn()
    # bodies, top to bottom (we look from slightly below, so lower parts are nearer)
    tin_soot = d.sooty(R["tin"], 0.35)
    if hatch > 0.01:
        # the kettle is the hatch: it tips back about 70 degrees on a hinge at the rim, on the
        # side away from the handle. Nothing else of this cel covers the space above the rim,
        # so a panda drawn BEFORE the rocket shows its head through the open mouth of the cup.
        side = -1.0 if math.sin(100 * D2R + pr.yaw) > 0 else 1.0
        pr.pv = pr.P(427, 72, side * math.pi / 2)[0]
        pr.pa = side * 70 * D2R * core.overshoot(hatch, 1.2) if hatch < 1 else side * 70 * D2R
        # the dark inside of the kettle, seen through its open base
        hs_, rs_ = _prof("kettle")
        a_ = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        cc = pr.P(hs_[0], 0, 0)[0]
        ax = pr.P(hs_[0], rs_[0], math.pi / 2)[0] - cc
        up = pr.P(hs_[0] + rs_[0] * 0.30, 0, 0)[0] - cc
        d.c.fill(cc + np.cos(a_)[:, None] * ax + np.sin(a_)[:, None] * up, d.shd(R["kettle"]) * 0.55)
        if pr.depth(80 * D2R) < 0:
            _spout(d)
    d.solid("knob", R["copper"], fire_frac=0.0)
    d.solid_ink()
    d.solid("kettle", R["kettle"], fire_frac=0.0)
    _kettle_details(d)
    d.solid_ink()
    if hatch > 0.01:
        if pr.depth(80 * D2R) >= 0:
            _spout(d)
        pv = pr.pv.copy()
        pr.pa = 0.0
        # the hinge: a little brass tab and pin on the rim
        brass = R["copper"] * 0.6 + R["star"] * 0.4
        d.fill(core.ellipse(pv[0], pv[1] - 1.5 * s, 5.2 * s, 6.5 * s, 14), d.lit(brass), ink=0.7, wob=0.2)
        d.c.dot(pv[0], pv[1] - 1.5 * s, 1.5 * s, d.ink)
    d.solid("cup", R["cup"], fire_frac=0.36)
    _cup_details(d)
    d.solid_ink()
    win_poly, win = _window(d)
    d.solid("saucer", R["cup"], fire_frac=1.0)
    d.solid_ink()
    d.solid("body", R["paper"], fire_frac=0.07, soot_frac=0.3)
    _body_details(d)
    d.solid_ink()
    d.solid("skirt", R["tin"], fire_frac=0.85, soot_frac=0.7)
    _skirt_details(d)
    d.solid_ink()
    inner = palette.mix(SOOT * 0.7, np.array([1.0, 0.86, 0.5], np.float32), min(1.0, st.flame * 1.2))
    d.solid("nozzle", tin_soot, fire_frac=0.9, under=inner, soot_frac=1.0)
    d.solid_ink()
    for dep, fn in parts:
        if dep >= -0.02:
            fn()
    _legs(d, behind=False)

    m = Image.new("L", (cel.w, cel.h), 0)
    ImageDraw.Draw(m).polygon([(float(a - cel.x0), float(b - cel.y0)) for a, b in win_poly], fill=255)
    noz = pr.P(56, 0, 0)[0]
    tip = pr.P(506, 0, 0)[0]
    lo = pr.P(0, 0, 0)[0]
    bb = np.vstack([pr.P(0, 160, -math.pi / 2), pr.P(0, 110, math.pi / 2), pr.P(520, 110, -math.pi / 2),
                    pr.P(520, 110, math.pi / 2), pr.P(300, 165, -math.pi / 2)])
    return dict(window=win, window_poly=win_poly, mask=np.asarray(m, np.float32) / 255.0,
                nozzle=(float(noz[0]), float(noz[1])), tip=(float(tip[0]), float(tip[1])),
                feet=(float(lo[0]), float(lo[1])),
                bbox=(float(bb[:, 0].min()), float(bb[:, 1].min()), float(bb[:, 0].max()), float(bb[:, 1].max())),
                _args=(x, y, s, rot, view, st))


def draw_hatch_back(cel, x, y, s=1.0, tod="dusk", drawing=0, rot=0.0, view="side", state=None):
    """Optional, for hatch_open > 0: the dark inside of the open teacup and its far rim.
    Draw this on a cel that goes BEHIND the panda (same arguments as draw), then the panda,
    then rocket.draw. Without it the opening simply shows whatever is behind the rocket."""
    st = state or State()
    pr = _Proj(x, y, s, rot, view, st)
    d = _Draw(cel, pr, tod, drawing, st)
    cc = pr.P(428, 0, 0)[0]
    ax = pr.P(428, 73, math.pi / 2)[0] - pr.P(428, 0, math.pi / 2)[0]
    up = pr.P(428 + 73 * 0.26, 0, math.pi / 2)[0] - pr.P(428, 0, math.pi / 2)[0]
    cc = cc + (pr.P(428, 73, math.pi / 2)[0] - pr.P(428, 0, math.pi / 2)[0]) * 0.0
    cc = (pr.P(428, 73, math.pi / 2)[0] + pr.P(428, 73, -math.pi / 2)[0]) / 2
    a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    pts = cc + np.cos(a)[:, None] * ax + np.sin(a)[:, None] * up
    d.c.fill(pts, d.shd(R["cup"]) * 0.50)
    far = cc + np.cos(a[:25])[:, None] * ax + np.sin(a[:25])[:, None] * up
    d.line(far, 0.8, wob=0.2, taper=(0.03, 0.03))
    d.c.stroke(cc + np.cos(a[2:23])[:, None] * ax * 0.93 + np.sin(a[2:23])[:, None] * up * 0.80, d.lw * 1.6,
               d.shd(R["cup"]) * 0.8, taper=(0.3, 0.3))


def draw_glass(cel, info, tod="dusk", drawing=0):
    """Glass reflections and the brass window rim, over whatever sits behind the glass."""
    x, y, s, rot, view, st = info["_args"]
    d = _Draw(cel, _Proj(x, y, s, rot, view, st), tod, drawing, st)
    _window(d, glass_only=True)


# ============================================================ cockpit close ups
# The same teacup and window, drawn large. Order for the outside view (s4b, s5a, s5c):
#     draw_cockpit_back(cel, ...)        inside of the cup seen through the hole
#     ... the panda, clipped with cockpit_mask(cx, cy, r) ...
#     draw_cockpit_front(cel, ...)       porcelain wall, brass rim, glass on top
# and for the inside view (s7b): sky, draw_cockpit_inside(cel, ...), then the panda in front.
WARM = np.array([1.34, 0.87, 0.47], np.float32)


class _Close:
    def __init__(self, cel, tod, drawing, r, light):
        self.c, self.dr, self.k = cel, int(drawing), r / 380.0
        self.T = palette.TOD[tod] if isinstance(tod, str) else tod
        self.ink = self.T["ink"]
        self.sun_r = self.T["sun"][0] >= 0
        self.sg = 1.0 if self.sun_r else -1.0
        self.name = tod if isinstance(tod, str) else str(tod.get("name", ""))
        self.light = float(light or 0.0)
        self.lw = 4.4 * max(0.5, self.k) ** 0.8
        self.n = 0

    def warm(self, c):
        if self.light <= 0.01:
            return np.asarray(c, np.float32)
        return palette.mix(c, np.clip(np.asarray(c, np.float32) * WARM * 1.25 + 0.04, 0, 1), min(1.0, self.light) * 0.8)

    def lit(self, c):
        return self.warm(palette.lit(c, self.T))

    def shd(self, c, k=1.0):
        return self.warm(palette.shade(c, self.T) * k)

    def rimc(self, c):
        return palette.rim(c, self.T)

    def wob(self, pts, amp=1.0, freq=0.012):
        self.n += 1
        return core.wobble(pts, amp * self.k, core.seed_of("cp", self.n, self.dr) % 9973, freq)

    def fill(self, pts, col, ink=0.0, amp=1.0):
        p = self.wob(pts, amp) if amp > 0 else np.asarray(pts, np.float64)
        self.c.fill(p, col)
        if ink > 0:
            self.c.stroke(p, self.lw * ink, self.ink, closed=True, seed=self.n)
        return p

    def line(self, pts, w=1.0, col=None, taper=(0.12, 0.18), amp=1.0):
        p = self.wob(pts, amp) if amp > 0 else np.asarray(pts, np.float64)
        self.c.stroke(p, self.lw * w, self.ink if col is None else col, taper=taper, seed=self.n)


def _sector(cx, cy, r0, r1, a0, a1, n=40):
    a = np.linspace(a0, a1, n)
    return np.vstack([np.c_[cx + np.cos(a) * r1, cy + np.sin(a) * r1],
                      np.c_[cx + np.cos(a[::-1]) * r0, cy + np.sin(a[::-1]) * r0]])


def _ring_poly(cx, cy, r0, r1, n=120, ox=0.0, oy=0.0):
    """An annulus as one keyhole polygon (outer circle may be offset by ox, oy)."""
    a = np.linspace(0, 2 * math.pi, n + 1)
    outer = np.c_[cx + ox + np.cos(a) * r1, cy + oy + np.sin(a) * r1]
    inner = np.c_[cx + np.cos(a[::-1]) * r0, cy + np.sin(a[::-1]) * r0]
    return np.vstack([outer, inner])


def _holed_frame(cel, cx, cy, r, pad=60, n=140):
    """The whole cel area with a round hole, as one keyhole polygon."""
    a = np.linspace(-math.pi / 2, 1.5 * math.pi, n + 1)
    hole = np.c_[cx + np.cos(a[::-1]) * r, cy + np.sin(a[::-1]) * r]
    x0, y0, x1, y1 = cel.x0 - pad, cel.y0 - pad, cel.x0 + cel.w + pad, cel.y0 + cel.h + pad
    rect = np.array([(cx, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0), (cx, y0)], np.float64)
    return np.vstack([rect, hole])


def cockpit_mask(cx=960, cy=560, r=380, w=core.W, h=core.H, grow=3.0):
    """H x W float alpha of the round window opening, to clip the panda behind the glass."""
    ss = 2
    m = Image.new("L", (w * ss, h * ss), 0)
    rr = (r + grow) * ss
    ImageDraw.Draw(m).ellipse([cx * ss - rr, cy * ss - rr, cx * ss + rr, cy * ss + rr], fill=255)
    return np.asarray(m.reduce(ss), np.float32) / 255.0


def _pinned(q, x, y, w, h, rot, kind, seed):
    """A paw drawing on a scrap of paper, held by a pin."""
    rg = core.rng("pinned", seed)
    c, s = math.cos(rot), math.sin(rot)

    def tr(pts):
        p = np.asarray(pts, np.float64) * (w * 0.5, h * 0.5)
        return np.c_[x + p[:, 0] * c - p[:, 1] * s, y + p[:, 0] * s + p[:, 1] * c]

    paper = R["paper"] if kind == "rocket" else R["sail"]
    corners = [(-1, -1), (1, -0.94), (0.96, 1), (0.55, 1.02), (-1.02, 0.95)]
    q.fill(tr(corners), q.shd(paper, 1.18), ink=0.55, amp=1.2)
    q.c.fill(tr([(0.55, 1.02), (0.96, 1), (0.98, 0.62)]), q.shd(paper, 0.86))          # a curled corner
    q.line(tr([(0.55, 1.02), (0.98, 0.62)]), 0.4, amp=0)
    dc = palette.mix(q.ink, q.shd(R["patch_blue"]), 0.45)
    if kind == "rocket":
        q.line(tr([(-0.15, 0.55), (-0.2, -0.25), (0.02, -0.62), (0.24, -0.25), (0.2, 0.55), (-0.15, 0.55)]), 0.5, dc)
        q.line(tr([(-0.17, 0.3), (-0.5, 0.66), (-0.16, 0.56)]), 0.45, dc)
        q.line(tr([(0.21, 0.3), (0.52, 0.66), (0.2, 0.56)]), 0.45, dc)
        q.c.fill(core.ellipse(*tr([(0.02, -0.12)])[0], w * 0.06, w * 0.06, 14), dc)
        for i in range(3):
            q.line(tr([(-0.06 + 0.08 * i, 0.62), (-0.1 + 0.12 * i, 0.86)]), 0.4, q.shd(R["patch_red"], 1.2), amp=0)
        st = _star(rg, 0.14) + (0.58, -0.6)
        q.c.fill(tr(st), q.shd(R["star"], 1.25))
    else:
        # a star map: dots joined by a wobbly line, one big star, a crescent
        pts = [(-0.7, 0.5), (-0.35, 0.1), (-0.05, 0.3), (0.3, -0.2), (0.62, -0.05)]
        q.line(tr(pts), 0.35, dc, taper=(0.0, 0.0))
        for p in pts:
            q.c.fill(core.ellipse(*tr([p])[0], w * 0.028, w * 0.028, 10), dc)
        q.c.fill(tr(_star(rg, 0.2) + (0.36, -0.58)), q.shd(R["star"], 1.25))
        q.line(tr([(-0.62, -0.62), (-0.78, -0.4), (-0.6, -0.22)]), 0.45, dc)
    pin = tr([(0.02, -0.86)])[0]
    q.c.fill(core.ellipse(pin[0], pin[1], w * 0.055, w * 0.055, 14), q.shd(R["patch_red"], 1.3))
    q.c.stroke(core.ellipse(pin[0], pin[1], w * 0.055, w * 0.055, 14), q.lw * 0.4, q.ink, closed=True)
    q.c.dot(pin[0] - w * 0.015, pin[1] - w * 0.02, w * 0.016, q.rimc(R["patch_red"]))


def _dial(q, x, y, rad):
    brass = R["copper"] * 0.55 + R["star"] * 0.45
    glow = np.array([1.0, 0.84, 0.50], np.float32)
    q.fill(core.ellipse(x, y, rad, rad, 40), q.shd(brass, 1.25), ink=0.7, amp=0.6)
    face = palette.mix(palette.shade(R["paper"], q.T), glow, 0.72)
    q.fill(core.ellipse(x, y, rad * 0.78, rad * 0.78, 36), face, ink=0.45, amp=0.4)
    q.c.fill(_sector(x, y, rad * 0.5, rad * 0.78, 0.2, 2.4, 16), palette.mix(face, q.shd(brass), 0.35))
    for a in np.linspace(math.pi * 0.85, math.pi * 2.15, 7):
        p0 = (x + math.cos(a) * rad * 0.56, y + math.sin(a) * rad * 0.56)
        p1 = (x + math.cos(a) * rad * 0.70, y + math.sin(a) * rad * 0.70)
        q.line(np.array([p0, p1]), 0.35, amp=0, taper=(0, 0))
    a = math.pi * 1.62 + 0.08 * core.noise1(q.dr * 0.37, 5)
    q.line(np.array([(x - math.cos(a) * rad * 0.12, y - math.sin(a) * rad * 0.12),
                     (x + math.cos(a) * rad * 0.62, y + math.sin(a) * rad * 0.62)]), 0.55,
           q.shd(R["patch_red"], 1.1), amp=0, taper=(0.0, 0.5))
    q.c.dot(x, y, rad * 0.09, q.ink)


def _strap(q, pts, width):
    poly, cl, nr = _ribbon(pts, width, width * 0.92, n=8)
    q.fill(poly, q.shd(R["sail"], 0.95), ink=0.6, amp=0.8)
    for sgn in (-1, 1):
        q.line(cl + nr * sgn * width * 0.33, 0.25, palette.mix(q.ink, q.shd(R["sail"]), 0.4), amp=0.3, taper=(0, 0))
    k = len(cl) // 2
    b = cl[k]
    brass = R["copper"] * 0.55 + R["star"] * 0.45
    q.fill(core.ellipse(b[0], b[1], width * 0.72, width * 0.5, 18, rot=math.atan2(nr[k][1], nr[k][0])),
           q.shd(brass, 1.2), ink=0.5, amp=0.3)


AMBER = np.array([1.0, 0.72, 0.40], np.float32)


def _to_cel(cel, arr):
    """A 1x float RGB field as a PIL image the size of the supersampled cel."""
    im = Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8), "RGB")
    return im.resize(cel.im.size, Image.BILINEAR)


def _apply_paint(cel, mult, hi, cx, cy, rad, outside):
    """Multiply and screen soft fields into what is already drawn on the cel, either outside
    or inside the circle (cx, cy, rad). The alpha of the cel is untouched."""
    rgb = cel.im.convert("RGB")
    al = cel.im.getchannel("A")
    new = ImageChops.multiply(rgb, _to_cel(cel, mult))
    if hi is not None:
        new = ImageChops.screen(new, _to_cel(cel, hi))
    m = Image.new("L", cel.im.size, 255 if outside else 0)
    ss = cel.ss
    box = [(cx - cel.x0 - rad) * ss, (cy - cel.y0 - rad) * ss, (cx - cel.x0 + rad) * ss, (cy - cel.y0 + rad) * ss]
    ImageDraw.Draw(m).ellipse(box, fill=0 if outside else 255)
    out = Image.composite(new, rgb, m)
    out.putalpha(al)
    cel.im = out
    cel.d = ImageDraw.Draw(cel.im)


def _grid(cel):
    yy, xx = np.mgrid[0:cel.h, 0:cel.w].astype(np.float32)
    return xx + cel.x0, yy + cel.y0


def _paint_wall(q, cel, cx, cy, r, hole_r, inside=False, flood=0.0):
    """Porcelain as a painted surface, not a flat fill: the cup is round, so the wall is lighter
    toward the light and falls off slowly to the shadow side; the glaze is mottled and carries a
    faint brush; the brass rim lays a soft contact shadow; one broad soft highlight sits on the glaze."""
    xx, yy = _grid(cel)
    k = q.k
    dist = np.hypot(xx - cx, yy - cy)
    cup = R["cup"]
    mott = core.fbm(cel.h, cel.w, 210.0 * k, 3, seed=71, ox=cel.x0, oy=cel.y0)
    brush = core.fbm(cel.h, cel.w, 10.0 * k, 2, seed=72, ox=cel.x0, oy=cel.y0, aspect=7.0)
    grain = core.fbm(cel.h, cel.w, 2.4, 2, seed=73)
    v = 1.0 - (0.10 * (1 - mott) + 0.055 * (1 - brush) + 0.03 * (1 - grain))
    extra = None
    dim = 1.0
    if not inside:
        # a round hull turning away from the light: darkest at the far edge, dark again at the near
        # edge and toward the bottom; the value runs from about 0.35 to 0.9 of the local colour
        ratio = np.clip(palette.shade(cup, q.T) / np.maximum(palette.lit(cup, q.T), 1e-3), 0, 1)
        ratio = ratio / max(float(ratio.mean()), 1e-3) * 0.36
        half = 0.5 * cel.w
        c = (xx - (cel.x0 + half + q.sg * 0.20 * half)) / (1.12 * half)
        c = c + 0.05 * ((yy - cy) / r) ** 2 * np.sign(c)
        lam = np.sqrt(np.clip(1 - c * c, 0, 1))
        vb = np.clip((yy - (cy + 0.1 * r)) / (1.7 * r), 0, 1)
        val = lam ** 1.5 * (1 - 0.55 * vb * vb)
        sa = math.atan2(q.T["sun"][1], q.T["sun"][0])
        dn = ((xx - cx) * math.cos(sa) + (yy - cy) * math.sin(sa)) / np.maximum(dist, 1.0)
        cs = np.exp(-np.clip(dist - 1.25 * r, 0, None) / (0.085 * r)) * (0.34 + 0.26 * (0.5 - 0.5 * dn))
        gt = np.clip(1 - val + cs, 0, 1)
        g = gt
        sag = 0.03 * r * ((xx - cx) / r) ** 2
        inband = ((yy > cy - 1.44 * r + sag) & (yy < cy - 0.80 * r + sag)).astype(np.float32)
        streak = core.fbm(cel.h, cel.w, 5.0 * k, 2, seed=74, ox=cel.x0, oy=cel.y0, aspect=14.0)
        v = v * (1 - inband * (0.20 * (1 - streak) + 0.08 * (1 - mott))) * 0.92
        hx, hy = cx + q.sg * 1.85 * r, cy - 0.10 * r
        hi = 0.27 * np.exp(-(((xx - hx) / (0.40 * r)) ** 2 + ((yy - hy) / (1.0 * r)) ** 2)) * (1 - 0.7 * inband)
        # a warm bounce from below, and the first rose of the sky along the edge that faces it
        bounce = 0.11 * np.exp(-((yy - (cel.y0 + cel.h)) / (0.55 * r)) ** 2) * (0.4 + 0.6 * lam)
        xe = cel.x0 + cel.w if q.sun_r else cel.x0
        edge = 0.15 * np.exp(-((xx - xe) / (0.42 * r)) ** 2)
        extra = bounce[..., None] * np.array([1.0, 0.66, 0.44], np.float32) + edge[..., None] * np.clip(q.T["rim"], 0, 1)
        dim = {"predawn": 0.72, "night": 0.62, "launch": 0.78}.get(q.name, 0.95)
        dim = dim + (1.0 - dim) * min(1.0, q.light) * 0.7
    else:
        ratio = np.array([0.52, 0.50, 0.62], np.float32)
        g = np.clip((dist - 1.15 * r) / (3.4 * r), 0, 1)
        g = g * g * (3 - 2 * g)
        side = np.clip((np.abs(xx - cx) - 1.6 * r) / (2.2 * r), 0, 1)
        cs = np.exp(-np.clip(dist - 1.16 * r, 0, None) / (0.06 * r)) * 0.22
        gt = np.clip(0.10 + 0.55 * g + 0.45 * side * side + cs * (1 - flood), 0, 1)
        hi = (0.10 + 0.30 * flood) * np.exp(-((dist - 1.30 * r) / (0.34 * r)) ** 2)
    mult = (1 - gt[..., None] * (1 - ratio)) * v[..., None] * dim
    hic = hi[..., None] * np.clip(palette.rim(cup, q.T), 0, 1) * dim
    if extra is not None:
        hic = hic + extra * dim
    _apply_paint(cel, mult, hic, cx, cy, hole_r, outside=True)


def _amber(q, cel, cx, cy, r):
    """The cockpit's own light on whatever sits behind the glass: a soft amber veil that is
    strongest toward the dial (lower right) and the lantern (above), so it falls on the face."""
    xx, yy = _grid(cel)
    d1 = np.hypot(xx - (cx + 0.63 * r), yy - (cy + 0.47 * r))
    d2 = np.hypot(xx - (cx - 0.30 * r), yy - (cy - 0.80 * r))
    al = 0.24 * np.exp(-(d1 / (0.85 * r)) ** 2) + 0.15 * np.exp(-(d2 / (0.8 * r)) ** 2)
    al = al * np.clip((r - np.hypot(xx - cx, yy - cy)) / 3.0, 0, 1)
    a8 = Image.fromarray((np.clip(al, 0, 1) * 255.0 + 0.5).astype(np.uint8), "L").resize(cel.im.size, Image.BILINEAR)
    ov = Image.new("RGBA", cel.im.size, _rgba_amber())
    ov.putalpha(a8)
    cel.im = Image.alpha_composite(cel.im, ov)
    cel.d = ImageDraw.Draw(cel.im)


def _rgba_amber():
    return (255, int(0.74 * 255), int(0.40 * 255), 255)


def draw_cockpit_back(cel, tod="predawn", drawing=0, cx=960, cy=560, r=380, light=None):
    """The inside of the teacup seen through the window: a warm, dim cockpit. The wall is
    lined with pasted paper above and scrap wood below, a small lantern hangs from the
    roof and the dial glows, both real amber light sources (their light is painted soft
    over the cel fills). Two pinned paw drawings, a strap. light 0..1 warms it more."""
    q = _Close(cel, tod, drawing, r, light)
    T = dict(q.T)
    T["amb"] = AMBER * 0.66                    # everything in here is lit by the lamp, not the sky
    T["key"] = AMBER * 1.05
    T["rim"] = np.array([1.0, 0.86, 0.56], np.float32)
    q.T = T
    rg = core.rng("cockpit lining")
    paper, wood = R["paper"], np.array([0.66, 0.46, 0.28], np.float32)
    q.c.fill(core.ellipse(cx, cy, r * 1.1, r * 1.1, 90), q.shd(paper, 0.80))
    # pasted paper: overlapping sheets, each its own value, a torn corner here and there
    for i in range(7):
        px, py = cx + (-1.05 + i * 0.36 + rg.normal(0, 0.05)) * r, cy + (-0.62 + rg.normal(0, 0.10)) * r
        w, h, a = r * rg.uniform(0.40, 0.58), r * rg.uniform(0.55, 0.78), rg.normal(0, 0.07)
        c, sn = math.cos(a), math.sin(a)
        quad = np.array([(-1, -1), (1, -0.95), (0.97, 1), (-0.6, 1.02), (-1.02, 0.7)]) * (w * 0.5, h * 0.5)
        quad = np.c_[px + quad[:, 0] * c - quad[:, 1] * sn, py + quad[:, 0] * sn + quad[:, 1] * c]
        q.fill(quad, q.shd(paper * rg.uniform(0.84, 1.06), 0.92), ink=0.28, amp=1.6)
        if i % 2 == 0:
            tp = np.array([(-0.5, -1.06), (0.1, -1.1), (0.14, -0.86), (-0.46, -0.82)]) * (w * 0.5, h * 0.5)
            q.c.fill(np.c_[px + tp[:, 0] * c - tp[:, 1] * sn, py + tp[:, 0] * sn + tp[:, 1] * c], q.shd(R["tape"], 1.0))
    # scrap wood: boards of different widths and heights, grain and a nail each
    x = cx - 1.12 * r
    j = 0
    while x < cx + 1.12 * r:
        w = r * rg.uniform(0.16, 0.30)
        top = cy + r * (0.20 + rg.uniform(-0.07, 0.08))
        col = q.shd(wood * rg.uniform(0.78, 1.14), 0.95)
        board = np.array([(x, top + rg.normal(0, 4) * q.k), (x + w, top + rg.normal(0, 4) * q.k), (x + w + 2, cy + 1.12 * r),
                          (x - 1, cy + 1.12 * r)])
        q.fill(board, col, ink=0.42, amp=1.0)
        for gline in range(2):
            gx = x + w * rg.uniform(0.2, 0.8)
            q.line(np.array([(gx, top + r * 0.04), (gx + rg.normal(0, 5) * q.k, top + r * rg.uniform(0.3, 0.8))]), 0.25,
                   col * 0.72, amp=1.5)
        q.c.dot(x + w * 0.5, top + r * 0.045, r * 0.012, q.ink)
        if j % 3 == 1:
            q.c.fill(core.ellipse(x + w * 0.55, top + r * rg.uniform(0.3, 0.6), w * 0.13, w * 0.2, 12), col * 0.6)   # a knot
        x += w
        j += 1
    _pinned(q, cx - r * 0.60, cy - r * 0.50, r * 0.36, r * 0.44, -0.13, "rocket", 0)
    _pinned(q, cx + r * 0.58, cy - r * 0.46, r * 0.34, r * 0.34, 0.16, "map", 1)
    # the lantern: a little tin frame on a wire, its glass alight
    lx, ly = cx - r * 0.30, cy - r * 0.80
    tin = R["tin"]
    q.line(np.array([(lx + r * 0.02, cy - r * 1.1), (lx, ly - r * 0.10)]), 0.4, amp=0.6)
    q.fill(np.array([(lx - r * 0.035, ly - r * 0.10), (lx + r * 0.035, ly - r * 0.10), (lx + r * 0.07, ly - r * 0.06),
                     (lx - r * 0.07, ly - r * 0.06)]), q.shd(tin, 0.9), ink=0.5, amp=0.4)
    q.fill(np.array([(lx - r * 0.058, ly - r * 0.06), (lx + r * 0.058, ly - r * 0.06), (lx + r * 0.048, ly + r * 0.075),
                     (lx - r * 0.048, ly + r * 0.075)]), np.array([1.0, 0.86, 0.52], np.float32), ink=0.5, amp=0.4)
    q.c.fill(core.ellipse(lx, ly + r * 0.012, r * 0.022, r * 0.042, 14), np.array([1.0, 0.97, 0.82], np.float32))
    q.fill(np.array([(lx - r * 0.065, ly + r * 0.075), (lx + r * 0.065, ly + r * 0.075), (lx + r * 0.05, ly + r * 0.105),
                     (lx - r * 0.05, ly + r * 0.105)]), q.shd(tin, 0.8), ink=0.5, amp=0.4)
    for sgn in (-1, 1):
        q.line(np.array([(lx + sgn * r * 0.056, ly - r * 0.06), (lx + sgn * r * 0.047, ly + r * 0.075)]), 0.45, amp=0)
    dx, dy, dr_ = cx + r * 0.63, cy + r * 0.47, r * 0.15
    _dial(q, dx, dy, dr_)
    _strap(q, [(cx - r * 1.02, cy - r * 0.08), (cx - r * 0.86, cy + r * 0.36), (cx - r * 0.62, cy + r * 0.72),
               (cx - r * 0.32, cy + r * 1.0)], r * 0.10)
    # the light itself: bright near the two lamps, falling away into brown dark; a paper and wood grain over all
    xx, yy = _grid(cel)
    d_l = np.hypot(xx - lx, yy - ly)
    d_d = np.hypot(xx - dx, yy - dy)
    lit_ = np.clip(np.exp(-(d_l / (0.78 * r)) ** 2) + 0.75 * np.exp(-(d_d / (0.55 * r)) ** 2), 0, 1)
    grain = core.fbm(cel.h, cel.w, 3.0, 2, seed=91) * 0.5 + core.fbm(cel.h, cel.w, 12.0 * q.k, 2, seed=92, aspect=0.2) * 0.5
    mult = ((0.40 + 0.60 * lit_) * (0.90 + 0.10 * grain))[..., None] * np.array([1.0, 0.96, 0.90], np.float32)
    hi = (0.30 * np.exp(-(d_l / (0.30 * r)) ** 2) + 0.20 * np.exp(-(d_d / (0.24 * r)) ** 2))[..., None] * AMBER
    _apply_paint(cel, mult, hi, cx, cy, r * 1.1, outside=False)


def _crazing(q, x, y, length, seed, n_branch=4):
    rg = core.rng("craze", seed)
    col = palette.mix(q.shd(R["cup"]), q.ink, 0.14)
    a = rg.uniform(0, 2 * math.pi)
    p = np.array([x, y], np.float64)
    trunk = [p.copy()]
    for i in range(7):
        a += rg.normal(0, 0.45)
        p = p + np.array([math.cos(a), math.sin(a)]) * length / 7
        trunk.append(p.copy())
    trunk = np.array(trunk)
    q.c.stroke(trunk, q.lw * 0.30, col, taper=(0.1, 0.6))
    for b in range(n_branch):
        k = rg.integers(1, 6)
        a2 = rg.uniform(0, 2 * math.pi)
        p = trunk[k].copy()
        br = [p.copy()]
        for i in range(4):
            a2 += rg.normal(0, 0.5)
            p = p + np.array([math.cos(a2), math.sin(a2)]) * length / 11
            br.append(p.copy())
        q.c.stroke(np.array(br), q.lw * 0.22, col, taper=(0.0, 0.7))


def _motifs(q, cx, cy, r, xs, lo):
    """Cobalt painted on by hand, as on real porcelain: a wavy line under the band and a few
    leaf sprigs, uneven, laid in before the glaze pass so they take its brush and its light."""
    cup, band = q.lit(R["cup"]), q.lit(R["cup_band"])
    cob, cob2 = palette.mix(cup, band, 0.80), palette.mix(cup, band, 0.52)
    rg = core.rng("cobalt motifs")
    for seg in (xs < cx - 1.34 * r, xs > cx + 1.34 * r):
        if seg.sum() > 3:
            x = xs[seg]
            y = lo[seg] + 0.13 * r + 0.035 * r * np.sin(x / (0.13 * r))
            pr_ = 0.55 + 0.45 * np.abs(np.sin(x / (0.26 * r) + 0.6))
            q.c.stroke(q.wob(np.c_[x, y], 2.0), 0.036 * r, cob, taper=(0.06, 0.1), pressure=pr_, seed=5)
            q.c.stroke(np.c_[x, y + 0.08 * r + 0.02 * r * np.sin(x / (0.13 * r) + 1.0)], 0.013 * r, cob2, taper=(0.1, 0.1), seed=6)
    for (ax, ay, sz, rot) in ((-1.78, 0.92, 0.17, 0.5), (2.02, 0.84, 0.15, -0.7), (-2.20, -0.26, 0.13, 1.9), (1.62, 1.32, 0.12, 0.2)):
        px, py = cx + ax * r, cy + ay * r
        n_leaf = 3 + int(rg.integers(0, 2))
        for j in range(n_leaf):
            a = rot + (j - (n_leaf - 1) / 2) * 0.75 + rg.normal(0, 0.12)
            ln = sz * r * rg.uniform(0.75, 1.1)
            leaf = np.array([(0, 0), (0.35, 0.22), (0.75, 0.16), (1, 0), (0.75, -0.16), (0.35, -0.2)]) * (ln, ln * rg.uniform(0.8, 1.2))
            q.c.fill(q.wob(core.xform(core.catmull(leaf, n=4), rot=a, tx=px, ty=py), 1.5), cob if j % 2 == 0 else cob2)
        q.c.dot(px, py, 0.016 * r, cob)
        stem = core.xform(np.array([(0.0, 0.0), (-0.5, 0.08), (-0.9, -0.05)]) * sz * r, rot=rot, tx=px, ty=py)
        q.c.stroke(stem, 0.015 * r, cob, taper=(0.0, 0.6))


def _wall_outside(q, cel, cx, cy, r):
    """The porcelain outside wall around the hole: blue band, drips, crazing, a chip."""
    cup, band = R["cup"], R["cup_band"]
    hole_r = r * 1.03
    q.c.fill(_holed_frame(cel, cx, cy, hole_r), q.lit(cup))
    x0, x1 = cel.x0 - 60.0, cel.x0 + cel.w + 60.0
    y0, y1 = cel.y0 - 60.0, cel.y0 + cel.h + 60.0
    # the one shadow tone: the wall turning away from the light
    ys = np.linspace(y0, y1, 20)
    xb = cx - q.sg * (r * 1.52 + r * 0.16 * np.cos((ys - cy) / r * 0.9))
    edge = np.c_[xb, ys]
    far = x0 if q.sun_r else x1
    del edge, far                              # the falloff to the shadow side is painted soft (_paint_wall)
    # the blue band: an arch across the top, cut where the window rises into it
    xs = np.linspace(x0, x1, 120)
    sag = 0.03 * r * ((xs - cx) / r) ** 2
    # a hand painted band: its edges wander
    up = cy - 1.44 * r + sag + (core.fbm_at(xs, np.zeros_like(xs), 130.0 * q.k, 3, 81) - 0.5) * 0.05 * r
    lo = cy - 0.80 * r + sag + (core.fbm_at(xs, np.zeros_like(xs) + 40.0, 110.0 * q.k, 3, 82) - 0.5) * 0.06 * r
    inside = np.abs(xs - cx) < hole_r
    circ = cy - np.sqrt(np.maximum(hole_r ** 2 - (xs - cx) ** 2, 0.0))
    lo2 = np.where(inside, np.minimum(lo, circ), lo)
    bandp = np.vstack([np.c_[xs, up], np.c_[xs, lo2][::-1]])
    q.c.fill(bandp, q.lit(band))
    xbb = cx - q.sg * r * 1.60
    sel = xs < xbb if q.sun_r else xs > xbb
    if sel.sum() > 2:
        pass
    # paint drips under the band, each its own length
    rg = core.rng("cupdrips")
    for i in range(0):                      # the drips read as cracks: gone
        x = x0 + (i + rg.uniform(0.15, 0.85)) * (x1 - x0) / 11
        if abs(x - cx) < r * 1.36:
            continue
        yb = cy - 0.80 * r + 0.03 * r * ((x - cx) / r) ** 2 - 3
        ln = rg.uniform(0.06, 0.30) * r
        wd = rg.uniform(0.030, 0.052) * r
        col = q.lit(band)
        q.c.stroke(np.array([(x, yb - 4), (x + rg.normal(0, 2), yb + ln)]), wd, col, taper=(0.0, 0.25), min_w=0.5)
        q.c.dot(x, yb + ln, wd * 0.62, col)
    _motifs(q, cx, cy, r, xs, lo)
    _paint_wall(q, cel, cx, cy, r, hole_r)
    q.line(np.c_[xs, up], 0.5, amp=1.2, taper=(0, 0))
    for seg in (xs < cx - hole_r * 0.72, xs > cx + hole_r * 0.72):
        if seg.sum() > 2:
            q.line(np.c_[xs[seg], lo[seg]], 0.5, amp=1.2, taper=(0.0, 0.0))
    # hairline crazing and a chip
    for i, (ax, ay, ln) in enumerate(((2.3, 0.40, 0.30),)):
        _crazing(q, cx + ax * r, cy + ay * r, ln * r, i)
    # the glaze: one long shine on the sun side and a short one under it
    gl = np.clip(palette.mix(q.lit(cup), q.rimc(cup), 0.55), 0, 1)
    for (ax, y_a, y_b, wd) in ((1.93, 0.66, 0.84, 0.026),):
        ysg = np.linspace(cy + y_a * r, cy + y_b * r, 14)
        xsg = cx + q.sg * (ax * r + 0.10 * r * ((ysg - cy) / r) ** 2)
        poly, _, _ = _ribbon(np.c_[xsg, ysg], 0.004 * r, wd * r, n=3)
        mid = _ribbon(np.c_[xsg, ysg], wd * r, wd * r * 0.9, n=3)[0]
        q.c.fill(q.wob(mid, 1.2), gl)
        q.c.dot(xsg[0], ysg[0], wd * r * 0.5, gl)
        q.c.dot(xsg[-1], ysg[-1], wd * r * 0.45, gl)
    chx, chy = cx + r * 1.72, cy + r * 0.98
    chip = np.array([(-1, -0.2), (-0.3, -0.9), (0.6, -0.6), (1.0, 0.2), (0.3, 0.8), (-0.6, 0.6)]) * r * 0.07 + (chx, chy)
    q.fill(chip, q.lit(np.array([0.60, 0.50, 0.44], np.float32)), ink=0.5, amp=1.0)
    q.c.fill(chip[[0, 1, 2]] * 0.6 + chip.mean(0) * 0.4, q.shd(np.array([0.60, 0.50, 0.44], np.float32)))


def _brass_rim(q, cx, cy, r, inside=False):
    brass = R["copper"] * 0.55 + R["star"] * 0.45
    r1 = r * (1.16 if inside else 1.25)
    sa = math.atan2(q.T["sun"][1], q.T["sun"][0])            # angle toward the sun
    if False:
        # (the contact shadow of the rim is painted soft by _paint_wall)
        q.c.fill(_ring_poly(cx, cy, r * 1.02, r1 * 1.0, ox=-math.cos(sa) * r * 0.075, oy=-math.sin(sa) * r * 0.075 + r * 0.02),
                 q.shd(R["cup"], 0.92))
    q.c.fill(_ring_poly(cx, cy, r, r1), q.lit(brass) if not inside else q.shd(brass, 1.15))
    # shadow tone on the half of the ring that faces away, inner bevel, one highlight arc
    # (a crescent that tapers to nothing at both ends, never a pie slice)
    dsh = (r1 - r * 1.045) * 0.62
    ph = np.linspace(math.pi * 0.5, math.pi * 1.5, 60)
    rho = dsh * np.cos(ph) + np.sqrt(np.maximum(r1 ** 2 - (dsh * np.sin(ph)) ** 2, 0.0))
    cres = np.vstack([np.c_[cx + np.cos(ph + sa) * r1, cy + np.sin(ph + sa) * r1],
                      np.c_[cx + np.cos(ph[::-1] + sa) * rho[::-1], cy + np.sin(ph[::-1] + sa) * rho[::-1]]])
    q.c.fill(q.wob(cres, 1.2), q.shd(brass))
    q.c.fill(_ring_poly(cx, cy, r, r * 1.045), q.shd(brass, 0.82))
    hi = np.clip(palette.mix(q.lit(brass), q.rimc(brass), 0.45), 0, 1)
    q.c.fill(_sector(cx, cy, r * 1.008, r * 1.04, sa + math.pi * 0.74, sa + math.pi * 1.22, 40), hi)
    if not inside:
        q.c.fill(_sector(cx, cy, r * 1.192, r * 1.208, sa - 0.62, sa + 0.34, 30), hi)
        q.c.fill(_sector(cx, cy, r * 1.192, r * 1.205, sa + 0.46, sa + 0.56, 8), hi)
    a = np.linspace(0, 2 * math.pi, 120, endpoint=False)
    for rr, w in ((r1, 1.0), (r, 0.9), (r * 1.045, 0.45)):
        q.c.stroke(q.wob(np.c_[cx + np.cos(a) * rr, cy + np.sin(a) * rr], 1.0), q.lw * w, q.ink, closed=True,
                   seed=int(rr) % 97)
    # rivets (nuts on the inside), no two set quite alike
    rg = core.rng("rimrivets", inside)
    rm = (r * 1.045 + r1) / 2
    for i in range(8):
        ang = 0.38 + i * math.pi / 4 + rg.normal(0, 0.03)
        if abs(((ang - math.pi + math.pi) % (2 * math.pi)) - math.pi) < 0.33 and not inside:
            continue                                         # the hinge sits here
        px, py = cx + math.cos(ang) * rm, cy + math.sin(ang) * rm
        rv = r * rg.uniform(0.027, 0.034)
        pts = core.ellipse(px, py, rv, rv, 6 if inside else 18, rot=rg.uniform(0, 1))
        q.c.fill(pts, q.shd(brass, 1.1))
        q.c.fill(core.ellipse(px + math.cos(sa) * rv * 0.25, py + math.sin(sa) * rv * 0.25, rv * 0.62, rv * 0.62, 12),
                 q.lit(brass))
        q.c.stroke(pts, q.lw * 0.45, q.ink, closed=True)
        q.c.dot(px + math.cos(sa) * rv * 0.4, py + math.sin(sa) * rv * 0.4, rv * 0.2, q.rimc(brass))
    return r1


def _hinge(q, cx, cy, r, side=-1.0):
    brass = R["copper"] * 0.7 + R["star"] * 0.3
    x_in, x_out = cx + side * r * 1.10, cx + side * r * 1.52
    hh = r * 0.17
    plate = np.array([(x_in, cy - hh), (x_out, cy - hh * 0.86), (x_out + side * r * 0.03, cy), (x_out, cy + hh * 0.9),
                      (x_in, cy + hh)])
    q.fill(plate, q.lit(brass), ink=0.8, amp=1.0)
    q.c.fill(np.array([(x_in, cy + hh * 0.2), (x_out, cy + hh * 0.3), (x_out, cy + hh * 0.9), (x_in, cy + hh)]), q.shd(brass))
    q.line(plate[[4, 3, 2, 1, 0]], 0.8, amp=0, taper=(0.0, 0.0))
    xk = cx + side * r * 1.29
    for j, yk in enumerate((cy - hh * 0.78, cy - hh * 0.05, cy + hh * 0.68)):          # the knuckles on the pin
        kn = core.ellipse(xk, yk, r * 0.036, hh * 0.30, 18)
        q.fill(kn, q.lit(brass) if j != 2 else q.shd(brass, 1.1), ink=0.6, amp=0.4)
    q.c.dot(xk, cy - hh * 1.12, r * 0.022, q.ink)
    for sx in (1.17, 1.43):
        px = cx + side * r * sx
        q.c.dot(px, cy - hh * 0.45, r * 0.02, q.ink)
        q.c.dot(px, cy + hh * 0.5, r * 0.02, q.ink)


def _glass(q, cx, cy, r, reflect=0.0, fog=0.0, strength=1.0):
    """Two diagonal reflection bands that slide with `reflect`, and breath fog."""
    rr = r * 0.985
    a = np.linspace(0, 2 * math.pi, 220, endpoint=False)
    circ = np.c_[np.cos(a), np.sin(a)]
    axis = np.array([0.74, 0.67])
    u = circ @ axis
    sky = palette.sky_color(q.T, 0.62)
    col = np.clip(palette.mix(sky, (1.0, 1.0, 1.0), 0.45), 0, 1)
    sh = (reflect % 1.0) * 1.9 if reflect else 0.0
    for lo, hi, al in ((-0.70, -0.40, 0.17), (-0.30, -0.21, 0.11)):
        lo2, hi2 = lo + sh, hi + sh
        if lo2 > 0.97:
            lo2, hi2 = lo2 - 1.95, hi2 - 1.95
        sel = (u > lo2) & (u < hi2)
        if sel.sum() >= 3:
            pts = circ[sel]
            # order the two arcs of the band into one outline
            v = pts @ np.array([-axis[1], axis[0]])
            near = pts[np.abs(pts @ axis - lo2) < np.abs(pts @ axis - hi2)]
            farp = pts[np.abs(pts @ axis - lo2) >= np.abs(pts @ axis - hi2)]
            near = near[np.argsort(near @ np.array([-axis[1], axis[0]]))]
            farp = farp[np.argsort(-(farp @ np.array([-axis[1], axis[0]])))]
            poly = np.vstack([near, farp]) * rr + (cx, cy)
            q.c.fill(q.wob(poly, 1.5), col, alpha=al * strength)
    # the thin bright crescent where the glass meets the rim, on the sky side
    sa = math.atan2(q.T["sun"][1], q.T["sun"][0])
    q.c.fill(_sector(cx, cy, rr * 0.955, rr, sa - 2.6, sa - 1.5, 30), col, alpha=0.24 * strength)
    if fog > 0.01:
        fx_, fy_ = cx + r * 0.06, cy + r * 0.44
        q.c.fill(q.wob(core.ellipse(fx_, fy_, r * 0.46 * (0.6 + 0.4 * fog), r * 0.33 * (0.6 + 0.4 * fog), 40), 9.0, 0.006),
                 (0.93, 0.95, 0.98), alpha=0.22 * fog)
        q.c.fill(q.wob(core.ellipse(fx_ + r * 0.03, fy_ + r * 0.02, r * 0.28 * (0.6 + 0.4 * fog), r * 0.19 * (0.6 + 0.4 * fog), 30),
                       6.0, 0.006), (0.96, 0.97, 1.0), alpha=0.20 * fog)


def draw_cockpit_front(cel, tod="predawn", drawing=0, cx=960, cy=560, r=380, reflect=0.0, light=None, fog=0.0):
    """The outside of the teacup filling the frame around the window hole, the thick brass
    rim with rivets and a hinge, and the glass on top. reflect 0..1 slides the reflections
    across the glass, fog 0..1 is breath on the glass, light 0..1 warms everything (fire)."""
    q = _Close(cel, tod, drawing, r, light)
    _wall_outside(q, cel, cx, cy, r)
    _hinge(q, cx, cy, r, -1.0 if q.sun_r else 1.0)
    _brass_rim(q, cx, cy, r)
    _amber(q, cel, cx, cy, r)
    _glass(q, cx, cy, r, reflect, fog)
    return dict(window=(cx, cy, r))


def _lantern(q, lx, ly, r, wire_to):
    tin = R["tin"]
    q.line(np.array([(lx + r * 0.02, wire_to), (lx, ly - r * 0.10)]), 0.4, amp=0.6)
    q.fill(np.array([(lx - r * 0.035, ly - r * 0.10), (lx + r * 0.035, ly - r * 0.10), (lx + r * 0.07, ly - r * 0.06),
                     (lx - r * 0.07, ly - r * 0.06)]), q.shd(tin, 0.9), ink=0.5, amp=0.4)
    q.fill(np.array([(lx - r * 0.058, ly - r * 0.06), (lx + r * 0.058, ly - r * 0.06), (lx + r * 0.048, ly + r * 0.075),
                     (lx - r * 0.048, ly + r * 0.075)]), np.array([1.0, 0.86, 0.52], np.float32), ink=0.5, amp=0.4)
    q.c.fill(core.ellipse(lx, ly + r * 0.012, r * 0.022, r * 0.042, 14), np.array([1.0, 0.97, 0.82], np.float32))
    q.fill(np.array([(lx - r * 0.065, ly + r * 0.075), (lx + r * 0.065, ly + r * 0.075), (lx + r * 0.05, ly + r * 0.105),
                     (lx - r * 0.05, ly + r * 0.105)]), q.shd(tin, 0.8), ink=0.5, amp=0.4)
    for sgn in (-1, 1):
        q.line(np.array([(lx + sgn * r * 0.056, ly - r * 0.06), (lx + sgn * r * 0.047, ly + r * 0.075)]), 0.45, amp=0)


def draw_cockpit_inside(cel, tod="sunrise", drawing=0, window=(1240, 430, 250), light=0.0, reflect=0.0):
    """Camera inside the teacup (s7b): the lined inner wall fills the frame (pasted paper above,
    scrap wood below, the same cockpit as the view through the window), lit by its lantern and
    curving away into dark at the sides, with a round hole where the sky shows (left transparent),
    the window frame from inside, pinned drawings, the dial, a strap. light 0..1 is the gold of the
    sunrise flooding in round the opening. Draw the sky first, this layer, then the panda in front."""
    wx, wy, r = window
    q = _Close(cel, tod, drawing, r, 0.0)
    T_out = q.T
    flood = float(light)
    Ta = dict(T_out)
    Ta["amb"], Ta["key"], Ta["rim"] = AMBER * 0.66, AMBER * 1.05, np.array([1.0, 0.86, 0.56], np.float32)
    q.T = Ta
    paper, wood = R["paper"], np.array([0.66, 0.46, 0.28], np.float32)
    rg = core.rng("cockpit lining inside")
    x0, x1 = cel.x0 - 60.0, cel.x0 + cel.w + 60.0
    y0, y1 = cel.y0 - 60.0, cel.y0 + cel.h + 60.0
    q.c.fill(_holed_frame(cel, wx, wy, r * 1.03), q.shd(paper, 0.80))
    keep = r * 1.22
    for row, yc in enumerate((wy - 1.25 * r, wy - 0.30 * r, wy + 0.65 * r)):
        for i in range(int((x1 - x0) / (r * 0.62)) + 2):
            px = x0 + (i + rg.normal(0, 0.08)) * r * 0.62 + (row % 2) * r * 0.3
            py = yc + rg.normal(0, 0.10) * r
            w, h, a = r * rg.uniform(0.62, 0.86), r * rg.uniform(0.82, 1.08), rg.normal(0, 0.07)
            tone = rg.uniform(0.84, 1.06)
            if math.hypot(px - wx, py - wy) < keep + 0.5 * math.hypot(w, h):
                continue
            c, sn = math.cos(a), math.sin(a)
            quad = np.array([(-1, -1), (1, -0.95), (0.97, 1), (-0.6, 1.02), (-1.02, 0.7)]) * (w * 0.5, h * 0.5)
            quad = np.c_[px + quad[:, 0] * c - quad[:, 1] * sn, py + quad[:, 0] * sn + quad[:, 1] * c]
            q.fill(quad, q.shd(paper * tone, 0.92), ink=0.25, amp=1.6)
            if i % 2 == 0:
                tp = np.array([(-0.5, -1.06), (0.1, -1.1), (0.14, -0.9), (-0.46, -0.86)]) * (w * 0.5, h * 0.5)
                q.c.fill(np.c_[px + tp[:, 0] * c - tp[:, 1] * sn, py + tp[:, 0] * sn + tp[:, 1] * c], q.shd(R["tape"], 1.0))
    x = x0
    j = 0
    while x < x1:
        w = r * rg.uniform(0.26, 0.48)
        top = wy + r * (1.58 + rg.uniform(-0.10, 0.10))
        col = q.shd(wood * rg.uniform(0.78, 1.14), 0.95)
        q.fill(np.array([(x, top + rg.normal(0, 4) * q.k), (x + w, top + rg.normal(0, 4) * q.k), (x + w + 2, y1), (x - 1, y1)]),
               col, ink=0.42, amp=1.0)
        for gline in range(2):
            gx = x + w * rg.uniform(0.2, 0.8)
            q.line(np.array([(gx, top + r * 0.06), (gx + rg.normal(0, 5) * q.k, top + r * rg.uniform(0.4, 1.0))]), 0.25,
                   col * 0.72, amp=1.5)
        q.c.dot(x + w * 0.5, top + r * 0.07, r * 0.018, q.ink)
        if j % 3 == 1:
            q.c.fill(core.ellipse(x + w * 0.55, top + r * rg.uniform(0.4, 0.8), w * 0.13, w * 0.2, 12), col * 0.6)
        x += w
        j += 1
    _pinned(q, wx - r * 2.05, wy - r * 0.30, r * 0.62, r * 0.76, -0.10, "rocket", 0)
    _pinned(q, wx - r * 2.75, wy + r * 0.62, r * 0.56, r * 0.56, 0.14, "map", 1)
    lx, ly = wx - r * 1.38, wy - r * 1.02
    _lantern(q, lx, ly, r * 1.5, y0)
    dx, dy, dr_ = wx + r * 1.65, wy + r * 1.30, r * 0.26
    _dial(q, dx, dy, dr_)
    _strap(q, [(wx + r * 1.7, y0), (wx + r * 1.95, wy - r * 0.6), (wx + r * 2.3, wy + r * 0.2), (x1 + 20, wy + r * 0.55)], r * 0.17)
    # the light: the lantern, the dial and what spills in round the opening; the wall curves away
    # into brown dark at the sides and toward the floor
    xx, yy = _grid(cel)
    dist = np.hypot(xx - wx, yy - wy)
    d_l = np.hypot(xx - lx, yy - ly)
    d_d = np.hypot(xx - dx, yy - dy)
    lit_ = np.clip(0.95 * np.exp(-(d_l / (1.7 * r)) ** 2) + 0.5 * np.exp(-(d_d / (0.8 * r)) ** 2)
                   + (0.22 + 0.78 * flood) * np.exp(-((dist - 1.1 * r) / (1.5 * r)) ** 2), 0, 1)
    half = 0.5 * cel.w
    c = (xx - (cel.x0 + half)) / (1.10 * half)
    lam = np.sqrt(np.clip(1 - c * c, 0, 1))
    vb = np.clip((yy - (wy + 0.9 * r)) / (2.2 * r), 0, 1)
    grain = core.fbm(cel.h, cel.w, 3.0, 2, seed=93) * 0.5 + core.fbm(cel.h, cel.w, 12.0 * q.k, 2, seed=94, aspect=0.2) * 0.5
    val = (0.34 + 0.62 * lit_) * (0.42 + 0.58 * lam ** 1.3) * (1 - 0.35 * vb * vb) * (0.90 + 0.10 * grain)
    mult = val[..., None] * np.array([1.0, 0.96, 0.90], np.float32)
    keyc = np.clip(np.asarray(T_out["key"], np.float32), 0, 1)
    hi = (0.28 * np.exp(-(d_l / (0.5 * r)) ** 2) + 0.16 * np.exp(-(d_d / (0.4 * r)) ** 2))[..., None] * AMBER \
        + ((0.06 + 0.34 * flood) * np.exp(-((dist - 1.25 * r) / (0.42 * r)) ** 2))[..., None] * keyc
    _apply_paint(cel, mult, hi, wx, wy, r * 1.03, outside=True)
    q.T = T_out
    _hinge(q, wx, wy, r * 0.93, 1.0 if q.sun_r else -1.0)
    _brass_rim(q, wx, wy, r, inside=True)
    _glass(q, wx, wy, r, reflect, 0.0, strength=0.6)
    return dict(window=(wx, wy, r))
