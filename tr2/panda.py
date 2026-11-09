"""The red panda: design, 2.5D rig and cel drawing.

Every drawing is rebuilt from a Pose. The head and torso are ellipsoid volumes;
face features are patches on the head surface, so they slide, foreshorten and
wrap behind the silhouette as the head turns. Limbs and tail are tubes solved to
targets. Each part is rasterised on its own small supersampled image, shaded as
one form (one hard shadow tone, a thin rim inside the line), inked, then laid
on the caller's Cel.

    from tr2 import panda
    panda.draw(cel, panda.POSES["stand"], x, y, s=1.0, tod="dusk", drawing=0)

Sides: `_l` is the side that sits on screen LEFT when the panda faces us.
Model units: 1 unit = 1 px at s = 1. The standing panda is 170 units tall.
"""
import math
from dataclasses import dataclass, fields, replace

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from . import core, palette

P = palette.PANDA
MOUTH_IN = np.array([0.40, 0.13, 0.13], np.float32)
LEATHER = np.array([0.45, 0.27, 0.17], np.float32)
DARK = np.array([0.37, 0.215, 0.185], np.float32)   # a notch above the ink so lines read on it
LOCAL = dict(dark=DARK, fur=np.array([0.83, 0.50, 0.27], np.float32), fur_dk=np.array([0.58, 0.27, 0.17], np.float32))
QUILT = [np.array(c, np.float32) for c in ((0.72, 0.36, 0.30), (0.40, 0.50, 0.66), (0.88, 0.82, 0.70), (0.56, 0.62, 0.45),
                                            (0.82, 0.64, 0.40), (0.62, 0.48, 0.60), (0.80, 0.72, 0.52))]


# ------------------------------------------------------------------- the pose
@dataclass
class Pose:
    # body
    dx: float = 0.0            # body offset from the ground point (units, y up)
    dy: float = 0.0
    lean: float = 0.0          # degrees, + leans toward screen right
    squash: float = 1.0        # < 1 squashed, > 1 stretched (volume kept)
    yaw: float = 0.0           # body turn: 0 front, 0.5 three quarter right, 1 profile, 1.5 three quarter back
    breath: float = 0.0        # 0..1
    curl: float = 0.0          # 1 = asleep, curled on the ground
    # head
    head_yaw: float = 0.0      # same scale as yaw, absolute (not relative to the body)
    pitch: float = 0.0         # -1 looks down .. +1 looks up
    tilt: float = 0.0          # degrees, + tips toward screen right
    head_dx: float = 0.0
    head_dy: float = 0.0
    # face
    eye_l: float = 1.0         # 0 shut .. 1 open .. 1.6 huge
    eye_r: float = 1.0
    eye_smile: float = 0.0     # lower lid pushed up; with shut eyes gives happy arcs
    squeeze: float = 0.0       # eyes screwed shut
    pupil: tuple = (0.0, 0.0)  # -1..1, x right, y up
    pupil_size: float = 1.0
    hl: float = 1.0            # highlight size
    brow_l: float = 0.0        # -1 low .. +1 raised
    brow_r: float = 0.0
    brow_tilt: float = 0.0     # + worried, - determined
    mouth_open: float = 0.0
    mouth_wide: float = 0.0    # -1 pursed .. +1 pulled wide
    smile: float = 0.35        # -1 frown .. +1 grin
    tongue: float = 0.0
    cheek: float = 0.0         # puffed cheeks
    ear_l: float = 0.0         # 0 alert .. 1 flat, negative perks forward
    ear_r: float = 0.0
    # limbs: targets in units from the ground point (x right, y up), None = relaxed
    paw_l: tuple = None
    paw_r: tuple = None
    foot_l: tuple = None
    foot_r: tuple = None
    tail: tuple = None         # control points ((x, y), ...) from the ground point
    # dressing
    goggles: float = -1.0      # -1 none, 0 pushed up on the forehead, 1 over the eyes
    goggles_askew: float = 0.0
    blanket: float = 0.0       # the patchwork quilt over the sleeper (curl pose)
    soot: float = 0.0
    wind: float = 0.0          # fur ruffle 0..1
    wind_dir: tuple = (0.0, 1.0)   # screen direction the fur streams toward
    smear: tuple = (0.0, 0.0)


def blend(a, b, u):
    """In between two poses. Numbers ease straight; paw, foot and tail targets
    travel on a shallow arc so nothing slides in a straight line."""
    out = {}
    for f in fields(Pose):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if va is None or vb is None:
            out[f.name] = va if u < 0.5 else vb
        elif isinstance(va, tuple):
            A, B = np.asarray(va, np.float64), np.asarray(vb, np.float64)
            if A.shape != B.shape:
                out[f.name] = va if u < 0.5 else vb
                continue
            v = A + (B - A) * u
            if f.name.startswith(("paw", "foot")):
                d = B - A
                v = v + np.array([-d[1], d[0]]) * 0.16 * math.sin(math.pi * u) * (1 if f.name.endswith("l") else -1)
            out[f.name] = tuple(map(tuple, v)) if v.ndim == 2 else tuple(float(k) for k in v)
        else:
            out[f.name] = va + (vb - va) * u
    return Pose(**out)


# ------------------------------------------------------------------ geometry
def _sph(lon, lat, r=1.0):
    lo = np.radians(np.atleast_1d(np.asarray(lon, np.float64)))
    la = np.radians(np.atleast_1d(np.asarray(lat, np.float64)))
    lo, la = np.broadcast_arrays(lo, la)
    return np.stack([np.cos(la) * np.sin(lo) * r, np.sin(la) * r, np.cos(la) * np.cos(lo) * r], -1)


class _Ball:
    """An ellipsoid volume seen in orthographic view. Surface points are given
    as (lon, lat) in degrees: lon 0 lat 0 is the front, lon + is screen right
    when facing us, lat + is up."""

    def __init__(self, cx, cy, rx, ry, yaw, pitch, roll):
        self.c = np.array([cx, cy], np.float64)
        self.rx, self.ry = rx, ry
        self.cy_, self.sy_ = math.cos(yaw), math.sin(yaw)
        self.cp, self.sp = math.cos(pitch), math.sin(pitch)
        self.cr, self.sr = math.cos(roll), math.sin(roll)

    def rot(self, v):
        v = np.asarray(v, np.float64).reshape(-1, 3)
        x = v[:, 0] * self.cy_ + v[:, 2] * self.sy_
        z = -v[:, 0] * self.sy_ + v[:, 2] * self.cy_
        y = v[:, 1] * self.cp + z * self.sp
        z2 = -v[:, 1] * self.sp + z * self.cp
        return x, y, z2

    def loc(self, x, y):
        """Local offsets in px (y down, before roll) to screen points."""
        x = np.atleast_1d(np.asarray(x, np.float64))
        y = np.atleast_1d(np.asarray(y, np.float64))
        return np.c_[self.c[0] + x * self.cr - y * self.sr, self.c[1] + x * self.sr + y * self.cr]

    def proj(self, v, clamp=False):
        x, y, z = self.rot(v)
        if clamp:
            n = np.hypot(x, y)
            k = np.where((z < 0) & (n > 1e-6), 1.0 / np.maximum(n, 1e-6), 1.0)
            x, y = x * k, y * k
        return self.loc(x * self.rx, -y * self.ry), z

    def patch(self, lon, lat, r=1.0):
        return self.proj(_sph(lon, lat, r), clamp=(r == 1.0))

    def pt(self, lon, lat, r=1.0):
        p, z = self.proj(_sph(lon, lat, r))
        return p[0], float(z[0])

    def frame(self, lon, lat, r=1.0, minx=0.5):
        """A little drawing board glued to the surface: origin, x axis, y axis
        (per unit, foreshortened with the turn) and depth."""
        o, z = self.pt(lon, lat, r)
        d = 3.0
        a, _ = self.pt(lon + d, lat, r)
        b, _ = self.pt(lon - d, lat, r)
        ex = (a - b) / (2 * math.radians(d) * self.rx * r * max(math.cos(math.radians(lat)), 0.2))
        u, _ = self.pt(lon, lat - d, r)
        w, _ = self.pt(lon, lat + d, r)
        ey = (u - w) / (2 * math.radians(d) * self.ry * r)
        n = math.hypot(*ex)
        if n < minx:
            base = np.array([self.cr, self.sr])
            ex = base * minx if n < 1e-3 else ex / n * minx
        m = math.hypot(*ey)
        if m < 0.6:
            ey = np.array([-self.sr, self.cr]) * 0.6
        return o, ex, ey, z


def _put(fr, pts, s):
    o, ex, ey = fr[0], fr[1], fr[2]
    p = np.asarray(pts, np.float64)
    return o + s * (p[:, :1] * ex + p[:, 1:2] * ey)


def _ell(clon, clat, w, h, n=30, rot=0.0, pw=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = w * np.sign(ct) * np.abs(ct) ** pw
    y = h * np.sign(st) * np.abs(st) ** pw
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    return clon + x * c - y * s, clat + x * s + y * c


def _arc3(a, m, b, k=7):
    t = np.linspace(0, 1, k)[:, None]
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * m + t * t * b


def _tube(ctrl, hw, n=40):
    """A limb: outline polygon of a tube of varying half width along a curve."""
    c = core.resample(core.catmull(ctrl, 10, closed=False), n)
    w = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(hw)), hw)
    t = np.gradient(c, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nr = np.c_[-t[:, 1], t[:, 0]]
    L, R = c + nr * w[:, None], c - nr * w[:, None]
    a = np.linspace(0, np.pi, 11)[1:-1]
    end = c[-1] + w[-1] * (np.cos(a)[:, None] * nr[-1] + np.sin(a)[:, None] * t[-1])
    start = c[0] + w[0] * (-np.cos(a)[:, None] * nr[0] - np.sin(a)[:, None] * t[0])
    return np.vstack([L, end, R[::-1], start]), c, L, R, t, w


def _runs(keep):
    """Index runs where keep is True on a closed loop."""
    n = len(keep)
    if keep.all():
        return [np.arange(n + 1) % n]
    if not keep.any():
        return []
    start = int(np.argmin(keep))
    idx = (np.arange(n) + start) % n
    out, cur = [], []
    for i in idx:
        if keep[i]:
            cur.append(i)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return [r for r in out if len(r) >= 3]


# -------------------------------------------------------------- light and ink
class _Env:
    def __init__(self, tod, s, drawing, light):
        self.t = palette.TOD[tod] if isinstance(tod, str) else tod
        self.s = s
        self.sun = np.asarray(self.t["sun"], np.float64)
        self.ink = np.asarray(self.t["ink"], np.float32)
        self.inkw = 1.5 * s ** 0.55
        self.rimw = 1.0 + 1.5 * s ** 0.8
        self.boil = 0.3 * s ** 0.45
        self.seed = core.seed_of("boil", int(drawing)) % 90000
        self.bfreq = 0.05 / s ** 0.5
        self.key = np.asarray(self.t["key"], np.float32)
        ratio = np.clip(self.t["amb"] / self.key * (1.12, 1.0, 1.0), 0, 1)
        self.sh8 = tuple(int(v * 255) for v in ratio)
        # the shadow tone is pushed toward violet by a small add of the ambient
        self.sh_add8 = tuple(int(v * 255) for v in np.clip(self.t["amb"] * 0.07 * np.array([0.9, 0.5, 1.3]), 0, 1))
        self.rim8 = tuple(int(v * 255) for v in np.clip(self.t["rim"], 0, 1))
        self.light = light
        self.pad = int(8 * s) + 6

    def c(self, name):
        if name == "white":
            # the mask stays a warm white in any light, never custard
            return np.clip(P["white"] * (0.38 * self.key + 0.62), 0, 1)
        return self.lit(LOCAL.get(name, P.get(name)))

    def lit(self, col):
        """Key light on a local colour, soft clamped and a little desaturated
        (the palette keys go above 1 and turn the fur neon otherwise)."""
        c = np.clip(np.asarray(col, np.float32) * self.key, 0, 1)
        g = float(c @ np.array([0.25, 0.6, 0.15], np.float32))
        return np.clip(g + (c - g) * 0.86, 0, 1)


class _Part:
    """One body part on its own small supersampled image."""

    def __init__(self, cel, box, env):
        ss = cel.ss
        pad = env.pad
        x0 = max(math.floor(box[0]), cel.x0 - pad)
        y0 = max(math.floor(box[1]), cel.y0 - pad)
        x1 = min(math.ceil(box[2]), cel.x0 + cel.w + pad)
        y1 = min(math.ceil(box[3]), cel.y0 + cel.h + pad)
        self.ok = x1 > x0 + 1 and y1 > y0 + 1
        self.cel, self.env, self.ss, self.x0, self.y0 = cel, env, ss, x0, y0
        self.inks = []
        if not self.ok:
            return
        self.size = ((x1 - x0) * ss, (y1 - y0) * ss)
        self.im = Image.new("RGBA", self.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        self.form = Image.new("L", self.size, 0)
        self.fd = ImageDraw.Draw(self.form)
        self.cast = Image.new("L", self.size, 0)
        self.cd = ImageDraw.Draw(self.cast)

    def _xy(self, pts):
        p = np.asarray(pts, np.float64)
        e = self.env
        if e.boil > 0:
            p = core.wobble(p, e.boil, e.seed, e.bfreq)
        return ((p - (self.x0, self.y0)) * self.ss).ravel().tolist()

    def mask(self, pts, clip=None):
        m = Image.new("L", self.size, 0)
        if len(pts) >= 3:
            ImageDraw.Draw(m).polygon(self._xy(pts), fill=255)
        if clip is not None:
            m = ImageChops.multiply(m, clip)
        return m

    def fill(self, pts, col, form=True, clip=None, alpha=1.0):
        if not self.ok or len(pts) < 3:
            return
        xy = self._xy(pts)
        c = core._rgba8(col)
        if clip is None and alpha >= 1.0:
            self.d.polygon(xy, fill=c)
        else:
            m = Image.new("L", self.size, 0)
            ImageDraw.Draw(m).polygon(xy, fill=255)
            if clip is not None:
                m = ImageChops.multiply(m, clip)
            if alpha < 1.0:
                m = m.point(lambda v: int(v * alpha))
            self.im.paste(c, (0, 0), m)
        if form:
            self.fd.polygon(xy, fill=255)

    def shadow(self, pts):
        if self.ok and len(pts) >= 3:
            self.cd.polygon(self._xy(pts), fill=255)

    def ink(self, pts, w, col=None, taper=(0.16, 0.2), pressure=None, closed=False, clip=None, lightdir=True):
        self.inks.append(lambda: self._stroke(pts, w, col, taper, pressure, closed, clip, lightdir))

    def later(self, pts, col, clip=None):
        self.inks.append(lambda: self.fill(pts, col, form=False, clip=clip))

    def _stroke(self, pts, w, col, taper, pressure, closed, clip, lightdir):
        p = np.asarray(pts, np.float64)
        if len(p) < 2 or not self.ok:
            return
        if closed:
            p = np.vstack([p, p[:1]])
        ln = np.hypot(*np.diff(p, axis=0).T).sum()
        n = max(8, int(ln / 1.6))
        p = core.resample(p, n)
        sv = np.linspace(0, 1, n)
        wv = np.ones(n)
        if not closed:
            a, b = taper
            if a > 0:
                wv *= np.clip(sv / a, 0, 1) ** 0.6
            if b > 0:
                wv *= np.clip((1 - sv) / b, 0, 1) ** 0.6
        if pressure is not None:
            pr = np.asarray(pressure, np.float64)
            wv *= np.interp(sv, np.linspace(0, 1, len(pr)), pr)
        t = np.gradient(p, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        if lightdir:
            # heavier line on the side away from the light
            out = p - p.mean(0)
            out /= np.maximum(np.hypot(out[:, 0], out[:, 1]), 1e-9)[:, None]
            wv *= 0.78 + 0.5 * np.clip(-(out @ self.env.sun), 0, 1)
        wv *= 0.86 + 0.3 * core.fbm_at(sv * 7.0 + 3.1, np.zeros(n) + 0.5, 1.0, 2, self.env.seed + int(ln) % 97)
        wv = np.maximum(wv, 0.14) * w * 0.5
        poly = np.vstack([p + nrm * wv[:, None], (p - nrm * wv[:, None])[::-1]])
        self.fill(poly, self.env.ink if col is None else col, form=False, clip=clip)

    def _shift(self, m, dx, dy):
        o = Image.new("L", self.size, 0)
        o.paste(m, (int(round(dx * self.ss)), int(round(dy * self.ss))))
        return o

    def finish(self, d_sh, d_rim=None, shade=True):
        if not self.ok:
            return
        e = self.env
        d_rim = e.rimw if d_rim is None else d_rim
        if shade:
            rgb = self.im.convert("RGB")
            al = self.im.getchannel("A")
            sx, sy = e.sun
            sh = ImageChops.subtract(self.form, self._shift(self.form, sx * d_sh, sy * d_sh))
            sh = ImageChops.lighter(sh, ImageChops.multiply(self.cast, self.form))
            dark = ImageChops.multiply(rgb, Image.new("RGB", self.size, e.sh8))
            dark = ImageChops.add(dark, Image.new("RGB", self.size, e.sh_add8))
            rgb = Image.composite(dark, rgb, sh)
            if e.light and e.light.get("amount", 0) > 0.01:
                ld = np.asarray(e.light["dir"], np.float64)
                ld = ld / max(np.hypot(*ld), 1e-6)
                am = e.light["amount"]
                m2 = ImageChops.subtract(self.form, self._shift(self.form, -ld[0] * d_sh * 1.7, -ld[1] * d_sh * 1.7))
                lc = tuple(int(core.clamp(v * am * 0.75) * 255) for v in e.light["color"])
                rgb = Image.composite(ImageChops.screen(rgb, Image.new("RGB", self.size, lc)), rgb, m2)
            rm = ImageChops.subtract(self.form, self._shift(self.form, -sx * d_rim, -sy * d_rim))
            rgb = Image.composite(Image.blend(rgb, Image.new("RGB", self.size, e.rim8), 0.62), rgb, rm)
            self.im = Image.merge("RGBA", (*rgb.split(), al))
            self.d = ImageDraw.Draw(self.im)
        for fn in self.inks:
            fn()
        cel, ss = self.cel, self.ss
        dx, dy = (self.x0 - cel.x0) * ss, (self.y0 - cel.y0) * ss
        cx0, cy0 = max(-dx, 0), max(-dy, 0)
        cx1 = min(self.size[0], cel.im.size[0] - dx)
        cy1 = min(self.size[1], cel.im.size[1] - dy)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        cel.im.alpha_composite(self.im.crop((cx0, cy0, cx1, cy1)), dest=(dx + cx0, dy + cy0))


def _bounds(pts, m):
    p = np.asarray(pts)
    return (p[:, 0].min() - m, p[:, 1].min() - m, p[:, 0].max() + m, p[:, 1].max() + m)


# ----------------------------------------------------------------------- ears
def _ear(ball, side, flat, s):
    A, za = ball.pt(side * 50, 67, 0.96)
    B, zb = ball.pt(side * 108, 25, 0.96)
    f = max(flat, -0.6)
    T, zt = ball.pt(side * (79 + 16 * f), 58 - 36 * f, 1.90 - 0.45 * max(f, 0))
    ln = math.hypot(*(T - (A + B) / 2))

    def bulge(a, b, other, k):
        d = b - a
        n = np.array([-d[1], d[0]])
        n /= max(math.hypot(*n), 1e-6)
        if n @ (other - a) > 0:
            n = -n
        return a + d * 0.52 + n * k * ln

    curve = core.catmull([A, bulge(A, T, B, 0.17), T, bulge(B, T, A, 0.22), B], 10, closed=False)
    base = (A + B) / 2 + (ball.c - (A + B) / 2) * 0.35
    G = (A + B + T) / 3
    _, _, nz = ball.rot(_sph(side * 38, 18))
    return dict(curve=curve, poly=np.vstack([curve, base]), G=G, base=base, z=(za + zb) / 2,
                nz=float(nz[0]), ln=ln, side=side, T=T, A=A, B=B)


def _ear_paint(p, e, env):
    s = env.s
    p.fill(e["poly"], env.c("fur"))
    if e["nz"] > 0.12:
        k = min(1.0, 0.45 + e["nz"])
        G, base = e["G"], e["base"]
        inner = G + (e["curve"] - G) * 0.80 + (base - G) * 0.05
        em = p.mask(e["poly"])
        p.fill(inner, env.c("white"), form=False, clip=em)
        # the hollow of the ear, its top edge broken by three tufts of white fur
        o_ = (e["A"] + e["B"]) / 2
        u_ = (e["B"] - e["A"]) / 2
        v_ = e["T"] - o_
        fl = [(-0.66, -0.1), (-0.50, 0.30), (-0.30, 0.15), (-0.08, 0.50), (0.12, 0.19), (0.32, 0.37), (0.62, -0.1)]
        hollow = core.catmull(np.array([o_ + u_ * a_ + v_ * b_ * k for a_, b_ in fl]), 5, closed=True, tension=0.42)
        p.fill(hollow, palette.mix(env.c("ear_in"), env.c("fur"), 0.25), form=False, clip=em)
    else:
        p.fill(e["poly"], palette.mix(env.c("fur"), env.c("fur_dk"), 0.55))
    p.ink(e["curve"], env.inkw * 1.05, taper=(0.10, 0.10))


# ----------------------------------------------------------------------- head
def _draw_head(cel, env, pose, hc, s, roll=0.0):
    yaw = pose.head_yaw * math.pi / 2
    RX = (38.0 - 3.5 * abs(math.sin(yaw))) * s
    RY = 31.0 * s
    ball = _Ball(hc[0], hc[1], RX, RY, yaw, math.radians(pose.pitch * 30), math.radians(pose.tilt) + roll)
    w = env.inkw
    box = (hc[0] - RX * 1.85, hc[1] - RY * 2.15, hc[0] + RX * 1.85, hc[1] + RY * 1.6)
    ears = [_ear(ball, -1, pose.ear_l, s), _ear(ball, 1, pose.ear_r, s)]
    far = [e for e in ears if e["z"] < -0.3]
    near = [e for e in ears if e["z"] >= -0.3]
    if far:
        pf = _Part(cel, box, env)
        for e in far:
            _ear_paint(pf, e, env)
        pf.finish(4 * s)
    p = _Part(cel, box, env)
    anchors = {}

    # ---- silhouette: a soft box with heavy cheeks and a few fur tufts
    a = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    ck = pose.cheek
    n = np.where(sa > 0, 2.75 + 0.5 * ck, 2.12)
    lx = RX * np.sign(ca) * np.abs(ca) ** (2 / n) * (1 + np.where(sa > 0, (0.07 + 0.13 * ck) * sa, 0.0))
    ly = RY * np.sign(sa) * np.abs(sa) ** (2 / n)
    wd = np.asarray(pose.wind_dir, np.float64) * pose.wind * 0.55
    deg = np.degrees(a)

    def tuft(ac, hw, L, lean):
        nonlocal lx, ly
        d = np.abs((deg - ac + 180) % 360 - 180)
        t = np.clip(1 - d / hw, 0, 1) ** 1.35
        lx = lx + s * L * t * (math.cos(math.radians(ac)) + lean[0])
        ly = ly + s * L * t * (math.sin(math.radians(ac)) + lean[1])

    k = 1 + 0.5 * pose.wind
    # the cheek ruff: four bold tufts a side, no two alike, the low ones overlap the neck
    for ac, hw_, L_, ly_ in ((5, 15, 7.0, 0.3), (27, 17, 11.5, 0.5), (50, 11, 6.5, 0.6),
                             (177, 13, 6.0, 0.25), (157, 18, 12.5, 0.45), (133, 12, 7.5, 0.6), (113, 8, 4.0, 0.7)):
        sd = 1 if ac < 90 else -1
        tuft(ac, hw_, L_ * k, (0.15 * sd + wd[0] * 1.2, ly_ + wd[1] * 1.2))
    cr = 270 + pose.head_yaw * 22
    tuft(cr - 7, 9.0, 4.8, (0.35 + wd[0], -0.1 + wd[1] * 0.5))
    tuft(cr + 6, 7.0, 3.2, (0.4 + wd[0], 0.0))
    outline = ball.loc(lx, ly)
    fur, white = env.c("fur"), env.c("white")
    p.fill(outline, fur)
    hm = p.mask(outline)
    for e in near:
        _ear_paint(p, e, env)

    # ---- white mask: muzzle, bridge and cheeks
    half = [(0, 19), (6, 17), (10, 7), (13, -7), (21, -16), (33, -18), (46, -14), (58, -5), (70, 3), (84, 2),
            (96, -10), (102, -40), (80, -72), (42, -86)]
    tears = []
    pts = half + [(0, -89)] + [(-lo, la) for lo, la in half[::-1]]
    ll = core.catmull(pts, 6, closed=True)
    mp, mz = ball.patch(ll[:, 0], ll[:, 1])
    p.fill(mp, white, form=False, clip=hm)
    # tear marks from the eye down the cheek
    tear = palette.mix(env.c("fur"), env.c("fur_dk"), 0.9)
    for sd in (-1, 1):
        cl = core.catmull([(sd * 17, -7), (sd * 25, -19), (sd * 29, -31), (sd * 25, -42), (sd * 17, -49)], 6, closed=False)
        hwd = np.interp(np.linspace(0, 1, len(cl)), [0, 0.35, 0.75, 1], [6.5, 5.6, 3.6, 0.8])
        t = np.gradient(cl, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        nr = np.c_[-t[:, 1], t[:, 0]]
        rb = np.vstack([cl + nr * hwd[:, None], (cl - nr * hwd[:, None])[::-1]])
        tp, tz = ball.patch(rb[:, 0], rb[:, 1])
        if tz.mean() > -0.25:
            tears.append(tp)

    # ---- muzzle: a small white bump that leaves the silhouette in profile
    mf = ball.frame(0, -25, 1.10, minx=0.6)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    mloc = np.c_[14.5 * np.cos(t), 10.5 * np.sin(t)]
    muz = _put(mf, mloc, s)
    show_face = mf[3] > -0.28
    if show_face:
        p.fill(muz, white)
    for tp in tears:
        p.fill(tp, tear, form=False, clip=hm)

    # ---- brows: the white dots do the acting
    for sd, br in ((-1, pose.brow_l), (1, pose.brow_r)):
        lo, la = _ell(sd * 28, 32 + 7 * br, 9.5, 5.2, 24, rot=-sd * (8 + 22 * pose.brow_tilt))
        bp, bz = ball.patch(lo, la)
        if bz.mean() > -0.1:
            p.fill(bp, white, form=False, clip=hm)

    # ---- eyes
    eyec = env.c("eye")
    lside = -1.0 if env.sun[0] < 0 else 1.0
    px, py = pose.pupil
    _eyes(p, ball, env, pose, hm, s, anchors)
    for sd, op in ():
        elon, elat = sd * 29.0, 1.0
        ew, eh = 12.8, 16.8
        if op > 1:
            eh *= 1 + 0.2 * (op - 1)
            ew *= 1 + 0.12 * (op - 1)
        o = min(op, 1.0)
        _, cz = ball.pt(elon, elat)
        if cz < 0.2:
            continue
        if o < 0.1 or pose.squeeze > 0.5:
            u = np.linspace(-1, 1, 22)
            up = pose.eye_smile > 0.3 or pose.squeeze > 0.5
            bow = 1.0 if up else -1.0
            lo = elon + ew * 1.1 * u
            la = elat - (0.15 if up else 0.35) * eh + bow * 0.46 * eh * (1 - u * u) - sd * 0.10 * ew * u
            cp_, _ = ball.patch(lo, la)
            p.ink(cp_, w * 2.7, taper=(0.2, 0.2), lightdir=False)
            if pose.squeeze > 0.5:
                c0 = np.array([elon + sd * ew * 1.02, la[-1] if sd > 0 else la[0]])
                for dlo, dla in ((7, 7), (8.5, -1), (6.5, -8)):
                    wp, _ = ball.patch([c0[0] + sd * 1.5, c0[0] + sd * dlo], [c0[1] + dla * 0.2, c0[1] + dla])
                    p.ink(wp, w * 1.1, taper=(0.1, 0.5), lightdir=False)
            continue
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ct, st = np.cos(t), np.sin(t)
        lo = elon + ew * np.sign(ct) * np.abs(ct) ** 0.86
        la = elat + eh * np.sign(st) * np.abs(st) ** 0.86
        un = (lo - elon) / ew
        la = np.minimum(la, elat - eh + 2 * eh * o + 0.10 * eh * (1 - o) * (1 - un * un))
        la = np.maximum(la, elat - eh + pose.eye_smile * eh * (0.25 + 0.75 * (1 - un * un)))
        la = la - sd * 0.10 * (lo - elon)
        ep, ez = ball.patch(lo, la)
        em = p.mask(ep, hm)
        p.fill(ep, eyec, form=False, clip=hm)
        ilo, ila = _ell(elon + px * 3.6, elat - 0.24 * eh + py * 4.0, 0.84 * ew, 0.70 * eh, 26)
        ip, _ = ball.patch(ilo, ila)
        p.fill(ip, env.c("iris"), form=False, clip=em)
        glo, gla = _ell(elon + px * 3.6, elat - 0.66 * eh + py * 4.0, 0.60 * ew, 0.32 * eh, 20)
        gp, _ = ball.patch(glo, gla)
        p.fill(gp, env.lit(palette.mix(P["iris"], (1.0, 0.72, 0.30), 0.55)), form=False, clip=em)
        plo, pla = _ell(elon + px * 3.6, elat + 0.02 * eh + py * 4.0, 0.46 * ew * pose.pupil_size, 0.50 * eh * pose.pupil_size, 22)
        pp, _ = ball.patch(plo, pla)
        p.fill(pp, eyec, form=False, clip=em)
        # highlights: a big one toward the light, a small one opposite
        wpx = np.ptp(ep[:, 0])
        hpx = np.ptp(ep[:, 1])
        h1, _ = ball.pt(elon + lside * 0.34 * ew + px * 1.5, elat + (0.36 * min(1.0, o * 1.2) - 0.9 * (1 - o)) * eh)
        r1 = 0.185 * wpx * pose.hl
        p.later(core.ellipse(h1[0], h1[1], r1, r1 * 1.12, 22, rot=0.3), (1, 1, 1), clip=em)
        h2, _ = ball.pt(elon - lside * 0.36 * ew + px * 1.5, elat - 0.42 * eh)
        r2 = 0.085 * wpx * pose.hl
        p.later(core.ellipse(h2[0], h2[1], r2, r2, 14), (1, 0.97, 0.9), clip=em)
        # the upper lid: a heavy line with a small flick at the outer corner
        idx = np.where(st > 0.12)[0]
        lid = ep[idx] if sd < 0 else ep[idx][::-1]      # runs from the outer corner inward
        flick = lid[0] + (lid[0] - lid[3]) * 0.9 + np.array([0, -0.06 * hpx])
        p.ink(np.vstack([flick, lid]), w * 2.3, taper=(0.3, 0.18), lightdir=False)
        low = np.where(st < -0.45)[0]
        p.ink(ep[low], w * 0.9, taper=(0.3, 0.3), lightdir=False)
        anchors["eye_l" if sd < 0 else "eye_r"] = ep.mean(0)

    # ---- nose and mouth
    if show_face:
        nf = ball.frame(0, -13, 1.31, minx=0.62)
        nose = core.catmull([(-4.0, -2.3), (0, -2.9), (4.0, -2.3), (4.3, -0.3), (1.7, 2.6), (0, 3.0), (-1.7, 2.6), (-4.3, -0.3)], 5)
        npoly = _put(nf, nose, s)
        p.fill(npoly, env.c("nose"), form=False)
        nh = _put(nf, np.array([[-1.2 * 1 + lside * 1.0, -1.5]]), s)[0]
        p.later(core.ellipse(nh[0], nh[1], 1.25 * s, 0.7 * s, 12), env.lit(palette.mix(P["nose"], (1, 1, 1), 0.6)))
        kf = ball.frame(0, -32, 1.17, minx=0.55)
        op, wd_, sm = pose.mouth_open, pose.mouth_wide, pose.smile
        cw = 6.4 + 2.6 * wd_
        cyv = 0.8 - 2.4 * sm
        nb = _put(nf, np.array([[0, 3.0]]), s)[0]
        ctr = _put(kf, np.array([[0, 0.0]]), s)[0]
        p.ink(np.array([nb, ctr]), w * 0.9, taper=(0.1, 0.3), lightdir=False)
        left = core.catmull([(0, 0), (-0.42 * cw, 2.3), (-0.8 * cw, 1.6 + cyv * 0.5), (-cw, cyv)], 6, closed=False)
        right = left * (-1, 1)
        if op > 0.06:
            dp = 2.5 + 9.0 * op
            lowc = core.catmull([(-cw * 0.92, cyv + 0.6), (-cw * 0.62, dp * 0.72), (0, dp), (cw * 0.62, dp * 0.72), (cw * 0.92, cyv + 0.6)], 6, closed=False)
            shape = np.vstack([left[::-1], right[1:], lowc[::-1]])
            mpoly = _put(kf, shape, s)
            mm = p.mask(mpoly)
            p.fill(mpoly, env.lit(MOUTH_IN), form=False)
            if op > 0.3:
                tg = _put(kf, core.ellipse(0, dp * 0.98, cw * 0.62, dp * 0.42, 20), s)
                p.fill(tg, env.c("tongue"), form=False, clip=mm)
            # one small fang
            fg = _put(kf, np.array([[-0.62 * cw, 1.7], [-0.36 * cw, 2.2], [-0.47 * cw, 4.6]]), s)
            p.fill(fg, white, form=False, clip=mm)
            p.ink(_put(kf, lowc, s), w * 0.95, taper=(0.12, 0.12), lightdir=False)
        if pose.tongue > 0.05:
            tq = _put(kf, core.ellipse(1.6, 2.2 + 2.6 * pose.tongue, 2.6, 2.4 + 1.6 * pose.tongue, 18), s)
            p.fill(tq, env.c("tongue"), form=False)
            p.ink(tq, w * 0.7, closed=True, lightdir=False)
        if pose.head_yaw < 0.62:
            p.ink(_put(kf, right, s), w * 1.05, taper=(0.05, 0.3), lightdir=False)
        if pose.head_yaw > -0.62:
            p.ink(_put(kf, left, s), w * 1.05, taper=(0.05, 0.3), lightdir=False)
        anchors["mouth"] = ctr
        anchors["mouth_l"] = _put(kf, np.array([[-cw * 0.7, 1.6]]), s)[0]
        anchors["mouth_r"] = _put(kf, np.array([[cw * 0.7, 1.6]]), s)[0]
        # whiskers
        for sd in (-1, 1):
            wf = ball.frame(sd * 22, -28, 1.04, minx=0.35)
            if wf[3] < 0.05:
                continue
            for j, (ang, ln_) in enumerate(((-12, 17), (4, 20), (20, 16))):
                r_ = math.radians(ang)
                q = np.array([[sd * 2.0, j * 1.6 - 1.6],
                              [sd * (2 + ln_ * 0.55 * math.cos(r_)), j * 1.6 - 1.6 + ln_ * 0.55 * math.sin(r_) - 0.8],
                              [sd * (2 + ln_ * math.cos(r_)), j * 1.6 - 1.6 + ln_ * math.sin(r_) + 0.6]])
                p.ink(core.catmull(_put(wf, q, s), 6, closed=False), w * 0.5, taper=(0.02, 0.7), lightdir=False)

    # ---- goggles
    if pose.goggles > -0.5:
        _goggles(p, ball, env, pose, hm, s)

    # ---- soot smudges
    if pose.soot > 0.05:
        soot = env.lit((0.20, 0.17, 0.17))
        for (lo0, la0, ww, hh, rt) in ((-40, -22, 15, 9, 20), (30, 30, 13, 8, -15), (8, -4, 8, 6, 0), (48, -40, 10, 7, 30)):
            lo, la = _ell(lo0, la0, ww, hh, 18, rot=rt)
            nz = core.fbm_at(lo * 0.2, la * 0.2, 1.0, 2, 77) - 0.5
            sp, sz = ball.patch(lo + nz * 6, la + nz * 5)
            if sz.mean() > 0:
                p.fill(sp, soot, form=False, clip=hm, alpha=0.6 * pose.soot)

    # ---- ink: the outline, broken where the muzzle crosses it
    if show_face:
        o, ex, ey = mf[0], mf[1], mf[2]
        M = np.c_[ex, ey] * s
        try:
            uv = np.linalg.solve(M, (outline - o).T).T
            inside = (uv[:, 0] / 14.5) ** 2 + (uv[:, 1] / 10.5) ** 2 < 0.9
        except np.linalg.LinAlgError:
            inside = np.zeros(len(outline), bool)
        for r in _runs(~inside):
            closed = len(r) == len(outline) + 1
            p.ink(outline[r], w * 1.2, taper=(0.0, 0.0) if closed else (0.04, 0.04))
        loc = (muz - ball.c) @ np.array([[ball.cr, -ball.sr], [ball.sr, ball.cr]])
        outside = (loc[:, 0] / RX) ** 2 + (loc[:, 1] / RY) ** 2 > 0.97
        for r in _runs(outside):
            p.ink(muz[r], w * 1.1, taper=(0.06, 0.06))
    else:
        p.ink(outline, w * 1.2, closed=True, taper=(0, 0))
    p.finish(8.5 * s)
    anchors["head"] = ball.c.copy()
    anchors["ball"] = ball
    anchors["outline"] = outline
    return anchors


def _eyes(p, ball, env, pose, hm, s, anchors):
    """Both eyes use one construction on a board glued to the head, so the far
    eye in a three quarter view is the same eye, only narrower."""
    w = env.inkw
    eyec = env.c("eye")
    iris = env.c("iris")
    lside = -1.0 if env.sun[0] < 0 else 1.0
    px, py = pose.pupil
    for sd, op in ((-1, pose.eye_l), (1, pose.eye_r)):
        fr = ball.frame(sd * 29.0, 1.0, 1.0, minx=0.64)
        if fr[3] < 0.18:
            continue
        o, ex, ey = fr[0], fr[1], fr[2]
        ew, eh = 8.7, 9.7
        if op > 1:
            eh *= 1 + 0.2 * (op - 1)
            ew *= 1 + 0.12 * (op - 1)
        opn = min(op, 1.0)
        # keep the whole eye on the face: slide it in from the silhouette
        ax = np.array([ball.cr, ball.sr])
        lxp = float((o - ball.c) @ ax)
        lim = ball.rx * 0.90 - ew * s * math.hypot(*ex)
        if abs(lxp) > lim:
            o = o - ax * (abs(lxp) - lim) * (1 if lxp > 0 else -1)
        frm = (o, ex, ey)

        def put(q, frm=frm):
            return _put(frm, np.asarray(q, np.float64).reshape(-1, 2), s)

        if opn < 0.1 or pose.squeeze > 0.5:
            u = np.linspace(-1, 1, 24)
            up = pose.eye_smile > 0.3 or pose.squeeze > 0.5
            xs = ew * 1.12 * u
            ys = (0.1 if up else 0.3) * eh - (1 if up else -1) * 0.5 * eh * (1 - u * u) + sd * 0.10 * xs
            line = np.c_[xs, ys]
            if sd > 0:
                line = line[::-1]
            p.ink(put(line), w * 2.8, taper=(0.18, 0.25), lightdir=False)
            for f_ in (0.05, 0.2):
                q0 = line[int(f_ * 23)]
                p.ink(put([q0, q0 + np.array([sd * 0.24 * ew, (0.2 if up else 0.3) * eh])]), w * 1.3,
                      taper=(0.05, 0.6), lightdir=False)
            if pose.squeeze > 0.5:
                c0 = line[0]
                for dx_, dy_ in ((0.5, -0.55), (0.66, -0.05)):
                    p.ink(put([c0 + np.array([sd * 0.1 * ew, 0]), c0 + np.array([sd * dx_ * ew, dy_ * eh])]),
                          w * 1.1, taper=(0.1, 0.5), lightdir=False)
            continue
        t = np.linspace(0, 2 * np.pi, 56, endpoint=False)
        ct, st = np.cos(t), np.sin(t)
        xs = ew * np.sign(ct) * np.abs(ct) ** 0.88
        ys = eh * np.sign(st) * np.abs(st) ** 0.88
        un = xs / ew
        top = -eh + 2 * eh * (1 - opn) - 0.12 * eh * (1 - opn) * (1 - un * un)
        ys = np.maximum(ys, top)
        ys = np.minimum(ys, eh - pose.eye_smile * eh * (0.25 + 0.75 * (1 - un * un)))
        ys = ys + sd * 0.10 * xs
        shape = np.c_[xs, ys]
        ep = put(shape)
        em = p.mask(ep, hm)
        p.fill(ep, eyec, form=False, clip=hm)
        cx_, cy_ = px * 2.4, 0.06 * eh - py * 2.4
        p.fill(put(core.ellipse(cx_, cy_, 0.86 * ew, 0.84 * eh, 28)), palette.mix(iris, eyec, 0.5), form=False, clip=em)
        p.fill(put(core.ellipse(cx_, cy_ + 0.40 * eh, 0.74 * ew, 0.50 * eh, 26)), iris, form=False, clip=em)
        p.fill(put(core.ellipse(cx_, cy_ + 0.68 * eh, 0.50 * ew, 0.21 * eh, 20)),
               env.lit(palette.mix(P["iris"], (1.0, 0.74, 0.32), 0.6)), form=False, clip=em)
        ps = pose.pupil_size
        p.fill(put(core.ellipse(cx_, cy_ - 0.04 * eh, 0.40 * ew * ps, 0.46 * eh * ps, 22)), eyec, form=False, clip=em)
        hy = max(-0.42 * eh, -eh + 2 * eh * (1 - opn) + 0.3 * eh)
        r1 = 0.27 * ew * pose.hl
        p.later(put(core.ellipse(lside * 0.36 * ew + px, hy, r1, r1 * 1.1, 22)), (1, 1, 1), clip=em)
        r2 = 0.12 * ew * pose.hl
        p.later(put(core.ellipse(-lside * 0.38 * ew + px, 0.46 * eh, r2, r2, 14)), (1, 0.97, 0.9), clip=em)
        lid = shape[np.where(st < -0.05)[0]]          # the top of the eye, x from -ew to +ew
        if sd > 0:
            lid = lid[::-1]                            # outer corner first
        flick = lid[0] + np.array([sd * 0.30 * ew, -0.10 * eh])
        p.ink(put(np.vstack([flick, lid])), w * 2.7, taper=(0.22, 0.2), lightdir=False)
        for f_, ln_ in ((0.10, 0.22), (0.25, 0.16)):
            q0 = lid[int(f_ * (len(lid) - 1))]
            p.ink(put([q0, q0 + np.array([sd * ln_ * ew * 0.9, -ln_ * eh])]), w * 1.25, taper=(0.02, 0.65), lightdir=False)
        p.ink(put(shape[np.where(st > 0.5)[0]]), w * 0.85, taper=(0.3, 0.3), lightdir=False)
        anchors["eye_l" if sd < 0 else "eye_r"] = ep.mean(0)


def _quilt(cel, env, body, x, y, s):
    """The patchwork blanket over the sleeper: follows the loaf of the body."""
    c0 = body.mean(0)
    big = c0 + (body - c0) * np.array([1.06, 1.08])
    big[:, 1] = np.minimum(big[:, 1], y + 1.0 * s)
    keep = big[:, 0] > x - 4 * s + (big[:, 1] - (y - 30 * s)) * 0.3
    runs = _runs(keep)
    if not runs:
        return
    quilt = big[max(runs, key=len)]
    p = _Part(cel, _bounds(quilt, 6 * s + 4), env)
    p.fill(quilt, env.lit(QUILT[2]))
    qm = p.mask(quilt)
    r = core.rng("quilt")
    x0, y0, x1, y1 = _bounds(quilt, 0)
    nx, ny = 4, 3
    g = np.zeros((ny + 1, nx + 1, 2))
    for j in range(ny + 1):
        for i in range(nx + 1):
            u = i / nx
            arch = -(1 - (2 * u - 1) ** 2) * (1 - j / ny) * (y1 - y0) * 0.10
            g[j, i] = (x0 + (x1 - x0) * u + r.uniform(-3, 3) * s + (j - 1.5) * 4 * s,
                       y0 + (y1 - y0) * j / ny + arch + r.uniform(-2.5, 2.5) * s)
    st_col = env.lit((0.95, 0.90, 0.78))
    k = 0
    for j in range(ny):
        for i in range(nx):
            quad = np.array([g[j, i], g[j, i + 1], g[j + 1, i + 1], g[j + 1, i]])
            p.fill(quad, env.lit(QUILT[(k * 3 + j) % len(QUILT)]), form=False, clip=qm)
            k += 1
            for a_, b_ in ((quad[0], quad[1]), (quad[0], quad[3])):
                for q in range(5):
                    m0 = a_ + (b_ - a_) * (q + 0.25) / 5 + np.array([1.5, 1.5]) * s
                    m1 = a_ + (b_ - a_) * (q + 0.7) / 5 + np.array([1.5, 1.5]) * s
                    p.ink(np.array([m0, m1]), env.inkw * 0.7, col=st_col, taper=(0.1, 0.1), lightdir=False, clip=qm)
    p.ink(quilt, env.inkw * 1.1, closed=True, taper=(0, 0))
    p.finish(6.0 * s)


def _goggles(p, ball, env, pose, hm, s):
    g = core.clamp(pose.goggles)
    la0 = 55 + (3 - 55) * g
    r0 = 16.0 + 3.0 * g
    rr = 1.07
    sk = pose.goggles_askew
    rimc = env.lit(palette.ROCKET["copper"])
    glass = env.lit(palette.ROCKET["glass"])
    strap = env.lit(LEATHER)
    lo = np.r_[np.linspace(-104, 104, 60), np.linspace(104, -104, 60)]
    la = np.r_[np.full(60, la0 + 6.0), np.full(60, la0 - 6.0)] + sk * np.r_[np.linspace(-1, 1, 60), np.linspace(1, -1, 60)] * 0.25
    sp, sz = ball.patch(lo, la)
    p.fill(sp, strap, form=False, clip=hm)
    for sd in (-1, 1):
        clo, cla = sd * (23 + 7 * g), la0 + sd * sk * 0.2
        _, cz = ball.pt(clo, cla)
        if cz < 0.05:
            continue
        lo, la = _ell(clo, cla, r0, r0, 30)
        op_, _ = ball.proj(_sph(lo, la, rr))
        p.fill(op_, rimc)
        lo, la = _ell(clo, cla, r0 * 0.74, r0 * 0.74, 26)
        ip_, _ = ball.proj(_sph(lo, la, rr))
        if g > 0.5:
            p.fill(ip_, glass, form=False, alpha=0.32)
        else:
            p.fill(ip_, glass, form=False)
        gm = p.mask(ip_)
        c0, _ = ball.proj(_sph(clo - 4, cla + 4, rr))
        q = np.array([[-0.26, -0.05], [0.06, -0.05], [0.26, 0.05], [-0.06, 0.05]]) * np.ptp(ip_[:, 0])
        q = np.c_[q[:, 0] + q[:, 1] * 1.2, q[:, 1] * 2.2 - q[:, 0] * 0.7]
        p.later(c0[0] + q, (1, 1, 1), clip=gm)
        p.ink(op_, env.inkw * 1.05, closed=True, taper=(0, 0))
        p.ink(ip_, env.inkw * 0.7, closed=True, taper=(0, 0), lightdir=False)
    lo, la = _ell(0, la0, 5.5 + 3 * g, 3.2, 14)
    bp, _ = ball.proj(_sph(lo, la, rr))
    p.fill(bp, rimc)


# ----------------------------------------------------------------------- body
def _limb(cel, env, ctrl, hw, col, s, toes=0, toe_dir=None, extra=None, d_sh=None):
    poly, c, L, R, t, w = _tube(ctrl, hw, n=34)
    box = _bounds(poly if extra is None else np.vstack([poly, extra]), 6 * s + 4)
    p = _Part(cel, box, env)
    p.fill(poly, col)
    if extra is not None:
        p.fill(extra, col)
        p.ink(extra, env.inkw, closed=True, taper=(0, 0))
    p.ink(poly, env.inkw, closed=True, taper=(0, 0))
    lt = palette.mix(col, env.lit(P["pad"]), 0.8)
    if toes:
        # a mitten paw: four toe bumps round the end
        tip, tv, nv, r = c[-1], t[-1], np.array([-t[-1][1], t[-1][0]]), w[-1]
        for ph, rr_ in ((-1.0, 0.33), (-0.36, 0.38), (0.30, 0.38), (0.94, 0.32)):
            dv = math.cos(ph) * tv + math.sin(ph) * nv
            cc = tip + dv * r * 0.80
            an = np.linspace(-1.5, 1.5, 12)
            pv = np.array([-dv[1], dv[0]])
            arc = cc + r * rr_ * (np.cos(an)[:, None] * dv + np.sin(an)[:, None] * pv)
            p.fill(np.vstack([arc, cc - dv * r * 0.3]), col)
            p.ink(arc, env.inkw * 0.9, taper=(0.2, 0.2))
            p.ink(np.array([cc - dv * r * 0.1 + pv * r * rr_ * 0.95, cc - dv * r * 0.55 + pv * r * rr_ * 0.7]),
                  env.inkw * 0.7, col=lt, taper=(0.05, 0.6), lightdir=False)
    if extra is not None:
        # toes on the foot
        ec = extra.mean(0)
        erx, ery = np.ptp(extra[:, 0]) / 2, np.ptp(extra[:, 1]) / 2
        for k in (-0.42, 0.0, 0.42):
            p.ink(np.array([ec + (k * erx, ery * 0.96), ec + (k * erx * 0.85, ery * 0.15)]), env.inkw * 0.75, col=lt,
                  taper=(0.05, 0.6), lightdir=False)
    p.finish((d_sh or 3.4) * s)
    return c


def _draw_tail(cel, env, ctrl, s, wind=0.0):
    hw = np.array([6.5, 11.5, 13.8, 13.4, 11.0, 6.5]) * s
    poly, c, L, R, t, w = _tube(ctrl, hw, n=70)
    # a pointed fur tip
    tipp = c[-1] + t[-1] * w[-1] * 1.9
    nrm = np.array([-t[-1][1], t[-1][0]])
    tip = np.array([c[-1] + nrm * w[-1] * 0.8, tipp, c[-1] - nrm * w[-1] * 0.8])
    p = _Part(cel, _bounds(np.vstack([poly, tip]), 6 * s + 4), env)
    fur, dk = env.c("fur"), palette.mix(env.c("fur"), env.c("fur_dk"), 0.85)
    p.fill(poly, fur)
    p.fill(tip, dk)
    tm = p.mask(poly)
    n = len(c)
    for a, b in ((0.13, 0.23), (0.35, 0.46), (0.58, 0.69), (0.81, 1.0)):
        i0, i1 = int(a * (n - 1)), int(b * (n - 1))
        fwd0 = _arc3(R[i0], c[i0] + t[i0] * w[i0] * 0.55, L[i0])
        if b >= 1.0:
            far = c[-1] + t[-1] * w[-1] * 2.5
            ring = np.vstack([L[i0:], far + nrm * w[-1] * 2, far - nrm * w[-1] * 2, R[i0:][::-1], fwd0])
        else:
            fwd1 = _arc3(L[i1], c[i1] + t[i1] * w[i1] * 0.55, R[i1])
            ring = np.vstack([L[i0:i1 + 1], fwd1, R[i0:i1 + 1][::-1], fwd0])
        p.fill(ring, dk, form=False, clip=tm)
    whole = np.vstack([L, tip, R[::-1], poly[-9:]])
    p.ink(whole, env.inkw * 1.1, closed=True, taper=(0, 0))
    p.finish(4.6 * s)
    return c[-1]


def _default_tail(pose):
    yr = pose.yaw * math.pi / 2
    sy, cy = math.sin(yr), math.cos(yr)
    if cy > 0.3 and abs(sy) < 0.5:
        d = 1.0 if sy <= 0.01 else -1.0
        if abs(sy) < 0.01:
            d = 1.0
        return ((d * 14, 32), (d * 36, 22), (d * 55, 30), (d * 64, 52), (d * 58, 76), (d * 46, 90))
    if cy >= -0.3:
        d = -1.0 if sy > 0 else 1.0
        return ((d * 20, 34), (d * 42, 26), (d * 62, 30), (d * 74, 48), (d * 72, 70), (d * 62, 86))
    d = 1.0 if sy > 0 else -1.0
    return ((d * 2, 36), (d * 8, 20), (d * 26, 9), (d * 48, 12), (d * 60, 30), (d * 58, 50))


def _frames(pose, x, y, s):
    sq = pose.squash
    sxk = 1.0 / math.sqrt(max(sq, 0.2))
    ln = math.radians(pose.lean)
    cl, sl = math.cos(ln), math.sin(ln)
    piv = 26.0

    def M(mx, my):
        vx, vy = mx * sxk, (my - piv) * sq
        X = vx * cl + vy * sl + pose.dx
        Y = -vx * sl + vy * cl + piv * sq + pose.dy
        return np.array([x + s * X, y - s * Y])

    def G(mx, my):
        return np.array([x + s * mx, y - s * my])

    return M, G, sxk, sq, ln


def draw(cel, pose, x, y, s=1.0, tod="dusk", drawing=0, light=None):
    """Draw the panda on a core.Cel. (x, y) is the screen position of the
    ground point between the feet. Returns a dict of screen anchors (mouth,
    paws, head, tail tip) for props such as the string."""
    env = _Env(tod, s, drawing, light)
    if pose.curl > 0.5:
        return _draw_curled(cel, env, pose, x, y, s)
    M, G, sxk, sq, ln = _frames(pose, x, y, s)
    yr = pose.yaw * math.pi / 2
    cyw, syw = math.cos(yr), math.sin(yr)
    br = pose.breath
    torso = _Ball(*M(0, 58 + 1.2 * br), (29 + 1.3 * br) * s * sxk, 37 * s * sq, yr, 0.0, ln)
    hc = M(pose.head_dx, 121 + pose.head_dy + 1.6 * br)
    dark = env.c("dark")
    anchors = {}
    front = cyw > -0.25

    # tail
    tl = pose.tail if pose.tail is not None else _default_tail(pose)
    tctrl = np.array([G(*q) for q in tl])
    if front:
        anchors["tail_tip"] = _draw_tail(cel, env, tctrl, s, pose.wind)

    # legs and feet
    for sd, ft in ((-1, pose.foot_l), (1, pose.foot_r)):
        hip = M(sd * 12 * cyw + 0, 31)
        if ft is None:
            ft = (sd * (13 * abs(cyw) + 4 * abs(syw)) * (1 if cyw >= 0 else -1), 0.0)
        f = G(ft[0], ft[1])
        ank = f + np.array([0, -6.5 * s])
        mid = (hip + ank) / 2 + np.array([sd * 1.5 * s * cyw, 0])
        foot = core.ellipse(f[0] + (syw * 5.0 + sd * 1.5 * cyw) * s, f[1] - 5.4 * s, (11.5 + 3.0 * abs(syw)) * s, 6.8 * s, 26)
        _limb(cel, env, [hip, mid, ank], np.array([9.5, 8.6, 7.4]) * s, dark, s, extra=foot)

    # shoulders, far arms first
    arms = []
    for sd, pw in ((-1, pose.paw_l), (1, pose.paw_r)):
        S, z = torso.pt(sd * 66, 42, 0.90)
        if pw is None:
            Pw = S + np.array([sd * 3.5 * cyw + syw * 3, 31.0]) * s
        else:
            Pw = G(pw[0], pw[1])
        d = Pw - S
        dist = math.hypot(*d)
        L1 = 18.6 * s
        h = math.sqrt(max(L1 * L1 - (dist / 2) ** 2, 0.0))
        nrm = np.array([-d[1], d[0]]) / max(dist, 1e-6)
        if nrm @ np.array([sd * (1 if front else -1), 0.45]) < 0:
            nrm = -nrm
        E = (S + Pw) / 2 + nrm * h
        # a real elbow: two points close together make the bend, the forearm runs straight to the wrist
        ctrl = [S, S + (E - S) * 0.6, E, E + (Pw - E) * 0.45, Pw]
        arms.append((z, sd, ctrl))
        anchors["paw_l" if sd < 0 else "paw_r"] = Pw
    hw_arm = np.array([8.0, 7.4, 6.6, 5.6, 5.4, 8.8]) * s
    for z, sd, ctrl in arms:
        if z < -0.05:
            _limb(cel, env, ctrl, hw_arm, dark, s, toes=4)

    # torso
    a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    # a pear: narrow shoulders, heavy hips, the belly pushed toward where it faces
    lx = torso.rx * ca * (1 + 0.24 * sa) * (1 - 0.06 * np.abs(sa) ** 3) + syw * torso.rx * 0.12 * np.maximum(sa + 0.2, 0)
    ly = torso.ry * sa
    # hip tufts break the egg
    for ac, hw_, L_ in ((38, 12, 3.6), (150, 13, 4.2)):
        d_ = np.abs((np.degrees(a) - ac + 180) % 360 - 180)
        t_ = np.clip(1 - d_ / hw_, 0, 1) ** 1.3
        lx = lx + s * L_ * t_ * math.cos(math.radians(ac))
        ly = ly + s * L_ * t_ * (math.sin(math.radians(ac)) + 0.5)
    tout = torso.loc(lx, ly)
    p = _Part(cel, _bounds(tout, 8 * s + 4), env)
    p.fill(tout, env.c("fur"))
    tm = p.mask(tout)
    # the dark belly wraps the form; its edge is tufted, not an ellipse
    tb = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    saw = np.abs(((tb * 8 / (2 * np.pi) + 0.35 * np.sin(tb * 3)) % 1.0) - 0.5) * 2
    rad = 1.0 + 0.08 * saw * (0.4 + 0.6 * np.abs(np.cos(tb)))
    lo = 40 * np.cos(tb) * rad
    la = -24 + 62 * np.sin(tb) * rad
    lo = lo * (0.66 + 0.34 * np.clip((55 - la) / 100, 0, 1))
    bp, bz = torso.patch(lo, la)
    if bz.mean() > -0.2:
        p.fill(bp, dark, form=False, clip=tm)
    # the chest ruff: fur falling from under the chin over the top of the belly
    rf = [(-60, 62), (-54, 45), (-45, 49), (-37, 34), (-26, 41), (-12, 23), (1, 35), (13, 27), (24, 41), (35, 35),
          (45, 48), (54, 45), (62, 62), (30, 86), (-30, 86)]
    rp, rz = torso.patch([q[0] for q in rf], [q[1] for q in rf])
    if rz.mean() > -0.15:
        p.fill(rp, palette.mix(env.c("fur"), env.c("white"), 0.10), form=False, clip=tm)
        p.ink(rp[1:12], env.inkw * 0.8, taper=(0.1, 0.1), lightdir=False, clip=tm)
    # designed shadows: the head and cheek ruff on the chest and shoulders, each near arm on the body
    off = np.array([-env.sun[0] * 7 * s, 11 * s - env.sun[1] * 3 * s])
    p.shadow(core.ellipse(hc[0] + off[0], hc[1] + off[1], 43 * s, 33 * s, 40))
    for z, sd, ctrl in arms:
        if z >= -0.05:
            ap = _tube(ctrl, hw_arm * 1.12, n=30)[0]
            p.shadow(ap + np.array([-env.sun[0] * 4.5 * s, 3.0 * s - env.sun[1] * 2.5 * s]))
    p.ink(tout, env.inkw * 1.15, closed=True, taper=(0, 0))
    p.finish(6.0 * s)

    if not front:
        anchors["tail_tip"] = _draw_tail(cel, env, tctrl, s, pose.wind)

    anchors.update(_draw_head(cel, env, pose, hc, s, roll=ln * 0.6))

    for z, sd, ctrl in arms:
        if z >= -0.05:
            _limb(cel, env, ctrl, hw_arm, dark, s, toes=4)
    return anchors


def _draw_curled(cel, env, pose, x, y, s):
    """Asleep: a round loaf of fur, head resting low, tail wrapped over the nose."""
    def G(mx, my):
        return np.array([x + s * mx, y - s * my])

    br = pose.breath
    a = np.linspace(0, 2 * np.pi, 110, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    bx = 8 + (45 + 1.0 * br) * np.sign(ca) * np.abs(ca) ** 0.9 * (1 + 0.05 * sa)
    by = 30 + 1.5 * br + (30 + 2.2 * br) * np.sign(sa) * np.abs(sa) ** 0.75
    body = np.c_[x + s * bx, y - s * by]
    p = _Part(cel, _bounds(body, 8 * s + 4), env)
    p.fill(body, env.c("fur"))
    hc = G(-27, 31 + 0.8 * br)
    p.shadow(core.ellipse(hc[0] - env.sun[0] * 6 * s, hc[1] + 8 * s, 38 * s, 30 * s, 36))
    p.ink(body, env.inkw * 1.15, closed=True, taper=(0, 0))
    hn = core.catmull([G(52, 30), G(40, 40), G(22, 38), G(12, 22), G(14, 6)], 8, closed=False)
    p.ink(hn, env.inkw, taper=(0.25, 0.3), lightdir=False)
    p.finish(5.5 * s)
    if pose.blanket > 0.05:
        _quilt(cel, env, body, x, y, s)
    anchors = _draw_head(cel, env, pose, hc, s)
    tl = pose.tail if pose.tail is not None else ((50, 13), (26, 6), (-2, 7), (-30, 12), (-55, 22), (-64, 40))
    anchors["tail_tip"] = _draw_tail(cel, env, np.array([G(*q) for q in tl]), s)
    return anchors


def bbox(pose, x, y, s):
    """Screen box that always contains the drawing, so a small Cel can be used."""
    pts = [(-70, -8), (70, 182)]
    for q in (pose.paw_l, pose.paw_r, pose.foot_l, pose.foot_r):
        if q is not None:
            pts += [(q[0] - 12, q[1] - 12), (q[0] + 12, q[1] + 12)]
    tl = pose.tail if pose.tail is not None else ((-95, 0), (95, 110))
    pts += [(q[0] - 20, q[1] - 20) for q in tl] + [(q[0] + 20, q[1] + 20) for q in tl]
    a = np.array(pts, np.float64)
    a[:, 0] += pose.dx
    a[:, 1] += pose.dy
    return (x + s * a[:, 0].min() - 4, y - s * a[:, 1].max() - 4, x + s * a[:, 0].max() + 4, y - s * a[:, 1].min() + 4)


# ------------------------------------------------------------ the pose library
POSES = {
    "stand": Pose(),
    "stand_34": Pose(yaw=0.42, head_yaw=0.36),
    "stand_profile": Pose(yaw=0.95, head_yaw=0.95),
    "stand_34back": Pose(yaw=1.45, head_yaw=1.45),
    "proud": Pose(yaw=0.2, head_yaw=0.3, pitch=0.25, lean=-3, paw_l=(-25, 62), paw_r=(27, 62), smile=0.8,
                  eye_smile=0.35, brow_l=0.3, brow_r=0.3, breath=1.0),
    "tie_knot": Pose(yaw=0.38, head_yaw=0.36, pitch=-0.12, tilt=-5, lean=-8, paw_l=(27, 69), paw_r=(45, 83),
                     mouth_open=0.14, smile=0.1, brow_tilt=-0.35, brow_l=-0.2, brow_r=-0.2, pupil=(0.35, -0.55),
                     eye_l=0.9, eye_r=0.9, ear_l=-0.1, ear_r=0.05),
    "pleased": Pose(head_yaw=0.3, yaw=0.3, eye_l=0.0, eye_r=0.0, eye_smile=1.0, smile=0.9, tilt=6, brow_l=0.4, brow_r=0.4),
    "sleep": Pose(curl=1.0, head_yaw=-0.18, tilt=-10, eye_l=0.0, eye_r=0.0, smile=0.2, ear_l=0.7, ear_r=-0.05,
                  brow_l=-0.3, brow_r=-0.3, blanket=1.0),
    "wind": Pose(pitch=0.2, eye_l=0.5, eye_r=0.45, eye_smile=0.2, brow_tilt=0.8, brow_l=0.5, brow_r=0.5, mouth_open=0.7,
                 mouth_wide=1.0, smile=-0.3, cheek=0.7, ear_l=1.0, ear_r=1.0, wind=1.0, wind_dir=(0.0, 0.75),
                 pupil_size=0.8, paw_l=(-22, 92), paw_r=(22, 92), goggles=1.0),
    "wonder": Pose(pitch=0.35, head_yaw=0.25, eye_l=1.55, eye_r=1.55, hl=1.5, brow_l=1.0, brow_r=1.0, brow_tilt=0.3,
                   mouth_open=0.55, mouth_wide=-0.5, smile=0.3, ear_l=-0.3, ear_r=-0.3, pupil=(0.3, 0.5),
                   paw_l=(-22, 92), paw_r=(22, 92), goggles=0.0),
    "determined": Pose(brow_tilt=-1.0, brow_l=-0.6, brow_r=-0.6, eye_l=0.82, eye_r=0.82, smile=-0.1, mouth_wide=-0.3,
                       pitch=-0.12, goggles=1.0),
    "nervous": Pose(brow_tilt=1.0, brow_l=0.5, brow_r=0.5, pupil_size=0.72, smile=-0.55, mouth_wide=-0.4, ear_l=0.5,
                    ear_r=0.5, pupil=(0.0, -0.3), goggles=1.0),
    "squeeze": Pose(squeeze=1.0, eye_l=0.0, eye_r=0.0, cheek=1.0, smile=-0.5, mouth_wide=-0.7, brow_tilt=-0.5,
                    brow_l=-0.8, brow_r=-0.8, ear_l=0.7, ear_r=0.7, goggles=0.0),
    "one_eye": Pose(squeeze=0.0, eye_l=0.0, eye_r=0.85, cheek=0.8, smile=-0.4, mouth_wide=-0.6, brow_l=-0.6,
                    brow_r=0.7, ear_l=0.6, ear_r=0.2, pupil=(0.5, 0.2), goggles=0.0),
    "look_up": Pose(pitch=0.6, head_yaw=-0.25, yaw=-0.15, smile=0.5, brow_l=0.5, brow_r=0.5, pupil=(-0.2, 0.6),
                    mouth_open=0.15, soot=1.0, goggles=0.0, goggles_askew=14),
    "camera_smile": Pose(tilt=-5, smile=0.9, eye_smile=0.25, brow_l=0.3, brow_r=0.3),
}
