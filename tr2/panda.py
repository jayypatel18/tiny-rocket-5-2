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
        self.rimw = max(1.0, 1.15 * s ** 0.75)
        self.boil = 0.3 * s ** 0.45
        self.seed = core.seed_of("boil", int(drawing)) % 90000
        self.bfreq = 0.05 / s ** 0.5
        ratio = np.clip(self.t["amb"] / self.t["key"], 0, 1)
        self.sh8 = tuple(int(v * 255) for v in ratio)
        self.rim8 = tuple(int(v * 255) for v in np.clip(self.t["rim"], 0, 1))
        self.light = light
        self.pad = int(8 * s) + 6

    def c(self, name):
        return palette.lit(P[name], self.t)

    def lit(self, col):
        return palette.lit(col, self.t)


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
    A, za = ball.pt(side * 56, 64, 0.96)
    B, zb = ball.pt(side * 104, 28, 0.96)
    f = max(flat, -0.6)
    T, zt = ball.pt(side * (80 + 26 * f), 58 - 46 * f, 1.80 - 0.26 * max(f, 0))
    ln = math.hypot(*(T - (A + B) / 2))

    def bulge(a, b, other, k):
        d = b - a
        n = np.array([-d[1], d[0]])
        n /= max(math.hypot(*n), 1e-6)
        if n @ (other - a) > 0:
            n = -n
        return a + d * 0.52 + n * k * ln

    tip_in = T + (A - T) * 0.10
    tip_out = T + (B - T) * 0.10
    curve = core.catmull([A, bulge(A, T, B, 0.15), tip_in, T + (T - (A + B) / 2) * 0.03, tip_out,
                          bulge(B, T, A, 0.20)[::1], B], 8, closed=False)
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
        inner = G + (e["curve"] - G) * 0.70 * np.array([1, 1]) + (base - G) * 0.10
        inner = np.vstack([inner, base + (G - base) * 0.25])
        em = p.mask(e["poly"])
        p.fill(inner, env.c("white"), form=False, clip=em)
        core_ = G + (e["curve"] - G) * 0.40 * k + (base - G) * 0.30
        p.fill(np.vstack([core_, base + (G - base) * 0.3]), env.c("ear_in"), form=False, clip=em)
        # two fur tufts growing from the inner base
        for j, (u, lnk) in enumerate(((0.30, 0.42), (0.52, 0.33))):
            a0 = e["A"] + (e["B"] - e["A"]) * u
            tipv = a0 + (e["T"] - a0) * lnk
            wv = (e["B"] - e["A"]) * 0.09
            p.fill(np.array([a0 - wv, tipv, a0 + wv]), env.c("white"), form=False, clip=em)
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
    wd = np.asarray(pose.wind_dir, np.float64) * pose.wind
    deg = np.degrees(a)

    def tuft(ac, hw, L, lean):
        nonlocal lx, ly
        d = np.abs((deg - ac + 180) % 360 - 180)
        t = np.clip(1 - d / hw, 0, 1) ** 1.35
        lx = lx + s * L * t * (math.cos(math.radians(ac)) + lean[0])
        ly = ly + s * L * t * (math.sin(math.radians(ac)) + lean[1])

    k = 1 + 1.4 * pose.wind
    for sd in (1, -1):
        base = 0 if sd == 1 else 180
        tuft(base + sd * 10, 13, 7.0 * k, (0.15 * sd + wd[0] * 1.3, 0.6 + wd[1] * 1.3))
        tuft(base + sd * 36, 11, 5.2 * k, (0.05 * sd + wd[0], 0.55 + wd[1]))
    cr = 270 + pose.head_yaw * 22
    tuft(cr - 7, 7.5, 4.6, (0.55 + wd[0], -0.1 + wd[1] * 0.5))
    tuft(cr + 5, 6.5, 3.0, (0.65 + wd[0], 0.0))
    outline = ball.loc(lx, ly)
    fur, white = env.c("fur"), env.c("white")
    p.fill(outline, fur)
    hm = p.mask(outline)
    for e in near:
        _ear_paint(p, e, env)

    # ---- white mask: muzzle, bridge and cheeks
    half = [(0, 16), (7, 14), (12, 5), (16, -8), (25, -16), (37, -18), (50, -14), (64, -5), (80, 1),
            (100, -6), (112, -40), (84, -70), (42, -85)]
    pts = half + [(0, -89)] + [(-lo, la) for lo, la in half[::-1]]
    ll = core.catmull(pts, 6, closed=True)
    mp, mz = ball.patch(ll[:, 0], ll[:, 1])
    p.fill(mp, white, form=False, clip=hm)
    # tear marks from the eye down the cheek
    tear = palette.mix(env.c("fur"), env.c("fur_dk"), 0.7)
    for sd in (-1, 1):
        cl = core.catmull([(sd * 31, -8), (sd * 42, -23), (sd * 52, -40), (sd * 58, -58), (sd * 60, -74)], 6, closed=False)
        hwd = np.interp(np.linspace(0, 1, len(cl)), [0, 0.35, 0.75, 1], [7.5, 6.0, 4.2, 2.0])
        t = np.gradient(cl, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        nr = np.c_[-t[:, 1], t[:, 0]]
        rb = np.vstack([cl + nr * hwd[:, None], (cl - nr * hwd[:, None])[::-1]])
        tp, tz = ball.patch(rb[:, 0], rb[:, 1])
        if tz.mean() > -0.25:
            p.fill(tp, tear, form=False, clip=hm)

    # ---- muzzle: a small white bump that leaves the silhouette in profile
    mf = ball.frame(0, -25, 1.02, minx=0.5)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    mloc = np.c_[14.5 * np.cos(t), 10.5 * np.sin(t)]
    muz = _put(mf, mloc, s)
    show_face = mf[3] > -0.28
    if show_face:
        p.fill(muz, white)

    # ---- brows: the white dots do the acting
    for sd, br in ((-1, pose.brow_l), (1, pose.brow_r)):
        lo, la = _ell(sd * 30, 34 + 7 * br, 8.5, 5.0, 24, rot=-sd * (8 + 22 * pose.brow_tilt))
        bp, bz = ball.patch(lo, la)
        if bz.mean() > -0.1:
            p.fill(bp, white, form=False, clip=hm)

    # ---- eyes
    eyec = env.c("eye")
    lside = -1.0 if env.sun[0] < 0 else 1.0
    px, py = pose.pupil
    for sd, op in ((-1, pose.eye_l), (1, pose.eye_r)):
        elon, elat = sd * 32.0, 3.0
        ew, eh = 12.5, 16.0
        if op > 1:
            eh *= 1 + 0.2 * (op - 1)
            ew *= 1 + 0.12 * (op - 1)
        o = min(op, 1.0)
        _, cz = ball.pt(elon, elat)
        if cz < -0.12:
            continue
        if o < 0.1 or pose.squeeze > 0.5:
            u = np.linspace(-1, 1, 22)
            up = pose.eye_smile > 0.3 or pose.squeeze > 0.5
            bow = 1.0 if up else -1.0
            lo = elon + ew * 1.02 * u
            la = elat - (0.05 if up else 0.35) * eh + bow * 0.40 * eh * (1 - u * u) - sd * 0.10 * ew * u
            cp_, _ = ball.patch(lo, la)
            p.ink(cp_, w * 2.1, taper=(0.22, 0.22), lightdir=False)
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
        h1, _ = ball.pt(elon + lside * 0.30 * ew + px * 1.5, elat + 0.40 * eh * min(1.0, o * 1.2))
        r1 = 0.25 * wpx * pose.hl
        p.later(core.ellipse(h1[0], h1[1], r1, r1 * 1.12, 22, rot=0.3), (1, 1, 1), clip=em)
        h2, _ = ball.pt(elon - lside * 0.36 * ew + px * 1.5, elat - 0.42 * eh)
        r2 = 0.105 * wpx * pose.hl
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
        nf = ball.frame(0, -13, 1.21, minx=0.62)
        nose = core.catmull([(-4.0, -2.3), (0, -2.9), (4.0, -2.3), (4.3, -0.3), (1.7, 2.6), (0, 3.0), (-1.7, 2.6), (-4.3, -0.3)], 5)
        npoly = _put(nf, nose, s)
        p.fill(npoly, env.c("nose"), form=False)
        nh = _put(nf, np.array([[-1.2 * 1 + lside * 1.0, -1.5]]), s)[0]
        p.later(core.ellipse(nh[0], nh[1], 1.25 * s, 0.7 * s, 12), env.lit(palette.mix(P["nose"], (1, 1, 1), 0.6)))
        kf = ball.frame(0, -32, 1.10, minx=0.55)
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
        p.ink(_put(kf, left, s), w * 1.05, taper=(0.05, 0.3), lightdir=False)
        p.ink(_put(kf, right, s), w * 1.05, taper=(0.05, 0.3), lightdir=False)
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
    p.finish(5.2 * s)
    anchors["head"] = ball.c.copy()
    anchors["ball"] = ball
    anchors["outline"] = outline
    return anchors


def _goggles(p, ball, env, pose, hm, s):
    g = core.clamp(pose.goggles)
    la0 = 55 + (3 - 55) * g
    r0 = 17.5 + 4.5 * g
    rr = 1.07
    sk = pose.goggles_askew
    rimc = env.lit(palette.ROCKET["copper"])
    glass = env.lit(palette.ROCKET["glass"])
    strap = env.lit(LEATHER)
    lo = np.r_[np.linspace(-178, 178, 60), np.linspace(178, -178, 60)]
    la = np.r_[np.full(60, la0 + 6.0), np.full(60, la0 - 6.0)] + sk * np.r_[np.linspace(-1, 1, 60), np.linspace(1, -1, 60)] * 0.25
    sp, sz = ball.patch(lo, la)
    p.fill(sp, strap, form=False, clip=hm)
    for sd in (-1, 1):
        clo, cla = sd * (24 + 8 * g), la0 + sd * sk * 0.2
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
        q = np.array([[-0.5, -0.12], [0.1, -0.12], [0.5, 0.12], [-0.1, 0.12]]) * np.ptp(ip_[:, 0])
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
    if toes:
        tip, tv, nv, r = c[-1], t[-1], np.array([-t[-1][1], t[-1][0]]), w[-1]
        lt = palette.mix(col, env.lit(P["pad"]), 0.75)
        for k in (-0.38, 0.38):
            a = tip + tv * r * 0.98 + nv * r * k
            b = tip + tv * r * 0.35 + nv * r * k * 0.8
            p.ink(np.array([a, b]), env.inkw * 0.75, col=lt, taper=(0.05, 0.6), lightdir=False)
    p.finish((d_sh or 3.2) * s)
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
        foot = core.ellipse(f[0] + syw * 4.0 * s, f[1] - 4.6 * s, (9.5 + 2.5 * abs(syw)) * s, 5.6 * s, 26)
        _limb(cel, env, [hip, mid, ank], np.array([8.5, 8.0, 7.2]) * s, dark, s, extra=foot)

    # shoulders, far arms first
    arms = []
    for sd, pw in ((-1, pose.paw_l), (1, pose.paw_r)):
        S, z = torso.pt(sd * 66, 42, 0.90)
        if pw is None:
            Pw = S + np.array([sd * 6.0 * cyw + syw * 3, 30.0]) * s
        else:
            Pw = G(pw[0], pw[1])
        d = Pw - S
        dist = math.hypot(*d)
        L1 = 18.5 * s
        h = math.sqrt(max(L1 * L1 - (dist / 2) ** 2, 0.0))
        nrm = np.array([-d[1], d[0]]) / max(dist, 1e-6)
        if nrm @ np.array([sd * (1 if front else -1), 0.45]) < 0:
            nrm = -nrm
        E = (S + Pw) / 2 + nrm * h * 0.9
        arms.append((z, sd, [S, E, Pw]))
        anchors["paw_l" if sd < 0 else "paw_r"] = Pw
    hw_arm = np.array([6.6, 6.0, 5.8, 7.9]) * s
    for z, sd, ctrl in arms:
        if z < -0.05:
            _limb(cel, env, ctrl, hw_arm, dark, s, toes=2)

    # torso
    a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    lx = torso.rx * ca * (1 + 0.13 * sa) + syw * torso.rx * 0.10 * np.maximum(sa + 0.2, 0)
    ly = torso.ry * sa
    tout = torso.loc(lx, ly)
    p = _Part(cel, _bounds(tout, 8 * s + 4), env)
    p.fill(tout, env.c("fur"))
    tm = p.mask(tout)
    lo, la = _ell(0, -18, 46, 66, 36, pw=0.9)
    lo = lo * (0.72 + 0.28 * np.clip((60 - la) / 100, 0, 1))
    bp, bz = torso.patch(lo, la)
    if bz.mean() > -0.2:
        p.fill(bp, dark, form=False, clip=tm)
    # the head throws a shadow on the chest
    hsh = core.ellipse(hc[0] - env.sun[0] * 7 * s, hc[1] + 9 * s - env.sun[1] * 3 * s, 37 * s, 31 * s, 40)
    p.shadow(hsh)
    p.ink(tout, env.inkw * 1.15, closed=True, taper=(0, 0))
    p.finish(5.0 * s)

    if not front:
        anchors["tail_tip"] = _draw_tail(cel, env, tctrl, s, pose.wind)

    anchors.update(_draw_head(cel, env, pose, hc, s, roll=ln * 0.6))

    for z, sd, ctrl in arms:
        if z >= -0.05:
            _limb(cel, env, ctrl, hw_arm, dark, s, toes=2)
    return anchors


def _draw_curled(cel, env, pose, x, y, s):
    """Asleep: a round loaf of fur, head resting low, tail wrapped over the nose."""
    def G(mx, my):
        return np.array([x + s * mx, y - s * my])

    br = pose.breath
    a = np.linspace(0, 2 * np.pi, 110, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    bx = 8 + (46 + 1.0 * br) * np.sign(ca) * np.abs(ca) ** 0.9 * (1 + 0.05 * sa)
    by = 27 + 1.5 * br - (27 + 2.2 * br) * np.sign(sa) * np.abs(sa) ** 0.8 * -1
    body = np.c_[x + s * bx, y - s * by]
    p = _Part(cel, _bounds(body, 8 * s + 4), env)
    p.fill(body, env.c("fur"))
    hc = G(-27, 31 + 0.8 * br)
    p.shadow(core.ellipse(hc[0] - env.sun[0] * 6 * s, hc[1] + 8 * s, 38 * s, 30 * s, 36))
    p.ink(body, env.inkw * 1.15, closed=True, taper=(0, 0))
    hn = core.catmull([G(52, 30), G(40, 40), G(22, 38), G(12, 22), G(14, 6)], 8, closed=False)
    p.ink(hn, env.inkw, taper=(0.25, 0.3), lightdir=False)
    p.finish(5.5 * s)
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
    "stand_34": Pose(yaw=0.42, head_yaw=0.42),
    "stand_profile": Pose(yaw=0.95, head_yaw=0.95),
    "stand_34back": Pose(yaw=1.45, head_yaw=1.45),
    "proud": Pose(yaw=0.2, head_yaw=0.3, pitch=0.25, lean=-3, paw_l=(-25, 62), paw_r=(27, 62), smile=0.8,
                  eye_smile=0.35, brow_l=0.3, brow_r=0.3, breath=1.0),
    "tie_knot": Pose(yaw=0.38, head_yaw=0.5, pitch=-0.28, tilt=5, lean=4, paw_l=(14, 74), paw_r=(40, 80),
                     mouth_open=0.22, smile=0.1, brow_tilt=-0.35, brow_l=-0.2, brow_r=-0.2, pupil=(0.35, -0.55),
                     eye_l=0.9, eye_r=0.9, ear_l=-0.1, ear_r=0.05),
    "pleased": Pose(head_yaw=0.3, yaw=0.3, eye_l=0.0, eye_r=0.0, eye_smile=1.0, smile=0.9, tilt=6, brow_l=0.4, brow_r=0.4),
    "sleep": Pose(curl=1.0, head_yaw=-0.18, tilt=-10, eye_l=0.0, eye_r=0.0, smile=0.2, ear_l=0.55, ear_r=0.5,
                  brow_l=-0.3, brow_r=-0.3),
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
