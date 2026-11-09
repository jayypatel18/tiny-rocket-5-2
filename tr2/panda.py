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
Model units: 1 unit = 1 px at s = 1. The standing panda is about 168 units
tall to the ear tips, 100 wide across the cheek tufts, feet at y = 0.

API
  draw(cel, pose, x, y, s=1.0, tod="dusk", drawing=0, light=None) -> anchors
      (x, y) = screen position of the ground point between the feet. tod is a
      palette name or a blended tod dict. drawing seeds the line boil (pass
      common.twos(fl); pass a constant for a dead still hold). light is an
      optional second light dict(color=(r, g, b), dir=(dx, dy), amount=0..1),
      dir pointing from the panda TO the light. Returns screen anchors: head,
      mouth, mouth_l, mouth_r, eye_l, eye_r, paw_l, paw_r, tail_tip.
      Good from s = 0.8 to s = 8. Cost about 0.15 s at s = 1, 1.3 s at s = 7.
  blend(a, b, u)            in between of two poses (paws and feet travel on arcs)
  pose_at(keys, t)          keys = [(time, Pose) or (time, Pose, ease)], eased key animation
  follow(pose_fn, fl)       adds ear and tail overlap from the motion history (see below)
  mirror(pose)              the same pose facing the other way
  replace(pose, **fields)   a copy with some fields changed (dataclasses.replace)
  bbox(pose, x, y, s)       screen box (x0, y0, x1, y1) that contains the drawing
  draw_paw_closeup(cel, x, y, s, tod, drawing, paint, press)   shot 9c

Pose fields (all optional)
  dx, dy          body offset from the ground point, units, y up (a hop is dy > 0)
  lean            degrees, + leans toward screen right (pivot at the hips)
  squash          1 normal, 0.85 squashed, 1.15 stretched (volume kept)
  yaw             body turn: 0 front, 0.4 three quarter right, 1 profile right,
                  1.45 three quarter back, negative = toward screen left
  breath          0..1 chest rise
  curl            1 = the curled sleeper (head at screen left, tail over the nose)
  blanket         1 = patchwork quilt over the sleeper
  head_yaw        head turn, same scale as yaw, absolute. Keep within 0.5 of yaw.
  pitch           -1 looks down .. +1 looks up
  tilt            degrees, + tips the head toward screen right
  head_dx, head_dy  head offset in units (neck stretch, chin bob)
  eye_l, eye_r    0 shut, 1 open, up to 1.6 huge (wonder)
  eye_smile       0..1 lower lid up; with shut eyes it makes happy arcs
  squeeze         1 = eyes screwed shut with wrinkles
  pupil           (x, y) each -1..1, x toward screen right, y up
  pupil_size      1 normal, 0.7 small (fear), 1.2 big
  hl              highlight size, 1 normal, 1.5 for wonder
  brow_l, brow_r  -1 low .. +1 raised (the white brow dots)
  brow_tilt       +1 worried, -1 determined
  mouth_open      0 shut .. 1 yawn
  mouth_wide      -1 pursed .. +1 pulled wide
  smile           -1 frown .. +1 grin (0.35 is the resting face)
  tongue          0..1 tongue tip out
  cheek           0..1 puffed cheeks (held breath)
  ear_l, ear_r    0 alert, 1 flat back, -0.5 perked forward
  paw_l, paw_r    (x, y) target of that paw in units from the ground point, or None = hangs relaxed
  palm_l, palm_r  1 = that palm faces us, pads drawn
  foot_l, foot_r  (x, y) target of that foot, or None = planted under the hip
  tail            6 control points ((x, y), ...) from the ground point, root first, or None = natural for the yaw
  goggles         -1 none, 0 pushed up on the forehead, 1 over the eyes (in betweens slide)
  goggles_askew   degrees of tilt of the goggles
  soot            0..1 soot smudges on the face
  wind, wind_dir  fur ruffle 0..1 and the screen direction it streams toward
  smear           (dx, dy) units: a smear drawing dragged out behind a fast move in that direction

POSES (names)
  standing        stand, stand_34, stand_profile, stand_34back, proud, look_up_soft, camera_smile, tail_curl
  walking         step_a, step_b, step_c (facing right; a b c b a ... walks forward; mirror() to face left)
  shot 1          tie_knot, pleased
  map             read_map, map_tap, paint (these face screen left)
  sleeping        sleep, ear_twitch
  waking, climb   wake_sit, yawn, climb_reach, climb_pull, climb_over (facing right)
  cockpit         cockpit, goggles_grab, goggles_down, determined, nervous, swallow, nod_down,
                  squeeze, one_eye, wind, wonder, climb_out, look_up (sooty)

Animating between keys with follow through:
    K = panda.POSES
    keys = [(0.0, K["stand"]), (0.4, panda.replace(K["stand"], squash=0.9, dy=-3)),   # anticipation
            (0.7, K["proud"], core.overshoot), (2.0, K["proud"])]
    def pose_fn(k):                       # key animation at frame k (hold before 0)
        return panda.pose_at(keys, max(k, 0) / 24.0)
    pose = panda.follow(pose_fn, common.twos(fl))      # ears and tail trail and settle
    panda.draw(cel, pose, x, y, s, "dusk", drawing=common.twos(fl))
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
    palm_l: float = 0.0        # 1 = that palm faces us: pads are drawn
    palm_r: float = 0.0
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
        ratio = np.clip(self.t["amb"] / self.key * (1.12, 1.0, 1.0), 0, 0.82)
        self.sh8 = tuple(int(v * 255) for v in ratio)
        # the shadow tone is pushed toward violet by a small add of the ambient
        ratio = ratio.mean() + (ratio - ratio.mean()) * 0.7
        self.sh8 = tuple(int(v * 255) for v in ratio)
        self.sh_add8 = tuple(int(v * 255) for v in np.clip(self.t["amb"] * 0.07 * np.array([0.9, 0.5, 1.0]), 0, 1))
        self.rim8 = tuple(int(v * 255) for v in np.clip(self.t["rim"], 0, 1))
        self.light = light
        self.pad = int(8 * s) + 6
        self.smear = None

    def set_smear(self, vec, centre):
        ln = float(np.hypot(*vec)) * self.s
        if ln > 0.5:
            self.smear = True
            self.smear_len = ln
            self.smear_dir = np.asarray(vec, np.float64) / np.hypot(*vec)
            self.smear_c = np.asarray(centre, np.float64)
            self.pad += int(ln * 1.7)

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
        if e.smear is not None:
            # a smear drawing: the trailing side of the form is dragged out behind the motion
            rel = (p - e.smear_c) @ e.smear_dir
            p = p - e.smear_dir[None, :] * (np.clip(-rel / (45.0 * e.s) + 0.2, 0, 1.6) * e.smear_len)[:, None]
        return ((p - (self.x0, self.y0)) * self.ss).ravel().tolist()

    def mask(self, pts, clip=None):
        if not self.ok:                     # a part outside the cel is a silent no-op
            return None
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
        if lightdir is not False and lightdir is not None:
            # a contour: heavy on the side away from the light, thin and sometimes broken on the lit edge.
            # lightdir is True (use the stroke's own centre) or the centre of the form it outlines.
            c0 = p.mean(0) if lightdir is True else np.asarray(lightdir, np.float64)
            out = p - c0
            out /= np.maximum(np.hypot(out[:, 0], out[:, 1]), 1e-9)[:, None]
            fl_ = out @ self.env.sun                       # +1 faces the light
            wv *= 0.50 + 0.62 * np.clip(0.55 - fl_, 0, 1)
            nz = core.fbm_at(np.arange(n) * 1.6 / (70.0 * self.env.s ** 0.5), np.zeros(n) + 1.5, 1.0, 2, self.env.seed + 5)
            wv *= 1.0 - 0.85 * np.clip(fl_ - 0.25, 0, 1) * np.clip((nz - 0.52) / 0.12, 0, 1)
        wv *= 0.86 + 0.3 * core.fbm_at(sv * 7.0 + 3.1, np.zeros(n) + 0.5, 1.0, 2, self.env.seed + int(ln) % 97)
        wv = np.maximum(wv, 0.10) * w * 0.5
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
            rgb = Image.composite(Image.blend(rgb, Image.new("RGB", self.size, e.rim8), 0.74), rgb, rm)
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
    """A wide round ear on the corner of the head, shorter than it is wide."""
    f = max(flat, -0.6)
    # base points; when they go behind the head they are pulled to the silhouette, so the far ear
    # rises from the outline instead of poking out as a spike
    pa, za = ball.proj(_sph(side * 34, 63), clamp=True)
    pb, zb = ball.proj(_sph(side * 101, 29), clamp=True)
    A, B, za, zb = pa[0], pb[0], float(za[0]), float(zb[0])
    T, zt = ball.pt(side * (72 + 14 * f), 57 - 34 * f, 1.56 - 0.42 * max(f, 0))
    mid = (A + B) / 2
    ln = math.hypot(*(T - mid))
    lu = math.hypot(*(B - A)) / 2
    if lu < 0.62 * ln:                      # the far ear never thins to a blade
        A, B = mid + (A - mid) / max(lu, 1e-6) * 0.62 * ln, mid + (B - mid) / max(lu, 1e-6) * 0.62 * ln

    # a round arch from A over the tip T to B, a little wider than its base
    u_ = (B - A) / 2
    v_ = T - mid
    th = np.linspace(0, np.pi, 41)[:, None]
    curve = mid - u_ * np.cos(th) * (1.0 + 0.12 * np.sin(th)) + v_ * np.sin(th) ** 0.82
    # a small tuft at the tip only
    up = v_ / max(ln, 1e-6)
    for j, k in ((-2, 0.2), (-1, 0.55), (0, 1.0), (1, 0.4), (2, 0.1)):
        curve[20 + j] = curve[20 + j] + (up * 0.13 - u_ / max(math.hypot(*u_), 1e-6) * 0.03) * ln * k
    _, _, nz = ball.rot(_sph(side * 38, 18))
    return dict(curve=curve, poly=curve, G=mid + v_ * 0.4, base=mid, z=(za + zb) / 2,
                nz=float(nz[0]), ln=ln, side=side, T=T, A=A, B=B, u=u_, v=v_)


def _ear_paint(p, e, env):
    if e["nz"] <= 0.12:
        # seen from behind: the dark red brown back
        p.fill(e["poly"], palette.mix(env.c("fur"), env.c("fur_dk"), 0.85))
    else:
        p.fill(e["poly"], env.c("fur"))
        base = e["base"]
        o_, u_, v_ = base, e["u"], e["v"]
        em = p.mask(e["poly"])
        # the broad white front, its lower edge broken into fur where it meets the head
        c1 = base + v_ * 0.10
        arch = c1 + (e["curve"] - c1) * 0.80
        low = [arch[-1] + (arch[0] - arch[-1]) * f_ + v_ * k_ for f_, k_ in ((0.22, -0.07), (0.4, 0.03), (0.58, -0.08), (0.78, 0.02))]
        p.fill(np.vstack([arch, low]), env.c("white"), form=False, clip=em)
        # the hollow, warm pink in light (it turns mauve only inside the shadow shape)
        pink = palette.mix(env.c("ear_in"), env.c("fur"), 0.15)
        c2 = base + v_ * 0.06
        p.fill(c2 + (e["curve"] - c2) * 0.46, pink, form=False, clip=em)
        for a0, a1, b1, wd_ in ((-0.50, -0.18, 0.42, 0.11), (-0.20, 0.10, 0.30, 0.10)):
            r0 = o_ + u_ * a0 - v_ * 0.02
            tp_ = o_ + u_ * a1 + v_ * b1
            wv = u_ * wd_
            md = (r0 + tp_) / 2 - u_ * 0.10
            p.fill(np.vstack([_arc3(r0 - wv, md - wv * 0.8, tp_), _arc3(tp_, md + wv * 0.9, r0 + wv)]),
                   env.c("white"), form=False, clip=em)
    p.ink(e["curve"], env.inkw * 1.8, taper=(0.06, 0.06), lightdir=e["G"])


def _inside(poly, pts):
    """Even odd point in polygon test for an array of points."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x0, y0 = poly[:, 0][None, :], poly[:, 1][None, :]
    x1, y1 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)
    return cond.sum(1) % 2 == 1


# ----------------------------------------------------------------------- head
def _draw_head(cel, env, pose, hc, s, roll=0.0):
    """The head: broad and round, wider than tall, flat on top, full cheeks, a
    small chin. White cheek tufts stick out sideways and are its widest point."""
    yaw = pose.head_yaw * math.pi / 2
    RX = (41.0 - 3.0 * abs(math.sin(yaw))) * s
    RY = 32.0 * s
    ball = _Ball(hc[0], hc[1], RX, RY, yaw, math.radians(pose.pitch * 30), math.radians(pose.tilt) + roll)
    w = env.inkw
    wo = w * 1.9                      # outer contour, about twice the inner lines
    wi = w * 0.8
    box = (hc[0] - RX * 1.8, hc[1] - RY * 2.0, hc[0] + RX * 1.8, hc[1] + RY * 1.6)
    ears = [_ear(ball, -1, pose.ear_l, s), _ear(ball, 1, pose.ear_r, s)]
    far = [e for e in ears if e["z"] < -0.3]
    near = [e for e in ears if e["z"] >= -0.3]
    if far:
        pf = _Part(cel, box, env)
        for e in far:
            _ear_paint(pf, e, env)
        pf.finish(4 * s, env.rimw * 1.4)
    p = _Part(cel, box, env)
    anchors = {}
    if not p.ok:
        anchors["head"] = ball.c.copy()
        return anchors

    # ---- silhouette
    a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    deg = np.degrees(a)
    ck = pose.cheek
    n = np.where(sa > 0, 2.45 + 0.4 * ck, 2.38)
    side_deg = np.where(ca > 0, deg % 360, 180 - (deg % 360))            # angle below the horizontal on either side
    side_deg = (side_deg + 180) % 360 - 180
    full = np.exp(-((side_deg - 26) / 30.0) ** 2)                         # full round cheeks
    lx = RX * np.sign(ca) * np.abs(ca) ** (2 / n) * (1 + (0.055 + 0.10 * ck) * full)
    ly = RY * np.sign(sa) * np.abs(sa) ** (2 / n)
    wd = np.asarray(pose.wind_dir, np.float64) * pose.wind * 0.55

    def tuft(ac, hw, L, lean):
        nonlocal lx, ly
        d = np.abs((deg - ac + 180) % 360 - 180)
        t = np.clip(1 - d / hw, 0, 1) ** 1.1
        lx = lx + s * L * t * (math.cos(math.radians(ac)) + lean[0])
        ly = ly + s * L * t * (math.sin(math.radians(ac)) + lean[1])

    cr = 270 + pose.head_yaw * 22
    tuft(cr - 7, 9.0, 4.2, (0.35 + wd[0], -0.1 + wd[1] * 0.5))
    tuft(cr + 6, 7.0, 2.8, (0.4 + wd[0], 0.0))
    outline = ball.loc(lx, ly)
    fur, white = env.c("fur"), env.c("white")
    p.fill(outline, fur)
    hm = p.mask(outline)
    for e in near:
        _ear_paint(p, e, env)

    # ---- white mask: muzzle and cheeks (no blaze up the nose: that reads as a fox)
    half = [(0, 9), (7, 8), (12, 1), (15, -9), (22, -17), (33, -19), (46, -15), (58, -6), (70, 3), (84, 3),
            (97, -8), (103, -38), (82, -70), (42, -86)]
    pts = half + [(0, -89)] + [(-lo, la) for lo, la in half[::-1]]
    ll = core.catmull(pts, 6, closed=True)
    mp, mz = ball.patch(ll[:, 0], ll[:, 1])
    p.fill(mp, white, form=False, clip=hm)

    # ---- white cheek tufts: soft lobes that stick out sideways at cheek height
    tufts = []
    for sd in (-1, 1):
        o, z = ball.proj(_sph(sd * 78, -19), clamp=True)
        o, z = o[0], float(z[0])
        k = (0.78 if z < 0 else 1.0) * core.clamp((z + 0.56) / 0.25) * (1 + 0.3 * pose.wind + 0.2 * ck)
        if k < 0.08:
            continue
        exv = np.array([ball.cr, ball.sr]) * sd * k
        eyv = np.array([-ball.sr, ball.cr])
        # soft clumps all the way round, shrinking toward the chin: no straight lower edge (it read as a jaw)
        q = np.array([(-8, -12), (6, -12.5), (4, -7.5), (11.5, -5.5), (7, -0.5), (11.5, 4), (6.5, 7), (9, 12.5),
                      (3.5, 12), (4.5, 17.5), (-1.5, 15), (-2.5, 19.5), (-10, 15)], np.float64)
        q[1::2] += wd * 9
        ctrl = o + s * (q[:, :1] * exv + q[:, 1:2] * eyv)
        edge = core.catmull(ctrl, 7, closed=False, tension=0.52)
        p.fill(edge, white)
        tufts.append(edge)

    # ---- tear marks: strong red brown, inner eye corner down and out to the mouth corner
    tear = env.c("fur_dk")
    tears = []
    for sd in (-1, 1):
        cl = core.catmull([(sd * 15, -4), (sd * 23, -16), (sd * 26, -28), (sd * 20, -39), (sd * 11, -46)], 6, closed=False)
        hwd = np.interp(np.linspace(0, 1, len(cl)), [0, 0.35, 0.75, 1], [6.6, 6.2, 4.6, 1.6])
        t = np.gradient(cl, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        nr = np.c_[-t[:, 1], t[:, 0]]
        rb = np.vstack([cl + nr * hwd[:, None], (cl - nr * hwd[:, None])[::-1]])
        tp, tz = ball.patch(rb[:, 0], rb[:, 1])
        if tz.mean() > -0.25:
            tears.append(tp)

    # ---- muzzle: short and small, a low white bump; it barely leaves the silhouette even in profile
    mf = ball.frame(0, -27, 1.03, minx=0.62)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    MRX, MRY = 12.5, 9.4
    muz = _put(mf, np.c_[MRX * np.cos(t), MRY * np.sin(t)], s)
    show_face = mf[3] > -0.28
    if show_face:
        p.fill(muz, white, clip=ImageChops.lighter(ImageChops.invert(hm), p.mask(mp, hm)))
    for tp in tears:
        p.fill(tp, tear, form=False, clip=hm)

    # ---- brows: the white dots do the acting
    for sd, br in ((-1, pose.brow_l), (1, pose.brow_r)):
        lo, la = _ell(sd * 28, 32 + 7 * br, 9.5, 5.2, 24, rot=-sd * (8 + 22 * pose.brow_tilt))
        bp, bz = ball.patch(lo, la)
        if bz.mean() > -0.1:
            p.fill(bp, white, form=False, clip=hm)

    _eyes(p, ball, env, pose, hm, s, anchors)

    # ---- nose and mouth
    lside = -1.0 if env.sun[0] < 0 else 1.0
    if show_face:
        nf = ball.frame(0, -16, 1.13, minx=0.62)
        nose = core.catmull([(-3.5, -2.0), (0, -2.5), (3.5, -2.0), (3.8, -0.3), (1.5, 2.3), (0, 2.6), (-1.5, 2.3), (-3.8, -0.3)], 5)
        p.fill(_put(nf, nose, s), env.c("nose"), form=False)
        nh = _put(nf, np.array([[lside * 1.1 - 0.6, -1.3]]), s)[0]
        p.later(core.ellipse(nh[0], nh[1], 1.1 * s, 0.6 * s, 12), env.lit(palette.mix(P["nose"], (1, 1, 1), 0.6)))
        kf = ball.frame(0, -33, 1.07, minx=0.55)
        op, wd_, sm = pose.mouth_open, pose.mouth_wide, pose.smile
        cw = 4.9 + 2.4 * wd_
        cyv = 0.7 - 2.0 * sm
        nb = _put(nf, np.array([[0, 2.6]]), s)[0]
        ctr = _put(kf, np.array([[0, 0.0]]), s)[0]
        p.ink(np.array([nb, ctr]), wi, taper=(0.1, 0.3), lightdir=False)
        left = core.catmull([(0, 0), (-0.42 * cw, 2.0), (-0.8 * cw, 1.4 + cyv * 0.5), (-cw, cyv)], 6, closed=False)
        right = left * (-1, 1)
        if op > 0.06:
            dp = 2.2 + 8.0 * op
            lowc = core.catmull([(-cw * 0.92, cyv + 0.6), (-cw * 0.62, dp * 0.72), (0, dp), (cw * 0.62, dp * 0.72),
                                 (cw * 0.92, cyv + 0.6)], 6, closed=False)
            mpoly = _put(kf, np.vstack([left[::-1], right[1:], lowc[::-1]]), s)
            mm = p.mask(mpoly)
            p.fill(mpoly, env.lit(MOUTH_IN), form=False)
            if op > 0.3:
                p.fill(_put(kf, core.ellipse(0, dp * 0.98, cw * 0.62, dp * 0.42, 20), s), env.c("tongue"), form=False, clip=mm)
            p.fill(_put(kf, np.array([[-0.62 * cw, 1.5], [-0.34 * cw, 2.0], [-0.47 * cw, 4.2]]), s), white, form=False, clip=mm)
            p.ink(_put(kf, lowc, s), wi, taper=(0.12, 0.12), lightdir=False)
        if pose.tongue > 0.05:
            tq = _put(kf, core.ellipse(1.4, 2.0 + 2.4 * pose.tongue, 2.3, 2.2 + 1.5 * pose.tongue, 18), s)
            p.fill(tq, env.c("tongue"), form=False)
            p.ink(tq, wi * 0.8, closed=True, lightdir=False)
        if pose.head_yaw < 0.62:
            p.ink(_put(kf, right, s), w * 1.0, taper=(0.05, 0.3), lightdir=False)
        if pose.head_yaw > -0.62:
            p.ink(_put(kf, left, s), w * 1.0, taper=(0.05, 0.3), lightdir=False)
        anchors["mouth"] = ctr
        anchors["mouth_l"] = _put(kf, np.array([[-cw * 0.7, 1.4]]), s)[0]
        anchors["mouth_r"] = _put(kf, np.array([[cw * 0.7, 1.4]]), s)[0]
        # whiskers
        for sd in (-1, 1):
            wf = ball.frame(sd * 15, -30, 1.03, minx=0.35)
            if wf[3] < 0.05:
                continue
            for j, (ang, ln_) in enumerate(((-10, 15), (5, 18), (20, 14))):
                r_ = math.radians(ang)
                q = np.array([[sd * 2.0, j * 1.5 - 1.5],
                              [sd * (2 + ln_ * 0.55 * math.cos(r_)), j * 1.5 - 1.5 + ln_ * 0.55 * math.sin(r_) - 0.8],
                              [sd * (2 + ln_ * math.cos(r_)), j * 1.5 - 1.5 + ln_ * math.sin(r_) + 0.6]])
                p.ink(core.catmull(_put(wf, q, s), 6, closed=False), w * 0.45, taper=(0.02, 0.7), lightdir=False)

    if pose.goggles > -0.5:
        _goggles(p, ball, env, pose, hm, s)

    if pose.soot > 0.05:
        soot = env.lit((0.20, 0.17, 0.17))
        for (lo0, la0, ww, hh, rt) in ((-40, -22, 15, 9, 20), (30, 30, 13, 8, -15), (8, -4, 8, 6, 0), (48, -40, 10, 7, 30)):
            lo, la = _ell(lo0, la0, ww, hh, 18, rot=rt)
            nz = core.fbm_at(lo * 0.2, la * 0.2, 1.0, 2, 77) - 0.5
            sp, sz = ball.patch(lo + nz * 6, la + nz * 5)
            if sz.mean() > 0:
                p.fill(sp, soot, form=False, clip=hm, alpha=0.6 * pose.soot)

    # ---- ink: a heavy outer contour, broken where the muzzle and the cheek tufts cross it
    keep = np.ones(len(outline), bool)
    for tf in tufts:
        keep &= ~_inside(tf, outline)
    if show_face:
        o, ex, ey = mf[0], mf[1], mf[2]
        try:
            uv = np.linalg.solve(np.c_[ex, ey] * s, (outline - o).T).T
            keep &= ~((uv[:, 0] / MRX) ** 2 + (uv[:, 1] / MRY) ** 2 < 0.9)
        except np.linalg.LinAlgError:
            pass
        loc = (muz - ball.c) @ np.array([[ball.cr, -ball.sr], [ball.sr, ball.cr]])
        for r in _runs((loc[:, 0] / RX) ** 2 + (loc[:, 1] / RY) ** 2 > 0.97):
            p.ink(muz[r], wo * 0.9, taper=(0.06, 0.06), lightdir=ball.c)
    for r in _runs(keep):
        closed = len(r) == len(outline) + 1
        p.ink(outline[r], wo, taper=(0.0, 0.0) if closed else (0.05, 0.05), lightdir=ball.c)
    for tf in tufts:
        p.ink(tf, wo * 0.85, taper=(0.10, 0.10), lightdir=ball.c)
    p.finish(9.0 * s, env.rimw * 1.5)
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

        if pose.squeeze > 0.5:
            # screwed shut: two heavy strokes that meet toward the nose, like > <
            tipq = np.array([-sd * 0.62 * ew, 0.10 * eh])
            for oy, wk in ((-0.62, 2.9), (0.60, 2.3)):
                o0 = np.array([sd * 0.98 * ew, oy * eh])
                midq = (o0 + tipq) / 2 + np.array([0.0, -oy * 0.16 * eh])
                p.ink(put(_arc3(o0, midq, tipq, 9)), w * wk, taper=(0.25, 0.08), lightdir=False)
            p.ink(put([np.array([sd * 1.1 * ew, -0.95 * eh]), np.array([sd * 1.5 * ew, -1.25 * eh])]), w * 1.1,
                  taper=(0.1, 0.5), lightdir=False)
            continue
        if opn < 0.1:
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
        q = np.array([[-0.16, -0.028], [0.04, -0.028], [0.16, 0.028], [-0.04, 0.028]]) * np.ptp(ip_[:, 0])
        q = np.c_[q[:, 0] + q[:, 1] * 1.2, q[:, 1] * 2.2 - q[:, 0] * 0.7]
        p.later(c0[0] + q, (1, 1, 1), clip=gm)
        p.ink(op_, env.inkw * 1.05, closed=True, taper=(0, 0))
        p.ink(ip_, env.inkw * 0.7, closed=True, taper=(0, 0), lightdir=False)
    lo, la = _ell(0, la0, 5.5 + 3 * g, 3.2, 14)
    bp, _ = ball.proj(_sph(lo, la, rr))
    p.fill(bp, rimc)


# ----------------------------------------------------------------------- body
def _limb(cel, env, ctrl, hw, col, s, toes=0, toe_dir=None, extra=None, d_sh=None, sleeve=0.0, sleeve_col=None,
          pads=0.0):
    poly, c, L, R, t, w = _tube(ctrl, hw, n=34)
    box = _bounds(poly if extra is None else np.vstack([poly, extra]), 6 * s + 4)
    p = _Part(cel, box, env)
    p.fill(poly, col)
    if toes:
        # a small clump of fur at the elbow breaks the line of the arm
        ie = int(0.52 * (len(c) - 1))
        ch = (c[0] + c[-1]) / 2                         # the outside of the bend is the side away from the chord
        side_pts = L if np.hypot(*(L[ie] - ch)) > np.hypot(*(R[ie] - ch)) else R
        od = side_pts[ie] - c[ie]
        od /= max(np.hypot(*od), 1e-6)
        p.fill(core.catmull([side_pts[ie - 4], side_pts[ie - 1] + od * 2.6 * s, side_pts[ie + 2] + od * 1.2 * s,
                             side_pts[ie + 5], c[ie]], 5), col)
    if sleeve > 0:
        # the russet shoulder runs a little way down the arm and ends in tufts
        i = max(2, int(sleeve * (len(c) - 1)))
        zig = np.array([L[i] + (R[i] - L[i]) * f + t[i] * w[i] * k_
                        for f, k_ in ((0.2, 0.9), (0.42, 0.15), (0.62, 1.0), (0.82, 0.2))])
        p.fill(np.vstack([L[:i + 1], zig, R[:i + 1][::-1], poly[-9:]]), sleeve_col, form=False, clip=p.mask(poly))
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
            if pads > 0.5:
                p.fill(core.ellipse(cc[0] + dv[0] * r * 0.06, cc[1] + dv[1] * r * 0.06, r * rr_ * 0.62, r * rr_ * 0.5, 14,
                                    rot=math.atan2(dv[1], dv[0]) + 1.57), env.lit(P["pad"]), form=False)
            else:
                p.ink(np.array([cc - dv * r * 0.1 + pv * r * rr_ * 0.95, cc - dv * r * 0.55 + pv * r * rr_ * 0.7]),
                      env.inkw * 0.7, col=lt, taper=(0.05, 0.6), lightdir=False)
        if pads > 0.5:
            # the palm faces us: one broad pad under the four toe pads
            pc = tip - tv * r * 0.12
            pad = core.catmull([pc + nv * r * 0.52 - tv * r * 0.05, pc + tv * r * 0.36, pc - nv * r * 0.52 - tv * r * 0.05,
                                pc - nv * r * 0.3 - tv * r * 0.42, pc + nv * r * 0.3 - tv * r * 0.42], 6)
            p.fill(pad, env.lit(P["pad"]), form=False)
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
    fur = env.c("fur")
    cream = palette.mix(env.c("fur"), env.c("white"), 0.62)
    tipc = palette.mix(env.c("fur_dk"), env.c("dark"), 0.7)
    p.fill(poly, fur)
    p.fill(tip, tipc)
    tm = p.mask(poly)
    n = len(c)
    for a, b in ((0.12, 0.21), (0.31, 0.40), (0.50, 0.59), (0.69, 0.77), (0.87, 1.0)):
        dk = tipc if b >= 1.0 else cream
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
    p.ink(whole, env.inkw * 1.7, closed=True, taper=(0, 0))
    p.finish(5.5 * s)
    return c[-1]


def _default_tail(pose):
    yr = pose.yaw * math.pi / 2
    sy, cy = math.sin(yr), math.cos(yr)
    # three shapes (front, side, back view) mixed by the turn, so the tail never pops: as the body
    # turns through the front the tail swings across behind it
    d = core.clamp(1 - 10 * pose.yaw, -1, 1)
    front = np.array([(d * 14, 32), (d * 36, 22), (d * 55, 30), (d * 64, 52), (d * 58, 76), (d * 46, 90)], np.float64)
    ds = -1.0 if sy > 0 else 1.0
    side = np.array([(ds * 20, 34), (ds * 42, 26), (ds * 62, 30), (ds * 74, 48), (ds * 72, 70), (ds * 62, 86)], np.float64)
    back = np.array([(-ds * 2, 36), (-ds * 8, 20), (-ds * 26, 9), (-ds * 48, 12), (-ds * 60, 30), (-ds * 58, 50)], np.float64)
    ws = core.smooth((abs(sy) - 0.38) / 0.24)
    wb = core.smooth((-cy - 0.15) / 0.3)
    out = (front * (1 - ws) + side * ws) * (1 - wb) + back * wb
    return tuple((float(q[0]), float(q[1])) for q in out)


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
    torso = _Ball(*M(0, 60 + 1.2 * br), (29 + 1.3 * br) * s * sxk, 39 * s * sq, yr, 0.0, ln)
    hc = M(pose.head_dx, 125 + pose.head_dy + 1.6 * br)
    dark = env.c("dark")
    anchors = {}
    front = cyw > -0.25
    env.set_smear(pose.smear, hc)

    # tail (the default tail rides with the body offset)
    if pose.tail is not None:
        tctrl = np.array([G(*q) for q in pose.tail])
    else:
        tctrl = np.array([G(q[0] + pose.dx, q[1] + pose.dy) for q in _default_tail(pose)])
    if front:
        anchors["tail_tip"] = _draw_tail(cel, env, tctrl, s, pose.wind)

    # legs and feet
    for sd, ft in ((-1, pose.foot_l), (1, pose.foot_r)):
        hip = M(sd * 10 * cyw + 0, 33)
        if ft is None:
            ft = (sd * (14 * abs(cyw) + 4 * abs(syw)) * (1 if cyw >= 0 else -1), 0.0)
        f = G(ft[0], ft[1])
        ank = f + np.array([0, -6.5 * s])
        # a round thigh that tapers to the ankle, never a straight slab
        mid = (hip + ank) / 2 + np.array([sd * 3.2 * s * cyw + syw * 2.0 * s, -1.0 * s])
        foot = core.ellipse(f[0] + (syw * 5.5 + sd * 2.0 * cyw) * s, f[1] - 5.2 * s, (12.0 + 3.0 * abs(syw)) * s, 6.6 * s, 26)
        _limb(cel, env, [hip, mid, ank], np.array([11.5, 10.5, 8.0, 6.2]) * s, dark, s, extra=foot)

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
        h = max(math.sqrt(max(L1 * L1 - (dist / 2) ** 2, 0.0)), 0.20 * dist)     # even a stretched arm keeps its elbow
        nrm = np.array([-d[1], d[0]]) / max(dist, 1e-6)
        if nrm @ np.array([sd * (1 if front else -1), 0.45]) < 0:
            nrm = -nrm
        E = (S + Pw) / 2 + nrm * h
        # a real elbow: the upper arm and the forearm are nearly straight, the bend is tight
        ctrl = [S, S + (E - S) * 0.72, E, E + (Pw - E) * 0.30, Pw]
        arms.append((z, sd, ctrl))
        anchors["paw_l" if sd < 0 else "paw_r"] = Pw
    hw_arm = np.array([8.4, 7.8, 7.4, 5.6, 5.0, 9.6]) * s
    for z, sd, ctrl in arms:
        if z < -0.05:
            _limb(cel, env, ctrl, hw_arm, dark, s, toes=4, sleeve=0.27, sleeve_col=env.c("fur"))

    # torso
    a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    # a pear: narrow shoulders, heavy hips, the belly pushed toward where it faces
    lx = torso.rx * ca * (1.04 + 0.30 * sa - 0.10 * sa * sa) + syw * torso.rx * 0.14 * np.maximum(sa + 0.2, 0)
    ly = torso.ry * sa
    # soft fur bumps at the hips and the flanks break the outline (round, never a hem)
    for ac, hw_, L_ in ((30, 20, 2.6), (58, 16, 2.0), (148, 22, 3.0), (118, 15, 1.8), (200, 18, 1.6), (338, 18, 1.8)):
        d_ = np.abs((np.degrees(a) - ac + 180) % 360 - 180)
        t_ = np.cos(np.clip(d_ / hw_, 0, 1) * np.pi / 2) ** 2
        lx = lx + s * L_ * t_ * math.cos(math.radians(ac))
        ly = ly + s * L_ * t_ * math.sin(math.radians(ac))
    tout = torso.loc(lx, ly)
    p = _Part(cel, _bounds(tout, 8 * s + 4), env)
    p.fill(tout, env.c("fur"))
    tm = p.mask(tout)
    # the dark belly wraps the form; its edge is tufted, not an ellipse
    tb = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    saw = np.abs(((tb * 8 / (2 * np.pi) + 0.35 * np.sin(tb * 3)) % 1.0) - 0.5) * 2
    rad = 1.0 + 0.08 * saw * (0.4 + 0.6 * np.abs(np.cos(tb)))
    lo = 42 * np.cos(tb) * rad
    la = -12 + 74 * np.sin(tb) * rad
    lo = lo * (0.78 + 0.22 * np.clip((55 - la) / 100, 0, 1))
    bp, bz = torso.patch(lo, la)
    if bz.mean() > -0.2:
        p.fill(bp, dark, form=False, clip=tm)
    # the chest ruff: fur falling from under the chin over the top of the belly
    rf = [(-60, 62), (-54, 45), (-45, 49), (-37, 34), (-26, 41), (-12, 23), (1, 35), (13, 27), (24, 41), (35, 35),
          (45, 48), (54, 45), (62, 62), (30, 86), (-30, 86)]
    rp, rz = torso.patch([q[0] for q in rf], [q[1] for q in rf])
    if False and rz.mean() > -0.15:      # the chest is dark like the belly; the russet collar is the neck ruff part
        p.fill(rp, palette.mix(env.c("fur"), env.c("white"), 0.10), form=False, clip=tm)
        p.ink(rp[1:12], env.inkw * 0.8, taper=(0.1, 0.1), lightdir=False, clip=tm)
    # designed shadows: the head and cheek ruff on the chest and shoulders, each near arm on the body
    off = np.array([-env.sun[0] * 7 * s, 11 * s - env.sun[1] * 3 * s])
    p.shadow(core.ellipse(hc[0] + off[0], hc[1] + off[1], 43 * s, 33 * s, 40))
    for z, sd, ctrl in arms:
        if z >= -0.05:
            ap = _tube(ctrl, hw_arm * 1.12, n=30)[0]
            p.shadow(ap + np.array([-env.sun[0] * 4.5 * s, 3.0 * s - env.sun[1] * 2.5 * s]))
    p.ink(tout, env.inkw * 1.8, closed=True, taper=(0, 0))
    p.finish(6.0 * s)

    if not front:
        anchors["tail_tip"] = _draw_tail(cel, env, tctrl, s, pose.wind)

    # the neck ruff: a collar of fur between the shoulders and the head, so the head sits ON the body
    nt = M(0, 94)
    nc = nt * 0.45 + (hc + np.array([0, 27 * s])) * 0.55
    na = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    nx_ = 25 * s * sxk * np.cos(na)
    ny_ = 12 * s * np.sin(na) * np.where(np.sin(na) > 0, 1.2, 0.8)
    for ac, hw_, L_ in ((20, 26, 3.4), (66, 24, 4.6), (112, 22, 3.6), (158, 26, 4.2)):
        d_ = np.abs((np.degrees(na) - ac + 180) % 360 - 180)
        t_ = np.cos(np.clip(d_ / hw_, 0, 1) * np.pi / 2) ** 2          # round clumps of fur, not a zigzag hem
        nx_ = nx_ + s * L_ * t_ * math.cos(math.radians(ac)) * 0.8
        ny_ = ny_ + s * L_ * t_ * 0.9
    neck = np.c_[nc[0] + nx_ + syw * 3 * s, nc[1] + ny_]
    pn = _Part(cel, _bounds(neck, 6 * s + 4), env)
    pn.fill(neck, env.c("fur"))
    pn.shadow(core.ellipse(hc[0] - env.sun[0] * 7 * s, hc[1] + 12 * s, 43 * s, 33 * s, 40))
    pn.ink(neck[2:39], env.inkw, taper=(0.12, 0.12))
    pn.finish(4.0 * s)

    anchors.update(_draw_head(cel, env, pose, hc, s, roll=ln * 0.6))

    for z, sd, ctrl in arms:
        if z >= -0.05:
            _limb(cel, env, ctrl, hw_arm, dark, s, toes=4, sleeve=0.27, sleeve_col=env.c("fur"),
                  pads=pose.palm_l if sd < 0 else pose.palm_r)
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


def mirror(p):
    """The same pose facing the other way (screen left and right swapped)."""
    def fx(t):
        return None if t is None else (-t[0], t[1])

    return replace(p, yaw=-p.yaw, head_yaw=-p.head_yaw, tilt=-p.tilt, lean=-p.lean, dx=-p.dx, head_dx=-p.head_dx,
                   eye_l=p.eye_r, eye_r=p.eye_l, brow_l=p.brow_r, brow_r=p.brow_l, ear_l=p.ear_r, ear_r=p.ear_l,
                   paw_l=fx(p.paw_r), paw_r=fx(p.paw_l), foot_l=fx(p.foot_r), foot_r=fx(p.foot_l),
                   palm_l=p.palm_r, palm_r=p.palm_l, pupil=(-p.pupil[0], p.pupil[1]),
                   tail=None if p.tail is None else tuple((-q[0], q[1]) for q in p.tail),
                   wind_dir=(-p.wind_dir[0], p.wind_dir[1]), smear=(-p.smear[0], p.smear[1]),
                   goggles_askew=-p.goggles_askew)


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
    "tie_knot": Pose(yaw=0.38, head_yaw=0.36, pitch=-0.12, tilt=-5, lean=-5, head_dx=5, paw_l=(27, 69), paw_r=(45, 83),
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
    "tail_curl": Pose(tilt=-5, smile=0.9, eye_smile=0.3, brow_l=0.3, brow_r=0.3,
                      tail=((14, 32), (38, 24), (58, 36), (62, 60), (46, 72), (36, 58))),
    # walking, facing screen right (use mirror() for the other way). a, b, c, b, a ... is a forward walk;
    # play it c, b, a while moving left for the backward steps of shot 2.
    "step_a": Pose(yaw=0.8, head_yaw=0.62, lean=3, dy=-1.5, foot_l=(-11, 0), foot_r=(11, 0), paw_l=(13, 57), paw_r=(-9, 55)),
    "step_b": Pose(yaw=0.8, head_yaw=0.62, lean=1, dy=1.8, foot_l=(1, 6), foot_r=(0, 0), paw_l=(3, 53), paw_r=(2, 53)),
    "step_c": Pose(yaw=0.8, head_yaw=0.62, lean=3, dy=-1.5, foot_l=(11, 0), foot_r=(-11, 0), paw_l=(-9, 55), paw_r=(13, 57)),
    "look_up_soft": Pose(pitch=0.62, head_yaw=-0.25, yaw=-0.15, smile=0.5, brow_l=0.5, brow_r=0.5, pupil=(-0.2, 0.6),
                         mouth_open=0.12, ear_l=-0.1, ear_r=-0.1),
    # the star map is on screen left of the panda in the film: these face left
    "read_map": Pose(yaw=-0.6, head_yaw=-0.62, lean=-9, pitch=-0.1, paw_l=(-33, 84), brow_l=0.2, brow_r=0.2,
                     pupil=(-0.4, 0.1), smile=0.3),
    "map_tap": Pose(yaw=-0.6, head_yaw=-0.62, lean=-11, pitch=-0.05, paw_l=(-41, 93), brow_l=0.4, brow_r=0.4,
                    pupil=(-0.5, 0.3), smile=0.5, mouth_open=0.12),
    "paint": Pose(yaw=-0.6, head_yaw=-0.6, lean=-13, pitch=-0.2, paw_l=(-40, 84), tongue=0.9, brow_tilt=-0.35,
                  brow_l=-0.2, brow_r=-0.2, pupil=(-0.5, -0.2), smile=0.2, eye_l=0.85, eye_r=0.85),
    # waking and climbing (facing right, toward the rocket)
    "wake_sit": Pose(dy=-9, squash=0.9, eye_l=0.35, eye_r=0.3, ear_l=0.45, ear_r=0.4, smile=0.0, tilt=7, yaw=0.3,
                     head_yaw=0.2, paw_l=(-20, 40), paw_r=(22, 42), foot_l=(-17, 0), foot_r=(19, 0)),
    "yawn": Pose(squash=1.12, pitch=0.55, mouth_open=1.0, mouth_wide=0.1, smile=0.0, eye_l=0.0, eye_r=0.0, ear_l=0.35,
                 ear_r=0.35, brow_l=0.6, brow_r=0.6, paw_l=(-31, 158), paw_r=(31, 160), palm_l=1.0, palm_r=1.0, lean=-3,
                 yaw=0.2, head_yaw=0.15),
    "climb_reach": Pose(yaw=0.9, head_yaw=0.8, pitch=0.6, squash=1.1, paw_r=(27, 152), paw_l=(20, 122), foot_r=(9, 13),
                        brow_l=0.4, brow_r=0.4, mouth_open=0.15),
    "climb_pull": Pose(yaw=0.9, head_yaw=0.8, pitch=0.3, squash=0.9, paw_r=(27, 128), paw_l=(22, 118), foot_l=(-5, 8),
                       foot_r=(10, 17), brow_tilt=-0.4, smile=0.0, eye_l=0.8, eye_r=0.8),
    "climb_over": Pose(yaw=1.0, head_yaw=0.9, lean=32, pitch=-0.3, paw_r=(36, 92), paw_l=(30, 98), foot_l=(-9, 11),
                       foot_r=(2, 5), ear_l=0.3, ear_r=0.3),
    # in the cockpit: busts (the cup hides everything below the shoulders)
    "cockpit": Pose(paw_l=(-25, 90), paw_r=(25, 90), smile=0.2),
    "goggles_grab": Pose(paw_l=(-33, 150), paw_r=(33, 150), smile=0.1, goggles=0.0, brow_l=0.3, brow_r=0.3, pitch=0.1),
    "goggles_down": Pose(paw_l=(-36, 127), paw_r=(36, 127), smile=0.1, goggles=1.0, pitch=-0.05),
    "swallow": Pose(paw_l=(-25, 90), paw_r=(25, 90), goggles=1.0, brow_tilt=1.0, brow_l=0.5, brow_r=0.5, pupil_size=0.75,
                    smile=-0.5, mouth_wide=-0.5, ear_l=0.5, ear_r=0.5, head_dy=-2.0, cheek=0.25, pitch=-0.1),
    "nod_down": Pose(paw_l=(-25, 90), paw_r=(25, 90), goggles=1.0, brow_tilt=-1.0, brow_l=-0.6, brow_r=-0.6, eye_l=0.8,
                     eye_r=0.8, smile=-0.1, mouth_wide=-0.3, pitch=-0.5, head_dy=-2.5),
    "climb_out": Pose(paw_l=(-25, 90), paw_r=(25, 90), soot=1.0, goggles=0.0, goggles_askew=16, tilt=8, eye_l=1.2,
                      eye_r=1.2, pupil_size=0.8, brow_l=0.6, brow_r=0.6, mouth_open=0.2, smile=0.2, ear_l=0.3, ear_r=-0.2),
    "ear_twitch": Pose(curl=1.0, head_yaw=-0.18, tilt=-10, eye_l=0.0, eye_r=0.0, smile=0.2, ear_l=0.7, ear_r=-0.55,
                       brow_l=-0.3, brow_r=-0.3, blanket=1.0),
}


# --------------------------------------------------- overlap and follow through
def _spring(target, f_hz, zeta, fps=24.0, sub=3):
    """Follow a sampled target (frames, k) with a damped spring; the follower at the last sample."""
    x = target[0].copy()
    v = np.zeros_like(x)
    w = 2 * math.pi * f_hz
    dt = 1.0 / fps / sub
    for T in target[1:]:
        for _ in range(sub):
            v += (w * w * (T - x) - 2 * zeta * w * v) * dt
            x += v * dt
    return x


def follow(pose_fn, fl, back=16, ears=1.0, tail=1.0):
    """Overlap and follow through for the ears and the tail.

    pose_fn(frame) -> Pose is your key animation WITHOUT secondary motion (it
    must work for any frame number, also negative ones: hold the first pose).
    Returns the pose for frame `fl` with the ears and the tail trailing the
    head and the body by a few frames and overshooting when they stop. It is
    a spring chain run over the last `back` frames, so any frame can be
    rendered alone and always comes out the same."""
    fr = [pose_fn(k) for k in range(fl - back, fl + 1)]
    cur = fr[-1]
    if cur.curl > 0.5:
        return cur

    def head_drv(p):
        return np.array([p.dx + p.head_dx + 100 * math.sin(math.radians(p.lean)) + 14 * p.head_yaw + 0.25 * p.tilt,
                         p.dy + p.head_dy + 9 * p.pitch + 60 * (p.squash - 1)])

    def body_drv(p):
        return np.array([p.dx - 30 * math.sin(math.radians(p.lean)) - 16 * p.yaw, p.dy + 20 * (p.squash - 1)])

    H = np.array([head_drv(p) for p in fr])
    B = np.array([body_drv(p) for p in fr])
    lag = H[-1] - _spring(H, 3.4, 0.30)
    add = (lag[1] * 0.06 + abs(lag[0]) * 0.03) * ears
    out = replace(cur, ear_l=core.clamp(cur.ear_l + add - lag[0] * 0.02 * ears, -0.6, 1.2),
                  ear_r=core.clamp(cur.ear_r + add + lag[0] * 0.02 * ears, -0.6, 1.2))
    if tail > 0:
        tl = np.array(cur.tail if cur.tail is not None else [(q[0] + cur.dx, q[1] + cur.dy) for q in _default_tail(cur)],
                      np.float64)
        l1 = B[-1] - _spring(B, 2.2, 0.38)
        l2 = B[-1] - _spring(B, 1.4, 0.30)
        wgt = np.linspace(0, 1, len(tl)) ** 1.3
        tl = tl - (l1[None, :] * (1 - wgt[:, None]) + l2[None, :] * wgt[:, None]) * wgt[:, None] * 1.6 * tail
        out = replace(out, tail=tuple((float(q[0]), float(q[1])) for q in tl))
    return out


def pose_at(keys, t, ease=core.smooth):
    """Animate through key poses: keys = [(time, Pose), ...] or [(time, Pose, ease), ...]
    sorted by time (the ease is used arriving INTO that key; core.overshoot gives
    a settle). Holds before the first key and after the last."""
    if t <= keys[0][0]:
        return keys[0][1]
    for i in range(1, len(keys)):
        if t <= keys[i][0]:
            t0, t1 = keys[i - 1][0], keys[i][0]
            e = keys[i][2] if len(keys[i]) > 2 else ease
            return blend(keys[i - 1][1], keys[i][1], e((t - t0) / (t1 - t0)) if t1 > t0 else 1.0)
    return keys[-1][1]


# ------------------------------------------------------------ shot 9c: the paw
def draw_paw_closeup(cel, x, y, s=1.0, tod="morning", drawing=0, paint=1.0, press=0.0):
    """A forearm and paw reaching in from the lower right. One toe is extended
    and its tip touches (x, y); it carries a dab of gold paint (paint 0..1).
    press 0..1 flattens the toe tip on the paper. About 620 px long at s = 1."""
    env = _Env(tod, 3.4 * s, drawing, None)
    env.pad = int(30 * s) + 8
    dark = env.c("dark")
    pad = env.lit(P["pad"])
    d = np.array([-0.80, -0.60])
    nv = np.array([0.60, -0.80])
    P0 = np.array([x, y], np.float64)

    def Lp(a, b):
        return P0 - d * a * s + nv * b * s

    p = _Part(cel, _bounds(np.array([Lp(-30, -170), Lp(-30, 170), Lp(700, -170), Lp(700, 170)]), 10), env)
    if not p.ok:
        return
    w = env.inkw
    arm, c, Lf, Rt, t, wv = _tube([Lp(660, 34), Lp(450, 16), Lp(270, 4), Lp(170, 0)], np.array([76, 68, 58, 64]) * s, n=40)
    p.fill(arm, dark)
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    palm = np.array([Lp(142 + 92 * math.cos(q), 76 * math.sin(q) * (1 + 0.1 * math.cos(q))) for q in th])
    p.fill(palm, dark)
    toes = []
    for a_, b_, r_ in ((66, 60, 31), (86, -64, 29), (50, -34, 30)):
        tp = core.ellipse(*Lp(a_, b_), r_ * s, r_ * s * 0.92, 24)
        p.fill(tp, dark)
        toes.append((tp, Lp(a_, b_), r_))
    k = 1 + 0.35 * press
    toe, tc, tL, tR, tt, tw = _tube([Lp(104, 16), Lp(58, 9), Lp(17 + 3 * press, 2)], np.array([31, 27, 17.5 * k]) * s, n=24)
    p.fill(toe, dark)
    # pads: under the toe tip, along the lower edge of the palm, on the curled toes
    p.fill(core.ellipse(*Lp(20, -9), 15 * s * k, 8 * s, 18, rot=math.atan2(d[1], d[0])), pad, form=False, clip=p.mask(toe))
    p.fill(core.ellipse(*Lp(132, -60), 50 * s, 17 * s, 22, rot=math.atan2(d[1], d[0])), pad, form=False, clip=p.mask(palm))
    for tp, cc, r_ in toes[1:]:
        p.fill(core.ellipse(cc[0] - nv[0] * r_ * 0.62 * s, cc[1] - nv[1] * r_ * 0.62 * s, r_ * 0.6 * s, r_ * 0.3 * s, 16,
                            rot=math.atan2(d[1], d[0])), pad, form=False, clip=p.mask(tp))
    # the dab of gold paint
    if paint > 0.02:
        r = core.rng("paintdab")
        ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        rad = (11 + 6 * paint) * s * (0.8 + 0.4 * r.random(14))
        c0 = Lp(7, -4)
        blob = core.catmull(np.c_[c0[0] + np.cos(ang) * rad, c0[1] + np.sin(ang) * rad * 0.8], 5)
        gold = env.lit(palette.ROCKET["star"])
        p.fill(blob, gold, form=False)
        p.later(core.ellipse(c0[0] - 3 * s, c0[1] - 3 * s, 3.2 * s, 2.0 * s, 10), (1.0, 0.97, 0.8))
        p.ink(blob, w * 0.6, col=env.lit((0.70, 0.48, 0.14)), closed=True, lightdir=False)
    cen = Lp(150, 0)
    p.ink(Lf, w * 1.8, taper=(0.02, 0.12), lightdir=cen)
    p.ink(Rt, w * 1.8, taper=(0.02, 0.12), lightdir=cen)
    p.ink(palm[np.r_[30:48, 0:8]], w * 1.8, taper=(0.1, 0.1), lightdir=cen)
    for tp, cc, r_ in toes:
        p.ink(tp[np.r_[14:24, 0:6]], w * 1.5, taper=(0.15, 0.15), lightdir=cen)
    p.ink(np.vstack([tL, toe[24:33], tR[::-1]]), w * 1.8, taper=(0.1, 0.1), lightdir=cen)
    lt = palette.mix(dark, pad, 0.6)
    for a_, b_ in ((420, 30), (350, -20), (300, 36), (230, -30), (200, 30)):
        p.ink(np.array([Lp(a_, b_), Lp(a_ - 34, b_ + 4)]), w * 0.7, col=lt, taper=(0.1, 0.6), lightdir=False)
    p.finish(20.0 * s, env.rimw * 1.5)
