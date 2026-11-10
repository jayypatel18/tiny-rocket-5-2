"""Shot 5a, the match (28.5 to 30.5 s, predawn). A dark frame: deep blue before
dawn, the panda a cool silhouette with a thin rim, looking over the dark edge of
the teacup. The doubt: it looks at the match, at the fuse, takes a small breath,
commits. The strike: the flare blooms and the tiny flame becomes the ONLY warm
light in the frame, on the paw, the muzzle, the whiskers, the cheek tufts, the
eyes, falling off fast. The match goes to the fuse; it catches and the fizz
runs down out of frame while the eyes follow it.

The panda is drawn twice per frame from the same pose and drawing number, once
in the dark light and once in the flame light, and the two cels are mixed by
distance from the flame: real local light on a flat cel."""
import math

import numpy as np

from .. import core, palette, panda
from .. import panda_shot as ps
from ..timeline import FPS
from . import common

NAME = "s5a_match"
TOD = "predawn"
K = panda.POSES
T_STRIKE = common.cue("match_strike", NAME)
T_LIT = common.cue("fuse_lit", NAME)
S = 7.8
HX, HY = 985.0, 440.0             # head centre on screen
RIM_Y = 935.0
FUSE = np.array([640.0, RIM_Y - 34.0])

DARK = dict(palette.TOD[TOD])
DARK.update(key=np.array([0.26, 0.34, 0.58], np.float32), amb=np.array([0.13, 0.17, 0.34], np.float32),
            rim=np.array([0.50, 0.70, 1.0], np.float32), ink=np.array([0.03, 0.04, 0.10], np.float32))
WARM = dict(palette.TOD[TOD])
WARM.update(key=np.array([1.22, 0.94, 0.62], np.float32), amb=np.array([0.50, 0.28, 0.30], np.float32),
            rim=np.array([1.0, 0.90, 0.62], np.float32), ink=np.array([0.26, 0.10, 0.08], np.float32), sun=(0.55, 0.83))

B = panda.replace(K["nervous"], goggles=1.0, paw_l=(-27, 60), yaw=0.08)
LOOK_M = panda.replace(B, paw_r=(31, 92), head_yaw=0.2, pitch=-0.12, pupil=(0.6, -0.35), tilt=3, ear_l=0.55, ear_r=0.5)
LOOK_F = panda.replace(LOOK_M, head_yaw=-0.12, pitch=-0.2, pupil=(-0.7, -0.6), tilt=-3, ear_l=0.7, ear_r=0.6,
                       paw_r=(30, 90), head_dy=-1.0)
BREATH = panda.replace(LOOK_M, breath=1.0, squash=1.035, pupil=(0.5, -0.3), head_yaw=0.15, brow_tilt=-0.5, brow_l=-0.3,
                       brow_r=-0.3, ear_l=0.25, ear_r=0.25, pupil_size=0.9, smile=-0.2, paw_r=(33, 90))
ANT = panda.replace(BREATH, paw_r=(47, 70), lean=2, eye_l=0.82, eye_r=0.82, squash=0.975, breath=0.6, pitch=-0.2)
STRIKE = panda.replace(BREATH, paw_r=(11, 72), lean=-3, pitch=0.06, eye_l=0.45, eye_r=0.45, ear_l=0.9, ear_r=0.9,
                       pupil=(0.1, -0.5), squash=1.0)
AWE = panda.replace(LOOK_M, paw_r=(27, 95), eye_l=1.4, eye_r=1.4, hl=1.7, brow_l=0.85, brow_r=0.85, brow_tilt=0.2,
                    pupil=(0.32, -0.25), mouth_open=0.28, mouth_wide=-0.45, smile=0.1, ear_l=0.05, ear_r=0.05,
                    head_yaw=0.15, pupil_size=1.05, tilt=4)
CARRY = panda.replace(AWE, paw_r=(-22, 76), head_yaw=-0.3, yaw=-0.08, pitch=-0.36, pupil=(-0.45, -0.55), eye_l=1.0,
                      eye_r=1.0, mouth_open=0.0, smile=-0.3, brow_tilt=0.7, tilt=-4, lean=-4, hl=1.4)
FLINCH = panda.replace(CARRY, paw_r=(-12, 86), lean=2, pitch=-0.1, eye_l=1.3, eye_r=1.3, ear_l=0.8, ear_r=0.8,
                       squash=1.03, pupil=(-0.6, -0.6), pupil_size=0.75)
FOLLOW = panda.replace(FLINCH, pitch=-0.42, pupil=(-0.8, -0.85), squash=1.0, ear_l=0.5, ear_r=0.5, tilt=-7, lean=-3)
fast = lambda u: core.ease_out(u, 3.0)      # noqa: E731
KEYS = [(0.0, LOOK_M), (0.10, LOOK_M), (0.19, LOOK_F, fast), (0.27, LOOK_F), (0.37, BREATH), (T_STRIKE - 0.07, ANT),
        (T_STRIKE + 0.06, STRIKE, fast), (T_STRIKE + 0.30, AWE, core.overshoot), (1.04, AWE), (T_LIT - 0.10, CARRY),
        (T_LIT + 0.08, FLINCH, fast), (T_LIT + 0.40, FOLLOW), (9.0, FOLLOW)]
UP, SCRAPE, TOFUSE = np.array([-0.28, -0.96]), np.array([-0.55, 0.83]), np.array([-0.85, 0.53])
MDIR = [(0.0, UP), (T_STRIKE - 0.16, UP), (T_STRIKE - 0.06, SCRAPE), (T_STRIKE + 0.07, SCRAPE), (T_STRIKE + 0.26, UP),
        (1.06, UP), (T_LIT - 0.12, TOFUSE), (9.0, TOFUSE)]


def pose_fn(k):
    return panda.pose_at(KEYS, max(k, 0) / FPS)


def _match(cel, paw, mdir, s, T, burnt):
    head = paw + mdir * 22.0 * s
    stick = np.array([paw - mdir * 4.0 * s, head])
    cel.stroke(stick, 1.9 * s, T["ink"], taper=(0, 0))
    cel.stroke(stick, 1.15 * s, palette.lit((0.86, 0.74, 0.52), T), taper=(0, 0))
    cel.fill(core.ellipse(head[0], head[1], 1.5 * s, 2.0 * s, 16, rot=math.atan2(mdir[1], mdir[0]) + 1.57),
             (0.10, 0.05, 0.05) if burnt else palette.lit((0.72, 0.22, 0.18), T))
    return head


def _flame(cel, head, s, level, flare, key):
    """A living flame: orange body, yellow heart, white core; every drawing differs."""
    r = core.rng("flame5a", key)
    hgt = (12.0 + 9.0 * flare + r.uniform(-1.4, 1.4)) * s * (0.6 + 0.4 * level)
    lean = r.uniform(-0.9, 0.9) * s
    for wdt, hh, col in ((6.6, 1.0, (1.0, 0.50, 0.14)), (4.4, 0.74, (1.0, 0.80, 0.32)), (2.3, 0.46, (1.0, 0.98, 0.86))):
        w_ = wdt * s * (1 + 0.5 * flare)
        j = r.uniform(-0.4, 0.4, 4) * s
        cel.fill(core.catmull([head + (-w_ * 0.5, 0.8 * s), head + (-w_ * 0.50 + j[0], -hgt * hh * 0.42),
                               head + (-w_ * 0.16 + lean * 0.6 + j[1], -hgt * hh * 0.80), head + (lean, -hgt * hh),
                               head + (w_ * 0.20 + lean * 0.5 + j[2], -hgt * hh * 0.74),
                               head + (w_ * 0.50 + j[3], -hgt * hh * 0.38), head + (w_ * 0.5, 0.8 * s),
                               head + (0, 2.2 * s)], 6), col)
    return head + np.array([lean * 0.4, -hgt * 0.4])


def _lmask(c, radius, level):
    xx, yy = ps.grid()
    return np.clip(level * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (radius * radius)), 0, 1).astype(np.float32)


def render(fl, n, f):
    t = fl / FPS
    ones = T_STRIKE - 0.15 <= t < T_STRIKE + 0.4 or T_LIT - 0.1 <= t < T_LIT + 0.3
    k = fl if ones else common.twos(fl)
    tk = k / FPS
    pose = panda.follow(pose_fn, k)
    if abs(k - round(T_STRIKE * FPS)) < 0.5:
        pose = panda.replace(pose, smear=(-7.0, 0.0))
    lit = tk >= T_STRIKE + 0.04
    flare = math.exp(-max(tk - T_STRIKE - 0.04, 0) * 8.0) if lit else 0.0
    level = (0.74 + 0.12 * core.noise1(t * 12.0, 3) + 0.06 * math.sin(fl * 2.4)) if lit else 0.0     # flickers on ones
    z = 1.0 + 0.05 * core.ease_io(core.span(t, 0.0, n / FPS), 1.5)
    dx, dy = common.drift(t, 2.5, 6)
    s = S * z
    px, py = HX + dx, HY + dy + 126 * s

    # the world: deep blue before dawn, far out of focus
    base = ps.soft_bg("s5a_bg_v1", TOD, cx=1500, cy=930, zoom=3.6, sigma=18.0)
    pic = core.warp(base, 1.02 * z, 960 * (1 - 1.02 * z) + dx * 0.4, 540 * (1 - 1.02 * z) + dy * 0.4)
    pic *= np.float32((0.22, 0.27, 0.42))
    pic += np.float32((0.004, 0.008, 0.028))

    # the panda and the match, once in the dark and once in the light of the flame
    mdir = core.keys(tk, MDIR)
    mdir = mdir / np.hypot(*mdir)
    layers = []
    for T in (DARK, WARM) if lit else (DARK,):
        cel = core.Cel(ss=2)
        an = panda.draw(cel, pose, px, py, s, T, drawing=k)
        head = _match(cel, an["paw_r"], mdir, s, T, lit and tk > T_STRIKE + 0.3)
        layers.append(cel.rgba())
    layers[0][..., :3] *= np.float32((0.52, 0.58, 0.78))      # the unlit panda sinks to a cool silhouette
    layer = layers[0]
    fc = head + np.array([0.0, -6.0 * s])
    m = None
    if lit:
        m = _lmask(fc, (440.0 + 380.0 * flare) * z, min(1.0, 1.25 * level + flare))
    burn = core.span(tk, T_LIT, T_LIT + 0.36)
    fz = np.array([FUSE + (dx, dy), FUSE + (-30 + dx, 18 + dy), FUSE + (-46 + dx, 90 + dy), FUSE + (-36 + dx, 260 + dy)])
    fuse = core.resample(core.catmull(fz, 12, closed=False), 60)
    ib = int(burn * 59)
    sparking = tk >= T_LIT and burn < 1.0
    if sparking:
        m = np.maximum(m, _lmask(fuse[ib], 240.0, 0.9))
    if m is not None:
        layer = layers[0] * (1 - m[..., None]) + layers[1] * m[..., None]
    core.over(pic, layer)

    # the cup rim: a dark curved edge; the fuse end hangs over it
    T = DARK
    cel = core.Cel(ss=2)
    xs = np.linspace(-80, core.W + 80, 60)
    top = np.c_[xs, RIM_Y + dy + 52 * ((xs - 960) / 1040.0) ** 2]
    cel.fill(np.vstack([top, [(core.W + 80, core.H + 40), (-80, core.H + 40)]]), (0.035, 0.05, 0.115))
    cel.fill(np.vstack([top + (0, 44), (top + (0, 80))[::-1]]), (0.03, 0.055, 0.16))
    cel.stroke(top, 5.0, (0.10, 0.15, 0.30), taper=(0, 0))
    cel.stroke(fuse, 9.0, T["ink"], taper=(0, 0))
    cel.stroke(fuse, 5.5, (0.20, 0.22, 0.30), taper=(0, 0))
    if ib > 1:
        cel.stroke(fuse[:ib + 1], 6.0, (0.02, 0.02, 0.03), taper=(0, 0))
    core.over(pic, cel.rgba())

    if lit or sparking:
        # what the flame touches outside the cel: the rim edge, the whiskers, the fuse
        lay = core.Cel(ss=1)
        lay.stroke(top + (0, 3), 9.0, (1.0, 0.72, 0.40), taper=(0, 0))
        lay.stroke(fuse[ib:] if sparking else fuse, 4.0, (0.95, 0.80, 0.55), taper=(0, 0))
        for nm, sd in (("mouth_l", -1), ("mouth_r", 1)):
            if nm in an:
                b0 = an[nm] + np.array([sd * 3.0 * s, -1.5 * s])
                for a_, ln_ in ((-13, 15), (4, 18), (20, 14)):
                    e0 = b0 + np.array([sd * math.cos(math.radians(a_)), math.sin(math.radians(a_))]) * ln_ * s
                    lay.stroke(core.catmull([b0, (b0 + e0) / 2 + (0, -0.8 * s), e0], 6, closed=False), 2.6, (1.0, 0.90, 0.68),
                               taper=(0.02, 0.7))
        core.add_light(pic, lay.rgba()[..., :3] * np.clip(m * 1.5, 0, 1)[..., None], amount=0.9)
    if lit:
        cel = core.Cel(ss=2)
        _flame(cel, head, s, level, flare, fl)
        fl_layer = cel.rgba()
        ps.glow(pic, fc[0], fc[1], (230 + 520 * flare) * z, (1.0, 0.62, 0.26), 0.50 * level + 0.75 * flare)
        ps.glow(pic, fc[0], fc[1], (60 + 90 * flare) * z, (1.0, 0.86, 0.6), 0.75)
        core.over(pic, fl_layer)
        core.add_light(pic, core.gblur(fl_layer[..., :3], 9.0), amount=0.7)
    if sparking:
        sp = fuse[ib]
        ps.glow(pic, sp[0], sp[1], 130, (1.0, 0.70, 0.30), 0.75)
        ps.glow(pic, sp[0], sp[1], 18, (1.0, 0.97, 0.85), 1.0)
    # sparks: a spray at the strike, a fizz at the fuse
    srcs = []
    if lit and flare > 0.10:
        srcs.append((head, int(34 * flare) + 2, 330 * flare + 50))
    if sparking:
        srcs.append((fuse[ib], 10, 100))
    if srcs:
        lay = core.Cel(ss=1)
        r = core.rng("sparks5a", fl)
        for c0, cnt, reach in srcs:
            for _ in range(cnt):
                a = r.uniform(0, 2 * math.pi)
                d0 = r.uniform(0.25, 1.0) * reach
                v = np.array([math.cos(a), math.sin(a) * 0.8 - 0.3])
                lay.stroke(np.array([c0 + v * d0, c0 + v * (d0 + r.uniform(14, 44))]), r.uniform(2.0, 4.2), (1.0, 0.86, 0.5),
                           taper=(0.1, 0.6))
        sl = lay.rgba()[..., :3]
        core.add_light(pic, sl, amount=1.0)
        core.add_light(pic, core.gblur(sl, 6.0), amount=0.8)
    return pic, dict(bloom_amt=0.35 + 0.45 * min(level + flare, 1.0), bloom_thr=0.70, vig=0.34)
