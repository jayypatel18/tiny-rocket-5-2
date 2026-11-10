"""Shot 5a, the match (28.5 to 30.5 s, predawn). Close: the teacup rim runs
across the lower third, the panda looks over it. The paw sweeps the match
along the rim, a flare bursts and lights the face warm, it settles to a small
flame. The match goes to the fuse end hanging over the rim; the fuse catches
and the fizz runs down out of frame while the eyes follow it."""
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
HX, HY = 1010.0, 425.0            # head centre on screen
RIM_Y = 860.0
FUSE = np.array([690.0, RIM_Y - 26.0])

B = panda.replace(K["nervous"], goggles=1.0, paw_l=(-27, 62), ear_l=0.3, ear_r=0.3)
WATCH = panda.replace(B, paw_r=(33, 84), head_yaw=0.22, yaw=0.1, pitch=-0.2, pupil=(0.55, -0.45), tilt=3)
ANT = panda.replace(WATCH, paw_r=(47, 70), lean=2, eye_l=0.8, eye_r=0.8, brow_tilt=-0.3, squash=0.98)
STRIKE = panda.replace(WATCH, paw_r=(10, 72), lean=-3, pitch=0.05, eye_l=0.45, eye_r=0.45, ear_l=0.9, ear_r=0.9,
                       pupil=(0.1, -0.5), brow_tilt=0.2)
AWE = panda.replace(WATCH, paw_r=(21, 92), eye_l=1.35, eye_r=1.35, hl=1.4, brow_l=0.8, brow_r=0.8, brow_tilt=0.2,
                    pupil=(0.3, -0.25), mouth_open=0.25, mouth_wide=-0.4, smile=0.1, ear_l=0.1, ear_r=0.1, head_yaw=0.15,
                    pupil_size=1.0)
CARRY = panda.replace(AWE, paw_r=(-22, 70), head_yaw=-0.3, yaw=-0.1, pitch=-0.36, pupil=(-0.45, -0.55), eye_l=1.0,
                      eye_r=1.0, mouth_open=0.0, smile=-0.3, brow_tilt=0.7, tilt=-4, lean=-4, hl=1.2)
FLINCH = panda.replace(CARRY, paw_r=(-10, 84), lean=2, pitch=-0.1, eye_l=1.3, eye_r=1.3, ear_l=0.8, ear_r=0.8,
                       squash=1.03, pupil=(-0.6, -0.6), pupil_size=0.75)
FOLLOW = panda.replace(FLINCH, pitch=-0.42, pupil=(-0.8, -0.85), squash=1.0, ear_l=0.5, ear_r=0.5, tilt=-7, lean=-3)
fast = lambda u: core.ease_out(u, 3.0)      # noqa: E731
KEYS = [(0.0, WATCH), (0.22, WATCH), (T_STRIKE - 0.08, ANT), (T_STRIKE + 0.06, STRIKE, fast),
        (T_STRIKE + 0.30, AWE, core.overshoot), (1.02, AWE), (T_LIT - 0.10, CARRY), (T_LIT + 0.08, FLINCH, fast),
        (T_LIT + 0.40, FOLLOW), (9.0, FOLLOW)]


def pose_fn(k):
    return panda.pose_at(KEYS, max(k, 0) / FPS)


def _rim(cel, back, warm, drawing):
    """The teacup rim: the far edge (behind the panda) or the near wall across the lower third."""
    T = palette.TOD[TOD]
    cup = palette.mix(palette.shade(palette.ROCKET["cup"], T), palette.lit(palette.ROCKET["cup"], T), 0.55)
    cup = palette.mix(cup, (1.0, 0.78, 0.5), 0.45 * warm)
    dk = palette.mix(palette.shade(palette.ROCKET["cup"], T) * 0.8, (0.75, 0.45, 0.3), 0.3 * warm)
    band = palette.mix(palette.shade(palette.ROCKET["cup_band"], T), (0.55, 0.40, 0.45), 0.3 * warm)
    xs = np.linspace(-80, core.W + 80, 60)
    u = (xs - 960) / 1040.0
    sd = core.seed_of("rim", int(drawing)) % 9000
    if back:
        top = np.c_[xs, RIM_Y - 110 - 60 * (1 - u * u)]
        cel.fill(np.vstack([top, [(core.W + 80, core.H + 40), (-80, core.H + 40)]]), dk)
        cel.stroke(core.wobble(top, 0.8, sd), 5.0, T["ink"], seed=3, taper=(0, 0))
        return
    top = core.wobble(np.c_[xs, RIM_Y + 56 * (1 - u * u)], 0.8, sd)
    lip = top + (0, 30)
    cel.fill(np.vstack([top, [(core.W + 80, core.H + 40), (-80, core.H + 40)]]), cup)
    cel.fill(np.vstack([top, lip[::-1]]), palette.mix(cup, (1, 1, 1), 0.22))
    cel.fill(np.vstack([lip + (0, 85), (lip + (0, 135))[::-1]]), band)
    cel.stroke(top, 7.0, T["ink"], seed=1, taper=(0, 0))
    cel.stroke(lip, 3.0, T["ink"], seed=2, taper=(0.05, 0.05))


def render(fl, n, f):
    t = fl / FPS
    k = fl if T_STRIKE - 0.15 <= t < T_STRIKE + 0.4 or T_LIT - 0.1 <= t < T_LIT + 0.3 else common.twos(fl)
    tk = k / FPS
    pose = panda.follow(pose_fn, k)
    if abs(k - round((T_STRIKE + 0.0) * FPS)) < 0.5:
        pose = panda.replace(pose, smear=(-7.0, 0.0))
    lit = tk >= T_STRIKE + 0.04
    # the flame: a burst, then a small steady light that flickers
    flare = math.exp(-max(tk - T_STRIKE - 0.04, 0) * 9.0) if lit else 0.0
    flame = (0.52 + 0.1 * core.noise1(t * 11.0, 3) + 0.05 * math.sin(t * 43)) if lit else 0.0
    warm = core.clamp(flame + 0.6 * flare)
    z = 1.0 + 0.04 * core.ease_io(core.span(t, 0.0, n / FPS), 1.5)
    dx, dy = common.drift(t, 2.5, 6)
    base = ps.soft_bg("s5a_bg_v1", TOD, cx=1500, cy=930, zoom=3.6, sigma=18.0)
    pic = core.warp(base, 1.02 * z, 960 * (1 - 1.02 * z) + dx * 0.4, 540 * (1 - 1.02 * z) + dy * 0.4)
    cel = core.Cel(ss=2)
    _rim(cel, True, warm, common.twos(fl))
    core.over(pic, cel.rgba())
    cel = core.Cel(ss=2)
    s = S * z
    px, py = HX + dx, HY + dy + 126 * s
    mdir = np.array([-0.35, -0.94]) if tk < T_LIT - 0.25 else np.array([-0.8, -0.6])
    pl = dict(color=(1.0, 0.66, 0.30), dir=(0.35, 0.6) if tk < 1.15 else (-0.5, 0.7), amount=warm) if warm > 0.01 else None
    an = panda.draw(cel, pose, px, py, s, TOD, drawing=k, light=pl)
    T = palette.TOD[TOD]
    paw = an["paw_r"]
    head = paw + mdir * 15.0 * s
    cel.stroke(np.array([paw - mdir * 3.0 * s, head]), 1.1 * s, T["ink"], taper=(0, 0))
    cel.stroke(np.array([paw - mdir * 3.0 * s, head]), 0.62 * s, palette.lit((0.86, 0.74, 0.52), T), taper=(0, 0))
    cel.fill(core.ellipse(head[0], head[1], 0.95 * s, 1.25 * s, 14, rot=0.3), (0.16, 0.08, 0.07) if lit else (0.62, 0.2, 0.16))
    if lit:
        r = core.rng("flame", k)
        hgt = (3.2 + 4.0 * flare + r.uniform(-0.5, 0.5)) * s
        for wdt, hh, col in ((1.9, 1.0, (1.0, 0.55, 0.16)), (1.25, 0.72, (1.0, 0.82, 0.36)), (0.62, 0.42, (1.0, 0.97, 0.82))):
            sway = r.uniform(-0.5, 0.5) * s
            cel.fill(core.catmull([head + (-wdt * s * 0.5, 0.3 * s), head + (-wdt * s * 0.42, -hgt * hh * 0.45),
                                   head + (sway, -hgt * hh), head + (wdt * s * 0.42, -hgt * hh * 0.4),
                                   head + (wdt * s * 0.5, 0.3 * s), head + (0, 0.9 * s)], 6), col)
    _rim(cel, False, warm, common.twos(fl))
    # the fuse end hanging over the rim, burning away once it catches
    burn = core.span(tk, T_LIT, T_LIT + 0.42)
    fz = np.array([FUSE + (dx, dy), FUSE + (-34 + dx, 22 + dy), FUSE + (-52 + dx, 120 + dy), FUSE + (-40 + dx, 330 + dy)])
    fc = core.resample(core.catmull(fz, 12, closed=False), 60)
    cel.stroke(fc, 9.0, T["ink"], taper=(0, 0))
    cel.stroke(fc, 5.5, palette.lit(palette.ROCKET["string"], T), taper=(0, 0))
    ib = int(burn * 59)
    if ib > 1:
        cel.stroke(fc[:ib + 1], 6.0, (0.07, 0.06, 0.07), taper=(0, 0))
    core.over(pic, cel.rgba())
    # light: the flame glow, the flare, the spark head on the fuse with its spray
    if lit:
        ps.glow(pic, head[0], head[1] - 2 * s, 280 + 760 * flare, (1.0, 0.70, 0.34), 0.55 * flame + 0.95 * flare)
        ps.glow(pic, head[0], head[1] - 2 * s, 34 + 60 * flare, (1.0, 0.95, 0.8), 0.9)
    if tk >= T_LIT and burn < 1.0:
        sp = fc[ib]
        ps.glow(pic, sp[0], sp[1], 120, (1.0, 0.72, 0.32), 0.7)
        ps.glow(pic, sp[0], sp[1], 16, (1.0, 0.97, 0.85), 1.0)
    buf = np.zeros((core.H, core.W), np.float32)
    lay = core.Cel(ss=1)
    r = core.rng("sparks5a", fl)
    srcs = []
    if lit and flare > 0.12:
        srcs.append((head, int(26 * flare) + 2, 260 * flare + 40))
    if tk >= T_LIT and burn < 1.0:
        srcs.append((fc[ib], 9, 90))
    for c0, cnt, reach in srcs:
        for _ in range(cnt):
            a = r.uniform(0, 2 * math.pi)
            d0 = r.uniform(0.25, 1.0) * reach
            v = np.array([math.cos(a), math.sin(a) * 0.8 - 0.25])
            lay.stroke(np.array([c0 + v * d0, c0 + v * (d0 + r.uniform(10, 34))]), r.uniform(1.6, 3.4), (1.0, 0.86, 0.5),
                       taper=(0.1, 0.6))
    if srcs:
        sl = lay.rgba()
        core.add_light(pic, sl[..., :3], amount=1.0)
        core.add_light(pic, core.gblur(sl[..., :3], 5.0), amount=0.8)
    del buf
    return pic, dict(bloom_amt=0.4 + 0.3 * warm, bloom_thr=0.76)
