"""Shot 1, The Last Knot (0.0 to 6.0 s, dusk). Close on the panda, three
quarter view facing screen left into the low sun. Frame 0 is mid pull: the
string end in its teeth, both paws hauling the knot tight on a bamboo leg of
the rocket. The tug, the string dropped, ear flicks, a tail swish, a lean in to
pat the knot, two blinks, then pleased with itself. Slow push in, a small rack
focus from the knot to the face."""
import math

import numpy as np

from .. import core, fireflies, palette, panda
from .. import panda_shot as ps
from ..timeline import FPS
from . import common

NAME = "s1_knot"
TOD = "dusk"
S0 = 7.0                       # panda scale at the start of the push
X0, Y0 = 1100.0, 1345.0        # ground point between the feet (below the frame)
PIV = (1060.0, 470.0)          # the push in aims at the face
KNOT = np.array([52.6, 80.0])  # the knot, in panda units, on the side it faces
SDIR = np.array([-0.30, -1.0]) / math.hypot(0.30, 1.0)     # the bamboo leg, on screen
MOUTH_REL = np.array([20.9, 109.1])   # where the mouth corner is when it lets go (units, measured)

P = panda.Pose
# Key poses, drawn facing RIGHT (knot at +x); the whole pose is mirrored for the screen.
STRAIN = P(yaw=0.38, head_yaw=0.36, pitch=0.0, tilt=-6, lean=-6, head_dx=5, paw_l=(27, 69), paw_r=(46, 72),
           mouth_open=0.2, smile=0.0, mouth_wide=0.45, brow_tilt=-0.9, brow_l=-0.5, brow_r=-0.5, pupil=(0.45, -0.45),
           eye_l=0.86, eye_r=0.78, eye_smile=0.38, ear_l=0.3, ear_r=0.35, cheek=0.15, breath=0.6)
STRAIN2 = panda.replace(STRAIN, lean=-7.5, paw_l=(25, 68), tilt=-7, eye_l=0.8, eye_r=0.7, eye_smile=0.45)
ANTIC = panda.replace(STRAIN, lean=-1.5, head_dx=8, pitch=-0.2, paw_l=(34, 73), tilt=-2, eye_l=0.95, eye_r=0.95,
                      ear_l=0.0, ear_r=0.0, squash=0.97, eye_smile=0.0)
TUG = panda.replace(STRAIN, lean=-14, head_dx=1, pitch=0.3, tilt=-11, paw_l=(13, 63), squeeze=1.0, eye_l=0.0,
                    eye_r=0.0, ear_l=-0.5, ear_r=-0.5, cheek=0.5, mouth_wide=0.6, squash=1.04, brow_l=-0.9, brow_r=-0.9)
TUG2 = panda.replace(TUG, lean=-12, ear_l=0.95, ear_r=0.95, paw_l=(15, 64), squash=0.985)
SETTLE = panda.replace(STRAIN, lean=-5, pitch=0.05, tilt=-4, eye_l=0.9, eye_r=0.9, eye_smile=0.0, paw_l=(22, 66),
                       brow_tilt=-0.2, brow_l=0.0, brow_r=0.0, cheek=0.0, ear_l=0.1, ear_r=0.1)
RELEASE = panda.replace(SETTLE, mouth_open=0.5, mouth_wide=-0.2, lean=-3, pitch=0.1, tilt=-1, brow_l=0.3, brow_r=0.3,
                        eye_l=1.0, eye_r=1.0, smile=0.1)
RELAX = P(yaw=0.38, head_yaw=0.36, pitch=-0.18, tilt=2, lean=-1, head_dx=3, paw_l=(24, 62), paw_r=(46, 72), smile=0.3,
          pupil=(0.5, -0.5), breath=0.3)
BACK = panda.replace(RELAX, lean=-3, pitch=-0.1, squash=0.985)
LEAN = panda.replace(RELAX, lean=7, head_dx=7, pitch=-0.38, tilt=5, paw_l=(40, 86), eye_l=1.1, eye_r=1.1, brow_l=0.4,
                     brow_r=0.4, pupil=(0.55, -0.6), smile=0.35)
PAT = panda.replace(LEAN, paw_l=(47, 81), lean=8, squash=0.99)
LOOK = panda.replace(LEAN, paw_l=(33, 68), lean=5, pitch=-0.3, eye_l=1.0, eye_r=1.0)
PRE = panda.replace(LOOK, squash=0.965, dy=-1.5, lean=3, pitch=-0.24, paw_l=(28, 64))
PLEASED = P(yaw=0.34, head_yaw=0.2, pitch=0.14, tilt=7, lean=-2, head_dx=2, paw_l=(22, 60), paw_r=(46, 71), eye_l=0.0,
            eye_r=0.0, eye_smile=1.0, smile=0.95, brow_l=0.5, brow_r=0.5, breath=1.0, squash=1.02, ear_l=-0.1,
            ear_r=-0.1)

T_TUG = common.cue("knot_tug", NAME)
T_FLICK = common.cue("ear_flicks", NAME)
T_SWISH = common.cue("tail_swish", NAME)
T_BLINK = common.cue("blinks", NAME)
T_PLEASED = common.cue("pleased_breath", NAME)
T_DROP = 1.60                   # the string leaves its mouth
T_DROP2 = 2.84                  # the pulling paw lets its end go

fast = lambda u: core.ease_out(u, 3.0)        # noqa: E731
KEYS = [(0.0, STRAIN), (0.95, STRAIN2), (T_TUG - 0.11, ANTIC, core.ease_io), (T_TUG, TUG, fast), (T_TUG + 0.10, TUG2),
        (T_TUG + 0.30, SETTLE, core.overshoot), (T_DROP + 0.02, RELEASE), (T_DROP + 0.26, RELAX, core.overshoot),
        (2.74, RELAX), (2.84, BACK), (3.00, LEAN, core.overshoot), (3.10, PAT, fast), (3.19, LEAN), (3.29, PAT, fast),
        (3.40, LEAN), (3.58, LOOK), (4.30, LOOK), (4.42, PRE), (T_PLEASED + 0.04, PLEASED, core.overshoot),
        (9.0, PLEASED)]

REST = np.array([(-20, 34), (-42, 26), (-62, 30), (-74, 48), (-72, 70), (-62, 86)], np.float64)
UP = np.array([(-20, 34), (-44, 36), (-64, 54), (-72, 80), (-62, 104), (-44, 116)], np.float64)
OUT = np.array([(-20, 34), (-46, 24), (-72, 22), (-94, 34), (-106, 56), (-104, 80)], np.float64)
CURL = np.array([(-20, 34), (-42, 27), (-62, 33), (-73, 53), (-65, 73), (-51, 72)], np.float64)


def tail_at(t):
    """The tail: a slow sway, the swish (a wave that runs from root to tip),
    and a small curl of the tip when it is pleased."""
    out = []
    for i in range(6):
        ti = t - i * 0.024
        kf = [(0.0, REST[i]), (T_SWISH - 0.24, REST[i]), (T_SWISH - 0.02, UP[i]), (T_SWISH + 0.20, OUT[i]),
              (T_SWISH + 0.66, REST[i], core.overshoot), (T_PLEASED - 0.1, REST[i]), (T_PLEASED + 0.5, CURL[i]),
              (T_PLEASED + 1.5, CURL[i])]
        v = core.keys(ti, kf)
        w = (i / 5.0) ** 1.5
        sw = math.sin(ti * 1.7 + 0.5) * 2.4 + math.sin(ti * 0.9 + 2.0) * 1.6
        out.append((float(v[0] + sw * w), float(v[1] + sw * 0.5 * w)))
    return tuple(out)


def pose_fn(k):
    """The key animation at frame k (may be negative: holds the first pose)."""
    t = max(k, 0) / FPS
    p = panda.pose_at(KEYS, t)
    e_l, e_r = p.ear_l, p.ear_r
    e_r -= 0.75 * ps.bump(t, T_FLICK[0], 0.05, 0.16)
    e_l -= 0.75 * ps.bump(t, T_FLICK[1], 0.05, 0.16)
    e_l -= 0.5 * ps.bump(t, 5.3, 0.05, 0.2)
    bl = ps.blink(t, T_BLINK)
    br = p.breath
    if T_DROP + 0.3 < t < T_PLEASED - 0.3:
        br = 0.3 + 0.25 * math.sin((t - 1.9) * 2.2)
    elif t >= T_PLEASED:
        # the proud breath: a big inhale, held a moment, a long satisfied exhale, then easy breathing
        br = core.keys(t, [(T_PLEASED, 0.5), (T_PLEASED + 0.38, 1.0), (T_PLEASED + 0.7, 0.95), (T_PLEASED + 1.5, 0.2),
                           (T_PLEASED + 2.3, 0.55), (T_PLEASED + 3.2, 0.25)])
    return panda.replace(p, ear_l=e_l, ear_r=e_r, eye_l=p.eye_l * bl, eye_r=p.eye_r * bl, breath=br, tail=tail_at(t),
                         tilt=p.tilt + 1.6 * (ps.bump(t, T_FLICK[0], 0.05, 0.2) - ps.bump(t, T_FLICK[1], 0.05, 0.2)))


# ------------------------------------------------------------------ background
def _base_bg():
    cam = core.Camera(cx0=1600, cy0=900, cx=1690, cy=1012, zoom=4.5)
    pic = common.hilltop(cam, TOD, 0, 0.0, defocus=lambda p: 16.0, front=False, band=None)
    pic = core.gblur(pic, 5.0)
    return pic


def _fore():
    """A few big soft grass blades in front of the lens, bottom corners."""
    cel = core.Cel(ss=1)
    r = core.rng("s1 fore")
    T = palette.TOD[TOD]
    for i in range(11):
        left = i < 6
        x0 = r.uniform(-60, 420) if left else r.uniform(1560, 2000)
        h = r.uniform(260, 620)
        lean = r.uniform(-0.45, 0.45) + (0.25 if left else -0.25)
        w = r.uniform(26, 54)
        base = np.array([x0, 1110.0])
        tip = base + np.array([lean * h, -h])
        midp = base + np.array([lean * h * 0.3, -h * 0.55])
        nv = np.array([1.0, 0.0])
        blade = np.vstack([ps_arc(base - nv * w, midp - nv * w * 0.7, tip), ps_arc(tip, midp + nv * w * 0.7, base + nv * w)])
        g = palette.mix((0.20, 0.30, 0.12), (0.42, 0.40, 0.16), r.uniform(0, 1))
        cel.fill(blade, palette.mix(palette.shade(g, T), palette.lit(g, T), r.uniform(0.1, 0.5)))
    return core.gblur(cel.rgba(), 13.0)


def ps_arc(a, m, b, k=9):
    t = np.linspace(0, 1, k)[:, None]
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * m + t * t * b


def background(t, z, dx, dy):
    base = ps.cached("s1_bg_v2", _base_bg)
    zb = 1.014 + (z - 1.0) * 0.5                  # the far world grows less than the panda: a real push
    pic = core.warp(base, zb, PIV[0] * (1 - zb) + dx * 0.4, PIV[1] * (1 - zb) + dy * 0.4)
    cam = core.Camera(cx0=1600, cy0=900, cx=1690, cy=1012, zoom=4.5 * zb)
    # pollen catching the low sun, far out of focus, drifting up; the first fireflies
    fireflies.draw(pic, cam, t + 3.0, n=46, region=(1440, 1950, 870, 1150), seed=11, color=(1.0, 0.86, 0.55),
                   amount=1.25, p_range=(0.5, 1.5), bokeh=15, blink=False, rise=14, twinkle=0.6)
    fireflies.draw(pic, cam, t + 9.0, n=10, region=(1460, 1930, 930, 1130), seed=12, color=(0.86, 1.0, 0.5),
                   amount=0.7, p_range=(0.6, 1.3), bokeh=7, blink=True)
    # the low sun out of frame on the left
    ps.glow(pic, -160, 330, 1050, (1.0, 0.74, 0.42), 0.34, squash=0.8)
    return pic


def render(fl, n, f):
    t = fl / FPS
    z = 1.0 + 0.08 * core.ease_io(core.span(t, 0.0, n / FPS), 1.5)
    dx, dy = common.drift(t, 3.0, 1)
    pic = background(t, z, dx, dy)

    ones = T_TUG - 0.2 <= t < T_DROP + 0.2            # the tug is animated on ones
    k = fl if ones else common.twos(fl)
    tk = k / FPS
    pose = panda.follow(pose_fn, k)
    # strain: tiny trembles that grow until the tug
    if tk < T_TUG - 0.1:
        r = core.rng("s1 tremble", k)
        amp = 0.35 + 0.55 * tk / T_TUG
        q = r.normal(0, 1, 5) * amp
        pose = panda.replace(pose, lean=pose.lean + q[0] * 0.5, head_dy=q[1] * 0.45, tilt=pose.tilt + q[4] * 0.5,
                             paw_l=(pose.paw_l[0] + q[2] * 0.8, pose.paw_l[1] + q[3] * 0.5))
    # one smear drawing on the fastest frame of the tug
    if abs(k - round((T_TUG - 0.04) * FPS)) < 0.5:
        pose = panda.replace(pose, smear=(-9.0, 2.0))
    pm = panda.mirror(pose)

    s = S0 * z
    px = PIV[0] + (X0 - PIV[0]) * z + dx
    py = PIV[1] + (Y0 - PIV[1]) * z + dy
    knot = np.array([px - KNOT[0] * s, py - KNOT[1] * s])
    # the leg shudders when the knot bites
    if tk > T_TUG:
        knot = knot + np.array([1.0, 0.25]) * 3.0 * math.exp(-(tk - T_TUG) * 8.0) * math.sin((tk - T_TUG) * 62.0) * s / 7.0
    cel = core.Cel()
    tight = core.smooth(core.span(tk, T_TUG - 0.03, T_TUG + 0.05))
    ps.draw_strut(cel, knot, SDIR, s, TOD, drawing=common.twos(fl), spread=2.9 - 0.7 * tight)
    an = panda.draw(cel, pm, px, py, s, TOD, drawing=k)

    # the strands: knot to the teeth, knot to the pulling paw
    def units(v):
        return np.array([px - v[0] * s, py - v[1] * s])

    if tk < T_DROP and "mouth_l" in an:
        vib = math.sin(tk * 90.0) * (1.2 + 2.0 * core.span(tk, T_TUG - 0.3, T_TUG)) * (0.3 if tk > T_TUG else 1.0)
        m = an["mouth_l"]
        nrm = np.array([-(m - knot)[1], (m - knot)[0]]) / max(np.hypot(*(m - knot)), 1e-6)
        ps.draw_string(cel, [knot, (knot + m) / 2 + nrm * vib, m], s, TOD)
    else:
        v0 = units(MOUTH_REL) - knot
        ps.draw_string(cel, ps.pendulum(knot, v0, tk - T_DROP), s, TOD)
    pw = an.get("paw_r")
    if tk < T_DROP2 and pw is not None:
        end = pw + (pw - knot) / max(np.hypot(*(pw - knot)), 1e-6) * 9.0 * s + np.array([0.0, 2.5 * s])
        ps.draw_string(cel, [knot, pw + np.array([0.0, -1.5 * s]), end], s, TOD, curve=False)
    else:
        v0 = units(np.array([16.5, 59.5])) - knot
        ps.draw_string(cel, ps.pendulum(knot, v0, tk - T_DROP2, damp=3.4, freq=7.5), s, TOD)

    # rack focus: opens on the knot and the paws, the face a little soft; it finds the face before the tug
    u = core.smooth(core.span(t, 0.30, 0.85))
    v = core.smooth(core.span(t, 2.8, 3.1))
    layer = ps.rack(cel.rgba(), knot + np.array([30.0, 10.0]), 300.0 * z, 2.7 * (1 - u), 2.4 * u * (1 - 0.8 * v))
    core.over(pic, layer)

    fore = ps.cached("s1_fore_v1", _fore)
    sway = math.sin(t * 0.9) * 7.0 + math.sin(t * 2.1 + 1.0) * 2.5
    zf = 1.02 + (z - 1.0) * 1.9
    core.over(pic, core.warp(fore, zf, PIV[0] * (1 - zf) + dx * 1.6 + sway, 1080 * (1 - zf) + dy * 1.6 + 10))
    # warm veil of the low sun over everything on the left
    ps.glow(pic, -260, 300, 760, (1.0, 0.80, 0.50), 0.16, squash=0.9)
    return pic, dict(bloom_amt=0.5, bloom_thr=0.80)
