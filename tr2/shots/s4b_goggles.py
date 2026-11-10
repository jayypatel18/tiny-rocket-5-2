"""Shot 4b, goggles (25.0 to 28.5 s, predawn). Outside the teacup looking in
through the round window. The cut opens sharp on the brass rim and the glass
with the face soft, and racks to the face. Both paws pull the goggles down
(they snap, the ears bounce), the eyes dart, a nervous swallow, a breath in,
one small determined nod."""
from .. import core, panda
from .. import panda_shot as ps
from ..timeline import FPS
from . import common

NAME = "s4b_goggles"
TOD = "predawn"
K = panda.POSES
T_SNAP = common.cue("goggles_snap", NAME)
T_SWALLOW = common.cue("swallow", NAME)
T_NOD = common.cue("nod", NAME)

GLASS = dict(paw_l=(-31, 64), paw_r=(31, 64), palm_l=1.0, palm_r=1.0)
START = panda.replace(K["cockpit"], goggles=0.0, brow_l=0.3, brow_r=0.3, brow_tilt=0.5, smile=0.0, pupil=(0.1, 0.2), **GLASS)
GRAB = panda.replace(START, paw_l=(-34, 151), paw_r=(34, 151), palm_l=0.0, palm_r=0.0, pitch=0.12, squash=1.03,
                     eye_l=1.1, eye_r=1.1)
DOWN = panda.replace(GRAB, goggles=1.0, paw_l=(-37, 124), paw_r=(37, 124), pitch=-0.2, head_dy=-2.5, squash=0.96,
                     eye_l=0.5, eye_r=0.5, smile=-0.2)
SET = panda.replace(K["nervous"], pupil=(0.0, 0.0), paw_l=(-37, 128), paw_r=(37, 128), pitch=-0.04)
NERV = panda.replace(K["nervous"], pupil=(-0.6, 0.0), **GLASS)
NERV2 = panda.replace(NERV, pupil=(0.6, 0.1), tilt=2)
NERV3 = panda.replace(NERV, pupil=(0.0, -0.3), tilt=0)
GULP = panda.replace(K["swallow"], **GLASS)
AFTER = panda.replace(NERV3, brow_tilt=0.5, ear_l=0.35, ear_r=0.35)
BRAVE = panda.replace(K["determined"], breath=1.0, squash=1.03, ear_l=-0.2, ear_r=-0.2, pitch=0.0, **GLASS)
NOD_UP = panda.replace(BRAVE, pitch=0.14, head_dy=1.0)
NOD = panda.replace(K["nod_down"], breath=0.6, ear_l=0.1, ear_r=0.1, **GLASS)
READY = panda.replace(BRAVE, pitch=-0.08, breath=0.5, squash=1.0, smile=0.15)
fast = lambda u: core.ease_out(u, 3.0)      # noqa: E731
KEYS = [(0.0, START), (0.26, START), (0.50, GRAB, core.overshoot), (T_SNAP - 0.04, GRAB), (T_SNAP + 0.05, DOWN, fast),
        (T_SNAP + 0.24, SET, core.overshoot), (1.02, SET), (1.24, NERV3), (1.36, NERV, fast), (1.56, NERV),
        (1.64, NERV2, fast), (1.78, NERV2), (1.86, NERV3), (T_SWALLOW + 0.10, GULP), (T_SWALLOW + 0.30, AFTER, core.overshoot),
        (2.36, AFTER), (2.62, BRAVE), (T_NOD - 0.10, NOD_UP), (T_NOD + 0.05, NOD, fast), (T_NOD + 0.26, READY, core.overshoot),
        (9.0, READY)]


def pose_fn(k):
    return panda.pose_at(KEYS, max(k, 0) / FPS)


def render(fl, n, f):
    t = fl / FPS
    ones = (T_SNAP - 0.15 <= t < T_SNAP + 0.35) or (T_NOD - 0.15 <= t < T_NOD + 0.35)
    k = fl if ones else common.twos(fl)
    pose = panda.follow(pose_fn, k)
    pose = ps.jitter(pose, fl, 0.5 * core.span(t, 1.05, 1.3) * (1 - core.span(t, 2.3, 2.6)))      # whiskers tremble
    u = core.smooth(core.span(t, 0.10, 0.50))                                                     # the rack focus
    z = 1.0 + 0.05 * core.ease_io(core.span(t, 0.0, n / FPS), 1.5)
    dx, dy = common.drift(t, 2.5, 4)
    fog = 0.22 * core.smooth(core.span(t, 2.62, 2.9)) * (1 - core.smooth(core.span(t, 3.0, 3.5)))
    pic, an = ps.window_shot(pose, TOD, k, r=405.0 * z, s=6.4 * z, cx=960 + dx, cy=552 + dy, reflect=0.1 + 0.22 * t,
                             ptod=ps.cockpit_tod(TOD), plight=ps.DIAL,
                             fog=fog, sig_face=6.0 * (1 - u), sig_front=1.5 * u, ss=2)
    return pic, dict(bloom_amt=0.4, bloom_thr=0.76)
