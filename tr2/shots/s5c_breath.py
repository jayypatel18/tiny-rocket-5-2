"""Shot 5c, the held breath (33.0 to 36.0 s, predawn). Close on the face through
the round window. The eyes squeeze shut, the fuse light flickers on the glass
and dies. One beat of total stillness: the same drawing held, no boil, no
drift. Then one eye opens and the pupil slides down to look. In the last two
frames the first orange of the ignition touches the glass."""
import math

from .. import core, panda
from .. import panda_shot as ps
from ..timeline import FPS
from . import common

NAME = "s5c_breath"
TOD = "predawn"
K = panda.POSES
T_SHUT = common.cue("eyes_shut", NAME)
T_END = common.cue("fuse_end", NAME)
T_STILL = common.cue("stillness", NAME)[0]
T_ONE = common.cue("one_eye_opens", NAME)

GRIP = dict(paw_l=(-31, 65), paw_r=(31, 65), palm_l=1.0, palm_r=1.0, goggles=1.0)
FEAR = panda.replace(K["nervous"], pupil=(-0.55, -0.5), eye_l=1.1, eye_r=1.1, pitch=-0.1, tilt=-3, **GRIP)
WIDE = panda.replace(FEAR, eye_l=1.35, eye_r=1.35, brow_l=0.9, brow_r=0.9, pupil_size=0.65, squash=1.02, ear_l=0.2, ear_r=0.2)
SHUT = panda.replace(K["squeeze"], head_dy=-2.5, pitch=-0.16, squash=0.965, breath=1.0, paw_l=(-28, 67), paw_r=(28, 67),
                     palm_l=1.0, palm_r=1.0, goggles=1.0, ear_l=0.95, ear_r=0.95)
PEEK = panda.replace(SHUT, eye_r=1.0, hl=1.4, brow_r=0.6, ear_r=0.45, pupil=(0.0, 0.15), pupil_size=0.8)
LOOK = panda.replace(PEEK, pupil=(-0.6, -0.6), eye_r=1.05)
fast = lambda u: core.ease_out(u, 3.0)      # noqa: E731
KEYS = [(0.0, FEAR), (T_SHUT - 0.09, WIDE), (T_SHUT + 0.05, SHUT, fast), (T_ONE, SHUT), (T_ONE + 0.125, PEEK),
        (T_ONE + 0.30, PEEK), (T_ONE + 0.50, LOOK), (9.0, LOOK)]


def pose_fn(k):
    return panda.pose_at(KEYS, max(k, 0) / FPS)


def render(fl, n, f):
    t = fl / FPS
    still = T_STILL <= t < T_ONE
    if t < T_STILL:
        k = common.twos(fl)
    elif still:
        k = int(round(T_STILL * FPS))                 # ONE drawing, held: nothing boils
    else:
        k = fl if t < T_ONE + 0.2 else common.twos(fl)   # the lid lifts on ones
    pose = panda.follow(pose_fn, k) if t < T_STILL else pose_fn(k)
    # the held breath trembles, less and less, and is gone by the stillness
    pose = ps.jitter(pose, k, 0.9 * core.span(t, T_SHUT, T_SHUT + 0.2) * (1 - core.span(t, 0.6, T_STILL - 0.04)))
    # the fuse light on the glass: flickers, fades as the spark runs out, dies
    lt = 0.0
    if t < T_END:
        lt = (0.42 + 0.3 * core.noise1(t * 13.0, 5) + 0.1 * math.sin(t * 50)) * (1 - core.smooth(core.span(t, 0.45, T_END)))
    if fl >= n - 2:
        lt = 0.45 if fl == n - 2 else 0.85            # ignition is one cut away
    lt = core.clamp(lt)
    z = 1.0 + 0.04 * core.ease_io(core.span(t, 0.0, T_STILL), 1.6)
    dk = 1 - core.smooth(core.span(t, T_STILL - 0.5, T_STILL))
    dx, dy = common.drift(t, 2.5 * dk, 3)
    pl = dict(color=(1.0, 0.60, 0.26), dir=(-0.5, 0.85), amount=lt) if lt > 0.01 else ps.DIAL
    pic, an = ps.window_shot(pose, TOD, k, r=470.0 * z, s=7.0 * z, cx=960 + dx, cy=545 + dy, light=lt, plight=pl, ptod=ps.cockpit_tod(TOD),
                             reflect=0.35 + 0.02 * min(t, T_STILL), fog=0.0, ss=2)
    return pic, dict(bloom_amt=0.35 + 0.4 * lt, bloom_thr=0.78)
