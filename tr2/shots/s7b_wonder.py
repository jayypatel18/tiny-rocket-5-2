"""Shot 7b, wonder (46.5 to 50.0 s, sunrise). Inside the teacup, close. The
wind face: cheeks pushed back, fur and ears streaming, eyes screwed shut, hard
vibration on ones, cloud streaking past the window. The shaking eases, a paw
pushes the goggles up, gold light sweeps across the face from the right, and
the eyes open huge. The small o of the mouth turns into a smile."""
import numpy as np

from .. import core, palette, panda
from .. import panda_shot as ps
from ..timeline import FPS
from . import common

NAME = "s7b_wonder"
TOD = "sunrise"
K = panda.POSES
T_WONDER = common.cue("wonder", NAME)

HOLD = dict(paw_l=(-27, 78), paw_r=(27, 78))
WIND = panda.replace(K["wind"], squeeze=1.0, eye_l=0.0, eye_r=0.0, yaw=-0.12, head_yaw=-0.1, pitch=0.12, **HOLD)
WIND2 = panda.replace(WIND, wind=0.5, cheek=0.35, mouth_open=0.35, mouth_wide=0.5, ear_l=0.6, ear_r=0.6, pitch=0.05)
LIFT = panda.replace(WIND2, wind=0.25, paw_r=(33, 136), goggles=0.55, mouth_open=0.2, cheek=0.15, ear_l=0.4, ear_r=0.4)
TIGHT = panda.replace(LIFT, goggles=0.0, paw_r=(34, 148), wind=0.15, brow_l=-0.6, brow_r=-0.6, squash=0.97, pitch=0.0)
WONDER = panda.replace(K["wonder"], yaw=-0.12, head_yaw=0.16, pitch=0.3, goggles=0.0, paw_r=(30, 96), paw_l=(-27, 80),
                       squash=1.03, pupil=(0.45, 0.4), tilt=-3)
JOY = panda.replace(WONDER, mouth_open=0.3, mouth_wide=0.25, smile=0.95, eye_l=1.4, eye_r=1.4, eye_smile=0.2, squash=1.0,
                    paw_r=(28, 84), tilt=-6, brow_tilt=0.0)
fast = lambda u: core.ease_out(u, 3.0)      # noqa: E731
KEYS = [(0.0, WIND), (1.30, WIND), (1.55, WIND2), (1.72, LIFT), (T_WONDER - 0.06, TIGHT), (T_WONDER + 0.10, WONDER, fast),
        (2.55, WONDER), (3.05, JOY, core.overshoot), (9.0, JOY)]


def pose_fn(k):
    return panda.pose_at(KEYS, max(k, 0) / FPS)


def sky(t, flood):
    """Through the window: cloud streaming down past the cup, then open gold sky."""
    ys = np.linspace(0.35, 0.95, core.H, dtype=np.float32)
    pic = np.repeat(palette.sky_color(TOD, ys)[:, None, :], core.W, 1).copy()
    T = palette.TOD[TOD]
    n = core.fbm(core.H, core.W, 520, 4, seed=7, oy=-t * 5200.0, aspect=0.22)
    cl = np.clip((n - 0.38 - 0.5 * flood) / 0.22, 0, 1)[..., None]
    cloud = palette.mix(np.asarray(T["fog"]) * 0.9, (1.0, 0.95, 0.88), 0.5 + 0.4 * flood)
    pic = pic * (1 - cl) + cloud * cl
    ps.glow(pic, 1640, 470, 420, (1.0, 0.86, 0.5), 0.75 * flood)
    return pic


def render(fl, n, f):
    t = fl / FPS
    rough = 1 - core.smooth(core.span(t, 1.25, 1.85))          # the vibration, easing as it breaks into open sky
    ones = t < T_WONDER + 0.3
    k = fl if ones else common.twos(fl)
    pose = panda.follow(pose_fn, k)
    pose = ps.jitter(pose, fl, 2.2 * rough)
    flood = core.smooth(core.span(t, 1.45, 2.3))
    z = 1.0 + 0.07 * core.ease_io(core.span(t, 0.0, n / FPS), 1.5)
    sx, sy = common.shake(f, 8.0 * rough)
    dx, dy = common.drift(t, 2.5, 5)
    pic = sky(t, flood)
    wx, wy, wr = 1520 + (1520 - 900) * (z - 1) + dx + sx * 0.6, 370 + dy + sy * 0.6, 250 * z
    cel = core.Cel(ss=2)
    from .. import rocket
    rocket.draw_cockpit_inside(cel, TOD, common.twos(fl), window=(wx, wy, wr), light=flood, reflect=0.2 + 0.15 * t)
    core.over(pic, cel.rgba())
    s = 8.2 * z
    cel = core.Cel(ss=2)
    pl = dict(color=(1.0, 0.84, 0.48), dir=(0.95, -0.2), amount=0.7 * flood) if flood > 0.01 else None
    panda.draw(cel, pose, 850 + dx + sx, 575 + dy + sy + 126 * s, s, TOD, drawing=k, light=pl)
    core.over(pic, cel.rgba())
    # the gold floods the whole cup
    ps.glow(pic, 1900, 420, 1300, (1.0, 0.80, 0.45), 0.22 * flood, squash=0.8)
    return pic, dict(bloom_amt=0.45 + 0.25 * flood, bloom_thr=0.78)
