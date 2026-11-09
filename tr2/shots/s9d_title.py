"""Shot 9d: the panda looks at us, its tail curls once, and the title brushes
in over the morning sky: "Tiny Rocket", the red hanko, "Fable 5.2". Hold, fade
to paper, then black."""
import numpy as np

from .. import core, title
from ..timeline import FPS
from . import common

NAME = "s9d_title"
PAPER = np.array([0.95, 0.91, 0.80], np.float32)


def camera(t, n_t=6.5):
    # close on the panda for the look and the tail curl, then pull back and up to give the sky to the title
    u = core.ease_io(core.span(t, 1.25, 3.1), 2.0)
    dx, dy = common.drift(t + 50.0, 2.2, 7)
    return core.Camera(cx0=1600, cy0=900, cx=core.lerp(1300.0, 1960.0, u) + dx, cy=core.lerp(985.0, 792.0, u) + dy,
                       zoom=core.lerp(2.5, 1.05, u))


def render(fl, n, f):
    from .. import rocket
    t = fl / FPS
    cam = camera(t)
    try:
        from .. import acting
        act = acting.wide(NAME, fl, n)
    except Exception as e:  # noqa
        if not isinstance(e, ImportError):
            print("acting s9d:", repr(e)[:200])
        act = None
    st = rocket.State(soot=1.0, hatch_open=1.0)

    def mid(pic):
        common.draw_map_prop(pic, cam, "morning", t, wind=0.7, star_t=5.0)
        common.draw_rocket(pic, cam, "morning", common.twos(fl), state=st)
        try:
            from . import s9a_home
            s9a_home.draw_parachute_rest(pic, cam, t, f)
        except Exception:
            pass
        common.draw_panda(pic, cam, "morning", act)

    pic = common.hilltop(cam, "morning", f, t, mid=mid)
    ta, tb = common.cue("title_brush", NAME)
    th = common.cue("hanko", NAME)
    ca, cb = common.cue("credit_brush", NAME)
    title.draw(pic, t, t_title=(ta, tb), t_hanko=th, t_credit=(ca, cb), x=835.0, y=262.0, size=150.0)
    # fade to paper, then to black
    tp = common.cue("fade_to_paper", NAME)
    tk = common.cue("fade_to_black", NAME)
    te = common.cue("end", NAME)
    opts = dict(bloom_amt=0.4)
    if t >= tk:
        k = core.smooth(core.span(t, tk, te - 0.12))
        opts["fade"] = (PAPER * (1 - k), 1.0)
    elif t >= tp:
        opts["fade"] = (PAPER, core.smooth(core.span(t, tp, tk)))
    return pic, opts
