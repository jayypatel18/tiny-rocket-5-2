"""Shot 9b: morning. The hatch tips open, the panda pops up sooty with its
goggles askew, climbs out, hops down and looks up at the one star still fading."""
from .. import core
from ..timeline import FPS
from . import common

NAME = "s9b_out"
STAR_SCREEN = (1310.0, 150.0)       # where the fading star sits on screen at the end of the cut


def camera(t, n_t=3.0):
    u = core.ease_io(core.span(t, 1.5, 2.9), 2.0)
    dx, dy = common.drift(t + 30.0, 2.0, 5)
    return core.Camera(cx0=1600, cy0=900, cx=core.lerp(1700.0, 1640.0, u) + dx, cy=core.lerp(850.0, 870.0, u) + dy,
                       zoom=core.lerp(1.62, 1.42, u), tilt=110.0 * u)


def render(fl, n, f):
    from .. import rocket
    t = fl / FPS
    cam = camera(t)
    try:
        from .. import acting
        act = acting.wide(NAME, fl, n)
    except Exception as e:  # noqa
        if not isinstance(e, ImportError):
            print("acting s9b:", repr(e)[:200])
        act = None
    tc = common.cue("climb_out", NAME)
    hatch = core.overshoot(core.span(t, tc - 0.32, tc - 0.05)) * (1 - 0.0)
    st = rocket.State(hatch_open=hatch, soot=1.0)
    inside = act is not None and act.get("layer") == "behind_rocket"
    tl = common.cue("looks_up", NAME)
    star_amt = 0.9 * core.smooth(core.span(t, 0.0, 0.6)) * (1 - 0.55 * core.smooth(core.span(t, tl, n / FPS + 0.5)))
    # the star is fixed in the sky: it moves on screen only with the tilt
    sx = STAR_SCREEN[0]
    sy = STAR_SCREEN[1] - 110.0 + cam.tilt

    def mid(pic):
        common.draw_map_prop(pic, cam, "morning", t, wind=0.8)
        if inside:
            common.draw_panda(pic, cam, "morning", act, shadow=False)
        common.draw_rocket(pic, cam, "morning", common.twos(fl), state=st)
        try:
            from . import s9a_home
            s9a_home.draw_parachute_rest(pic, cam, t, f)
        except Exception:
            pass
        if not inside:
            common.draw_panda(pic, cam, "morning", act)

    pic = common.hilltop(cam, "morning", f, t, mid=mid, sky_kw=dict(fade_star=(sx, sy, star_amt)))
    return pic, dict(bloom_amt=0.45)
