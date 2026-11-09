"""Shot 9b: morning. The hatch tips open, the panda pops up sooty with its
goggles askew, climbs out, hops down and looks up at the one star still fading.
The camera starts close on the cup, comes down with the panda, then tilts up
with its gaze."""
from .. import core
from ..timeline import FPS
from . import common

NAME = "s9b_out"
STAR_SKY = (1180.0, -150.0)       # where the fading star sits before the tilt, in screen px


def camera(t, n_t=3.0):
    tc = common.cue("climb_out", NAME)
    p1, p2 = common.cue("paws_grass_home", NAME)
    tl = common.cue("looks_up", NAME)
    dx, dy = common.drift(t + 30.0, 2.0, 5)
    cx = core.keys(t, [(0.0, 1722.0), (tc + 0.7, 1716.0), (p1 + 0.1, 1668.0), (p2 + 0.2, 1640.0)], core.ease_io)
    cy = core.keys(t, [(0.0, 700.0), (tc + 0.7, 708.0), (p1 + 0.12, 968.0), (n_t, 960.0)], core.ease_io)
    zoom = core.keys(t, [(0.0, 2.15), (tc + 0.7, 2.1), (p1 + 0.1, 2.0), (tl - 0.1, 2.0), (n_t + 0.3, 1.72)], core.ease_io)
    tilt = 230.0 * core.ease_io(core.span(t, tl - 0.05, n_t + 0.3), 1.8)
    return core.Camera(cx0=1600, cy0=900, cx=cx + dx, cy=cy + dy, zoom=zoom, tilt=tilt)


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
    hatch = core.overshoot(core.span(t, tc - 0.32, tc - 0.05))
    st = rocket.State(hatch_open=hatch, soot=1.0)
    inside = act is not None and act.get("layer") == "behind_rocket"
    tl = common.cue("looks_up", NAME)
    star_amt = 0.95 * (1 - 0.5 * core.smooth(core.span(t, tl + 0.3, n / FPS + 0.6)))
    sx, sy = STAR_SKY[0], STAR_SKY[1] + cam.tilt + 160.0      # it rides the sky as the camera tilts

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
