"""Shot 4a, First Light: blue before dawn. The panda wakes, stretches, and
climbs into the teacup cockpit. Starts on the exact framing shot 3 ended on
(the dissolve is time passing in place), then pushes in and rises with the climb."""
from .. import core, fireflies
from ..timeline import FPS
from . import common, s3_night

NAME = "s4a_wake"


def camera(t, n_t=3.5):
    e = s3_night.END
    u = core.ease_io(core.span(t, 0.2, 3.3), 1.8)
    dx, dy = common.drift(t + 15.5, 2.0, 2)
    return core.Camera(cx0=1600, cy0=900, cx=core.lerp(e["cx"], 1690.0, u) + dx, cy=core.lerp(e["cy"], 872.0, u) + dy,
                       zoom=core.lerp(e["zoom"], 1.56, u))


def render(fl, n, f):
    from .. import rocket
    t = fl / FPS
    cam = camera(t)
    try:
        from .. import acting
        act = acting.wide(NAME, fl, n)
    except Exception as e:  # noqa
        if not isinstance(e, ImportError):
            print("acting s4a:", repr(e)[:200])
        act = None
    ti = common.cue("into_cup", NAME)
    hatch = core.ease_out(core.span(t, ti - 0.75, ti - 0.45), 2.0) * (1 - core.overshoot(core.span(t, ti + 0.22, ti + 0.42)))
    st = rocket.State(hatch_open=hatch)
    inside = act is not None and act.get("layer") == "behind_rocket"

    def mid(pic):
        common.draw_map_prop(pic, cam, "predawn", t, wind=0.4)
        if inside:
            common.draw_panda(pic, cam, "predawn", act, shadow=False)
        common.draw_rocket(pic, cam, "predawn", common.twos(fl), state=st)
        if not inside:
            common.draw_panda(pic, cam, "predawn", act)

    pic = common.hilltop(cam, "predawn", f, t, mid=mid, lights=0.12, sky_kw=dict(stars=0.35))
    fireflies.draw(pic, cam, t + 40.0, n=14, region=(980, 2380, 960, 1290), seed=4, amount=0.5)
    return pic, dict(bloom_amt=0.45, bloom_thr=0.72)
