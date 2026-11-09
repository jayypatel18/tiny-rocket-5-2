"""Shot 4a, First Light: blue before dawn. The panda wakes, stretches, and
climbs into the teacup cockpit. Starts on the exact framing shot 3 ended on
(the dissolve is time passing in place), then the camera rises with the climb."""
from .. import core, fireflies
from ..timeline import FPS
from . import common, s3_night

NAME = "s4a_wake"
DAWN_RIM = dict(color=(0.66, 0.80, 1.0), dir=(0.6, -0.8), amount=0.7)


def camera(t, n_t=3.5):
    e = s3_night.END
    c1, c2 = common.cue("climb_paws", NAME)
    ti = common.cue("into_cup", NAME)
    dx, dy = common.drift(t + 15.5, 2.0, 2)
    cx = core.keys(t, [(0.0, e["cx"]), (c1 - 0.3, e["cx"] - 8), (ti, 1716.0)], core.ease_io)
    cy = core.keys(t, [(0.0, e["cy"]), (c1 - 0.45, e["cy"] - 14), (c2, 835.0), (ti + 0.1, 716.0), (ti + 0.6, 706.0)], core.ease_io)
    zoom = core.keys(t, [(0.0, e["zoom"]), (c1 - 0.3, e["zoom"] + 0.06), (ti + 0.2, 1.92)], core.ease_io)
    return core.Camera(cx0=1600, cy0=900, cx=cx + dx, cy=cy + dy, zoom=zoom)


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
    glow = core.smooth(core.span(t, ti - 0.1, ti + 0.35))            # the little dial lamp inside wakes up
    st = rocket.State(hatch_open=hatch, window_glow=0.25 + 0.6 * glow)
    inside = act is not None and act.get("layer") == "behind_rocket"
    light = dict(DAWN_RIM)
    if t > ti - 0.3:
        k = core.smooth(core.span(t, ti - 0.3, ti))
        light = dict(color=(1.0, 0.82, 0.55), dir=(0.1, 1.0), amount=0.75 * k)     # warm cockpit glow from below

    def mid(pic):
        common.draw_map_prop(pic, cam, "predawn", t, wind=0.4)
        if inside:
            common.draw_panda(pic, cam, "predawn", act, light=light, shadow=False)
        common.draw_rocket(pic, cam, "predawn", common.twos(fl), state=st)
        if not inside:
            common.draw_panda(pic, cam, "predawn", act, light=light)

    pic = common.hilltop(cam, "predawn", f, t, mid=mid, lights=0.12, sky_kw=dict(stars=0.35))
    fireflies.draw(pic, cam, t + 40.0, n=14, region=(980, 2380, 960, 1290), seed=4, amount=0.5)
    return pic, dict(bloom_amt=0.45, bloom_thr=0.72)
