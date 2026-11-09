"""Shot 2, The Hill: the wide dusk shot. A slow lateral dolly reveals the
valley; village lights come on one by one; the panda steps back to admire the
rocket, then checks its star map and looks up at the first stars."""
from .. import core
from ..timeline import FPS
from . import common

NAME = "s2_hill"


SKY_TILT = 900.0    # how far the camera tilts up to rest on the sky; shot 3 starts from the same view


def camera(t, n_t=8.0):
    """Lateral dolly across the hill, then (as the panda looks up) a tilt up to
    the first stars. The dissolve to night happens on the sky; shot 3 calls
    this same function for its first frames so the two views match exactly."""
    u = core.ease_io(core.span(t, 0.0, 7.2), 1.7)
    v = core.ease_io(core.span(t, 6.75, 8.55), 2.0)
    dx, dy = common.drift(t, 2.0, 2)
    cx = core.lerp(1470.0, 1705.0, u)
    cy = core.lerp(905.0, 925.0, u) - 40.0 * v
    return core.Camera(cx0=1600, cy0=900, cx=cx + dx, cy=cy + dy, zoom=1.12, dolly=core.lerp(1.05, 1.0, u),
                       tilt=SKY_TILT * v)


def render(fl, n, f):
    t = fl / FPS
    n_t = n / FPS
    cam = camera(t, n_t)
    la, lb = common.cue("village_lights", NAME)
    lights = core.clamp((t - la) / (lb - la))
    stars = 0.04 + 0.26 * core.smooth(core.span(t, 1.5, n_t))
    try:
        from .. import acting
        act = acting.wide(NAME, fl, n)
    except Exception as e:  # noqa
        if not isinstance(e, ImportError):
            print("acting s2:", repr(e)[:200])
        act = None
    tr = common.cue("map_rustle", NAME)
    gust = 0.55 + 0.9 * core.smooth(core.span(t, tr - 0.1, tr + 0.15)) * (1 - core.smooth(core.span(t, tr + 0.3, tr + 1.2)))

    def mid(pic):
        common.draw_map_prop(pic, cam, "dusk", t, wind=gust)
        common.draw_rocket(pic, cam, "dusk", common.twos(fl))
        common.draw_panda(pic, cam, "dusk", act)

    pic = common.hilltop(cam, "dusk", f, t, mid=mid, lights=lights, sky_kw=dict(stars=stars))
    return pic, dict(bloom_amt=0.55, bloom_thr=0.80)
