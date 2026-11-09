"""Shot 2, The Hill: the wide dusk shot. A slow lateral dolly reveals the
valley; village lights come on one by one; the panda steps back to admire the
rocket, then checks its star map and looks up at the first stars."""
from .. import core
from ..timeline import FPS
from . import common

NAME = "s2_hill"


SKY_TILT = 1560.0   # how far the camera tilts up to rest on the sky; shot 3 starts from the same view
ZOOM_END = 1.5
END_CX, END_CY = 1262.0, 962.0   # where this camera rests (shot 3 picks up from here)


def camera(t, n_t=8.0):
    """Lateral dolly across the hill, then (as the panda looks up) a tilt up to
    the first stars. The dissolve to night happens on the sky; shot 3 calls
    this same function for its first frames so the two views match exactly."""
    v = core.ease_io(core.span(t, 6.75, 8.55), 2.0)
    dx, dy = common.drift(t, 2.0, 2)
    # wide establishing dolly, then push in for the proud beat, follow to the map, tilt up through the leaves
    cx = core.keys(t, [(0.0, 1400.0), (2.3, 1500.0, core.ease_out), (3.5, 1575.0), (4.35, 1575.0), (5.9, END_CX)], core.ease_io)
    cy = core.keys(t, [(0.0, 880.0), (2.3, 890.0), (3.5, 962.0), (8.6, END_CY)], core.ease_io)
    zoom = core.keys(t, [(0.0, 1.25), (2.2, 1.32), (3.5, 2.05), (6.75, 2.05), (8.55, ZOOM_END)], core.ease_io)
    return core.Camera(cx0=1600, cy0=900, cx=cx + dx, cy=cy + dy, zoom=zoom, tilt=SKY_TILT * v)


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
