"""Shot 3, Waiting for Morning: starts on the turning star field, a shooting
star crosses, the camera tilts down and pushes in to the panda asleep against
the rocket under its patchwork blanket. Fireflies. One ear twitches."""
from .. import core, fireflies
from ..timeline import FPS
from . import common, s2_hill

NAME = "s3_night"


END = dict(cx=1650.0, cy=952.0, zoom=1.36)     # shot 4a starts from this view (match dissolve)


def camera(t, n_t=7.5):
    if t < 0.62:
        return s2_hill.camera(t + 8.0)         # the same view as the end of shot 2, for the dissolve
    u = core.ease_io(core.span(t, 3.3, 6.0), 2.2)
    creep = core.smooth(core.span(t, 5.6, n_t + 0.7))
    dx, dy = common.drift(t + 8.0, 2.0, 2)
    cx = core.lerp(1705.0, END["cx"], u)
    cy = core.lerp(885.0, END["cy"] + 6, u) - 6 * creep
    zoom = core.lerp(1.12, END["zoom"] - 0.03, u) + 0.03 * creep
    return core.Camera(cx0=1600, cy0=900, cx=cx + dx, cy=cy + dy, zoom=zoom, tilt=s2_hill.SKY_TILT * (1 - u))


def render(fl, n, f):
    t = fl / FPS
    n_t = n / FPS
    cam = camera(t, n_t)
    ta, tb = common.cue("shooting_star_night", NAME)
    sky_kw = dict(stars=1.0, star_turn=0.013 * (t + 4.0),
                  shooting=(ta, 1480.0, 170.0, 700.0, 560.0, tb - ta))
    try:
        from .. import acting
        act = acting.wide(NAME, fl, n)
    except Exception as e:  # noqa
        if not isinstance(e, ImportError):
            print("acting s3:", repr(e)[:200])
        act = None

    def mid(pic):
        common.draw_map_prop(pic, cam, "night", t, wind=0.35)
        common.draw_rocket(pic, cam, "night", common.twos(fl))
        common.draw_panda(pic, cam, "night", act)

    pic = common.hilltop(cam, "night", f, t, mid=mid, lights=0.34, sky_kw=sky_kw)
    fireflies.draw(pic, cam, t + 20.0, n=46, region=(980, 2380, 960, 1290), seed=3, amount=0.95)
    return pic, dict(bloom_amt=0.6, bloom_thr=0.62, vig=0.3)
