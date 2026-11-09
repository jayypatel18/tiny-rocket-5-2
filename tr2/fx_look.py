"""Review stills for the rocket and the effects.

    python3 ROOT/tr2/fx_look.py sheet     rocket-sheet.png (side and low three quarter on paper)
    python3 ROOT/tr2/fx_look.py launch    launch-look.png (one lit frame, 0.6 s after ignition)
    python3 ROOT/tr2/fx_look.py cockpit   cockpit-look.png, cockpit-warm.png, cockpit-inside.png
                                          (with a flat stand in for the panda)
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tr2 import core, palette, rocket  # noqa: E402

REVIEW = os.path.join(ROOT, "build", "review")
PAPER = np.array([0.93, 0.89, 0.80], np.float32)


def paper_bg():
    pic = np.empty((core.H, core.W, 3), np.float32)
    pic[:] = PAPER
    g = core.fbm(core.H, core.W, 6.0, 3, seed=5, aspect=3.0)
    pic *= (0.965 + 0.07 * g)[..., None]
    return pic


def rocket_sheet(path=None, tod="morning"):
    pic = paper_bg()
    cel = core.Cel()
    info = rocket.draw(cel, 470, 1010, s=1.8, tod=tod, drawing=0, view="side", state=rocket.State())
    rocket.draw_glass(cel, info, tod, 0)
    info2 = rocket.draw(cel, 1130, 1040, s=1.8, tod=tod, drawing=0, view="low34", state=rocket.State())
    rocket.draw_glass(cel, info2, tod, 0)
    st = rocket.State(light=1.0, flame=1.0, fin_rattle=0.6, squash=-0.3)
    info3 = rocket.draw(cel, 1640, 1000, s=1.15, tod="launch", drawing=2, view="below", state=st, rot=0.06)
    rocket.draw_glass(cel, info3, "launch", 2)
    core.over(pic, cel.rgba())
    core.save(path or os.path.join(REVIEW, "rocket-sheet.png"), pic)
    return pic


def _stand_in(cx, cy, r, tod):
    """A flat russet head and shoulders so the cockpit layers can be judged before the panda exists."""
    cel = core.Cel(ss=2)
    fur = palette.lit(palette.PANDA["fur"], tod)
    wh = palette.lit(palette.PANDA["white"], tod)
    cel.fill(core.ellipse(cx, cy + r * 0.95, r * 0.8, r * 0.5, 40), palette.shade(palette.PANDA["dark"], tod))
    cel.fill(core.ellipse(cx, cy + r * 0.1, r * 0.62, r * 0.52, 50), fur)
    cel.fill(core.ellipse(cx, cy + r * 0.22, r * 0.44, r * 0.3, 40), wh)
    for sg in (-1, 1):
        cel.fill(core.ellipse(cx + sg * r * 0.5, cy - r * 0.36, r * 0.2, r * 0.24, 30), wh)
        cel.fill(core.ellipse(cx + sg * r * 0.24, cy + r * 0.05, r * 0.07, r * 0.09, 20), palette.PANDA["eye"])
    return cel.rgba()


def cockpit_look():
    t0 = time.time()
    for name, tod, light, reflect, fog in (("cockpit-look.png", "predawn", 0.0, 0.0, 0.0),
                                           ("cockpit-warm.png", "predawn", 0.8, 0.35, 0.7)):
        pic = np.zeros((core.H, core.W, 3), np.float32)
        cx, cy, r = 960, 560, 380
        back = core.Cel()
        rocket.draw_cockpit_back(back, tod, 0, cx, cy, r, light=light)
        core.over(pic, back.rgba())
        core.over(pic, _stand_in(cx, cy, r, tod) * rocket.cockpit_mask(cx, cy, r)[..., None])
        front = core.Cel()
        rocket.draw_cockpit_front(front, tod, 0, cx, cy, r, reflect=reflect, light=light, fog=fog)
        core.over(pic, front.rgba())
        core.save(os.path.join(REVIEW, name), pic)
        print(name, "%.1f s" % (time.time() - t0))
    tod = "sunrise"
    yy = np.linspace(0.35, 1.0, core.H, dtype=np.float32)[:, None].repeat(core.W, 1)
    pic = palette.sky_color(tod, yy).astype(np.float32)
    cel = core.Cel()
    rocket.draw_cockpit_inside(cel, tod, 0, window=(1240, 430, 250), light=0.7)
    core.over(pic, cel.rgba())
    core.over(pic, _stand_in(760, 640, 420, tod))
    core.save(os.path.join(REVIEW, "cockpit-inside.png"), pic)
    print("cockpit-inside.png %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    t0 = time.time()
    if mode == "sheet":
        rocket_sheet()
    elif mode == "launch":
        from tr2 import fx_launch_look
        fx_launch_look.render()
    elif mode == "cockpit":
        cockpit_look()
    print(mode, "done in %.1f s" % (time.time() - t0))
