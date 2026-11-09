"""Review stills for the rocket and the effects.

    python3 ROOT/tr2/fx_look.py sheet     rocket-sheet.png (side and low three quarter on paper)
    python3 ROOT/tr2/fx_look.py launch    launch-look.png (one lit frame, 0.6 s after ignition)
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


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    t0 = time.time()
    if mode == "sheet":
        rocket_sheet()
    elif mode == "launch":
        from tr2 import fx_launch_look
        fx_launch_look.render()
    print(mode, "done in %.1f s" % (time.time() - t0))
