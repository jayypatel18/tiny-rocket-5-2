"""Review stills for the panda: the model sheet and the shot 1 close up.

    python3 /abs/ROOT/tr2/panda_sheet.py sheet|closeup|all
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tr2 import core, palette, panda  # noqa: E402

OUT = os.path.join(ROOT, "build", "review")
PAPER = np.array([0.93, 0.885, 0.79], np.float32)


def paper():
    bg = np.empty((core.H, core.W, 3), np.float32)
    bg[:] = PAPER
    n = core.fbm(core.H, core.W, 90, 4, seed=5) - 0.5
    g = core.fbm(core.H, core.W, 3, 2, seed=9) - 0.5
    bg += (n * 0.05 + g * 0.025)[..., None]
    return bg


def sheet(path, tod="dusk"):
    t0 = time.time()
    bg = paper()
    cel = core.Cel()
    Pz = panda.POSES
    # turnaround
    for k, name in enumerate(("stand", "stand_34", "stand_profile", "stand_34back")):
        panda.draw(cel, Pz[name], 150 + 285 * k, 452, 2.5, tod, drawing=k)
    panda.draw(cel, Pz["tie_knot"], 1330, 452, 2.5, tod, drawing=5)
    panda.draw(cel, Pz["proud"], 1700, 452, 2.5, tod, drawing=6)
    # expression heads: draw the whole figure, the paper strip below hides the body
    s = 3.3
    names = ("stand", "pleased", "determined", "nervous", "squeeze", "one_eye")
    strip = core.Cel()
    for k, name in enumerate(names):
        cx = 165 + 318 * k
        ps = panda.replace(Pz[name], yaw=0.0, paw_l=None, paw_r=None, lean=0.0)
        panda.draw(strip, ps, cx, 598 + 121 * s, s, tod, drawing=10 + k)
    lay = strip.rgba()
    lay[716:] = 0
    core.over(bg, cel.rgba())
    core.over(bg, lay)
    # actions, bottom row
    cel = core.Cel()
    panda.draw(cel, Pz["sleep"], 250, 1045, 3.0, "night", drawing=20)
    panda.draw(cel, Pz["wind"], 760, 1262, 3.0, "sunrise", drawing=21)
    panda.draw(cel, Pz["wonder"], 1190, 1262, 3.0, "sunrise", drawing=22,
               light=dict(color=(1.0, 0.85, 0.5), dir=(0.8, 0.2), amount=0.6))
    panda.draw(cel, Pz["look_up"], 1620, 1262, 3.0, "morning", drawing=23)
    lay = cel.rgba()
    lay[:735] = 0
    core.over(bg, lay)
    core.save(path, bg)
    print("sheet", round(time.time() - t0, 1), "s")


def closeup(path, tod="dusk", drawing=0):
    t0 = time.time()
    v = np.linspace(0.25, 1.0, core.H)[:, None].repeat(core.W, 1)
    bg = palette.sky_color(tod, v).copy()
    n = core.fbm(core.H, core.W, 500, 4, seed=3) - 0.5
    bg += n[..., None] * 0.06
    s = 6.5
    x, y = 860.0, 1232.0
    pose = panda.POSES["tie_knot"]
    cel = core.Cel()
    ink = palette.TOD[tod]["ink"]
    # the bamboo strut the knot goes on (behind the paws)
    bam = palette.lit(palette.ROCKET["bamboo"], tod)
    a = np.array([x + 49 * s, y - 0 * s])
    b = np.array([x + 57 * s, y - 175 * s])
    d = (b - a) / np.hypot(*(b - a))
    nrm = np.array([-d[1], d[0]]) * 4.6 * s
    pole = np.array([a + nrm, b + nrm, b - nrm, a - nrm])
    cel.fill(pole, bam)
    cel.crescent(pole, palette.shade(palette.ROCKET["bamboo"], tod), np.array(palette.TOD[tod]["sun"]) * 2.6 * s)
    cel.stroke(np.array([a + nrm, b + nrm]), 4.0, ink, seed=1)
    cel.stroke(np.array([a - nrm, b - nrm]), 5.0, ink, seed=2)
    for u in (0.2, 0.52, 0.86):
        c = a + (b - a) * u
        cel.stroke(np.array([c + nrm, c + d * 5 - nrm * 0.0, c - nrm]), 3.2, ink, seed=3)
    # string lashing round the strut
    strc = palette.lit(palette.ROCKET["string"], tod)
    knot = np.array([x + 51.5 * s, y - 77 * s])
    for j in range(5):
        c = knot + d * (j - 2) * 2.3 * s
        cel.stroke(np.array([c + nrm * 1.08 + d * 3, c - nrm * 1.08 - d * 3]), 0.22 * s * 6.5, ink)
        cel.stroke(np.array([c + nrm * 1.05 + d * 3, c - nrm * 1.05 - d * 3]), 0.15 * s * 6.5, strc)
    an = panda.draw(cel, pose, x, y, s, tod, drawing=drawing)
    # the string: from the knot up to the teeth, and the short end in the pulling paw
    m = an["mouth_r"]
    pl = an["paw_l"]
    for pts in (core.catmull([knot, (knot + m) / 2 + np.array([10, 14]), m], 10, closed=False),
                core.catmull([knot, (knot + pl) / 2 + np.array([0, 10]), pl + np.array([4, -6])], 10, closed=False)):
        cel.stroke(pts, 7.5, ink, taper=(0.0, 0.0))
        cel.stroke(pts, 4.6, strc, taper=(0.0, 0.0))
    core.over(bg, cel.rgba())
    core.save(path, bg)
    print("closeup", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what in ("sheet", "all"):
        sheet(os.path.join(OUT, "panda-sheet.png"))
    if what in ("closeup", "all"):
        closeup(os.path.join(OUT, "panda-closeup.png"))
