"""Quick checks for the panda: every pose on one sheet, the paw close up, a
follow through strip.   python3 /abs/ROOT/tr2/panda_test.py poses|paw|follow"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tr2 import core, palette, panda  # noqa: E402

OUT = os.path.join(ROOT, "build", "cache", "panda")
PAPER = np.array([0.93, 0.885, 0.79], np.float32)


def poses():
    names = list(panda.POSES)
    cols, cw, ch = 8, 240, 270
    rows = (len(names) + cols - 1) // cols
    pic = np.empty((rows * ch, cols * cw, 3), np.float32)
    pic[:] = PAPER
    t0 = time.time()
    for k, nm in enumerate(names):
        cel = core.Cel(cw, ch, 3)
        ps = panda.POSES[nm]
        panda.draw(cel, ps, cw / 2, ch - 30 if ps.curl < 0.5 else ch - 60, 1.25, "morning", drawing=k)
        core.over(pic, cel.rgba(), (k % cols) * cw, (k // cols) * ch)
    core.save(os.path.join(OUT, "poses.png"), pic)
    print("poses", len(names), round(time.time() - t0, 1), "s")


def paw():
    pic = np.empty((core.H, core.W, 3), np.float32)
    pic[:] = PAPER
    cel = core.Cel()
    panda.draw_paw_closeup(cel, 900, 500, 1.0, "morning", 0, paint=1.0, press=0.5)
    core.over(pic, cel.rgba())
    core.save(os.path.join(OUT, "paw.png"), pic)


def follow():
    K = panda.POSES
    keys = [(0.0, K["stand"]), (0.3, panda.replace(K["stand"], squash=0.88, dy=-4)),
            (0.5, panda.replace(K["proud"], dy=14, squash=1.12)), (0.75, K["proud"], core.overshoot), (2.0, K["proud"])]

    def fn(k):
        return panda.pose_at(keys, max(k, 0) / 24.0)

    imgs = []
    for fl in range(0, 30, 2):
        pic = np.empty((360, 320, 3), np.float32)
        pic[:] = PAPER
        cel = core.Cel(320, 360, 3)
        panda.draw(cel, panda.follow(fn, fl), 150, 330, 1.5, "morning", drawing=fl)
        core.over(pic, cel.rgba())
        imgs.append(pic)
    core.sheet(imgs, 8, 240, os.path.join(OUT, "follow.jpg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for w in sys.argv[1:] or ["poses", "paw", "follow"]:
        globals()[w]()
