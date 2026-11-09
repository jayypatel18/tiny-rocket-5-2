"""Contact sheets of chosen frames of a cut, at half size, two workers.
    python3 /abs/ROOT/tr2/panda_strip.py s1_knot 0 6 12 ...     (local frame numbers)
    python3 /abs/ROOT/tr2/panda_strip.py s1_knot range 24 44 2  (from, to, step)
Writes ROOT/build/review/strip-<shot>[-tag].jpg"""
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_NUM_THREADS", "2")


def _work(f):
    import numpy as np
    from PIL import Image
    from tr2 import film
    img = film.render_u8(f)
    return np.asarray(Image.fromarray(img).resize((960, 540), Image.LANCZOS))


def main():
    from tr2 import core, timeline
    shot = sys.argv[1]
    f0, f1 = timeline.shot_range(shot)
    args = sys.argv[2:]
    tag = ""
    if args and args[0] == "range":
        a, b, st = int(args[1]), int(args[2]), int(args[3])
        fls = list(range(a, b, st))
        tag = "-%d-%d" % (a, b)
    elif args:
        fls = [int(x) for x in args]
        tag = "-pick"
    else:
        fls = [int((f1 - f0 - 1) * i / 15) for i in range(16)]
    with mp.get_context("spawn").Pool(2) as pool:
        res = pool.map(_work, [f0 + k for k in fls])
    imgs = [r.astype("float32") / 255.0 for r in res]
    out = os.path.join(ROOT, "build", "review", "strip-%s%s.jpg" % (shot, tag))
    core.sheet(imgs, 4, 480, out)
    print(out, fls)


if __name__ == "__main__":
    main()
