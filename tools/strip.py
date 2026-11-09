"""Motion check: render consecutive frames, crop the same box from each, and
lay them out as a strip, so in betweens can be read side by side.

  python3 tools/strip.py F0 F1 STEP X0 Y0 X1 Y1 COLS OUT.jpg [CELL_W]
"""
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")


def work(f):
    from tr2 import film
    return film.render_frame(f)


if __name__ == "__main__":
    f0, f1, step, x0, y0, x1, y1, cols = [int(v) for v in sys.argv[1:9]]
    out = sys.argv[9]
    cell = int(sys.argv[10]) if len(sys.argv) > 10 else 320
    fs = list(range(f0, f1, step))
    with mp.get_context("spawn").Pool(6) as pool:
        pics = pool.map(work, fs)
    from tr2 import core
    core.sheet([p[y0:y1, x0:x1] for p in pics], cols, cell, out)
    print(out, len(fs), "frames")
