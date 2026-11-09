"""Render the film (or part of it) and stream the frames straight into ffmpeg.

  python3 render.py                      whole film -> build/seg/full.mp4
  python3 render.py --shot s6_lift       one cut    -> build/seg/s6_lift.mp4
  python3 render.py --range 864 1032     frames     -> build/seg/f864-1032.mp4
  python3 render.py --still 900          one frame  -> build/review/f0900.png
  python3 render.py --sheet s6_lift 12   contact sheet of 12 frames of a cut
  options: --jobs N (worker processes), --scale 0.5 (animatic size), --crf 12, --out path, --step k

Frames never touch the disk: workers render in parallel, the parent writes raw
RGB to an ffmpeg pipe in order. Every frame is deterministic, so cuts can be
re rendered alone and joined with concat.py.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _work(args):
    f, scale = args
    import numpy as np
    from PIL import Image
    from tr2 import film
    img = film.render_u8(f)
    if scale != 1.0:
        w = int(round(img.shape[1] * scale / 2)) * 2
        h = int(round(img.shape[0] * scale / 2)) * 2
        img = np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))
    return f, img.tobytes(), img.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot")
    ap.add_argument("--range", nargs=2, type=int)
    ap.add_argument("--still", type=int)
    ap.add_argument("--sheet", nargs=2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()

    from tr2 import core, film, timeline

    if a.still is not None:
        out = a.out or os.path.join(ROOT, "build", "review", "f%04d.png" % a.still)
        t0 = time.time()
        core.save(out, film.render_frame(a.still), seed=a.still)
        print(out, "%.1fs" % (time.time() - t0))
        return
    if a.sheet:
        f0, f1 = timeline.shot_range(a.sheet[0])
        k = int(a.sheet[1])
        fs = [int(f0 + (f1 - 1 - f0) * i / max(k - 1, 1)) for i in range(k)]
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(a.jobs, k)) as pool:
            res = pool.map(_work, [(f, 1.0) for f in fs])
        import numpy as np
        imgs = [np.frombuffer(b, np.uint8).reshape(sh).astype(np.float32) / 255.0 for _, b, sh in res]
        out = a.out or os.path.join(ROOT, "build", "review", "sheet-%s.jpg" % a.sheet[0])
        core.sheet(imgs, 4 if k > 6 else 3, 480, out)
        print(out)
        return

    if a.shot:
        f0, f1 = timeline.shot_range(a.shot)
        name = a.shot
    elif a.range:
        f0, f1 = a.range
        name = "f%d-%d" % (f0, f1)
    else:
        f0, f1 = 0, timeline.NFRAMES
        name = "full"
    os.makedirs(os.path.join(ROOT, "build", "seg"), exist_ok=True)
    out = a.out or os.path.join(ROOT, "build", "seg", name + ".mp4")
    frames = list(range(f0, f1, a.step))
    w = int(round(core.W * a.scale / 2)) * 2
    h = int(round(core.H * a.scale / 2)) * 2
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
           "-r", str(timeline.FPS / a.step), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-movflags", "+faststart", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        for k, (f, buf, sh) in enumerate(pool.imap(_work, [(f, a.scale) for f in frames], chunksize=1)):
            ff.stdin.write(buf)
            if k % 24 == 0:
                el = time.time() - t0
                print("frame %d (%d/%d) %.0fs elapsed, %.2f s/frame" % (f, k + 1, len(frames), el, el / (k + 1)), flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
