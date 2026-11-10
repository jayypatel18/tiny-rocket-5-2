"""Motion check without eyes: decode the master small and grey, measure how much each frame
differs from the one before, and list frames that jump far more than their neighbours inside a
cut (a pop, a flash, a dropped drawing). Hard cuts and the intended flashes are listed apart.

  python3 tools/motioncheck.py [path to mp4]
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tr2.timeline import CUES, FPS, SHOTS  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "master-silent.mp4")
W, H = 320, 180
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=%d:%d,format=gray" % (W, H),
                      "-f", "rawvideo", "-"], capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32) / 255.0
d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))          # d[k] = change from frame k to k+1
cuts = {int(round(a * FPS)) for n, a, b, tod in SHOTS}
flash = {int(round(CUES["ignition"] * FPS)), int(round(CUES["star_ignites"] * FPS))}
print("frames", len(fr), "mean change %.4f" % d.mean())
out = []
for k in range(len(d)):
    f = k + 1
    lo, hi = max(0, k - 8), min(len(d), k + 9)
    nb = np.r_[d[lo:k], d[k + 1:hi]]
    med = np.median(nb) + 1e-4
    ratio = d[k] / med
    near_cut = any(abs(f - c) <= 1 for c in cuts)
    near_flash = any(-1 <= f - c <= 4 for c in flash)
    if ratio > 4.0 and d[k] > 0.012 and not near_cut and not near_flash:
        out.append((ratio, f, d[k], med))
out.sort(reverse=True)
print("suspicious jumps inside cuts (ratio to neighbours, frame, change, neighbour median):")
for r, f, v, m in out[:30]:
    name = [n for n, a, b, tod in SHOTS if a * FPS <= f < b * FPS][0]
    print("  x%.1f  frame %d (%s, t=%.2f)  %.4f vs %.4f" % (r, f, name, f / FPS, v, m))
# frozen stretches (no change at all) outside the stillness beat
still = (int(CUES["stillness"][0] * FPS), int(CUES["one_eye_opens"] * FPS))
z = np.where(d < 0.0006)[0]
z = [int(k + 1) for k in z if not (still[0] - 2 <= k + 1 <= still[1] + 2)]
print("frames with almost no change outside the stillness beat:", len(z), z[:40])
# per cut activity
for n, a, b, tod in SHOTS:
    s = d[int(a * FPS):int(b * FPS) - 1]
    print("  %-12s mean %.4f  max %.4f" % (n, s.mean(), s.max()))
