"""The film: frame number in, finished picture out.

Each cut is a module in tr2/shots/ named after the shot (see timeline.SHOTS)
with one function:

    render(fl, n, f) -> picture (float32 H x W x 3) or (picture, options)

fl = frame inside the shot (may run past n during a dissolve), n = shot length
in frames, f = global frame. options is a dict that may override the finishing
pass: tod (name or blended dict), bloom_amt, bloom_thr, vig, grain_amt, paper,
fade=(colour, amount), finished=True (the shot already ran its own finish).
"""
import importlib
import os

import numpy as np

from . import core, finish, palette
from .timeline import FPS, NFRAMES, SHOTS

# Dissolves between cuts: (index of the outgoing shot, seconds). All other cuts are hard.
DISSOLVES = {1: 1.2,    # dusk hill -> night: time passes
             2: 1.2,    # night -> first light
             11: 0.75}  # the star -> home


def _shot_frame(i, f):
    name, a, b, tod = SHOTS[i]
    f0 = int(round(a * FPS))
    n = int(round((b - a) * FPS))
    fl = f - f0
    try:
        mod = importlib.import_module("tr2.shots." + name)
    except ModuleNotFoundError:
        mod = None
    opts = {}
    if mod is None:
        ys = np.linspace(0, 1, core.H, dtype=np.float32)
        pic = np.repeat(palette.sky_color(tod, ys)[:, None, :], core.W, 1).copy()
    else:
        try:
            res = mod.render(fl, n, f)
        except Exception as e:  # noqa
            # animatic mode (TR2_SAFE=1): a cut that is still being written shows as plain sky
            if os.environ.get("TR2_SAFE") != "1":
                raise
            print("cut %s failed at frame %d: %r" % (name, f, e), flush=True)
            ys = np.linspace(0, 1, core.H, dtype=np.float32)
            res = np.repeat(palette.sky_color(tod, ys)[:, None, :], core.W, 1).copy()
        if isinstance(res, tuple):
            pic, opts = res
        else:
            pic = res
    if not opts.get("finished"):
        kw = {k: opts[k] for k in ("bloom_amt", "bloom_thr", "vig", "grain_amt", "paper", "fade") if k in opts}
        finish.finish(pic, opts.get("tod", tod), f, **kw)
    return pic


def render_frame(f):
    """Finished float picture for global frame f."""
    t = f / FPS
    idx = len(SHOTS) - 1
    for i, (name, a, b, tod) in enumerate(SHOTS):
        if a - 1e-9 <= t < b - 1e-9:
            idx = i
            break
    # inside a dissolve? The dissolve is centred on the cut.
    for i, d in DISSOLVES.items():
        cut = SHOTS[i][2]
        if cut - d / 2 <= t < cut + d / 2:
            u = core.smooth((t - (cut - d / 2)) / d)
            a_pic = _shot_frame(i, f)
            b_pic = _shot_frame(i + 1, f)
            return a_pic * (1 - u) + b_pic * u
    return _shot_frame(idx, f)


def render_u8(f):
    return core.to8(render_frame(f), seed=f)
