"""The finishing pass every frame goes through, like a compositing department:
bloom, one grade per time of day, vignette, a held paper tooth, fine animated
film grain. Dither happens in core.to8."""
import numpy as np

from . import core, palette

_static = {}


def _paper():
    if "paper" not in _static:
        a = core.fbm(core.H, core.W, 3.0, 2, seed=901)
        b = core.fbm(core.H, core.W, 90.0, 3, seed=902)
        _static["paper"] = ((a - 0.5) * 0.030 + (b - 0.5) * 0.020).astype(np.float32)
    return _static["paper"]


def _vig():
    if "vig" not in _static:
        yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
        r = np.hypot((xx - core.W / 2) / (core.W / 2), (yy - core.H / 2) / (core.H / 2) * 0.92)
        _static["vig"] = np.clip((r - 0.55) / 0.95, 0, 1) ** 1.8
    return _static["vig"]


def bloom(pic, threshold=0.78, amount=0.55, radius=1.0, tint=(1.0, 0.92, 0.80)):
    """Soft glow around the brightest parts of the picture, in place."""
    lum = pic.max(axis=2)
    hot = np.clip((lum - threshold) / (1 - threshold + 1e-6), 0, 1) ** 1.5
    src = pic * hot[..., None]
    small = src[::2, ::2]
    g = 0.5 * core.gblur(small, 5 * radius) + 0.32 * core.gblur(small, 16 * radius) + 0.18 * core.gblur(small, 44 * radius)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)[:pic.shape[0], :pic.shape[1]]
    g = core.gblur(g, 1.2)
    g *= np.asarray(tint, np.float32) * amount
    pic[:] = 1 - (1 - np.clip(pic, 0, 1)) * (1 - np.clip(g, 0, 1))
    return pic


def grade(pic, tod, strength=1.0):
    """Lift, gain and saturation from the palette, plus a gentle filmic curve."""
    t = palette.TOD[tod] if isinstance(tod, str) else tod
    lift, gain, sat = t["grade"]
    out = pic * (1 + (gain - 1) * strength) + lift * strength * (1 - pic)
    g = out.mean(axis=2, keepdims=True)
    out = g + (out - g) * (1 + (sat - 1) * strength)
    out = np.clip(out, 0, 1)
    # soft S curve: a little more contrast in the mids, protected ends
    out = out + 0.10 * strength * (out - 0.5) * (1 - (2 * out - 1) ** 2)
    # the dark cuts (night, first light, launch) get deeper shadows: values under about 0.35
    # are pulled down a little, so the frame has a real black point instead of a blue wash
    name = tod if isinstance(tod, str) else None
    deep = {"night": 0.16, "predawn": 0.13, "launch": 0.15}.get(name, 0.0)
    if deep > 0:
        k = np.clip(1 - out / 0.38, 0, 1)
        out = out * (1 - deep * k)
    pic[:] = np.clip(out, 0, 1)
    return pic


def vignette(pic, amount=0.22, color=(0.05, 0.03, 0.12)):
    v = (_vig() * amount)[..., None]
    pic *= (1 - v)
    pic += np.asarray(color, np.float32) * v
    return pic


def grain(pic, f, amount=0.022, paper=1.0):
    r = core.rng("grain", int(f))
    n = r.standard_normal((core.H // 2 + 1, core.W // 2 + 1), dtype=np.float32)
    n = np.repeat(np.repeat(n, 2, 0), 2, 1)[:core.H, :core.W]
    n2 = r.standard_normal((core.H, core.W), dtype=np.float32)
    g = (0.45 * n + 0.75 * n2) * amount
    lum = pic.mean(axis=2)
    w = 0.35 + 0.65 * np.clip(1 - np.abs(lum - 0.45) * 1.6, 0, 1)   # strongest in the mids
    pic += (g * w)[..., None]
    if paper > 0:
        pic += (_paper() * paper)[..., None]
    np.clip(pic, 0, 1, out=pic)
    return pic


def finish(pic, tod, f, bloom_amt=0.5, bloom_thr=0.78, vig=0.22, grain_amt=0.013, paper=0.8, fade=None):
    """The whole chain. `fade` = (colour, amount) for fades to paper or black."""
    if bloom_amt > 0:
        bloom(pic, bloom_thr, bloom_amt)
    grade(pic, tod)
    if vig > 0:
        vignette(pic, vig)
    if fade is not None and fade[1] > 0:
        c, a = fade
        pic *= (1 - a)
        pic += np.asarray(c, np.float32) * a
    grain(pic, f, grain_amt, paper)
    return pic
