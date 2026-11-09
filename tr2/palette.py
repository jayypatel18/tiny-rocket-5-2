"""One palette for the whole film. Every module takes its colours from here so
characters, props, effects and backgrounds always sit in the same light.

A time of day ("tod") is a dict:
  sky      list of (height 0..1 from top, rgb) stops for the sky gradient
  sun      screen direction the key light comes FROM, as a unit (dx, dy)
           (y grows down, so a low sun on the left is about (-0.9, -0.4))
  key      colour of the key light (multiplies local colour on lit areas)
  amb      colour of the ambient light (multiplies local colour in shadow)
  rim      colour of the rim light on the edge facing the sun
  fog      atmosphere colour that far things fade toward
  fog_k    how strong the atmosphere is (0..1) at the far valley
  ink      colour of the ink line in this light
  grade    (lift rgb, gain rgb, saturation) for the final grade
"""
import numpy as np


def _c(*v):
    return np.array(v, np.float32)


TOD = {
    # Shots 1 and 2: the last warm hour. Sun low on the left, long violet shadows.
    "dusk": dict(
        sky=[(0.0, _c(0.29, 0.36, 0.60)), (0.35, _c(0.62, 0.55, 0.66)), (0.62, _c(0.97, 0.69, 0.50)),
             (0.82, _c(1.00, 0.82, 0.55)), (1.0, _c(1.00, 0.90, 0.68))],
        sun=(-0.88, -0.47), key=_c(1.10, 0.86, 0.62), amb=_c(0.60, 0.50, 0.70), rim=_c(1.0, 0.86, 0.55),
        fog=_c(0.98, 0.72, 0.56), fog_k=0.75, ink=_c(0.25, 0.12, 0.10),
        grade=(_c(0.02, 0.00, 0.03), _c(1.04, 0.99, 0.94), 1.06)),
    # Shot 3: deep starry blue. Moon high on the right, village lights warm below.
    "night": dict(
        sky=[(0.0, _c(0.02, 0.035, 0.11)), (0.45, _c(0.045, 0.08, 0.21)), (0.8, _c(0.09, 0.16, 0.33)),
             (1.0, _c(0.15, 0.24, 0.40))],
        sun=(0.55, -0.83), key=_c(0.50, 0.62, 0.92), amb=_c(0.20, 0.25, 0.47), rim=_c(0.62, 0.78, 1.0),
        fog=_c(0.10, 0.17, 0.34), fog_k=0.8, ink=_c(0.05, 0.05, 0.13),
        grade=(_c(0.00, 0.005, 0.02), _c(0.96, 1.0, 1.06), 1.0)),
    # Shots 4 and 5: cold blue before dawn, one rose band low in the east (right).
    "predawn": dict(
        sky=[(0.0, _c(0.08, 0.13, 0.30)), (0.45, _c(0.20, 0.30, 0.52)), (0.72, _c(0.46, 0.50, 0.68)),
             (0.88, _c(0.78, 0.60, 0.66)), (1.0, _c(0.95, 0.72, 0.62))],
        sun=(0.92, -0.38), key=_c(0.70, 0.80, 1.00), amb=_c(0.36, 0.44, 0.66), rim=_c(1.0, 0.76, 0.70),
        fog=_c(0.50, 0.56, 0.74), fog_k=0.8, ink=_c(0.10, 0.09, 0.18),
        grade=(_c(0.00, 0.01, 0.03), _c(0.96, 1.0, 1.05), 0.96)),
    # Shot 6: the launch. Same dawn, but the fire is the key light (warm, from below).
    "launch": dict(
        sky=[(0.0, _c(0.10, 0.17, 0.38)), (0.45, _c(0.26, 0.36, 0.60)), (0.72, _c(0.58, 0.56, 0.72)),
             (0.88, _c(0.94, 0.66, 0.62)), (1.0, _c(1.00, 0.80, 0.60))],
        sun=(0.92, -0.38), key=_c(0.78, 0.84, 1.00), amb=_c(0.38, 0.44, 0.66), rim=_c(1.0, 0.80, 0.50),
        fog=_c(0.60, 0.60, 0.76), fog_k=0.75, ink=_c(0.12, 0.09, 0.16),
        fire=_c(1.0, 0.72, 0.30),
        grade=(_c(0.01, 0.00, 0.02), _c(1.02, 1.0, 1.0), 1.05)),
    # Shots 7 and 8: overwhelming golden sunrise above the clouds.
    "sunrise": dict(
        sky=[(0.0, _c(0.10, 0.16, 0.42)), (0.30, _c(0.26, 0.38, 0.70)), (0.55, _c(0.62, 0.62, 0.82)),
             (0.72, _c(0.99, 0.72, 0.62)), (0.86, _c(1.00, 0.84, 0.52)), (1.0, _c(1.00, 0.94, 0.72))],
        sun=(0.80, 0.25), key=_c(1.12, 0.92, 0.66), amb=_c(0.52, 0.48, 0.74), rim=_c(1.0, 0.90, 0.60),
        fog=_c(1.00, 0.80, 0.62), fog_k=0.7, ink=_c(0.24, 0.12, 0.12),
        grade=(_c(0.02, 0.00, 0.02), _c(1.05, 1.0, 0.95), 1.08)),
    # Shot 9: clear gentle morning. Sun on the right, cool violet shadows.
    "morning": dict(
        sky=[(0.0, _c(0.30, 0.53, 0.86)), (0.5, _c(0.56, 0.76, 0.95)), (0.85, _c(0.86, 0.93, 0.97)),
             (1.0, _c(0.98, 0.95, 0.88))],
        sun=(0.80, -0.60), key=_c(1.06, 1.00, 0.90), amb=_c(0.62, 0.64, 0.86), rim=_c(1.0, 0.97, 0.84),
        fog=_c(0.80, 0.88, 0.96), fog_k=0.6, ink=_c(0.22, 0.12, 0.10),
        grade=(_c(0.01, 0.01, 0.02), _c(1.02, 1.0, 0.98), 1.05)),
}

# Local (unlit) colours of the cast. Light them with lit() and shade().
PANDA = dict(
    fur=_c(0.84, 0.40, 0.17),        # russet
    fur_dk=_c(0.66, 0.27, 0.12),     # tear marks, tail rings, back of ears
    dark=_c(0.26, 0.15, 0.13),       # legs, belly, paws
    white=_c(0.99, 0.95, 0.87),      # face mask, ear rims, brows, muzzle
    ear_in=_c(0.96, 0.72, 0.66),
    nose=_c(0.16, 0.10, 0.11),
    eye=_c(0.20, 0.11, 0.09),
    iris=_c(0.62, 0.36, 0.14),
    pad=_c(0.52, 0.30, 0.30),
    tongue=_c(0.93, 0.50, 0.48),
)
ROCKET = dict(
    paper=_c(0.93, 0.86, 0.70), paper_dk=_c(0.84, 0.74, 0.56), tin=_c(0.62, 0.67, 0.70),
    kettle=_c(0.50, 0.58, 0.64), copper=_c(0.76, 0.46, 0.26), bamboo=_c(0.72, 0.68, 0.36),
    sail=_c(0.90, 0.84, 0.74), patch_red=_c(0.78, 0.30, 0.24), patch_blue=_c(0.34, 0.46, 0.66),
    cup=_c(0.95, 0.93, 0.88), cup_band=_c(0.30, 0.44, 0.70), star=_c(0.98, 0.80, 0.30),
    tape=_c(0.80, 0.72, 0.50), glass=_c(0.66, 0.84, 0.90), string=_c(0.86, 0.78, 0.62),
)


def lit(color, tod):
    """Local colour on the lit side, in this light."""
    t = TOD[tod] if isinstance(tod, str) else tod
    return np.clip(np.asarray(color, np.float32) * t["key"], 0, 1)


def shade(color, tod):
    """Local colour in the cel shadow: ambient light, slightly richer."""
    t = TOD[tod] if isinstance(tod, str) else tod
    c = np.asarray(color, np.float32) * t["amb"]
    g = c.mean()
    return np.clip(g + (c - g) * 1.12, 0, 1)


def rim(color, tod):
    """The thin lit edge facing the sun."""
    t = TOD[tod] if isinstance(tod, str) else tod
    return np.clip(0.45 * np.asarray(color, np.float32) * t["key"] + 0.62 * t["rim"], 0, 1)


def mix(a, b, t):
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def blend_tod(a, b, t):
    """A time of day part way between two others (for light that changes in shot)."""
    A, B = TOD[a], TOD[b]
    out = {}
    for k in A:
        if k == "sky":
            out[k] = [(ha, mix(ca, cb, t)) for (ha, ca), (hb, cb) in zip(A[k], B[k])] if len(A[k]) == len(B[k]) else (A[k] if t < 0.5 else B[k])
        elif k == "sun":
            v = mix(A[k], B[k], t)
            out[k] = tuple(v / max(np.hypot(*v), 1e-6))
        elif k == "grade":
            out[k] = (mix(A[k][0], B[k][0], t), mix(A[k][1], B[k][1], t), A[k][2] * (1 - t) + B[k][2] * t)
        elif k in B:
            out[k] = mix(A[k], B[k], t)
    return out


def sky_color(tod, v):
    """Sky gradient colour at height v (0 top .. 1 horizon); v may be an array."""
    t = TOD[tod] if isinstance(tod, str) else tod
    hs = np.array([s[0] for s in t["sky"]], np.float32)
    cs = np.stack([s[1] for s in t["sky"]])
    v = np.clip(np.asarray(v, np.float32), 0, 1)
    return np.stack([np.interp(v, hs, cs[:, k]) for k in range(3)], -1).astype(np.float32)
