"""Small birds: the flock that bursts from the camphor tree at lift off, and a
few morning birds. Each bird is rebuilt per frame from its flight state: a
teardrop body, a tail, and two wings whose shape follows the flap phase
(up stroke narrow and swept, down stroke broad). No two birds share a size,
a flap rate or a path."""
import math

import numpy as np

from . import core


class Flock:
    """Birds that burst outward from `origin_pts` (set coordinates) at time 0,
    scatter, then gather into a loose flock heading `heading` (dx, dy)."""

    def __init__(self, origin_pts, n=34, heading=(0.85, -0.52), seed=0, speed=520.0, spread=1.0):
        r = core.rng("flock", seed)
        o = np.asarray(origin_pts, np.float64)
        self.n = n
        self.p0 = o[r.integers(0, len(o), n)] + r.normal(0, 26, (n, 2))
        ang = r.uniform(-math.pi, 0.15 * math.pi, n)                  # mostly upward and sideways
        v0 = r.uniform(380, 900, n) * spread
        self.v0 = np.c_[np.cos(ang) * v0, np.sin(ang) * v0 * 0.9 - 120]
        h = np.asarray(heading, np.float64)
        h = h / np.hypot(*h)
        hv = r.uniform(0.75, 1.3, n) * speed
        jit = r.normal(0, 0.16, n)
        self.vf = np.c_[(h[0] * np.cos(jit) - h[1] * np.sin(jit)) * hv, (h[0] * np.sin(jit) + h[1] * np.cos(jit)) * hv]
        self.k = r.uniform(1.6, 3.2, n)                                # how fast each one settles to the flock
        self.delay = np.sort(r.uniform(0, 0.55, n) ** 1.6 * 0.9)       # they do not all leave at once
        self.size = r.uniform(0.75, 1.5, n)
        self.rate = r.uniform(8.5, 12.5, n)                            # wing beats per second
        self.phase = r.uniform(0, 1, n)
        self.wseed = r.integers(0, 10000, n)
        self.depth = r.uniform(0.85, 1.25, n)                          # parallax of each bird

    def state(self, i, t):
        """Position (set coords), velocity, flap phase for bird i at time t (s)."""
        t = t - self.delay[i]
        if t <= 0:
            return None
        k = self.k[i]
        e = math.exp(-k * t)
        a = (1 - e) / k
        p = self.p0[i] + self.v0[i] * a + self.vf[i] * (t - a)
        v = self.v0[i] * e + self.vf[i] * (1 - e)
        w = self.wseed[i]
        p = p + np.array([core.noise1(t * 1.3 + 5, w) * 26, core.noise1(t * 1.7 + 9, w + 3) * 20])
        # frantic beats at first, then steadier with short glides
        rate = self.rate[i] * (1.25 - 0.35 * core.smooth(t / 1.6))
        ph = self.phase[i] + rate * t
        glide = core.smooth((core.noise1(t * 0.9, w + 8) - 0.25) * 4) * core.smooth((t - 1.4) / 0.8)
        return p, v, ph, glide


def draw_bird(cel, x, y, size, vx, vy, ph, glide, col_body, col_wing, col_light=None):
    """One bird at screen (x, y). size = body length in px. ph = flap phase in
    turns; glide 0..1 holds the wings out flat."""
    sp = math.hypot(vx, vy) + 1e-6
    fx, fy = vx / sp, vy / sp                    # forward
    ux, uy = fy, -fx                             # the bird's "up" on screen
    if uy > 0:
        ux, uy = -ux, -uy
    L = size

    def P(a, b):                                 # forward a, up b
        return (x + fx * a * L + ux * b * L, y + fy * a * L + uy * b * L)

    c = math.cos(ph * 2 * math.pi)
    s = math.sin(ph * 2 * math.pi)
    lift = (0.15 + 0.85 * (0.5 - 0.5 * c)) * (1 - glide) + 0.42 * glide      # 0 = wings down, 1 = up
    # down stroke is broad, up stroke is folded and swept back
    broad = (0.55 + 0.45 * max(s, 0.0)) * (1 - glide) + 0.95 * glide
    h = (lift * 2 - 1) * 0.95                    # wing tip height
    span = 1.35 * broad + 0.45                   # long wings: the V and M shapes are what read as "bird"
    sweep = -0.22 - 0.40 * (1 - broad)

    def wing(k, dz):
        """A long pointed wing with a bend at the wrist. k scales it (the far wing is smaller)."""
        tip = (-0.10 + sweep * 1.2, h * span * 1.15 * k + dz)
        wrist = (0.10 + sweep * 0.35, h * span * 0.62 * k + dz + 0.10 * (1 - abs(h)))
        return [P(0.22, 0.05 + dz), P(*wrist), P(*tip), P(tip[0] - 0.10, tip[1] - 0.10 * np.sign(h) if h else tip[1]),
                P(-0.20 + sweep * 0.5, h * span * 0.40 * k + dz - 0.03), P(-0.16, 0.0 + dz)]

    # far wing first (smaller, darker), then body, then near wing
    cel.fill(wing(0.82, 0.05), col_wing)
    body = [P(0.52, 0.02), P(0.30, 0.16), P(-0.05, 0.17), P(-0.42, 0.06), P(-0.78, 0.10), P(-0.80, -0.02),
            P(-0.42, -0.07), P(-0.02, -0.15), P(0.32, -0.10)]
    cel.fill(core.catmull(body, 6), col_body)
    if col_light is not None:
        belly = [P(0.40, -0.02), P(0.10, -0.01), P(-0.30, -0.03), P(-0.05, -0.14), P(0.30, -0.10)]
        cel.fill(core.catmull(belly, 5), col_light)
    cel.fill(wing(1.0, 0.0), col_wing)
    cel.fill([P(0.50, 0.03), P(0.70, -0.01), P(0.50, -0.04)], col_wing)       # beak


def draw_flock(flock, cam, t, f, tod_ink, col=(0.13, 0.12, 0.22), light=(0.95, 0.72, 0.55), base_size=20.0):
    """All birds of a flock at time t as a premultiplied RGBA screen layer."""
    cel = core.Cel(ss=3)
    col = np.asarray(col, np.float32)
    for i in range(flock.n):
        st = flock.state(i, t)
        if st is None:
            continue
        p, v, ph, glide = st
        d = flock.depth[i]
        sx, sy = cam.pt(p[0], p[1], d)
        sc = cam.xf(d)[0]
        if sx < -80 or sx > core.W + 80 or sy < -80 or sy > core.H + 80:
            continue
        size = base_size * flock.size[i] * sc * (0.8 + 0.25 * d)
        wing = col * (0.85 + 0.3 * ((i * 37) % 10) / 10.0)
        draw_bird(cel, sx, sy, size, v[0], v[1], ph, glide, col, np.clip(wing * 1.25, 0, 1),
                  np.asarray(light, np.float32) * 0.75 if i % 3 == 0 else None)
    return cel.rgba()
