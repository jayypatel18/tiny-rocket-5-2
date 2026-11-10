"""Billowing cel smoke, version 2: lobes that ride a simulated flow.

The fluid (fx.SmokeSim) gives the motion; the picture is made of LOBES, tracer
particles that are born small at the fire, are carried outward by the flow,
swell as they age and shrink away at the end of their life. Each lobe is shaded
as a low dome in flat tones. Nothing in the look depends on the frame number
except a slow drift of the edge noise, so the smoke moves smoothly: no popping
(a lobe's place in the stack never changes) and no boiling.

    lob = Lobes(seed)
    lob.spawn(t, x, y, n, r=(12, 60), jit=(40, 16), vel=(0, 0), vjit=(0, 0), life=(2.5, 5.0),
              rise=-12.0, grow=1.0, follow=(0.25, 1.0), key=0)        # all in SET px and seconds
    lob.step(t, dt, sim)            # advect (sim may be None: free lobes)
    arr = lob.alive(t)              # float32 N x 12, cache this per frame

    out = render(arr, t, f, tod, xf=(s, tx, ty), fire=(x, y, amount, reach), seed=0, hw=(H, W))
        xf: set px to screen px (core.Camera.xf(1.0)); fire in SCREEN px
        returns dict(layer=RGBA premultiplied (smoke with its haze), alpha=coverage of the solid smoke)
"""
import math

import numpy as np
from numba import njit

from . import core, palette

W, H = core.W, core.H
# columns
X, Y, R0, TB, RND, VX, VY, LIFE, RISE, GROW, FOL, Z = range(12)


@njit(cache=True, fastmath=True)
def _advance(lob, n, t, dt, u, v, has_sim, ox, oy, cell, gyc, has_ground, drag, lift_k):
    ny, nx = u.shape
    for k in range(n):
        if t < lob[k, TB] or t > lob[k, TB] + lob[k, LIFE]:
            continue
        uu = 0.0
        vv = 0.0
        if has_sim:
            ci = (lob[k, X] - ox) / cell - 0.5
            cj = (lob[k, Y] - oy) / cell - 0.5
            if ci > 0.0 and cj > 0.0 and ci < nx - 1.001 and cj < ny - 1.001:
                i0 = int(ci)
                j0 = int(cj)
                fx = ci - i0
                fy = cj - j0
                uu = ((u[j0, i0] * (1 - fx) + u[j0, i0 + 1] * fx) * (1 - fy) +
                      (u[j0 + 1, i0] * (1 - fx) + u[j0 + 1, i0 + 1] * fx) * fy) * cell
                vv = ((v[j0, i0] * (1 - fx) + v[j0, i0 + 1] * fx) * (1 - fy) +
                      (v[j0 + 1, i0] * (1 - fx) + v[j0 + 1, i0 + 1] * fx) * fy) * cell
        fol = lob[k, FOL]
        if vv < 0.0:
            vv *= lift_k                     # smoke near the ground rolls out far more than it climbs
        lob[k, X] += (uu * fol + lob[k, VX]) * dt
        lob[k, Y] += (vv * fol + lob[k, VY] + lob[k, RISE]) * dt
        lob[k, VX] *= (1 - drag * dt)
        lob[k, VY] *= (1 - drag * dt)
        if has_ground:
            ci = int((lob[k, X] - ox) / cell)
            if ci >= 0 and ci < nx:
                g = oy + gyc[ci] * cell - lob[k, R0] * 0.55
                if lob[k, Y] > g:
                    lob[k, Y] = g
                    if lob[k, VY] > 0:
                        lob[k, VY] = 0.0


class Lobes:
    def __init__(self, seed=0, cap=6000):
        self.a = np.zeros((cap, 12), np.float64)
        self.n = 0
        self.seed = seed
        self.count = 0

    def spawn(self, t, x, y, n, r=(12.0, 60.0), jit=(40.0, 16.0), vel=(0.0, 0.0), vjit=(0.0, 0.0),
              life=(2.5, 5.0), rise=-12.0, grow=1.0, follow=(0.25, 1.0), key=0, pw=3.0, up_only=True,
              kids=0, kid_min=30.0):
        if n <= 0:
            return
        self.count += 1
        rg = core.rng("lobes2", self.seed, key, self.count)
        m = self.n
        if m + n > len(self.a):
            self.a = np.vstack([self.a, np.zeros((max(n, 2000), 12))])
        a = self.a[m:m + n]
        a[:, X] = x + rg.normal(0, jit[0], n)
        jy = rg.normal(0, jit[1], n)
        a[:, Y] = y - (np.abs(jy) if up_only else jy)
        a[:, R0] = r[0] + (rg.random(n) ** pw) * (r[1] - r[0])
        a[:, TB] = t
        a[:, RND] = rg.random(n)
        a[:, VX] = vel[0] + rg.normal(0, vjit[0], n) if vjit[0] > 0 else vel[0]
        a[:, VY] = vel[1] + rg.normal(0, vjit[1], n) if vjit[1] > 0 else vel[1]
        a[:, LIFE] = rg.uniform(life[0], life[1], n)
        a[:, RISE] = rise * rg.uniform(0.5, 1.5, n)
        a[:, GROW] = grow * rg.uniform(0.8, 1.25, n)
        a[:, FOL] = follow[0] + rg.random(n) * (follow[1] - follow[0])
        # a lobe's place in the stack is fixed at birth: small ones ride in front
        a[:, Z] = rg.random(n) * 300.0 + 60.0 * (1.0 - np.clip(a[:, R0] / 60.0, 0, 1))
        self.n += n
        if kids > 0:
            # bumps on bumps: every big lobe carries smaller ones around its crown. They share
            # its birth, speed and life, so they ride with it and the mass reads as cauliflower.
            big = np.nonzero(a[:, R0] >= kid_min)[0]
            for i in big:
                par = a[i].copy()
                nk = int(kids + rg.integers(0, 3))
                if self.n + nk > len(self.a):
                    self.a = np.vstack([self.a, np.zeros((2000, 12))])
                ang = rg.uniform(-0.15 * math.pi, 1.15 * math.pi, nk)
                rad = par[R0] * rg.uniform(0.62, 0.95, nk)
                c = np.repeat(par[None, :], nk, 0)
                c[:, X] = par[X] + np.cos(ang) * rad
                c[:, Y] = par[Y] - np.sin(ang) * rad * 0.85
                c[:, R0] = par[R0] * rg.uniform(0.26, 0.58, nk)
                c[:, Z] = par[Z] + rg.uniform(4.0, 40.0, nk)
                c[:, GROW] = par[GROW] * rg.uniform(0.8, 1.3, nk)
                self.a[self.n:self.n + nk] = c
                self.n += nk

    def step(self, t, dt, sim=None, drag=1.6, lift_k=0.4):
        if self.n == 0:
            return
        if sim is not None:
            gyc = sim.gy if sim.gy is not None else np.zeros(sim.nx)
            _advance(self.a, self.n, t, dt, sim.u, sim.v, True, sim.origin[0], sim.origin[1], sim.cell,
                     np.ascontiguousarray(gyc, np.float64), sim.gy is not None, drag, lift_k)
        else:
            z = np.zeros((2, 2), np.float32)
            _advance(self.a, self.n, t, dt, z, z, False, 0.0, 0.0, 1.0, np.zeros(1), False, drag, lift_k)

    def alive(self, t):
        a = self.a[:self.n]
        ok = (t >= a[:, TB]) & (t <= a[:, TB] + a[:, LIFE])
        return a[ok].astype(np.float32)

    def prune(self, t):
        a = self.a[:self.n]
        ok = t <= a[:, TB] + a[:, LIFE]
        k = int(ok.sum())
        self.a[:k] = a[ok]
        self.n = k


def radius(arr, t):
    """Lobe radius in set px at time t: born tiny, swells fast, keeps expanding slowly,
    shrinks away smoothly at the end of its life."""
    age = np.maximum(t - arr[:, TB], 0.0)
    u = np.clip(age / np.maximum(arr[:, LIFE], 1e-3), 0, 1)
    swell = 0.10 + 0.90 * (1 - np.exp(-age / 0.28))
    slow = 1.0 + 0.75 * arr[:, GROW] * u
    e = np.clip((u - 0.62) / 0.38, 0, 1)
    fade = 1 - e * e * (3 - 2 * e)
    return arr[:, R0] * swell * slow * fade


@njit(cache=True, fastmath=True)
def _lobes(xs, ys, rs, zs, wx, wy, hgt, nx, ny, nz, sd, tvs, tv):
    hh, ww = hgt.shape
    for k in range(xs.shape[0]):
        r = rs[k]
        if r < 0.6:
            continue
        pad = 20
        x0 = max(0, int(xs[k] - r - pad))
        x1 = min(ww, int(xs[k] + r + pad) + 1)
        y0 = max(0, int(ys[k] - r - pad))
        y1 = min(hh, int(ys[k] + r + pad) + 1)
        for j in range(y0, y1):
            for i in range(x0, x1):
                dx = i + wx[j, i] - xs[k]
                dy = j + wy[j, i] - ys[k]
                q2 = dx * dx + dy * dy
                s_ = r - math.sqrt(q2)
                if s_ > sd[j, i]:
                    sd[j, i] = s_
                q = r * r - q2
                if q > 0:
                    hz = math.sqrt(q)
                    h = hz * 0.3 + zs[k]
                    if h > hgt[j, i]:
                        hgt[j, i] = h
                        nx[j, i] = dx / r
                        ny[j, i] = dy / r
                        nz[j, i] = hz / r
                        tv[j, i] = tvs[k]


TONES = dict(
    hot_hi=np.array([1.00, 0.83, 0.46], np.float32),
    hot_mid=np.array([0.98, 0.60, 0.27], np.float32),
    light=np.array([0.97, 0.95, 0.93], np.float32),
)


def render(arr, t, f, tod="launch", xf=(1.0, 0.0, 0.0), fire=None, seed=0, hw=(H, W), haze=0.26,
           warm_top=0.12, soft=3.0, grow=1.0, dark=None):
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    hh, ww = hw
    s, tx, ty = xf
    out_empty = dict(layer=np.zeros((hh, ww, 4), np.float32), alpha=np.zeros((hh, ww), np.float32))
    if arr is None or len(arr) == 0:
        return out_empty
    arr = arr.astype(np.float64)
    r = radius(arr, t) * s * grow
    xs = arr[:, X] * s + tx
    ys = arr[:, Y] * s + ty
    vis = (r > 0.6) & (xs > -r - 30) & (xs < ww + r + 30) & (ys > -r - 30) & (ys < hh + r + 30)
    if not vis.any():
        return out_empty
    xs, ys, r, zs = xs[vis], ys[vis], r[vis], arr[vis, Z]
    tvs = np.ascontiguousarray(arr[vis, RND])
    # the hand in the outline: a warp field anchored to the SET, so lobes glide through it
    # (no boil); it only drifts slowly in time
    z = f * 0.012
    k = max(s, 0.2)
    wx = ((core.fbm(hh, ww, 95.0 * k, 2, seed=seed + 11, ox=-tx, oy=-ty, z=z) - 0.5) * 26.0 * k).astype(np.float32)
    wy = ((core.fbm(hh, ww, 95.0 * k, 2, seed=seed + 12, ox=-tx, oy=-ty, z=z) - 0.5) * 26.0 * k).astype(np.float32)
    fine = core.fbm(hh, ww, 26.0 * k, 2, seed=seed + 13, ox=-tx, oy=-ty, z=z)
    wx += ((fine - 0.5) * 8.0 * k).astype(np.float32)
    hgt = np.zeros((hh, ww), np.float32)
    nx = np.zeros((hh, ww), np.float32)
    ny = np.zeros((hh, ww), np.float32)
    nz = np.ones((hh, ww), np.float32)
    sd = np.full((hh, ww), -1e3, np.float32)
    tv = np.full((hh, ww), 0.5, np.float32)
    _lobes(xs, ys, r, zs, wx, wy, hgt, nx, ny, nz, sd, tvs, tv)
    # silhouette: a soft falloff of a few px, and wispy translucent bites out of the edge
    alpha = np.clip(sd / max(soft * k, 0.8) + 0.5, 0, 1)
    edge = np.clip(1 - sd / (13.0 * k), 0, 1)
    en = np.clip((fine - 0.44) / 0.22, 0, 1)
    alpha = alpha * (1 - 0.62 * edge * en)
    jit = (core.fbm(hh, ww, 70.0 * k, 2, seed=seed + 22, ox=-tx, oy=-ty, z=z) - 0.5) * 0.12
    sx, sy = T["sun"]
    L = np.array([sx * 0.55, sy * 0.3 - 0.78, 0.40])
    L /= np.linalg.norm(L)
    lv = nx * L[0] + ny * L[1] + nz * L[2] + jit
    # every lobe turns the light a little and sets its own threshold: no two highlights are
    # the same oval (some are crescents, some broad caps, some lobes get none)
    ang = (tv - 0.5) * 1.4
    ca, sn = np.cos(ang), np.sin(ang)
    tv2 = np.modf(tv * 7.31)[0]
    lvt = nx * (L[0] * ca - L[1] * sn) + ny * (L[0] * sn + L[1] * ca) + nz * L[2] + jit * 1.8
    a_top = np.clip((lvt - (0.64 + (tv2 - 0.5) * 0.34)) * 60.0 + 0.5, 0, 1)
    a_deep = np.clip((0.06 - lv) * 60.0 + 0.5, 0, 1)
    S = TONES
    c_top = np.clip(palette.mix(palette.lit(S["light"], T), T["sky"][-1][1], warm_top) * 1.07, 0, 1)
    c_body = palette.mix(T["fog"], np.array([0.60, 0.57, 0.74], np.float32), 0.5) * 0.92
    c_deep = palette.mix(palette.shade(S["light"], T), np.array([0.40, 0.33, 0.62], np.float32), 0.6)
    rgb = np.empty((hh, ww, 3), np.float32)
    rgb[:] = c_body
    rgb += (c_top - rgb) * a_top[..., None]
    rgb += (c_deep - rgb) * a_deep[..., None]
    if dark is not None:
        # depth: the mass goes from pale at its lit top to near black violet at the front bottom
        y0, y1, amt = dark
        dk = np.clip((np.arange(hh, dtype=np.float32)[:, None] - y0) / max(y1 - y0, 1.0), 0, 1)
        dk = dk * dk * (3 - 2 * dk) * amt * (0.80 + 0.20 * a_deep + 0.9 * jit)
        dcol = np.array([0.085, 0.07, 0.17], np.float32) * (0.6 + 0.9 * T["amb"])
        rgb += (dcol - rgb) * np.clip(dk, 0, 1)[..., None]
    if fire is not None and fire[2] > 0.01:
        # the fire lights the smoke from inside. The light is read off a SMOOTHED surface, so it
        # falls in a few broad faces on the undersides and inner sides near the flame (never one
        # flake per small lobe), orange near, rose farther out, then nothing: the tops stay cool.
        fxs, fys, amt, reach = fire
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        dx_, dy_ = fxs - xx, fys - yy
        dist = np.maximum(np.hypot(dx_, dy_), 1.0)
        am = (sd > 0).astype(np.float32)
        sg_ = 20.0 * k
        w_ = np.maximum(core.gblur(am, sg_), 1e-3)
        nxs = core.gblur(nx * am, sg_) / w_
        nys = core.gblur(ny * am, sg_) / w_
        fl = np.clip((nxs * dx_ + nys * dy_) / dist * 2.2 + 0.5 * np.clip(nys, 0, 1), 0, 1)
        fall = np.clip(1.12 - dist / reach, 0, 1)
        prox = np.clip(1 - dist / (reach * 0.52), 0, 1)
        up = np.clip(1.0 + 1.7 * np.minimum(nys, 0.0), 0.12, 1.0)          # broad faces that look up stay cool
        gain = 0.55 + 0.9 * core.gblur(tv * am, sg_) / w_                  # lobe by lobe, so the edge of the light is never a circle
        val = (0.70 * fl * fall + 0.80 * prox) * up * (1 - 0.85 * a_top) * gain * amt + jit * 0.35
        rose = np.array([0.74, 0.44, 0.56], np.float32)
        a_ro = np.clip((val - 0.17) * 60.0 + 0.5, 0, 1)
        a_or = np.clip((val - 0.25) * 60.0 + 0.5, 0, 1)
        a_go = np.clip((val - 0.55) * 60.0 + 0.5, 0, 1)
        rgb += (rose - rgb) * a_ro[..., None]
        rgb += (S["hot_mid"] - rgb) * a_or[..., None]
        rgb += (S["hot_hi"] - rgb) * a_go[..., None]
        hz_col = palette.mix(c_body, S["hot_mid"], np.clip((1 - dist / (reach * 0.56)) * amt, 0, 1)[..., None] * 0.7)
    else:
        hz_col = c_body
    rgb = core.gblur(rgb, 0.55)
    layer = core.premul(np.clip(rgb, 0, 1), alpha.astype(np.float32))
    if haze > 0:
        # a faint outer haze: the solid shape breathed out soft, under the smoke itself
        hz = core.gblur(np.clip(sd / (6.0 * k) + 0.5, 0, 1), 18.0 * k) * haze
        hz = hz * (1 - alpha)
        layer[..., :3] += hz_col * hz[..., None]
        layer[..., 3] += hz
    return dict(layer=layer, alpha=alpha)


FLAME = dict(red=np.array([0.80, 0.17, 0.07], np.float32), orange=np.array([1.00, 0.50, 0.12], np.float32), yellow=np.array([1.00, 0.84, 0.32], np.float32),
             white=np.array([1.00, 0.985, 0.88], np.float32))


def flame(f, xf=(1.0, 0.0, 0.0), F=None, grid=None, jet=None, seed=0, hw=(H, W), amount=1.0, focus=None):
    """The fire in three flat tones (white core, yellow, orange tongue), on ones.
    F, grid=(origin_x, origin_y, cell): the simulated flame field in set px (the splash on the ground).
    jet=(x0, y0, x1, y1, w0, w1, foot): the tongue from the nozzle in SCREEN px; foot 1 gives the
    bell where it hits the ground, foot 0 a free tail tapering to a point.
    Returns dict(flame=RGBA, jet=RGBA (the tongue alone), field=HxW)."""
    from . import fx
    from scipy import ndimage
    hh, ww = hw
    s, tx, ty = xf
    k = max(s, 0.2)
    z = f * 0.61
    wx = ((core.fbm(hh, ww, 46.0 * k, 3, seed=seed + 51, z=z) - 0.5) * 22.0 * k).astype(np.float32)
    wy = ((core.fbm(hh, ww, 46.0 * k, 3, seed=seed + 52, z=z) - 0.5) * 22.0 * k).astype(np.float32)
    n_fine = core.fbm(hh, ww, 34.0 * k, 3, seed=seed + 53, z=z)
    Fs = np.zeros((hh, ww), np.float32)
    if F is not None and float(F.max()) > 0.02:
        ox, oy, cell = grid
        sc = cell * s
        g2 = ndimage.zoom(F.astype(np.float32), 2, order=3, mode="nearest", grid_mode=True)
        fx._sample(np.ascontiguousarray(g2, np.float32), Fs, 2.0 / sc, ox * s + tx + 0.25 * sc, oy * s + ty + 0.25 * sc, wx, wy)
        Fs = np.clip(Fs, 0, 1.5)
        if focus is not None:                       # (x, y, radius) in screen px: no loose flakes far from the fire
            yy_, xx_ = np.mgrid[0:hh, 0:ww].astype(np.float32)
            Fs *= np.exp(-(((xx_ - focus[0]) / focus[2]) ** 2 + ((yy_ - focus[1]) / (focus[2] * 0.6)) ** 2))
    jf = None
    if jet is not None:
        x0, y0, x1, y1, w0, w1, foot = jet
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        ax, ay = x1 - x0, y1 - y0
        ln = max(math.hypot(ax, ay), 1.0)
        ax, ay = ax / ln, ay / ln
        px, py = xx + wx * 0.7 - x0, yy + wy * 0.5 - y0
        along = (px * ax + py * ay) / ln                 # 0 at the nozzle, 1 at the end
        across = -px * ay + py * ax
        u = np.clip(along, 0, 1)
        lick = 0.72 + 0.56 * core.fbm(hh, ww, 60.0 * k, 2, seed=seed + 41, z=z, aspect=0.35)
        wd_foot = (w0 + (w1 - w0) * u) * (1.0 + 1.1 * np.clip((u - 0.72) / 0.28, 0, 1) ** 2)
        wd_tail = w0 * (0.25 + 0.95 * u) * np.clip(1.0 - u, 0, 1) ** 0.55 * 1.55 + 1.0
        wd = (wd_foot * foot + wd_tail * (1 - foot)) * lick
        inside = np.clip((along + 0.03) * 40.0, 0, 1) * np.clip((1.0 + 0.12 * foot - along) * (8.0 + 22.0 * (1 - foot)), 0, 1)
        jf = (1.15 / (1.0 + 1.6 * (across / np.maximum(wd, 0.5)) ** 2) ** 1.5 * inside).astype(np.float32)
        jf *= (0.55 + 0.45 * (1 - u * (1 - foot)))
        Fs = np.maximum(Fs, jf)
    Fw = core.gblur(Fs, 1.2 * k) * (0.62 + 0.8 * n_fine) * amount
    a_rd = fx._hard(Fw, 0.10)
    a_or = fx._hard(Fw, 0.21)
    a_ye = fx._hard(Fw, 0.40)
    a_wh = fx._hard(Fw, 0.70)
    rgb = np.empty((hh, ww, 3), np.float32)
    rgb[:] = FLAME["red"]
    rgb += (FLAME["orange"] - rgb) * a_or[..., None]
    rgb += (FLAME["yellow"] - rgb) * a_ye[..., None]
    rgb += (FLAME["white"] - rgb) * a_wh[..., None]
    lay = core.premul(rgb, a_rd.astype(np.float32))
    out = dict(flame=lay, field=Fs)
    if jf is not None:
        out["jet"] = lay * np.clip(jf * 6.0, 0, 1)[..., None]
    return out
