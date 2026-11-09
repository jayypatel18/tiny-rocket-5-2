"""Effects: simulated smoke and fire in cel tones, sparks, light.

API (screen coordinates unless a Camera is named; f = film frame):

  Smoke and fire
    sim = SmokeSim(nx=384, ny=216, cell=5.0, origin=(0, 0), ground=fn_or_None)
        a 2D fluid (semi Lagrangian advection, pressure projection, buoyancy,
        vorticity confinement) with density d, temperature T and flame F.
        origin and cell map grid cells to SET (or screen) pixels.
    sim.step(dt, emit=dict(x, y, r, jet, rate, burst))      one substep
    fields = sim.fields()                                    dict(d, T, F) float32
    save_fields(path, fields) / load_fields(path)            float16, compressed, one file per frame
    lay = smoke_layers(fields, f, tod, scale, ox, oy, seed=0)
        scale, ox, oy: grid cell (i, j) lands at screen (ox + i*scale, oy + j*scale)
        returns dict(smoke=RGBA layer, flame=RGBA layer, heat=HxW 0..1 fire light field,
                     alpha=HxW smoke coverage)
  Sparks
    lay = sparks_layer(t, origin, seed, n=260, ground_y=None, spread=(8, 172), speed=(350, 1500),
                       burst=0.35, shutter=1/40, size=1.0)   t = seconds since ignition
  Light
    bloom(pic, threshold=0.78, radius=14, amount=0.6)
    glow_at(pic, x, y, radius, color, amount, squash=1.0)
    ground_light(pic, x, y, radius, color, amount, mask=None)   warm light ON surfaces (multiplies)
    light_shafts(pic, sun_xy, occlusion_alpha, amount=0.5, color=(1, .9, .7), length=0.6)
"""
import math
import os

import numpy as np
from numba import njit, prange
from scipy import ndimage

from . import core, palette

W, H = core.W, core.H


# =============================================================== fluid solver
@njit(cache=True, fastmath=True, parallel=True)
def _advect(q, u, v, dt, out, solid):
    ny, nx = q.shape
    for j in prange(ny):
        for i in range(nx):
            if solid[j, i]:
                out[j, i] = 0.0
                continue
            x = i - u[j, i] * dt
            y = j - v[j, i] * dt
            if x < 0.0:
                x = 0.0
            if x > nx - 1.001:
                x = nx - 1.001
            if y < 0.0:
                y = 0.0
            if y > ny - 1.001:
                y = ny - 1.001
            i0 = int(x)
            j0 = int(y)
            fx = x - i0
            fy = y - j0
            out[j, i] = (q[j0, i0] * (1 - fx) + q[j0, i0 + 1] * fx) * (1 - fy) + \
                        (q[j0 + 1, i0] * (1 - fx) + q[j0 + 1, i0 + 1] * fx) * fy


@njit(cache=True, fastmath=True)
def _project(u, v, solid, p, div, src, iters):
    ny, nx = u.shape
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if solid[j, i]:
                div[j, i] = 0.0
                continue
            ur = 0.0 if solid[j, i + 1] else u[j, i + 1]
            ul = 0.0 if solid[j, i - 1] else u[j, i - 1]
            vd = 0.0 if solid[j + 1, i] else v[j + 1, i]
            vu = 0.0 if solid[j - 1, i] else v[j - 1, i]
            div[j, i] = 0.5 * (ur - ul + vd - vu) - src[j, i]
    for it in range(iters):
        for col in range(2):
            for j in range(1, ny - 1):
                for i in range(1 + ((j + col) & 1), nx - 1, 2):
                    if solid[j, i]:
                        continue
                    pc = p[j, i]
                    pr = pc if solid[j, i + 1] else p[j, i + 1]
                    pl = pc if solid[j, i - 1] else p[j, i - 1]
                    pd = pc if solid[j + 1, i] else p[j + 1, i]
                    pu = pc if solid[j - 1, i] else p[j - 1, i]
                    pn = (pr + pl + pd + pu - div[j, i]) * 0.25
                    p[j, i] = pc + 1.7 * (pn - pc)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if solid[j, i]:
                u[j, i] = 0.0
                v[j, i] = 0.0
                continue
            pc = p[j, i]
            pr = pc if solid[j, i + 1] else p[j, i + 1]
            pl = pc if solid[j, i - 1] else p[j, i - 1]
            pd = pc if solid[j + 1, i] else p[j + 1, i]
            pu = pc if solid[j - 1, i] else p[j - 1, i]
            u[j, i] -= 0.5 * (pr - pl)
            v[j, i] -= 0.5 * (pd - pu)


@njit(cache=True, fastmath=True, parallel=True)
def _forces(u, v, T, d, dt, beta, alpha, eps, wk):
    ny, nx = u.shape
    for j in prange(1, ny - 1):
        for i in range(1, nx - 1):
            wk[j, i] = 0.5 * ((v[j, i + 1] - v[j, i - 1]) - (u[j + 1, i] - u[j - 1, i]))
    for j in prange(2, ny - 2):
        for i in range(2, nx - 2):
            gx = 0.5 * (abs(wk[j, i + 1]) - abs(wk[j, i - 1]))
            gy = 0.5 * (abs(wk[j + 1, i]) - abs(wk[j - 1, i]))
            m = math.sqrt(gx * gx + gy * gy) + 1e-5
            gx /= m
            gy /= m
            w = wk[j, i]
            u[j, i] += eps * gy * w * dt
            v[j, i] += -eps * gx * w * dt
            v[j, i] += (-beta * T[j, i] + alpha * d[j, i]) * dt


@njit(cache=True, fastmath=True)
def _trace(lob, u, v, dt, gyc):
    ny, nx = u.shape
    for k in range(lob.shape[0]):
        x = min(max(lob[k, 0], 0.0), nx - 1.001)
        y = min(max(lob[k, 1], 0.0), ny - 1.001)
        i0 = int(x)
        j0 = int(y)
        fx = x - i0
        fy = y - j0
        uu = (u[j0, i0] * (1 - fx) + u[j0, i0 + 1] * fx) * (1 - fy) + (u[j0 + 1, i0] * (1 - fx) + u[j0 + 1, i0 + 1] * fx) * fy
        vv = (v[j0, i0] * (1 - fx) + v[j0, i0 + 1] * fx) * (1 - fy) + (v[j0 + 1, i0] * (1 - fx) + v[j0 + 1, i0 + 1] * fx) * fy
        fol = 0.22 + 0.78 * lob[k, 4]            # heavy lobes lag behind the flow and stay near the fire
        lob[k, 0] += (uu * fol + lob[k, 5]) * dt
        lob[k, 1] += (vv * fol + lob[k, 6]) * dt
        lob[k, 5] *= (1 - 1.5 * dt)
        lob[k, 6] *= (1 - 1.5 * dt)
        g = gyc[min(max(int(lob[k, 0]), 0), nx - 1)] - lob[k, 2] * 0.75
        if lob[k, 1] > g:
            lob[k, 1] = g


@njit(cache=True, fastmath=True)
def _lobes(xs, ys, rs, zs, wx, wy, hgt, nx, ny, nz, sd):
    hh, ww = hgt.shape
    pad = 16
    for k in range(xs.shape[0]):
        r = rs[k]
        if r < 0.5:
            continue
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


def lobe_smoke(fields, f, tod="launch", scale=5.0, ox=0.0, oy=0.0, seed=0, out_hw=(H, W), fire_xy=None,
               fire_amt=1.0, fire_reach=500.0, grow=1.0):
    """Cel smoke built of rounded lobes that ride the simulated flow. Each lobe is shaded as
    a little dome: warm off white where the sky lights it, a grey violet body, a deep violet
    crescent underneath, and an orange to gold band on the faces that look at the fire.
    Returns (RGBA layer, alpha)."""
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    hh, ww = out_hw
    lob = fields["lobes"].astype(np.float64)
    t = float(fields["t"])
    z = core.on_n(int(f), 2) * 0.37
    wx = ((core.fbm(hh, ww, 95.0, 2, seed=seed + 11, z=z) - 0.5) * 26.0).astype(np.float32)
    wy = ((core.fbm(hh, ww, 95.0, 2, seed=seed + 12, z=z) - 0.5) * 26.0).astype(np.float32)
    wx += ((core.fbm(hh, ww, 26.0, 2, seed=seed + 13, z=z) - 0.5) * 9.0).astype(np.float32)
    wy += ((core.fbm(hh, ww, 26.0, 2, seed=seed + 14, z=z) - 0.5) * 9.0).astype(np.float32)
    age = np.maximum(t - lob[:, 3], 0.0)
    r = lob[:, 2] * scale * (0.42 + 1.55 * (1 - np.exp(-age / 0.30))) * grow
    xs = ox + (lob[:, 0] + 0.5) * scale
    ys = oy + (lob[:, 1] + 0.5) * scale
    # painter's order: lower (nearer) lobes sit in front, so most lobes show a whole round face
    zs = (ys - ys.min()) / max(float(np.ptp(ys)), 1.0) * 260.0 + lob[:, 4] * 150.0 + r * 0.5
    hgt = np.zeros((hh, ww), np.float32)
    nx = np.zeros((hh, ww), np.float32)
    ny = np.zeros((hh, ww), np.float32)
    nz = np.ones((hh, ww), np.float32)
    sd = np.full((hh, ww), -1e3, np.float32)
    _lobes(xs, ys, r, zs, wx, wy, hgt, nx, ny, nz, sd)
    alpha = np.clip(sd + 0.5, 0, 1)
    jit = (core.fbm(hh, ww, 70.0, 2, seed=seed + 22, z=z) - 0.5) * 0.12
    sx, sy = T["sun"]
    L = np.array([sx * 0.55, sy * 0.3 - 0.78, 0.40])
    L /= np.linalg.norm(L)
    lv = nx * L[0] + ny * L[1] + nz * L[2] + jit
    a_top = np.clip((lv - 0.72) * 60.0 + 0.5, 0, 1)
    a_deep = np.clip((0.06 - lv) * 60.0 + 0.5, 0, 1)
    S = SMOKE
    c_top = palette.mix(palette.lit(S["light"], T), T["sky"][-1][1], 0.38) * 1.02
    c_body = palette.mix(T["fog"], np.array([0.60, 0.57, 0.74], np.float32), 0.5)
    c_deep = palette.mix(palette.shade(S["light"], T), np.array([0.40, 0.33, 0.62], np.float32), 0.6)
    rgb = np.empty((hh, ww, 3), np.float32)
    rgb[:] = c_body
    rgb += (c_top - rgb) * a_top[..., None]
    rgb += (c_deep - rgb) * a_deep[..., None]
    if fire_xy is not None and fire_amt > 0.01:
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        tx, ty = fire_xy[0] - xx, fire_xy[1] - yy
        dist = np.maximum(np.hypot(tx, ty), 1.0)
        fl = (nx * tx + ny * ty) / dist * 0.93 + nz * 0.30
        fall = np.clip(1.22 - dist / fire_reach, 0, 1)
        near = np.clip(1 - dist / (fire_reach * 0.56), 0, 1)
        val = (np.clip(fl, -0.2, 1) * 0.85 * fall + 0.50 * near) * fire_amt + jit * 0.5
        a_or = np.clip((val - 0.24) * 60.0 + 0.5, 0, 1)
        a_go = np.clip((val - 0.66) * 60.0 + 0.5, 0, 1)
        rgb += (S["hot_mid"] - rgb) * a_or[..., None]
        rgb += (S["hot_hi"] - rgb) * a_go[..., None]
    rgb *= (0.975 + 0.05 * core.fbm(hh, ww, 9.0, 2, seed=seed + 31, aspect=4.0))[..., None]
    rgb = core.gblur(rgb, 0.55)
    return core.premul(np.clip(rgb, 0, 1), alpha.astype(np.float32)), alpha


class SmokeSim:
    def __init__(self, nx=384, ny=216, cell=5.0, origin=(0.0, 0.0), ground=None, seed=0):
        self.nx, self.ny, self.cell, self.origin = nx, ny, float(cell), origin
        z = lambda: np.zeros((ny, nx), np.float32)
        self.u, self.v, self.d, self.T, self.F = z(), z(), z(), z(), z()
        self.p, self.div, self.src, self.wk, self.tmp = z(), z(), z(), z(), z()
        self.solid = np.zeros((ny, nx), np.bool_)
        self.gy = None
        if ground is not None:
            xs = origin[0] + (np.arange(nx) + 0.5) * cell
            self.gy = (np.asarray([ground(x) for x in xs]) - origin[1]) / cell
            jj = np.arange(ny)[:, None]
            self.solid = np.ascontiguousarray(jj >= self.gy[None, :])
        self.t = 0.0
        self.seed = seed
        self.lob = np.zeros((0, 7), np.float64)     # x, y (cells), r0 (cells), birth, rand, du, dv
        self.beta, self.alpha, self.eps = 120.0, 14.0, 9.0
        self.cool_T, self.cool_F, self.fade_d = 1.1, 6.5, 0.10
        self.turb = 28.0
        jj, ii = np.mgrid[0:ny, 0:nx]
        self._jj, self._ii = jj.astype(np.float32), ii.astype(np.float32)

    def cell_of(self, x, y):
        return (x - self.origin[0]) / self.cell - 0.5, (y - self.origin[1]) / self.cell - 0.5

    def spawn(self, x, y, n, r=(18.0, 50.0), jit=(30.0, 14.0), stray=0.09, key=0):
        """Add n smoke lobes (tracers that ride the flow) around a point, sizes in px."""
        if n <= 0:
            return
        rg = core.rng("lobe", self.seed, key, int(round(self.t * 960)))
        ci, cj = self.cell_of(x, y)
        xs = ci + rg.normal(0, jit[0], n) / self.cell
        ys = cj - np.abs(rg.normal(0, jit[1], n)) / self.cell
        rr = (r[0] + (rg.random(n) ** 1.8) * (r[1] - r[0])) / self.cell
        rnd = rg.random(n)
        st = rnd > 1 - stray
        du = np.where(st, rg.normal(0, 30, n), 0.0) / self.cell * 8
        dv = np.where(st, -rg.uniform(20, 70, n), 0.0) / self.cell * 6
        rr = np.where(st, rr * 0.4, rr)
        new = np.c_[xs, ys, rr, np.full(n, self.t), rnd, du, dv]
        if st.any():
            # a stray is never one bead: give it two smaller companions so it reads as a wisp
            base = new[st]
            for q in range(2):
                c = base.copy()
                ang = rg.uniform(0, 2 * math.pi, len(c))
                c[:, 0] += np.cos(ang) * c[:, 2] * (1.2 + 0.5 * q)
                c[:, 1] += np.sin(ang) * c[:, 2] * (0.8 + 0.3 * q)
                c[:, 2] *= 0.72 - 0.2 * q
                c[:, 4] = np.clip(c[:, 4] - 0.004 * (q + 1), 0, 1)
                new = np.vstack([new, c])
        self.lob = np.vstack([self.lob, new])

    def step(self, dt, emit=None):
        """emit: dict(x, y) in the same space as origin; r = jet radius px; jet = speed px/s
        (downward, or (jx, jy)); rate = smoke per second; burst = radial push (px/s per cell)
        with burst_r px and burst_y the height of its centre."""
        u, v, d, T, F = self.u, self.v, self.d, self.T, self.F
        self.src[:] = 0
        k = int(round(self.t * 96))
        if emit is not None:
            ci, cj = self.cell_of(emit["x"], emit["y"])
            r = emit.get("r", 30.0) / self.cell
            jet = emit.get("jet", 1200.0)
            jx, jy = (0.0, jet) if np.isscalar(jet) else jet
            jx /= self.cell
            jy /= self.cell
            el = ((self._ii - ci) / r) ** 2 + ((self._jj - cj) / (r * 1.5)) ** 2
            m = el < 1.0
            rg = core.rng("emit", self.seed, k)
            nz = rg.normal(0, 1, 2).astype(np.float32)
            core_w = np.clip(1.25 - el, 0, 1)[m]
            v[m] = jy * (0.75 + 0.25 * core_w)
            u[m] = jx + (self._ii[m] - ci) / r * abs(jy) * 0.22 + nz[0] * abs(jy) * 0.10
            F[m] = np.maximum(F[m], emit.get("heat", 1.0) * core_w)
            T[m] = np.maximum(T[m], 0.8)
            d[m] += emit.get("rate", 9.0) * dt
            b = emit.get("burst", 0.0)
            if b > 0:
                br = emit.get("burst_r", 60.0) / self.cell
                bi, bj = self.cell_of(emit["x"], emit.get("burst_y", emit["y"]))
                g = np.exp(-(((self._ii - bi) / br) ** 2 + ((self._jj - bj) / (br * 0.6)) ** 2))
                self.src += (b / self.cell) * g * dt * 96
                d += g * emit.get("burst_d", 30.0) * dt
                T[:] = np.maximum(T, g * 0.9)
                F[:] = np.maximum(F, g * emit.get("burst_heat", 0.9))
            self.spawn(emit["x"], emit.get("lobe_y", emit["y"]), emit.get("lobes", 0), emit.get("lobe_r", (24.0, 62.0)),
                       emit.get("lobe_jit", (55.0, 22.0)), key=1)
            for q_, pf in enumerate(emit.get("puffs", ())):
                self.spawn(pf["x"], pf["y"] + 20.0, pf.get("lobes", 0), pf.get("lobe_r", (16.0, 42.0)), (26.0, 16.0),
                           key=10 + q_)
            for pf in emit.get("puffs", ()):
                # a moving ground puff (the ignition ring seen from the side): smoke, a little
                # heat and a push, laid down as a soft blob
                pi, pj = self.cell_of(pf["x"], pf["y"])
                pr_ = pf.get("r", 40.0) / self.cell
                g = np.exp(-(((self._ii - pi) / pr_) ** 2 + ((self._jj - pj) / (pr_ * pf.get("flat", 0.6))) ** 2))
                d += g * pf.get("d", 20.0) * dt
                T[:] = np.maximum(T, g * pf.get("T", 0.3))
                if pf.get("F", 0.0) > 0:
                    F[:] = np.maximum(F, g * pf["F"])
                k_ = np.clip(g * 1.5, 0, 1)
                u += (pf.get("u", 0.0) / self.cell - u) * k_ * pf.get("grip", 0.5)
                v += (pf.get("v", 0.0) / self.cell - v) * k_ * pf.get("grip", 0.5)
        # turbulence: a curl noise push wherever there is smoke
        if self.turb > 0:
            psi = core.fbm(self.ny, self.nx, 16.0, 3, seed=self.seed + 3, z=self.t * 2.2)
            gy_, gx_ = np.gradient(psi)
            wgt = np.clip(d * 1.5, 0, 1) * self.turb * 60 * dt
            u += gy_ * wgt
            v -= gx_ * wgt
        _forces(u, v, T, d, dt, self.beta, self.alpha, self.eps, self.wk)
        _advect(u, u, v, dt, self.tmp, self.solid)
        un = self.tmp.copy()
        _advect(v, u, v, dt, self.tmp, self.solid)
        self.u, self.v = un, self.tmp.copy()
        _project(self.u, self.v, self.solid, self.p, self.div, self.src, 50)
        for name in ("d", "T", "F"):
            q = getattr(self, name)
            out = np.empty_like(q)
            _advect(q, self.u, self.v, dt, out, self.solid)
            setattr(self, name, out)
        if len(self.lob):
            gyc = self.gy if self.gy is not None else np.full(self.nx, 1e9)
            _trace(self.lob, self.u, self.v, dt, np.ascontiguousarray(gyc, np.float64))
        # fire turns to smoke as it burns out; heat leaks away
        self.d += self.F * 5.0 * dt
        self.F *= math.exp(-self.cool_F * dt)
        self.T *= math.exp(-self.cool_T * dt)
        self.d *= math.exp(-self.fade_d * dt)
        np.clip(self.d, 0, 3.0, out=self.d)
        self.t += dt

    def fields(self):
        return dict(d=self.d.copy(), T=self.T.copy(), F=self.F.copy(), lobes=self.lob.astype(np.float32),
                    t=np.float32(self.t))


def save_fields(path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in fields.items()})


def load_fields(path):
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


# ============================================================ painted render
@njit(cache=True, fastmath=True, parallel=True)
def _sample(src, out, inv, ox, oy, wx, wy):
    sh, sw = src.shape
    oh, ow = out.shape
    for j in prange(oh):
        for i in range(ow):
            x = (i + wx[j, i] - ox) * inv
            y = (j + wy[j, i] - oy) * inv
            if x < 0.0 or y < 0.0 or x > sw - 1.001 or y > sh - 1.001:
                out[j, i] = 0.0
                continue
            i0 = int(x)
            j0 = int(y)
            fx = x - i0
            fy = y - j0
            out[j, i] = (src[j0, i0] * (1 - fx) + src[j0, i0 + 1] * fx) * (1 - fy) + \
                        (src[j0 + 1, i0] * (1 - fx) + src[j0 + 1, i0 + 1] * fx) * fy


def _hard(field, thr, aa=1.1, gmin=0.004):
    """A crisp anti aliased step of a smooth field at a threshold: the edge is
    about `aa` pixels wide wherever the field has a real slope."""
    gy, gx = np.gradient(field)
    g = np.maximum(np.hypot(gx, gy), gmin)
    return np.clip(0.5 + (field - thr) / (g * aa * 2), 0, 1)


SMOKE = dict(
    light=np.array([0.97, 0.95, 0.93], np.float32),
    hot_hi=np.array([1.00, 0.83, 0.46], np.float32),
    hot_mid=np.array([0.98, 0.60, 0.27], np.float32),
    hot_lo=np.array([0.88, 0.43, 0.22], np.float32),
    warm_hi=np.array([0.96, 0.82, 0.78], np.float32),
    warm_mid=np.array([0.80, 0.57, 0.64], np.float32),
    warm_lo=np.array([0.52, 0.35, 0.56], np.float32),
    fl_or=np.array([1.00, 0.50, 0.12], np.float32),
    fl_ye=np.array([1.00, 0.84, 0.32], np.float32),
    fl_wh=np.array([1.00, 0.985, 0.88], np.float32),
)


def _density_smoke(d, heat, n_fine, T, z, seed, thr, fire_xy, dim, hh, ww):
    """Fallback look when a field has no lobe tracers: tones from a billow relief of the density."""
    n_lobe = core.fbm(hh, ww, 135.0, 3, seed=seed + 21, z=z * 0.6)
    bil = np.abs(2.0 * n_lobe - 1.0)
    dd = d * (0.62 + 0.7 * n_fine) * (0.50 + 1.25 * bil)
    alpha = _hard(core.gblur(dd, 1.6), thr)
    # form: treat the smoke as a relief of big round lobes with sharp creases between them
    # (billow noise), and light that relief from the sky side. One light, three flat tones.
    body = core.gblur(np.minimum(dd, 1.0), 9.0)
    wide = core.gblur(np.minimum(dd, 1.0), 34.0)
    relief = body * (0.45 + 1.1 * bil) + 0.9 * wide
    gy, gx = np.gradient(core.gblur(relief, 3.0))
    sx, sy = T["sun"]
    sx, sy = sx * 0.75, sy * 0.75 - 0.70
    nrm = math.hypot(sx, sy)
    lv = core.gblur(-(gx * sx + gy * sy) / nrm * 80.0 + 0.18, 5.0)
    a_mid = 1 - _hard(lv, -0.04, 1.2, 0.01)       # away from the light
    a_lo = 1 - _hard(lv, -0.52, 1.2, 0.01)
    # fire light from inside: near the flame the tones turn warm, then hot, most of all on
    # the faces of the lobes that look toward the fire
    hn = heat * (0.85 + 0.3 * n_lobe)
    if fire_xy is not None:
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        tx, ty = fire_xy[0] - xx, fire_xy[1] - yy
        tl = np.maximum(np.hypot(tx, ty), 1.0)
        facing = core.gblur((gx * tx + gy * ty) / tl * -95.0, 3.0)
        hn = hn * np.clip(0.62 + 0.75 * facing, 0.25, 1.7)
    a_warm = _hard(hn, 0.20, 1.3, 0.002)
    a_hot = _hard(hn, 0.52, 1.3, 0.002)
    S = SMOKE
    c_hi = palette.lit(S["light"], T) * dim
    c_lo = np.array([0.44, 0.40, 0.70], np.float32) * (0.62 + 0.5 * T["amb"]) * (0.5 + 0.5 * dim)
    c_mid = palette.mix(c_hi, c_lo, 0.55) * np.array([0.97, 0.97, 1.06], np.float32)

    def tones(hi, mid, lo):
        rgb = np.empty((hh, ww, 3), np.float32)
        rgb[:] = hi
        rgb += (mid - hi) * a_mid[..., None]
        rgb += (lo - mid) * a_lo[..., None]
        return rgb

    rgb = tones(c_hi, c_mid, c_lo)
    rgb += (tones(S["warm_hi"], S["warm_mid"], S["warm_lo"]) - rgb) * a_warm[..., None]
    rgb += (tones(S["hot_hi"], S["hot_mid"], S["hot_lo"]) - rgb) * a_hot[..., None]
    # dry brush: a faint value jitter along the lobes so the tones feel painted, not filled
    tex = core.fbm(hh, ww, 9.0, 2, seed=seed + 31, aspect=4.0)
    rgb *= (0.975 + 0.05 * tex)[..., None]
    smoke = core.premul(np.clip(rgb, 0, 1), alpha.astype(np.float32))
    return smoke, alpha


def smoke_layers(fields, f, tod="launch", scale=5.0, ox=0.0, oy=0.0, seed=0, out_hw=(H, W), thr=0.22,
                 fire_xy=None, dim=0.86, jet=None, fire_amt=1.0):
    T = palette.TOD[tod] if isinstance(tod, str) else tod
    hh, ww = out_hw
    z = core.on_n(int(f), 2) * 0.37
    # upsample: cubic to a 2x grid, then bilinear with a noise warp so edges are alive and sharp
    wn1 = core.fbm(hh, ww, 46.0, 3, seed=seed + 11, z=z)
    wn2 = core.fbm(hh, ww, 46.0, 3, seed=seed + 12, z=z)
    wx = ((wn1 - 0.5) * 22.0).astype(np.float32)
    wy = ((wn2 - 0.5) * 22.0).astype(np.float32)
    up = {}
    for k in ("d", "F"):
        g2 = ndimage.zoom(fields[k], 2, order=3, mode="nearest", grid_mode=True)
        o = np.empty((hh, ww), np.float32)
        _sample(np.ascontiguousarray(g2, np.float32), o, 2.0 / scale, ox + 0.25 * scale, oy + 0.25 * scale, wx, wy)
        up[k] = o
    d, F = np.clip(up["d"], 0, 3), np.clip(up["F"], 0, 1.5)
    if jet is not None:
        # the hard jet from the nozzle down to where it splashes: a tapering tongue
        x0, y0, x1, y1, w0, w1 = jet
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        u_ = np.clip((yy + wy * 0.5 - y0) / max(y1 - y0, 1.0), 0, 1)
        xc = x0 + (x1 - x0) * u_
        wd = (w0 + (w1 - w0) * u_) * (1.0 + 1.1 * np.clip((u_ - 0.72) / 0.28, 0, 1) ** 2)
        wd = wd * (0.72 + 0.56 * core.fbm(hh, ww, 60.0, 2, seed=seed + 41, z=int(f) * 0.61, aspect=0.35))
        inside = ((yy > y0 - 4) & (yy < y1 + 30)).astype(np.float32)
        jf = 1.15 / (1.0 + 1.6 * ((xx + wx * 0.7 - xc) / wd) ** 2) ** 1.5 * inside
        jf *= np.clip((y1 + 34 - yy) / 30.0, 0, 1)
        F = np.maximum(F, jf)
        jet_a = np.clip(jf * 6.0, 0, 1)
    n_fine = core.fbm(hh, ww, 34.0, 3, seed=seed + 22, z=z)
    heat = np.clip(core.gblur(F, 26.0) * 2.2 + core.gblur(F, 95.0) * 3.6 + core.gblur(F, 200.0) * 5.0, 0, 2.0)
    S = SMOKE
    if "lobes" in fields and len(fields["lobes"]):
        smoke, alpha = lobe_smoke(fields, f, tod, scale, ox, oy, seed, out_hw, fire_xy, fire_amt)
    else:
        smoke, alpha = _density_smoke(d, heat, n_fine, T, z, seed, thr, fire_xy, dim, hh, ww)
    # the flame: three flat tones with hard inner shapes
    Fw = core.gblur(F, 1.2) * (0.62 + 0.8 * n_fine)
    a_or = _hard(Fw, 0.20)
    a_ye = _hard(Fw, 0.42)
    a_wh = _hard(Fw, 0.74)
    frgb = np.empty((hh, ww, 3), np.float32)
    frgb[:] = S["fl_or"]
    frgb += (S["fl_ye"] - S["fl_or"]) * a_ye[..., None]
    frgb += (S["fl_wh"] - frgb) * a_wh[..., None]
    flame = core.premul(frgb, a_or.astype(np.float32))
    out = dict(smoke=smoke, flame=flame, heat=np.clip(heat, 0, 1), alpha=alpha)
    if jet is not None:
        out["jet"] = flame * jet_a[..., None]      # the nozzle jet alone, to lay over front smoke
    return out


# ===================================================================== sparks
@njit(cache=True, fastmath=True)
def _spark_sim(px, py, vx, vy, tb, life, bounce, gy, t, g, drag, ox, oy):
    n = px.shape[0]
    dt = 1.0 / 240.0
    for k in range(n):
        age = t - tb[k]
        if age <= 0 or age > life[k]:
            px[k] = -1e6
            continue
        x, y, a, b = ox[k], oy[k], vx[k], vy[k]
        steps = int(age / dt)
        for s in range(steps):
            b += g * dt
            a *= (1 - drag * dt)
            b *= (1 - drag * dt)
            x += a * dt
            y += b * dt
            if y > gy[k] and b > 0:
                if bounce[k] > 0:
                    y = gy[k]
                    b = -b * bounce[k]
                    a *= 0.7
                else:
                    a *= 0.9
                    b = 0.0
                    y = gy[k]
        px[k], py[k], vx[k], vy[k] = x, y, a, b


def sparks_state(t, origin, seed=0, n=260, ground_y=None, spread=(8.0, 172.0), speed=(350.0, 1500.0),
                 burst=0.35, gravity=1500.0, drag=1.6):
    """Positions and velocities of the sparks t seconds after ignition."""
    rg = core.rng("sparks", seed)
    tb = rg.random(n) * max(t, burst) - 0.02 * rg.random(n)
    tb[: n // 3] = rg.random(n // 3) * 0.06                       # the first burst
    side = rg.random(n) < 0.5
    lo, hi = spread
    a = lo + (rg.random(n) ** 1.6) * (90 - lo) * 0.95
    a = np.where(side, a, 180 - a)
    a = np.clip(a + rg.normal(0, 6, n), min(lo, 3), max(hi, 177)) * math.pi / 180
    sp = speed[0] + (rg.random(n) ** 1.5) * (speed[1] - speed[0])
    vx = np.cos(a) * sp
    vy = -np.sin(a) * sp
    life = 0.22 + rg.random(n) ** 1.5 * 0.9
    bounce = np.where(rg.random(n) < 0.3, rg.uniform(0.3, 0.55, n), 0.0)
    ox = origin[0] + rg.normal(0, 14, n)
    oy = origin[1] + rg.normal(0, 6, n)
    if ground_y is None:
        gy = np.full(n, 1e9)
    elif callable(ground_y):
        gy = np.array([ground_y(x + v * 0.3) for x, v in zip(ox, vx)], np.float64)
    else:
        gy = np.full(n, float(ground_y))
    px, py = np.empty(n), np.empty(n)
    _spark_sim(px, py, vx, vy, tb, life, bounce, gy, float(t), gravity, drag, ox, oy)
    age = np.clip((t - tb) / life, 0, 1)
    return px, py, vx, vy, age, rg


def sparks_layer(t, origin, seed=0, n=260, ground_y=None, spread=(8.0, 172.0), speed=(500.0, 2300.0),
                 burst=0.35, shutter=1.0 / 24.0, size=1.0, frame=None, **kw):
    """Motion blurred streaks: white hot heads, orange tails, every one its own size."""
    px, py, vx, vy, age, rg = sparks_state(t, origin, seed, n, ground_y, spread, speed, burst, **kw)
    fr = int(round(t * 24)) if frame is None else frame
    fl = core.rng("sparkflick", seed, fr).random(n)
    szs = rg.uniform(0.45, 1.5, n) ** 2.2
    cel = core.Cel(ss=2)
    c_tail = np.array([1.0, 0.42, 0.10], np.float32)
    c_mid = np.array([1.0, 0.74, 0.24], np.float32)
    c_head = np.array([1.0, 0.97, 0.80], np.float32)
    order = np.argsort(szs)
    for k in order:
        if px[k] < -1e5 or not (-200 < px[k] < W + 200 and -200 < py[k] < H + 200):
            continue
        fade = 1 - age[k] ** 2.2
        wdt = size * szs[k] * (0.55 + 0.75 * fl[k]) * (0.35 + 0.65 * fade) * 3.6
        v = np.array([vx[k], vy[k]])
        L = max(np.hypot(*v) * shutter, 2.5 * wdt)
        dirn = v / max(np.hypot(*v), 1e-6)
        head = np.array([px[k], py[k]])
        tail = head - dirn * L
        cel.stroke(np.array([tail, head]), wdt, c_tail, taper=(0.75, 0.0), min_w=0.05)
        cel.stroke(np.array([head - dirn * L * 0.55, head]), wdt * 0.8, c_mid, taper=(0.6, 0.0), min_w=0.05)
        if fade > 0.25:
            cel.stroke(np.array([head - dirn * max(L * 0.22, wdt), head]), wdt * 0.62, c_head, taper=(0.4, 0.05))
    return cel.rgba()


# ====================================================================== light
def bloom(pic, threshold=0.78, radius=14.0, amount=0.6):
    """Highlights bleed into their surroundings (two radii, screened on)."""
    lum = pic.max(axis=2)
    k = np.clip((lum - threshold) / max(1 - threshold, 1e-3), 0, 1) ** 1.5
    hi = pic * k[..., None]
    g = core.gblur(hi, radius) * 0.7 + core.gblur(hi, radius * 4.0) * 0.6
    return core.add_light(pic, g, amount=amount)


def _blob(x, y, radius, squash=1.0, hw=(H, W)):
    hh, ww = hw
    r = radius * 3.0
    x0, x1 = int(max(0, x - r)), int(min(ww, x + r))
    y0, y1 = int(max(0, y - r * squash)), int(min(hh, y + r * squash))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-(((xx - x) / radius) ** 2 + ((yy - y) / (radius * squash)) ** 2) * 0.5)
    return (slice(y0, y1), slice(x0, x1)), g


def glow_at(pic, x, y, radius, color=(1.0, 0.8, 0.5), amount=0.6, squash=1.0):
    b = _blob(x, y, radius, squash, pic.shape[:2])
    if b is None:
        return pic
    sl, g = b
    core.add_light(pic[sl], g, color, amount)
    return pic


def ground_light(pic, x, y, radius, color=(1.0, 0.72, 0.30), amount=1.0, squash=0.45, mask=None):
    """Light landing ON surfaces: it multiplies up what is there (so grass stays grass,
    only warmer and brighter), plus a small additive lift. mask (HxW) limits it."""
    b = _blob(x, y, radius, squash, pic.shape[:2])
    if b is None:
        return pic
    sl, g = b
    if mask is not None:
        g = g * mask[sl]
    g = (g * amount)[..., None]
    c = np.asarray(color, np.float32)
    p = pic[sl]
    p[:] = np.clip(p * (1 + g * (c * 3.2 - 0.4)) + g * c * 0.10, 0, 1)
    return pic


def light_shafts(pic, sun_xy, occlusion_alpha, amount=0.5, color=(1.0, 0.9, 0.7), length=0.6, steps=28):
    """Radial shafts from the sun through the gaps of an occluder (clouds, tree, smoke)."""
    q = 4
    hh, ww = pic.shape[:2]
    lum = pic.max(axis=2)
    src = np.clip((lum - 0.55) * 2.2, 0, 1) * (1 - np.clip(occlusion_alpha, 0, 1))
    small = np.ascontiguousarray(src[::q, ::q])
    sx, sy = sun_xy[0] / q, sun_xy[1] / q
    acc = np.zeros_like(small)
    for k in range(steps):
        s = 1.0 + length * (k + 0.5) / steps
        w = 1.0 - k / steps
        acc += core.warp(small, s, sx * (1 - s), sy * (1 - s), small.shape)[..., 0] * w
    acc /= steps * 0.5
    big = ndimage.zoom(core.gblur(acc, 1.2), q, order=1)[:hh, :ww]
    return core.add_light(pic, big, color, amount)
