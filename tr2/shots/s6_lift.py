"""Shot 6, Lift: the showpiece. One continuous move: low and close under the
rocket, the ignition hits like a wall (white flash, blown out bloom, heavy
shake), smoke rolls out along the ground, the rocket squashes, shudders, leaps,
and the camera tilts up after it past the tree into the dawn sky and the first
clouds.

    python3 -m tr2.shots.s6_lift sim         precompute the smoke for the whole cut (cache)
    python3 -m tr2.shots.s6_lift smoketest   2.6 s of the smoke alone at half size (motion check)
"""
import math
import os
import sys

import numpy as np
from scipy import ndimage

from .. import core, fx, fx_smoke, palette, rocket
from ..timeline import FPS
from . import common

NAME = "s6_lift"
TOD = "launch"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "build", "cache", "fx", "s6")
SIM_VERSION = 7
RX = common.ROCKET_X
IGN = common.cue("ignition", NAME)
LIFT = common.cue("liftoff", NAME)
BIRDS = common.cue("birds_burst", NAME)
PASS = common.cue("passes_tree", NAME)
F_IGN = int(round(IGN * FPS))
ZOOM0 = 1.75
GRID = dict(origin=(980.0, 420.0), nx=300, ny=176, cell=5.0)
N_SIM = 172


def gy(x=RX):
    return float(common.ground_y(x))


# ------------------------------------------------------------- choreography
def height(t):
    """Set px the feet have left the ground."""
    tau = t - LIFT
    if tau <= 0:
        return 0.0
    return 420.0 * (tau - 0.9 * (1 - math.exp(-tau / 0.9))) + 60.0 * (1 - math.exp(-tau / 0.12))


def motion(t, fl=None):
    """Where the rocket is and what state it is in at local time t."""
    ti, tau = t - IGN, t - LIFT
    H = height(t)
    scale = core.lerp(1.0, 0.30, core.smooth(core.span(tau, 1.2, 5.4)))
    X = RX + 150.0 * core.ease_in(core.span(tau, 0.3, 5.4), 1.6)
    if ti < 0:
        sq, rattle, amp, light = 0.0, 0.0, 0.0, 0.0
    elif tau < 0:
        build = core.span(t, IGN, LIFT)
        sq = 0.50 * core.overshoot(core.span(ti, 0.0, 0.16)) + 0.10 * math.sin(ti * 41.0) * (0.4 + 0.6 * build) \
            + 0.38 * core.smooth(core.span(t, LIFT - 0.32, LIFT))
        rattle, amp, light = 0.3 + 0.7 * build, 1.5 + 5.5 * build, 1.0
    else:
        sq = core.keys(tau, [(0.0, 0.9), (0.085, -0.70), (0.5, -0.28), (1.3, -0.10)])
        rattle, amp, light = 1.0, core.lerp(5.0, 2.0, core.span(tau, 0.0, 1.5)), 1.0
    dx = dy = dr = 0.0
    if fl is not None and amp > 0:
        r = core.rng("rk shake", fl)
        a = r.normal(0, 1, 3)
        dx, dy, dr = a[0] * amp, a[1] * amp * 0.6, a[2] * amp * 0.0022
    rot = 0.11 * core.smooth(core.span(tau, 0.6, 4.6)) + 0.02 * core.noise1(t * 1.3, 31) * core.span(tau, 0, 1)
    return dict(H=H, X=X, scale=scale, squash=sq, rattle=rattle, light=light, rot=rot, shake=(dx, dy, dr),
                legs=0.75 * core.smooth(core.span(tau, 0.15, 1.1)), soot=0.3 * core.smooth(core.span(ti, 0.0, 1.2)),
                view=rocket.mix_views("low34", "below", core.smooth(core.span(tau, 0.3, 2.8))),
                flame=1.0 if ti >= 0 else 0.0)


def camera(t, fl):
    ti, tau = t - IGN, t - LIFT
    zoom = core.lerp(ZOOM0, 1.5, core.smooth(core.span(tau, 0.8, 5.0)))
    lag = 0.42 * (1 - core.smooth(core.span(tau, 0.7, 3.0)))
    tilt = zoom * height(t - lag) - 430.0 * core.ease_in(core.span(tau, 3.9, 5.4), 2.0)
    if ti < 0:
        amp = 0.0
    elif tau < 0:
        amp = 36.0 * math.exp(-ti / 0.20) + 3.0 + 8.0 * core.span(t, IGN + 0.3, LIFT)
    else:
        amp = 13.0 * math.exp(-tau / 0.55) + 2.2
    sx, sy = common.shake(fl, amp, seed=6) if amp > 0 else (0.0, 0.0)
    dx, dy = common.drift(t, 2.0, 6)
    return core.Camera(cx0=1600.0, cy0=900.0, cx=1705.0 + dx + 70.0 * core.smooth(core.span(tau, 0.5, 5.0)),
                       cy=890.0 + dy - 25.0 * core.smooth(core.span(tau, 0.0, 4.0)), zoom=zoom, tilt=tilt,
                       shake=(sx, sy))


# ------------------------------------------------------------ the simulation
def _path(fl):
    return os.path.join(CACHE, "v%d_f%03d.npz" % (SIM_VERSION, fl))


def build_sim(n_frames=N_SIM, log=True, force=False):
    """Run the fluid and its lobes for the whole cut and cache one small file per frame."""
    import time
    os.makedirs(CACHE, exist_ok=True)
    if not force and all(os.path.exists(_path(k)) for k in range(n_frames)):
        return
    t0 = time.time()
    sim = fx.SmokeSim(GRID["nx"], GRID["ny"], GRID["cell"], GRID["origin"], ground=common.ground_y, seed=6)
    sim.beta, sim.turb = 70.0, 24.0
    lob = fx_smoke.Lobes(seed=6)
    dt = 1.0 / 96.0
    acc = np.zeros(8)
    fronts = ((-1, 640.0, 0.26), (1, 580.0, 0.22))
    fast = ((-1, 1080.0, 0.30), (1, 1000.0, 0.27))
    k = 0
    for fl in range(n_frames):
        t_end = fl / FPS
        while k * dt < t_end - 1e-6:
            t = k * dt
            k += 1
            ti, tau = t - IGN, t - LIFT
            if ti < 0:
                sim.t = t + dt
                continue
            m = motion(t)
            g0 = gy(m["X"])
            noz = (m["X"], g0 - m["H"] - 56.0 * m["scale"])
            sc = m["scale"]
            emit = dict(x=noz[0], y=noz[1] + 24 * sc, r=42.0 * sc, jet=(40.0, 1500.0), rate=9.0, lobes=0)
            if ti < 0.14:
                emit.update(burst=150.0 * (1 - ti / 0.14), burst_r=120.0, burst_y=g0 - 22.0, burst_d=24.0, burst_heat=0.85)
            puffs = []
            for q, (sgn, reach, tc) in enumerate(fronts):
                if ti < 1.2:
                    Xf = reach * (1 - math.exp(-ti / tc))
                    V = reach / tc * math.exp(-ti / tc)
                    px = noz[0] + sgn * Xf
                    puffs.append(dict(x=px, y=gy(px) - 30.0, r=58.0, d=30.0 * math.exp(-ti / 0.5), T=0.10,
                                      u=sgn * V * 0.8, v=-25.0, grip=0.6, flat=0.7))
                    # the ring: lobes shrink as the front runs out, so the mass tapers
                    acc[q] += 130.0 * math.exp(-ti / 0.6) * dt
                    n = int(acc[q])
                    acc[q] -= n
                    tap = 0.26 + 0.74 * math.exp(-ti / 0.45)
                    lob.spawn(t, px, gy(px) - 6.0, n, r=(9.0 * tap + 3.0, 84.0 * tap + 8.0), jit=(34.0, 14.0),
                              life=(2.6, 5.0), rise=-5.0, follow=(0.10, 0.95), key=2 + q, pw=2.4, kids=3)
            for q, (sgn, reach, tc) in enumerate(fast):
                if ti < 0.8:
                    Xf = reach * (1 - math.exp(-ti / tc))
                    V = reach / tc * math.exp(-ti / tc)
                    px = noz[0] + sgn * Xf
                    puffs.append(dict(x=px, y=gy(px) - 14.0, r=34.0, d=10.0 * math.exp(-ti / 0.3), T=0.08,
                                      u=sgn * V * 0.9, v=-20.0, grip=0.5, flat=0.45))
                    # the thin fast layer that races out ahead and thins into wisps
                    acc[4 + q] += 95.0 * dt
                    n = int(acc[4 + q])
                    acc[4 + q] -= n
                    lob.spawn(t, px - sgn * 30.0, gy(px) - 3.0, n, r=(4.0, 19.0), jit=(34.0, 5.0), life=(0.8, 1.8),
                              rise=-5.0, grow=1.7, follow=(0.75, 1.0), key=6 + q, pw=2.2)
            emit["puffs"] = puffs
            if noz[1] < GRID["origin"][1] + 30:
                emit = None                          # the nozzle has left the simulated box
            sim.step(dt, emit)
            # the heavy lobes at the base: a few big, many small, born while the fire is near the ground
            if m["H"] < 280:
                rate = 330.0 if ti < 0.25 else 150.0
                if tau > 0:
                    rate = 170.0 * math.exp(-tau / 0.45)
                acc[2] += rate * dt
                n = int(acc[2])
                acc[2] -= n
                lob.spawn(t, noz[0], g0 - 5.0, n, r=(12.0, 135.0), jit=(115.0, 30.0), life=(3.0, 5.6), rise=-9.0,
                          follow=(0.10, 0.9), key=1, pw=2.7, kids=4)
            if tau > 0:
                # the column under the climbing rocket: born in the jet, thrown down, then it hangs and widens
                acc[3] += 170.0 * dt
                n = int(acc[3])
                acc[3] -= n
                # born a little "earlier" so they are already swollen where they leave the flame; they hang
                # where they were made, so the column is continuous, widens with age and tapers to the rocket
                lob.spawn(t - 0.12, noz[0], noz[1] + 70.0 * sc, n, r=(22.0 * sc, 92.0 * sc), jit=(30.0 * sc, 34.0 * sc),
                          vel=(0.0, 150.0 * sc), vjit=(95.0 * sc, 80.0 * sc), life=(3.0, 5.6), rise=-6.0, grow=2.4,
                          follow=(0.10, 0.35), key=9, pw=1.6, up_only=False, kids=3, kid_min=46.0 * sc)
            lob.step(t, dt, sim)
        t = k * dt
        tmp = _path(fl) + ".tmp.npz"
        np.savez_compressed(tmp, lobes=lob.alive(t), F=sim.F.astype(np.float16), t=np.float32(t))
        os.replace(tmp, _path(fl))
        if fl % 24 == 0:
            lob.prune(t)
            if log:
                print("  sim frame %d  lobes %d  %.1fs" % (fl, lob.n, time.time() - t0), flush=True)


_cache = {}


def fields(fl):
    fl = int(min(max(fl, 0), N_SIM - 1))
    if fl not in _cache:
        if not os.path.exists(_path(fl)):
            build_sim()
        z = np.load(_path(fl))
        if len(_cache) > 6:
            _cache.clear()
        _cache[fl] = dict(lobes=z["lobes"], F=z["F"].astype(np.float32), t=float(z["t"]))
    return _cache[fl]


# ----------------------------------------------------------------- the clouds
_bank = None


def _dawn_sprite(kind, seed):
    """One of the painted sky sprites (tr2/sky_paint.py), relit for the dawn seen
    from below: cool violet tops that keep their form, an underside that glows
    rose then gold. Returns the kind name to give to sky_paint.place."""
    from .. import sky_paint as sp
    name = kind + "@dawn"
    if (name, seed, False) not in sp._MIPS:
        F = sp.cloud_sprite(kind, seed).copy()
        L, U = F[..., 1], np.clip(F[..., 2] * 1.15, 0, 1)
        u = U * U * (3 - 2 * U)
        F[..., 1] = np.clip(0.27 + 0.42 * (L - 0.45) + 0.56 * u, 0, 1)
        sp.add_variant(name, seed, F)
    return name


def cloud_bank():
    """The first clouds and the bank the rocket punches into: painted cloud sprites
    at three depths (parallax 0.55, 1.0, 1.5) in set coords. Returns (clouds, base)."""
    global _bank
    if _bank is None:
        from .. import sky_paint as sp
        rg = core.rng("s6 clouds")
        y_end = gy() - height(7.0)
        base = y_end + 190.0
        its = []

        def add(p, x, ye, kind, seed, sc, op=1.0, bank=False):
            # ye is where the cloud would sit at parallax 1; other depths are shifted so that
            # they cross the middle of the frame at the same moment of the tilt
            y = ye - (p - 1.0) * 0.35 * (890.0 - ye)
            its.append(dict(p=p, x=float(x), y=float(y), kind=_dawn_sprite(kind, seed), seed=int(seed), sc=float(sc), op=op,
                            h=sp.SIZES[kind][1], bank=bank))

        # the bank: heaped cumulus right across, a taller tier behind, undersides scalloped
        x, k = 380.0, 0
        while x < 3350.0:
            scb = rg.uniform(0.72, 1.0)
            kind = "cumulus"
            rg.random()
            dip = 70.0 * math.sin(k * 0.9) + rg.normal(0, 26)
            h = sp.SIZES[kind][1] * scb
            if k % 2 == 0:
                add(1.0, x + rg.normal(0, 60), base + dip - 290.0 - 0.34 * 540 * scb * 1.15, "cumulus",
                    int(rg.integers(0, 7)), scb * 1.15, 0.97, bank=True)
            add(1.0, x, base + dip - 0.34 * h + 24.0, kind, int(rg.integers(0, 7 if kind == "cumulus" else 4)), scb, 0.95, bank=True)
            x += 235.0 * scb * rg.uniform(0.8, 1.2)
            k += 1
        # the first small clouds on the way up
        for (x, dy, kind, seed, sc) in ((1180.0, 560.0, "cumulus", 2, 0.50), (2360.0, 400.0, "cumulus", 4, 0.56),
                                        (2180.0, 820.0, "bar", 1, 0.36), (1440.0, 230.0, "bar", 3, 0.50),
                                        (2560.0, 1000.0, "cumulus", 5, 0.40)):
            add(1.0, x, base + dy, kind, seed, sc)
        # far: small, hazy, slower; kept clear of the rocket's column
        for j in range(9):
            x = rg.uniform(620.0, 1380.0) if j % 2 == 0 else rg.uniform(2250.0, 3050.0)
            kind = ("cumulus", "bar", "cumulus", "band")[j % 4]
            add(0.55, x, base + rg.uniform(-150.0, 1150.0), kind, int(rg.integers(0, 6)), rg.uniform(0.28, 0.46))
        # near: big, out of focus, sliding past faster; the last one is mist across the rocket itself
        for (x, dy, kind, seed, sc, op) in ((800.0, 330.0, "cumulus", 1, 1.25, 0.92), (2800.0, 700.0, "cumulus", 3, 1.1, 0.92),
                                            (2520.0, 120.0, "cumulus", 6, 1.3, 0.92), (980.0, -80.0, "cumulus", 0, 1.4, 0.92),
                                            (1800.0, -260.0, "wisp", 2, 1.7, 0.55)):
            add(1.5, x, base + dy, kind, seed, sc, op)
        its.sort(key=lambda it: it["y"])
        # the bank is one painting: compose its sprites once as cloud FIELDS (alpha, light,
        # underside) on a strip in set coords, so a frame colours and places it in one go
        bk = [it for it in its if it["bank"]]
        X0 = 250.0
        Y0 = min(it["y"] - it["h"] * it["sc"] * 0.5 for it in bk) - 10.0
        Y1 = max(it["y"] + it["h"] * it["sc"] * 0.5 for it in bk) + 10.0
        Fb = np.zeros((int(Y1 - Y0), 3400, 3), np.float32)
        for it in bk:
            Fm = sp.mips(it["kind"], it["seed"])[0]
            hh, ww = Fm.shape[:2]
            fx0, fy0 = it["x"] - X0 - ww * it["sc"] / 2, it["y"] - Y0 - hh * it["sc"] / 2
            xa, ya = max(int(math.floor(fx0)), 0), max(int(math.floor(fy0)), 0)
            xb, yb = min(int(fx0 + ww * it["sc"]) + 2, Fb.shape[1]), min(int(fy0 + hh * it["sc"]) + 2, Fb.shape[0])
            if xb <= xa or yb <= ya:
                continue
            Fw = core.warp(Fm, it["sc"], fx0 - xa, fy0 - ya, out_hw=(yb - ya, xb - xa))
            reg = Fb[ya:yb, xa:xb]
            a_s = Fw[..., 0:1]
            a_n = a_s + reg[..., 0:1] * (1 - a_s)
            reg[..., 1:] = (Fw[..., 1:] * a_s + reg[..., 1:] * reg[..., 0:1] * (1 - a_s)) / np.maximum(a_n, 1e-4)
            reg[..., 0:1] = a_n
        global _bank_strip
        _bank_strip = (Fb, X0, Y0)
        _bank = ([it for it in its if not it["bank"]], base)
    return _bank


_bank_strip = None
DAWN_TONES = [np.array(c, np.float32) for c in ((0.33, 0.37, 0.60), (0.49, 0.51, 0.73), (0.78, 0.62, 0.74),
                                                (1.00, 0.76, 0.70), (1.00, 0.89, 0.66))]


def ascent_clouds(pic, cam, bank, noz, hid, s):
    """Draw the cloud_bank() clouds over the picture, three depths back to front. The faces
    nearest the flame (screen point noz) take its warm light; hid 0..1 = how deep the rocket
    is inside the bank (glow through the mist)."""
    from .. import sky_paint as sp
    skyc = np.array([0.38, 0.46, 0.72], np.float32)
    lays = {0.55: np.zeros((core.H, core.W, 4), np.float32), 1.0: np.zeros((core.H, core.W, 4), np.float32),
            1.5: np.zeros((core.H // 2, core.W // 2, 4), np.float32)}
    haze = {0.55: 0.46, 1.0: 0.08, 1.5: 0.0}
    Fb, X0, Y0 = _bank_strip
    s1, tx1, ty1 = cam.xf(1.0)
    ya, yb = max(int(math.floor(Y0 * s1 + ty1)), 0), min(int((Y0 + Fb.shape[0]) * s1 + ty1) + 2, core.H)
    if yb > ya:
        Fw = core.warp(Fb, s1, tx1 + X0 * s1, ty1 + Y0 * s1 - ya, out_hw=(yb - ya, core.W))
        cols = [palette.mix(c, skyc, haze[1.0] * (0.95 if i < 2 else 0.7)) for i, c in enumerate(DAWN_TONES)]
        # soft mist behind the bank: it closes the small gaps between heaps (a notch of dark sky
        # showed between two of them) and gives the bank a glowing edge
        am = (np.clip(core.gblur(Fw[..., 0], 26.0) * 1.7, 0, 1) * 0.9).astype(np.float32)
        mc = palette.mix(cols[2], cols[3], 0.5)
        core.over_rgba(lays[1.0][ya:yb], core.premul(np.broadcast_to(mc, Fw.shape).astype(np.float32).copy(), am))
        core.over_rgba(lays[1.0][ya:yb], sp.colourise(Fw, cols, 0.95))
    for it in bank:
        p = it["p"]
        sp_, _, _ = cam.xf(p)
        x, y = cam.pt(it["x"], it["y"], p)
        y += (p - 1.0) * 0.35 * cam.tilt                      # nearer cloud slides down faster in the tilt
        hh = it["h"] * it["sc"] * sp_ * 0.6
        if y < -hh or y > core.H + hh or x < -900 or x > core.W + 900:
            continue
        cols = [palette.mix(c, skyc, haze[p] * (0.95 if i < 2 else 0.7)) for i, c in enumerate(DAWN_TONES)]
        res = 2.0 if p == 1.5 else 1.0
        sp.place(lays[p], it["kind"], it["seed"], x / res, y / res, it["sc"] * sp_ / res, cols, op=it["op"])
    core.over(pic, lays[0.55])
    mid = lays[1.0]
    # the flame lights the faces nearest to it
    yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
    g = np.exp(-(((xx - noz[0]) ** 2 + (yy - noz[1]) ** 2) / (300.0 * s) ** 2))
    warm = np.array([1.0, 0.62, 0.30], np.float32)
    mid[..., :3] = np.minimum(mid[..., :3] + (g * 0.55)[..., None] * warm * mid[..., 3:4], mid[..., 3:4] * 1.0)
    if hid > 0.02:
        fx.glow_at(pic, noz[0], noz[1], 150.0 * s, (1.0, 0.70, 0.40), 0.5 * hid)
    core.over(pic, mid)
    near = core.warp(core.gblur(lays[1.5], 2.6), 2.0, 0.5, 0.5)
    near[..., :3] = np.minimum(near[..., :3] + (g * 0.35)[..., None] * warm * near[..., 3:4], near[..., 3:4])
    core.over(pic, near)
    if hid > 0.02:
        fx.glow_at(pic, noz[0], noz[1] - 30 * s, 170.0 * s, (1.0, 0.78, 0.52), 0.30 * hid)


def leaves_layer(cam, t, fl):
    """A dozen leaves torn out of the canopy as the rocket passes: each whips up after it,
    spins and flips, and is smeared along its path (on ones)."""
    rg = core.rng("s6 leaves")
    T = palette.TOD[TOD]
    cel = None
    for i in range(7):
        t0 = PASS - 0.32 + rg.uniform(0, 0.8)
        p0 = np.array([rg.uniform(1330.0, 1560.0), rg.uniform(400.0, 770.0)])
        v0 = np.array([rg.uniform(160.0, 640.0), -rg.uniform(420.0, 980.0)])
        # director: leaf sized (about 30 to 45 px long on screen), not paddle sized
        kd, size, spin = rg.uniform(1.4, 2.4), rg.uniform(14.0, 22.0), rg.uniform(3.2, 6.5) * rg.choice([-1.0, 1.0])
        ph, flut, par, pale = rg.uniform(0, 6.28), rg.uniform(20.0, 60.0), rg.uniform(0.95, 1.3), rg.random() < 0.4
        tau = t - t0
        if tau <= 0 or tau > 2.8:
            continue

        def pos(tt):
            e = math.exp(-kd * tt)
            return p0 + v0 * (1 - e) / kd + np.array([flut * math.sin(tt * 5.0 + ph) + 70.0 * tt, 130.0 * tt * tt])

        a = np.array(cam.pt(*pos(tau), par))
        b = np.array(cam.pt(*pos(max(tau - 0.04, 0.0)), par))
        if not (-60 < a[0] < core.W + 60 and -60 < a[1] < core.H + 60):
            continue
        if cel is None:
            cel = core.Cel(ss=2)
        sc = cam.xf(par)[0] * size
        rot = ph + spin * tau
        flip = 0.22 + 0.78 * abs(math.cos(tau * spin * 0.7 + 2 * ph))
        base = np.array([0.20, 0.40, 0.20] if pale else [0.14, 0.31, 0.17], np.float32)
        col = palette.lit(base, T)
        edge = np.array([0.62, 0.72, 0.36], np.float32)
        dk = palette.shade(base, T) * 0.8
        cel.stroke(np.array([b, a]), max(sc * 0.55 * flip, 1.5), dk, taper=(0.8, 0.0), min_w=0.08)      # the smear
        leaf = np.array([(-1, 0), (-0.4, 0.42 * flip), (0.35, 0.36 * flip), (1, 0), (0.35, -0.36 * flip), (-0.4, -0.42 * flip)])
        leaf = core.xform(core.catmull(leaf, n=5), scale=sc, rot=rot, tx=a[0], ty=a[1])
        cel.fill(leaf, col)
        cel.stroke(leaf[:len(leaf) // 2 + 1], max(sc * 0.14, 1.4), edge, taper=(0.25, 0.25))      # the lit edge
        cel.stroke(core.xform(np.array([(-1.0, 0.0), (1.0, 0.0)]), scale=sc, rot=rot, tx=a[0], ty=a[1]), 1.3, dk)
    return None if cel is None else cel.rgba()


# ------------------------------------------------------------------ the frame
def render(fl, n, f):
    t = fl / FPS
    ti, tau = t - IGN, t - LIFT
    cam = camera(t, fl)
    m = motion(t, fl)
    s, tx, ty = cam.xf(1.0)
    g0 = gy(m["X"])
    burning = ti >= 0
    flick = 0.84 + 0.16 * core.rng("fire flick", fl).random()
    near = math.exp(-m["H"] / 330.0)
    fire_amt = (flick * near) if burning else 0.0
    flat = None
    if burning:
        flat = (RX, 70.0 + 560.0 * (1 - math.exp(-ti / 0.25)),
                min(1.0, ti / 0.08) * (1 - 0.65 * core.smooth(core.span(tau, 0.6, 2.6))))
    blast = (0.0 if not burning else 1.0 + 3.5 * core.smooth(core.span(t, IGN, IGN + 0.3))
             * (1 - 0.7 * core.smooth(core.span(tau, 1.6, 3.5))))
    wind = 1.0 + blast
    pic = common.sky_pic(TOD, cam, f, t)
    hs = None
    ground_on = ty + (g0 - 500.0) * s < core.H + 60          # is any of the hill in frame
    try:
        from .. import hillset as hs
        if ground_on:
            hs.render_back(pic, cam, TOD, f, wind=wind, flatten=flat,
                           rocket_shadow=1.0 - core.smooth(core.span(t, IGN, LIFT)),
                           fire=(RX, g0 - 10.0, 300.0, 0.8 * fire_amt) if fire_amt > 0.01 else None)
            if burning:
                # the near slope falls into dark so the fire and the lit smoke carry the frame
                rows = np.arange(core.H, dtype=np.float32)
                fall_ = np.clip((rows - ((g0 + 70.0) * s + ty)) / (300.0 * s), 0, 1)
                pic *= (1.0 - 0.5 * fall_ * fall_ * (3 - 2 * fall_))[:, None, None]
    except Exception as e:  # noqa
        print("s6 hillset:", repr(e)[:200])
        hs = None

    # the rocket cel
    st = rocket.State(shake=m["shake"], squash=m["squash"], fin_rattle=m["rattle"], flame=m["flame"], soot=m["soot"],
                      legs_fold=m["legs"], light=m["light"])
    sc = s * m["scale"]
    fx_, fy_ = m["X"] * s + tx, (g0 - m["H"]) * s + ty
    cel = core.Cel(ss=2 if sc > 1.5 else 3)
    info = rocket.draw(cel, fx_, fy_, s=sc, tod=TOD, drawing=fl if (burning and tau < 1.0) else common.twos(fl),
                       rot=m["rot"], view=m["view"], state=st)
    rocket.draw_glass(cel, info, TOD, common.twos(fl))
    noz = info["nozzle"]
    gsx, gsy = noz[0], g0 * s + ty                            # the ground under the nozzle, on screen

    # smoke, flame, sparks
    fd = fields(fl)
    foot = 1 - core.smooth(core.span(m["H"], 40.0, 170.0))
    tail = 215.0 * sc * (0.85 + 0.3 * core.rng("tail", fl).random())
    ax, ay = -math.sin(m["rot"]), math.cos(m["rot"])
    end = (noz[0] + ax * tail, noz[1] + ay * tail)
    if foot > 0.01:
        end = (core.lerp(end[0], noz[0] + 6 * sc, foot), core.lerp(end[1], max(gsy - 10 * s, noz[1] + 30 * sc), foot))
    fire_xy = (end[0], end[1] - 14 * s * foot)
    smoke = None
    if burning and len(fd["lobes"]):
        smoke = fx_smoke.render(fd["lobes"], fd["t"], f, TOD, (s, tx, ty), seed=6, soft=4.5, haze=0.36, warm_top=0.0,
                                dark=((g0 - 215.0) * s + ty, (g0 + 40.0) * s + ty, 0.88),
                                fire=(fire_xy[0], fire_xy[1], flick * (0.55 + 0.45 * near), 410.0 * s * max(m["scale"], 0.5)))
    fl_lay = None
    if burning:
        fl_lay = fx_smoke.flame(fl, (s, tx, ty), F=fd["F"] if (ground_on and m["H"] < 200.0) else None,
                                grid=(GRID["origin"][0], GRID["origin"][1], GRID["cell"]),
                                jet=(noz[0], noz[1] - 4 * sc, end[0], end[1], 24.0 * sc, 44.0 * sc, foot), seed=6,
                                focus=(gsx, gsy - 20 * s, 300.0 * s))
    if not burning:
        # the stillness is ending: the fuse fizzes under the skirt
        k = 0.25 + 0.75 * t / max(IGN, 1e-3)
        fx.glow_at(pic, noz[0], noz[1] + 10 * s, 46.0 * s, (1.0, 0.66, 0.30), (0.22 + 0.2 * flick) * k)
    else:
        fx.glow_at(pic, fire_xy[0], fire_xy[1] - 30 * s, 400.0 * s * max(m["scale"], 0.4), (1.0, 0.62, 0.30),
                   0.30 * flick, squash=0.75)
    if hs is not None and ground_on and burning:
        try:
            core.over(pic, hs.grass_band(cam, TOD, f, 1150, 2300, flatten=flat))     # the smoke rolls over the crest grass
        except Exception as e:  # noqa
            print("s6 band:", repr(e)[:200])
    if smoke is not None:
        core.over(pic, smoke["layer"])
    if hs is not None and ground_on and tau < 2.0:
        common.contact_shadow(pic, cam, RX, g0 + 6, 190, TOD, 0.42 * (1 - core.span(tau, 0.0, 0.6)))
    core.over(pic, cel.rgba())
    if fl_lay is not None:
        core.over(pic, fl_lay["flame"])
        if smoke is not None and ground_on:
            # the low front rolls swallow the legs: the smoke again, in front, below a wandering line
            yy = np.arange(core.H, dtype=np.float32)[:, None]
            xx = np.arange(core.W, dtype=np.float32)[None, :]
            wob = core.fbm(core.H, core.W, 90.0 * s, 3, seed=44, ox=-tx, oy=-ty)
            line = (g0 - 74.0) * s + ty + (wob - 0.5) * 70.0 * s + 62.0 * s * np.exp(-((xx - gsx) / (130.0 * s)) ** 2)
            fm = np.clip((yy - line) / 2.0, 0, 1).astype(np.float32)
            core.over(pic, smoke["layer"] * fm[..., None])
        core.over(pic, fl_lay["jet"])
    if not burning:
        sp = fx.sparks_layer(t + 0.5, (RX + 4.0, g0 - 60.0), seed=9, n=46, ground_y=lambda x: gy(x) + 4.0, frame=f,
                             xf=(s, tx, ty), speed=(120.0, 620.0), size=0.5, dur=1.2, gravity=1100.0)
        core.over(pic, sp)
    elif ground_on and ti < 3.2:
        sp = fx.sparks_layer(ti, (RX + 4.0, g0 - 10.0), seed=5, n=170, ground_y=lambda x: gy(x) + 6.0, frame=f,
                             xf=(s, tx, ty), dur=LIFT - IGN + 0.35, size=0.5, shutter=1.0 / 15.0)
        core.over(pic, sp)
    if burning:
        fx.glow_at(pic, noz[0], noz[1] + 60 * sc, 190.0 * sc, (1.0, 0.66, 0.30), 0.40)
        fx.glow_at(pic, noz[0], noz[1] + 34 * sc, 64.0 * sc, (1.0, 0.92, 0.70), 0.50)

    # the birds burst out of the camphor tree
    if t >= BIRDS - 0.05:
        try:
            from .. import birds
            flock = birds.Flock([(1260.0, 470.0), (1390.0, 530.0), (1150.0, 440.0), (1450.0, 610.0), (1320.0, 590.0),
                                 (1060.0, 520.0)], n=22, heading=(0.72, -0.62), seed=6)
            core.over(pic, birds.draw_flock(flock, cam, t - BIRDS, f, None, col=(0.50, 0.43, 0.55),
                                            light=(1.28, 1.06, 1.02), base_size=32.0))
        except Exception as e:  # noqa
            print("s6 birds:", repr(e)[:200])

    lv_ = leaves_layer(cam, t, fl)
    if lv_ is not None:
        core.over(pic, lv_)

    # grass in front of everything on the hill
    if hs is not None and ground_on:
        try:
            if not burning:
                core.over(pic, hs.grass_band(cam, TOD, f, 1150, 2300, flatten=flat))
            fr = hs.render_front(cam, TOD, f, wind=wind, defocus=common.dof(1.0, 5.0), flatten=flat)
            fr[..., :3] *= 0.46 if burning else 0.72          # the near grass stands dark against the fire
            core.over(pic, fr)
        except Exception as e:  # noqa
            print("s6 front:", repr(e)[:200])

    # the first clouds, then the bank that swallows the rocket
    bank, base = cloud_bank()
    if (base + 1500.0) * s + ty > -40:
        hid = core.smooth(core.span((base - (g0 - m["H"])), -40.0, 120.0))     # how deep the rocket is in the bank
        ascent_clouds(pic, cam, bank, noz, hid, s)

    # ignition hits like a wall: one white frame, two blown out, then it settles
    opts = dict(bloom_amt=0.62, bloom_thr=0.80)
    if fl == F_IGN:
        yy, xx = np.mgrid[0:core.H, 0:core.W].astype(np.float32)
        d = np.hypot(xx - gsx, (yy - gsy) * 1.3) / 1500.0
        w = np.clip(1.0 - 0.22 * d, 0.7, 1.0)[..., None]
        pic[:] = pic * (1 - w) + np.array([1.0, 0.985, 0.93], np.float32) * w
        opts = dict(bloom_amt=0.9, bloom_thr=0.6, vig=0.05)
    elif fl in (F_IGN + 1, F_IGN + 2):
        kx = 1.0 if fl == F_IGN + 1 else 0.55
        fx.glow_at(pic, gsx, gsy - 60 * s, 560.0 * s, (1.0, 0.88, 0.66), 0.50 * kx)
        pic[:] = np.clip(pic * (1.0 + 0.38 * kx) + 0.03 * kx, 0, 1)
        opts = dict(bloom_amt=0.75 + 0.50 * kx, bloom_thr=0.72 - 0.12 * kx, vig=0.10)
    elif burning and ti < 0.6:
        opts = dict(bloom_amt=0.62 + 0.3 * (1 - ti / 0.6), bloom_thr=0.76)
    return pic, opts


# ---------------------------------------------------------------- test modes
def smoketest(frames=(6, 68), scale=0.5):
    """The smoke and flame alone on a flat ground at half size: a clip and a sheet of
    consecutive frames, to judge the motion (shimmer, popping, boiling) before anything else."""
    import subprocess
    build_sim()
    hh, ww = int(core.H * scale), int(core.W * scale)
    out = os.path.join(ROOT, "build", "review", "s6-smoketest.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                          "%dx%d" % (ww, hh), "-r", "24", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                          out], stdin=subprocess.PIPE)
    pics = []
    cam = core.Camera(cx0=1600.0, cy0=900.0, cx=1705.0, cy=890.0, zoom=ZOOM0)
    s, tx, ty = cam.xf(1.0)
    s, tx, ty = s * scale, tx * scale, ty * scale
    ys = np.linspace(0.3, 1, hh, dtype=np.float32)
    bg = np.repeat(palette.sky_color(TOD, ys)[:, None, :], ww, 1).copy()
    gys = np.array([gy((x - tx) / s) * s + ty for x in range(ww)], np.float32)
    gm = np.clip(np.arange(hh, dtype=np.float32)[:, None] - gys[None, :], 0, 1)[..., None]
    bg = bg * (1 - gm) + np.array([0.10, 0.17, 0.19], np.float32) * gm
    for fl in range(*frames):
        t = fl / FPS
        fd = fields(fl)
        m = motion(t)
        pic = bg.copy()
        nz = (m["X"] * s + tx, (gy(m["X"]) - m["H"] - 56.0 * m["scale"]) * s + ty)
        gsy = gy(m["X"]) * s + ty
        if t >= IGN:
            sm = fx_smoke.render(fd["lobes"], fd["t"], fl, TOD, (s, tx, ty), seed=6, hw=(hh, ww),
                                 fire=(nz[0], min(gsy - 10 * s, nz[1] + 120 * s), 1.0, 500.0 * s))
            core.over(pic, sm["layer"])
            fla = fx_smoke.flame(fl, (s, tx, ty), F=fd["F"], grid=(GRID["origin"][0], GRID["origin"][1], GRID["cell"]),
                                 jet=(nz[0], nz[1], nz[0], min(gsy - 8 * s, nz[1] + 200 * s), 24.0 * s, 44.0 * s,
                                      1.0 if m["H"] < 60 else 0.0), seed=6, hw=(hh, ww))
            core.over(pic, fla["flame"])
        pics.append(pic)
        p.stdin.write(core.to8(pic, fl).tobytes())
    p.stdin.close()
    p.wait()
    core.sheet(pics[6:30], 6, 480, os.path.join(ROOT, "build", "review", "s6-smoketest-sheet.jpg"))
    print(out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    if mode == "sim":
        build_sim(force="force" in sys.argv)
    elif mode == "smoketest":
        smoketest()
