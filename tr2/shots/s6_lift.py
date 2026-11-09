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

from .. import core, fx, fx_smoke, palette, rocket
from ..timeline import FPS
from . import common

NAME = "s6_lift"
TOD = "launch"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "build", "cache", "fx", "s6")
SIM_VERSION = 5
RX = common.ROCKET_X
IGN = common.cue("ignition", NAME)
LIFT = common.cue("liftoff", NAME)
BIRDS = common.cue("birds_burst", NAME)
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
                acc[3] += 165.0 * dt
                n = int(acc[3])
                acc[3] -= n
                lob.spawn(t, noz[0], noz[1] + 26.0 * sc, n, r=(8.0 * sc, 48.0 * sc), jit=(12.0 * sc, 10.0 * sc),
                          vel=(0.0, 540.0 * sc), vjit=(75.0 * sc, 120.0 * sc), life=(3.2, 6.0), rise=-8.0, grow=1.7,
                          follow=(0.5, 0.9), key=9, pw=2.5, up_only=False, kids=2, kid_min=22.0 * sc)
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


def cloud_bank():
    """The first clouds and the bank the rocket punches into, as static lobes in set coords."""
    global _bank
    if _bank is None:
        rg = core.rng("s6 clouds")
        lob = fx_smoke.Lobes(seed=77)
        y_end = gy() - height(7.0)
        base = y_end + 190.0
        # the bank: a deck of big lobes, its underside scalloped
        for i in range(34):
            x = 700.0 + i * 78.0 + rg.normal(0, 30)
            dip = 70.0 * math.sin(i * 0.9) + rg.normal(0, 26)
            lob.spawn(-100.0, x, base + dip - 130.0, 4, r=(60.0, 230.0), jit=(60.0, 110.0), life=(1e6, 1e6),
                      rise=0.0, grow=0.0, key=i, pw=1.6)
            lob.spawn(-100.0, x, base + dip - 420.0, 3, r=(150.0, 330.0), jit=(80.0, 150.0), life=(1e6, 1e6),
                      rise=0.0, grow=0.0, key=100 + i, pw=1.2)
        # the first small clouds on the way up
        for j, (x, y, n, rr) in enumerate(((1230.0, base + 520.0, 9, 70.0), (2330.0, base + 380.0, 11, 80.0),
                                           (1560.0, base + 250.0, 7, 46.0), (2120.0, base + 760.0, 6, 40.0))):
            lob.spawn(-100.0, x, y, n, r=(rr * 0.3, rr), jit=(rr * 1.9, rr * 0.4), life=(1e6, 1e6), rise=0.0,
                      grow=0.0, key=300 + j, pw=1.5, up_only=False)
        _bank = (lob.alive(0.0), base)
    return _bank


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
            hs.render_back(pic, cam, TOD, f, wind=wind, flatten=flat, rocket_shadow=0.0 if burning else 1.0,
                           fire=(RX, g0 - 10.0, 430.0, fire_amt) if fire_amt > 0.01 else None)
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
        smoke = fx_smoke.render(fd["lobes"], fd["t"], f, TOD, (s, tx, ty), seed=6,
                                fire=(fire_xy[0], fire_xy[1], flick * (0.55 + 0.45 * near), 410.0 * s * max(m["scale"], 0.5)))
    fl_lay = None
    if burning:
        fl_lay = fx_smoke.flame(fl, (s, tx, ty), F=fd["F"] if ground_on else None,
                                grid=(GRID["origin"][0], GRID["origin"][1], GRID["cell"]),
                                jet=(noz[0], noz[1] - 4 * sc, end[0], end[1], 24.0 * sc, 44.0 * sc, foot), seed=6)
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
        sp = fx.sparks_layer(ti, (RX + 4.0, g0 - 10.0), seed=5, n=420, ground_y=lambda x: gy(x) + 6.0, frame=f,
                             xf=(s, tx, ty), dur=LIFT - IGN + 0.35, size=0.95)
        core.over(pic, sp)
    if burning:
        fx.glow_at(pic, noz[0], noz[1] + 60 * sc, 120.0 * sc, (1.0, 0.72, 0.36), 0.26)

    # the birds burst out of the camphor tree
    if t >= BIRDS - 0.05:
        try:
            from .. import birds
            flock = birds.Flock([(1260.0, 470.0), (1390.0, 530.0), (1150.0, 440.0), (1450.0, 610.0), (1320.0, 590.0),
                                 (1060.0, 520.0)], n=36, heading=(0.72, -0.62), seed=6)
            core.over(pic, birds.draw_flock(flock, cam, t - BIRDS, f, None))
        except Exception as e:  # noqa
            print("s6 birds:", repr(e)[:200])

    # grass in front of everything on the hill
    if hs is not None and ground_on:
        try:
            if not burning:
                core.over(pic, hs.grass_band(cam, TOD, f, 1150, 2300, flatten=flat))
            core.over(pic, hs.render_front(cam, TOD, f, wind=wind, defocus=common.dof(1.0, 5.0), flatten=flat))
        except Exception as e:  # noqa
            print("s6 front:", repr(e)[:200])

    # the first clouds, then the bank that swallows the rocket
    bank, base = cloud_bank()
    if (base + 900.0) * s + ty > -40:
        ctod = palette.blend_tod("launch", "sunrise", 0.45)
        cl = fx_smoke.render(bank, 0.0, f, ctod, (s, tx, ty), seed=31, soft=5.0, haze=0.40, warm_top=0.5)
        hid = core.smooth(core.span((base - (g0 - m["H"])), -40.0, 120.0))     # how deep the rocket is in the bank
        if hid > 0.02:
            fx.glow_at(pic, noz[0], noz[1], 150.0 * s, (1.0, 0.70, 0.40), 0.5 * hid)
        core.over(pic, cl["layer"])
        if hid > 0.02:
            fx.glow_at(pic, noz[0], noz[1] - 30 * sc, 170.0 * s, (1.0, 0.78, 0.52), 0.30 * hid)

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
        fx.glow_at(pic, gsx, gsy - 60 * s, 620.0 * s, (1.0, 0.90, 0.70), 0.85 * kx)
        pic[:] = np.clip(pic * (1.0 + 0.9 * kx) + 0.10 * kx, 0, 1)
        opts = dict(bloom_amt=0.75 + 0.75 * kx, bloom_thr=0.70 - 0.22 * kx, vig=0.10)
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
