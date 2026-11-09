"""Checkpoint still: one lit frame of the launch, about 0.6 s after ignition, on a
placeholder hill (until the real set exists)."""
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tr2 import core, fx, palette, rocket  # noqa: E402

REVIEW = os.path.join(ROOT, "build", "review")
CACHE = os.path.join(ROOT, "build", "cache", "fx")
W, H = core.W, core.H
RX = 1010.0          # rocket feet x
S = 1.36             # rocket scale (about 620 px tall in the low view)


def ground(x):
    return 868.0 + 0.000085 * (x - 1040.0) ** 2 + 5.0 * math.sin(x * 0.011)


def placeholder_bg(tod="launch"):
    T = palette.TOD[tod]
    yy = np.arange(H, dtype=np.float32)
    v = np.clip(yy / 840.0, 0, 1) ** 1.15
    n = core.fbm(H, W, 420.0, 4, seed=3, aspect=2.6)
    vv = np.clip(v[:, None] + (n - 0.5) * 0.16, 0, 1)
    pic = palette.sky_color(tod, vv).astype(np.float32)
    # long brushed cloud bars low in the east
    cl = core.fbm(H, W, 150.0, 5, seed=8, aspect=5.5)
    band = np.exp(-((yy[:, None] - 640.0) / 150.0) ** 2)
    a = np.clip((cl - 0.52) * 5.0, 0, 1) * band * 0.55
    ccol = palette.mix(T["fog"], np.array([1.0, 0.74, 0.62], np.float32), 0.6)
    pic += (ccol - pic) * a[..., None]
    pic *= (0.98 + 0.04 * core.fbm(H, W, 7.0, 2, seed=9, aspect=5.0))[..., None]
    # far ridge in the fog
    xs = np.arange(W)
    ridge = 792 - 60 * core.fbm_at(xs, np.zeros(W), 380.0, 4, seed=4) - 0.00004 * (xs - 400.0) ** 2
    m = np.clip((yy[:, None] - ridge[None, :]) / 1.5, 0, 1)
    rc = palette.mix(T["fog"] * 0.62, np.array([0.30, 0.34, 0.56], np.float32), 0.5)
    pic += (rc - pic) * m[..., None]
    # the hill: dark predawn grass, painted with broken strokes
    gy = np.array([ground(x) for x in xs], np.float32)
    gm = np.clip((yy[:, None] - gy[None, :]) / 1.2, 0, 1)
    g1 = core.fbm(H, W, 60.0, 4, seed=5, aspect=2.5)
    g2 = core.fbm(H, W, 5.0, 2, seed=6, aspect=0.25)
    base = np.array([0.105, 0.19, 0.215], np.float32)
    hi = np.array([0.20, 0.31, 0.30], np.float32)
    depth = np.clip((yy[:, None] - gy[None, :]) / 220.0, 0, 1)
    grass = base[None, None, :] * (0.72 + 0.5 * g1 - 0.25 * depth)[..., None] \
        + (hi - base)[None, None, :] * (np.clip((g2 - 0.56) * 5, 0, 1) * (0.25 + 0.6 * g1))[..., None]
    pic += (grass - pic) * gm[..., None]
    # blades along the crest
    cel = core.Cel(ss=2)
    rg = core.rng("blades")
    for k in range(520):
        x = rg.uniform(-20, W + 20)
        y = ground(x) + rg.uniform(-1, 26) ** 1.0
        hgt = rg.uniform(10, 34) * (1.0 + 0.5 * (y - ground(x)) / 26.0)
        lean = rg.normal(0.0, 0.35)
        tip = (x + lean * hgt, y - hgt)
        c = base * rg.uniform(0.7, 1.5)
        cel.stroke(np.array([(x, y + 3), (x + lean * hgt * 0.35, y - hgt * 0.55), tip]), rg.uniform(2.2, 4.2), c,
                   taper=(0.0, 0.8))
    core.over(pic, cel.rgba())
    return pic, gm


def run_sim(t_end=0.6, feet_y=None, noz=None):
    sim = fx.SmokeSim(384, 216, 5.0, (0.0, 0.0), ground=ground, seed=1)
    dt = 1.0 / 96.0
    steps = int(round(t_end / dt))
    gyc = ground(noz[0])
    for k in range(steps):
        t = k * dt
        emit = dict(x=noz[0], y=noz[1] + 26, r=44.0, jet=(70.0, 1500.0), rate=9.0,
                    burst=150.0 * max(0.0, 1 - t / 0.14), burst_r=120.0, burst_y=gyc - 22.0,
                    burst_d=24.0, burst_heat=0.85)
        # the ignition ring: two fronts racing outward along the ground, slowing as they go
        puffs = []
        for sgn, reach, tau, sd in ((-1, 640.0, 0.26, 0), (1, 560.0, 0.22, 1)):
            X = reach * (1 - math.exp(-t / tau))
            V = reach / tau * math.exp(-t / tau)
            px = noz[0] + sgn * X
            puffs.append(dict(x=px, y=ground(px) - 30.0, r=58.0, d=30.0 * math.exp(-t / 0.5), T=0.22,
                              u=sgn * V * 0.8, v=-60.0, grip=0.6, flat=0.7))
        emit["puffs"] = puffs
        sim.step(dt, emit)
    return sim


def render(path=None, t=0.6, f=886, lift=34.0, save_layers=False):
    t0 = time.time()
    tod = "launch"
    pic, gmask = placeholder_bg(tod)
    feet = (RX, ground(RX) - lift)
    st = rocket.State(light=1.0, flame=1.0, fin_rattle=0.8, squash=-0.25, shake=(3.0, -2.0, 0.012))
    cel = core.Cel()
    info = rocket.draw(cel, feet[0], feet[1], s=S, tod=tod, drawing=f, view="low34", state=st)
    rocket.draw_glass(cel, info, tod, f)
    rk = cel.rgba()
    noz = info["nozzle"]
    print("rocket %.1fs  nozzle %s tip %s" % (time.time() - t0, noz, info["tip"]))
    sim = run_sim(t, feet[1], noz)
    print("sim %.1fs  dmax %.2f Fmax %.2f" % (time.time() - t0, sim.d.max(), sim.F.max()))
    gx, gy = noz[0], ground(noz[0])
    lay = fx.smoke_layers(sim.fields(), f, tod, 5.0, 0.0, 0.0, seed=1, fire_xy=(gx, gy - 30))
    print("smoke render %.1fs" % (time.time() - t0))
    heat = lay["heat"]
    gx, gy = noz[0], ground(noz[0])
    # fire light on the grass, and a little on everything near it
    fx.ground_light(pic, gx, gy + 30, 560.0, (1.0, 0.62, 0.26), 1.0, squash=0.42, mask=gmask)
    fx.glow_at(pic, gx, gy - 40, 420.0, (1.0, 0.62, 0.30), 0.30, squash=0.7)
    # back smoke, the rocket, then the low front rolls that swallow the legs
    core.over(pic, lay["smoke"])
    core.over(pic, rk)
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    wob = core.fbm(H, W, 90.0, 3, seed=44)
    front_line = gy - 74.0 + (wob - 0.5) * 70.0 - 26.0 * np.exp(-((xx - gx) / 170.0) ** 2)
    fm = np.clip((yy - front_line) / 1.5, 0, 1).astype(np.float32)
    core.over(pic, lay["smoke"] * fm[..., None])
    core.over(pic, lay["flame"])
    sp = fx.sparks_layer(t, (gx, gy - 12), seed=5, n=330, ground_y=lambda x: ground(x) + 6, frame=f)
    core.over(pic, sp)
    fx.glow_at(pic, noz[0], noz[1] + 60, 110.0, (1.0, 0.80, 0.45), 0.55)
    fx.bloom(pic, 0.80, 12.0, 0.62)
    # a light grade and vignette so the still reads as a frame
    lift_, gain, sat = palette.TOD[tod]["grade"]
    pic[:] = np.clip(pic * gain + lift_, 0, 1)
    g = pic.mean(axis=2, keepdims=True)
    pic[:] = np.clip(g + (pic - g) * sat, 0, 1)
    vg = 1 - 0.22 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.2
    pic *= vg[..., None]
    core.save(path or os.path.join(REVIEW, "launch-look.png"), pic, seed=f)
    print("total %.1fs" % (time.time() - t0))
    return pic


if __name__ == "__main__":
    render()
