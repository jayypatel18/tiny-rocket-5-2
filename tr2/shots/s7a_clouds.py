"""Shot 7a, Through the Clouds. The rocket climbs in the middle of the frame,
shaking, fins rattling, flame and smoke streaming behind it, while six depths
of cloud stream down past the camera. The light climbs from blue grey through
rose to gold. At `break_into_open_sky` it bursts out of the last deck in a
spray of mist: the sun flares from the right and god rays fan out."""
import math

import numpy as np

from .. import cloudset, core, palette
from .. import sky_paint as sp
from ..timeline import FPS
from . import common

NAME = "s7a_clouds"
W, H = core.W, core.H


def climb_at(t, T, tb):
    """The journey 0..1. The climb accelerates, reaches the top of the last deck
    exactly at the break, then eases into calm open sky."""
    D = cloudset.DECK
    if t <= tb:
        u = max(t, 0.0) / tb
        return D * (0.05 + 0.95 * (0.30 * u + 0.70 * u * u))
    return D + (1.0 - D) * core.ease_out(core.span(t, tb, T + 0.25), 1.8)


def _flame(cel, x, y, rot, s, drawing, power=1.0):
    """A cel flame streaming back from the nozzle: three hard tones, new every drawing."""
    r = core.rng("flame7", drawing)
    ax, ay = -math.sin(rot), math.cos(rot)          # down the rocket's axis
    px, py = ay, -ax
    L = 300.0 * s * power * r.uniform(0.86, 1.14)
    for k, (wid, ln, col) in enumerate(((30.0, 1.0, (1.0, 0.50, 0.16)), (21.0, 0.74, (1.0, 0.80, 0.30)),
                                        (11.0, 0.46, (1.0, 0.98, 0.84)))):
        u = np.linspace(0, 1, 26)
        w = wid * s * (0.55 + 0.9 * np.sin(np.pi * np.clip(u, 0, 1) ** 0.55) ** 1.0) * (1 - u ** 2.2)
        w *= 1 + 0.16 * np.sin(u * r.uniform(9, 15) + r.uniform(0, 6.28))
        sway = (r.uniform(-1, 1) * 9.0 * s) * u ** 1.5
        cx = x + ax * u * L * ln + px * sway
        cy = y + ay * u * L * ln + py * sway
        left = np.c_[cx - px * w, cy - py * w]
        right = np.c_[cx + px * w, cy + py * w][::-1]
        poly = core.wobble(np.vstack([left, right]), 1.6 * s, drawing * 7 + k, 0.06)
        cel.fill(poly, col)
    # loose licks torn off the tail
    for q in range(int(r.integers(2, 5))):
        u = r.uniform(0.75, 1.25)
        o = r.uniform(-16, 16) * s
        cx, cy = x + ax * u * L + px * o, y + ay * u * L + py * o
        rr = r.uniform(3.5, 8.0) * s
        cel.fill(core.ellipse(cx, cy, rr * 0.7, rr * 1.5, 14, rot), (1.0, 0.62, 0.22))
    return x + ax * L * 0.8, y + ay * L * 0.8


def _smoke(pic, x_of, y0, A, light, drawing, s):
    """The smoke column left behind: cel lobes strung down the frame, each one
    left where it was born, so the whole column streams down with the clouds.
    One silhouette, one hard shadow on the side away from the sun, a thin rim."""
    from PIL import ImageChops
    SP = 40.0
    bx0, by0 = int(x_of(0) - 460), int(max(0, y0 - 80))
    cel = core.Cel(w=920, h=max(H - by0, 8), ss=2, x0=bx0, y0=by0)
    tones = cloudset._tones(min(1.0, light * 0.8 + 0.2))
    base, shade, rimc = tones[3], tones[1], tones[4]
    hot = np.array([1.0, 0.80, 0.50], np.float32)
    k1 = int(math.floor(A / SP))
    lobes = []
    for k in range(k1, k1 - 44, -1):
        age = A - k * SP                                    # px since this puff was shed
        y = y0 + age
        r = core.rng("smoke7", k)
        rad = (24.0 + 4.2 * age ** 0.60) * s * r.uniform(0.8, 1.25)
        if y - rad > H + 20:
            break
        x = x_of(k) + r.uniform(-1, 1) * (10.0 + 0.16 * age) * s + 30.0 * s * math.sin(k * 0.7) * min(1.0, age / 300.0)
        lobes.append((x, y, rad, age, k))
    union = None
    half = None
    for (x, y, rad, age, k) in reversed(lobes):             # oldest first, the fresh ones on top
        r = core.rng("smoke7b", k, drawing // 2 % 3)
        heat = math.exp(-age / 170.0)
        col = palette.mix(base, hot, 0.9 * heat)
        for q in range(3):
            a = r.uniform(0, 6.28)
            d = rad * r.uniform(0.1, 0.6)
            rr = rad * r.uniform(0.6, 0.9)
            pts = core.wobble(core.ellipse(x + math.cos(a) * d * 1.2, y + math.sin(a) * d * 0.7, rr,
                                           rr * r.uniform(0.86, 1.0), 28), rad * 0.06, k * 13 + q, 0.02)
            cel.fill(pts, col)
            m = cel.mask(pts)
            union = m if union is None else ImageChops.lighter(union, m)
            if (k + q) % 3 == 0:
                half = m if half is None else ImageChops.lighter(half, m)
    if union is not None:
        ss = cel.ss
        scol = palette.mix(shade, np.array([0.80, 0.50, 0.40], np.float32), 0.25)
        dx, dy = int(26 * s * ss), int(-12 * s * ss)
        sh = ImageChops.subtract(union, ImageChops.offset(union, dx, dy))
        cel.paste(scol, sh)
        if half is not None:
            sh2 = ImageChops.multiply(ImageChops.subtract(half, ImageChops.offset(half, int(dx * 0.6), int(dy * 0.6))), union)
            cel.paste(scol, sh2, 0.75)
        rim = ImageChops.subtract(union, ImageChops.offset(union, int(-5 * s * ss), int(3 * s * ss)))
        cel.paste(palette.mix(rimc, hot, 0.3), rim)
    lay = cel.rgba()
    lay *= 0.96
    core.over(pic, lay, bx0, by0)


def render(fl, n, f):
    t = fl / FPS
    T = n / FPS
    tb = common.cue("break_into_open_sky", NAME)
    climb = climb_at(t, T, tb)
    A = climb * cloudset.TRAVEL
    light = float(sp.ss(-0.05, 1.0, climb / cloudset.DECK)) ** 1.15
    since = t - tb
    burst = math.exp(-since / 0.35) if since >= 0 else 0.0
    # camera: rides with the rocket, shaken hard, a kick at the break
    rate = (climb_at(t + 0.02, T, tb) - climb_at(t - 0.02, T, tb)) / 0.04
    calm = 1.0 if since < 0 else max(0.18, math.exp(-since / 0.22))
    amp = (4.0 + 7.0 * core.smooth(core.span(t, 0.0, tb))) * calm + 9.0 * burst
    sh = common.shake(f, amp, 7)
    cam = core.Camera(shake=sh, zoom=1.02 + 0.03 * core.smooth(t / T) - 0.02 * burst)

    pic = cloudset.render(cam, f, t, climb, rate=rate)

    # the rocket: sways, rattles; it surges up the frame as it breaks out
    tod = palette.blend_tod("predawn", "sunrise", core.clamp(0.25 + 0.75 * light))
    sway_x = 74.0 * core.noise1(t * 0.8, 71) + 26.0 * core.noise1(t * 2.3, 72)
    surge = 90.0 * core.overshoot(core.span(since, -0.12, 0.45), 2.2) if since > -0.12 else 0.0
    rx = W / 2 + sway_x
    ry = 742.0 - 30.0 * core.smooth(t / T) + 30.0 * core.noise1(t * 1.2, 73) - surge
    rot = 0.075 + 0.05 * core.noise1(t * 1.0, 74) + 0.03 * core.noise1(t * 2.6, 75)      # leaning a few degrees
    rr = core.rng("rattle7", fl)                                                         # hard shake, on ones
    from .. import rocket
    kshake = 0.35 + 0.65 * calm
    st = rocket.State(fin_rattle=1.0, flame=1.0, light=1.0,
                      shake=(float(rr.normal(0, 5.0)) * kshake, float(rr.normal(0, 4.0)) * kshake,
                             float(rr.normal(0, 0.022)) * kshake))
    s = 0.96                                         # about 480 px feet to tip
    sx, tx, ty = cam.xf(1.0)
    px, py = rx * sx + tx, ry * sx + ty
    cx0, cy0 = int(px - 360), int(py - 760)
    cel = core.Cel(w=720, h=920, ss=2, x0=cx0, y0=cy0)
    try:
        info = rocket.draw(cel, px, py, s=s * sx, tod=tod, drawing=fl, rot=rot, view="side", state=st)
    except Exception as e:  # noqa
        print("s7a rocket with blended light:", repr(e)[:160])
        cel = core.Cel(w=720, h=920, ss=2, x0=cx0, y0=cy0)
        info = rocket.draw(cel, px, py, s=s * sx, tod="sunrise", drawing=fl, rot=rot, view="side", state=st)
    nx, ny = info["nozzle"]
    # smoke and flame go behind the rocket
    fx0, fy0 = int(nx - 280), int(ny - 80)
    fcel = core.Cel(w=560, h=720, ss=2, x0=fx0, y0=fy0)
    tail = _flame(fcel, nx, ny, rot, s * sx * 1.25, fl, power=1.0 + 0.25 * burst)
    _smoke(pic, lambda k: px + 12.0 * math.sin(k * 0.23), tail[1] - 30.0, A, light, fl, s * sx)
    from .. import fx
    fx.glow_at(pic, nx, ny + 60 * s, 150.0 * s, (1.0, 0.62, 0.25), 0.55)
    core.over(pic, fcel.rgba(), fx0, fy0)
    fx.glow_at(pic, nx, ny + 30 * s, 46.0 * s, (1.0, 0.9, 0.6), 0.6)
    # the spray of mist it drags out of the deck
    if -0.05 < since < 0.8:
        lay = np.zeros((H, W, 4), np.float32)
        r = core.rng("spray7")
        cols = [np.asarray(c, np.float32) for c in cloudset._tones(0.95)]
        tau = since + 0.05
        for k in range(22):
            ang = r.uniform(-2.9, -0.25)
            v = r.uniform(380.0, 1150.0)
            kind = ("cumulus", "wisp", "cumulus")[k % 3]
            x = px + math.cos(ang) * v * (1 - math.exp(-tau * 3.0)) / 3.0 * 1.4
            y = py - 120.0 + math.sin(ang) * v * (1 - math.exp(-tau * 3.0)) / 3.0 + 1500.0 * tau * tau + 250.0 * tau
            sc = r.uniform(0.24, 0.6) * (1 + 1.8 * tau)
            sp.place(lay, kind, int(r.integers(0, 6)), x, y, sc, cols, op=max(0.0, 1 - tau / 0.8) ** 1.1,
                     flip=bool(k % 2))
        core.over(pic, core.gblur(lay, 1.0 + 3.0 * tau))

    try:
        rocket.draw_glass(cel, info, tod if isinstance(tod, str) else "sunrise", fl)
    except Exception as e:  # noqa
        print("s7a glass:", repr(e)[:120])
    core.over(pic, cel.rgba(), cx0, cy0)

    # near cloud and wisps whip past the lens
    core.over(pic, cloudset.render_front(cam, f, t, climb, rate=rate))

    # out in the open: the sun flares from the right, god rays fan out
    sa = cloudset.sun_amount(climb)
    if sa > 0.02:
        sxy = cloudset.sun_xy(cam, climb)
        occ = cloudset.last_alpha()
        fx.light_shafts(pic, sxy, occ if occ is not None else np.zeros((H, W), np.float32),
                        amount=0.30 * sa + 0.35 * burst, color=(1.0, 0.88, 0.62), length=0.9)
        fx.glow_at(pic, sxy[0], sxy[1], 260.0, (1.0, 0.86, 0.56), 0.28 * sa + 0.5 * burst)
        if burst > 0.02:
            xs = np.arange(W, dtype=np.float32)[None, :]
            ys = np.arange(H, dtype=np.float32)[:, None]
            streak = np.exp(-np.abs(xs - sxy[0]) / 700.0) * np.exp(-((ys - sxy[1]) / 5.0) ** 2)
            core.add_light(pic, streak.astype(np.float32), (1.0, 0.94, 0.78), 0.9 * burst)
            wash = np.exp(-np.hypot(xs - sxy[0], ys - sxy[1]) / 900.0)
            core.add_light(pic, wash.astype(np.float32), (1.0, 0.93, 0.76), 0.30 * burst)
    return pic, dict(tod=tod, bloom_amt=0.5, bloom_thr=0.80, vig=0.24)
