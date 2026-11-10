"""Shot 9a: morning. The scorched rocket drifts down under a patched paper
umbrella, swinging like a pendulum (the rocket lags the canopy), the camera
tilting down with it. Touchdown: the legs touch, squash, a small bounce, a
settle; the strings slip free, the umbrella folds and drops to one side with
follow through; grass puffs; a wisp of smoke from the nozzle.

Also exports draw_parachute_rest(pic, cam, t, f): the folded umbrella lying in
the grass just right of the rocket (s9b and s9d call it).

The cut is dissolved into from shot 8, so render() accepts negative fl: every
motion here is a plain function of time.
"""
import math

import numpy as np

from .. import core, palette
from ..timeline import FPS
from . import common

NAME = "s9a_home"
TOD = "morning"
RX = common.ROCKET_X
RIM_H, RIM_R = 428.0, 72.0        # cup rim of the rocket (rocket units = set px at scale 1)
L_STR = 150.0                     # canopy rim plane above the cup rim in flight
LR, NR, HANDLE = 133.0, 8, 62.0  # rib length, rib count, shaft below the rim plane
STR_A = (-1.36, -0.50, 0.50, 1.36)            # view angles of the four strings
REST = dict(X=1886.0, dY=-5.0, rot=1.80, open=0.08, spin=0.55)
FALL = (0.08, 0.56)               # the umbrella drops between these times after touchdown


def _td():
    return common.cue("touchdown", NAME)


def camera(t):
    td = _td()
    dx, dy = common.drift(t + 11.0, 2.0, 9)
    cx = core.keys(t, [(-0.8, 1625.0), (td + 0.2, 1700.0), (3.7, 1706.0)], core.ease_io)
    zoom = core.keys(t, [(-0.8, 0.84), (td, 0.98), (3.7, 1.02)], core.ease_io)
    tilt = core.keys(t, [(-0.8, 380.0), (td + 0.15, 0.0)], lambda u: core.ease_io(u, 1.6))
    u = t - td
    bump = 9.0 * math.exp(-u * 9.0) * math.cos(u * 34.0) if u > 0 else 0.0     # the thump of the landing
    return core.Camera(cx0=1600, cy0=900, cx=cx + dx, cy=862.0 + dy, zoom=zoom, tilt=tilt, shake=(0.0, bump))


# ------------------------------------------------------------------ the motion
def _theta(t, td):
    amp = 0.06 + 0.16 * core.clamp((td - t) / 3.4)
    return amp * math.sin(2 * math.pi * (t - td) / 1.9 + 0.9)


def _fly(t, td):
    """Hub of the canopy, cup rim centre and feet of the rocket while it hangs
    (relative to the landing point), and the two swing angles."""
    h = 150.0 * (td - t)
    th = _theta(t, td)
    thr = 1.25 * _theta(t - 0.13, td)                      # the rocket lags the canopy
    hub = np.array([-0.20 * h + 10 * math.sin(t * 1.3 + 1.0), -h - RIM_H - L_STR])
    rim = hub + L_STR * np.array([math.sin(th), math.cos(th)])
    feet = rim + RIM_H * np.array([-math.sin(thr), math.cos(thr)])
    return hub, feet, th, thr


def _dims(opn, t):
    al = math.radians(12 + 58 * opn)
    rt = LR * math.sin(al) * (1 + 0.03 * opn * math.sin(t * 5.3))
    dz = (LR * math.cos(al) * (1 + 0.25 * opn) + 18 * opn) * (1 + 0.035 * opn * math.sin(t * 5.3 + 1.2))
    return rt, dz


def _local(hub, rho, a, b):
    """Umbrella local (a across, b up the axis) to set coords."""
    cr, sr = math.cos(rho), math.sin(rho)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return np.c_[(hub[0] + a * cr + b * sr).ravel(), (hub[1] + a * sr - b * cr).ravel()]


def _rim_pt(X, Yf, rot, a, squash=0.0):
    """A point on the cup rim of the rocket (set coords) at view angle a."""
    e = 0.03 + 0.10 * RIM_H / 500.0
    lx = RIM_R * math.sin(a) * (1 + 0.11 * squash)
    ly = (-RIM_H - RIM_R * e * math.cos(a)) * (1 - 0.15 * squash)
    c, s = math.cos(rot), math.sin(rot)
    return np.array([X + lx * c - ly * s, Yf + lx * s + ly * c])


def _rest_hub():
    return np.array([REST["X"], common.ground_y(REST["X"]) + REST["dY"]])


def pose(t):
    """Everything about the rocket and the umbrella at time t (seconds in the cut)."""
    td = _td()
    gy = common.ground_y(RX)
    land = np.array([RX, gy])
    _, f0, _, _ = _fly(td, td)
    u = t - td
    if u < 0:
        hub, feet, th, thr = _fly(t, td)
        hub, feet = hub - f0 + land, feet - f0 + land
        P = dict(X=float(feet[0]), Yf=float(feet[1]), rot=thr, hub=hub, rho=-0.6 * _theta(t + 0.1, td) + 0.035 * math.sin(2.3 * t),
                 open=1.0, spin=0.2 + 0.45 * t, attach=1.0, dY_extra=0.0,
                 squash=-0.18 * core.smooth(core.span(t, td - 0.5, td - 0.04)), rattle=0.3,
                 flat=0.0, shadow=core.smooth(core.span(t, td - 0.45, td)) * 0.6,
                 contact=core.smooth(core.span(t, td - 0.9, td)))
        return P
    hub_td, _, _, thr_td = _fly(td, td)
    hub_td = hub_td - f0 + land
    rho_td = -0.6 * _theta(td + 0.1, td) + 0.035 * math.sin(2.3 * td)
    hop = -27.0 * math.sin(math.pi * core.span(u, 0.15, 0.42)) if 0.15 < u < 0.42 else 0.0
    squash = core.keys(u, [(0.0, -0.18), (0.04, 1.45), (0.11, 1.25), (0.18, -0.38), (0.36, -0.10), (0.45, 0.62), (0.57, -0.07), (0.68, 0.0)])
    rot = thr_td * math.exp(-u * 8.0) * math.cos(u * 16.0)
    # the umbrella: a beat of follow through, then it tips over, folds and drops to the right
    u1 = core.span(u, FALL[0], FALL[1])
    ti = max(u - FALL[1], 0.0)
    start = hub_td + np.array([8.0, 16.0]) * core.smooth(core.span(u, 0.0, 0.12))
    rest = _rest_hub()
    hub = np.array([start[0] + (rest[0] - start[0]) * core.ease_out(u1, 1.7),
                    start[1] + (rest[1] - start[1]) * core.ease_in(u1, 2.1)])
    hit = math.exp(-ti * 9.0) * (1.0 if u > FALL[1] else 0.0)
    hub[1] -= 17.0 * hit * abs(math.sin(ti * 15.0))
    rho = rho_td + (REST["rot"] - rho_td) * core.ease_io(u1, 1.8) + 0.22 * hit * math.sin(ti * 21.0)
    opn = 1.0 + (REST["open"] - 1.0) * core.smooth(core.span(u, 0.10, 0.52)) + 0.10 * math.sin(math.pi * core.span(u, 0.0, 0.14)) + 0.09 * hit * math.cos(ti * 17.0)
    spin = (0.2 + 0.45 * td) + (REST["spin"] - (0.2 + 0.45 * td)) * core.smooth(core.span(u, 0.0, 0.5))
    return dict(X=RX, Yf=gy + hop, rot=rot, hub=hub, rho=rho, open=opn, spin=spin,
                attach=1.0 - core.smooth(core.span(u, 0.02, 0.12)), squash=squash,
                rattle=0.5 * math.exp(-u * 6.0),
                flat=1.0 * core.ease_out(core.span(u, 0.0, 0.07)) * (1 - core.smooth(core.span(u, 0.26, 0.68))),
                shadow=0.6 + 0.4 * core.smooth(core.span(u, 0.0, 0.35)), contact=1.0)


# ---------------------------------------------------------------- the umbrella
def _strings(P, t):
    """Four strings as set space polylines: taut to the cup rim while attached,
    then slipping free and trailing from the canopy."""
    rt, dz = _dims(P["open"], t)
    out = []
    for k, a in enumerate(STR_A):
        top = _local(P["hub"], P["rho"], rt * math.sin(a), 0.20 * rt * math.cos(a))[0]
        rim = _rim_pt(P["X"], P["Yf"], P["rot"], a, P["squash"])
        # where the loose end goes: it hangs from the canopy and curls on the grass
        cr, sr = math.cos(P["rho"]), math.sin(P["rho"])
        loose = top + np.array([-26.0 - 9.0 * k, 15.0 + 5.0 * (k % 2)]) * (0.55 + 0.45 * abs(sr)) + np.array([0.0, 26.0]) * abs(cr)
        at = P["attach"]
        end = loose + (rim - loose) * at
        mid = (top + end) / 2 + np.array([(-10.0 + 7.0 * k) * (1 - at), 4.0 * at + (22.0 - 5.0 * k) * (1 - at)])
        mid[0] += 2.5 * math.sin(t * 7.0 + k) * at
        out.append(core.catmull([top, mid, end], 7, closed=False))
    return out


def _draw_umbrella(pic, cam, hub, rho, opn, t, drawing, spin, strings, tod=TOD):
    T = palette.TOD[tod]
    s, tx, ty = cam.xf(1.0)

    def S(a, b):
        p = _local(hub, rho, a, b)
        return np.c_[p[:, 0] * s + tx, p[:, 1] * s + ty]

    rt, dz = _dims(opn, t)
    ev = 0.20
    u = np.linspace(0, 1, 9)
    phis = spin + np.arange(NR) * (2 * math.pi / NR)
    ribs = []
    for i, ph in enumerate(phis):
        fl_ = 0.035 * opn * math.sin(t * 9.0 + i * 1.9)
        r = rt * np.sin(u * math.pi / 2) ** 0.9 * (1 + fl_ * u)
        z = dz * (1 - np.cos(u * math.pi / 2)) ** 0.9
        ribs.append(np.c_[r * math.sin(ph), dz - z + ev * r * math.cos(ph)])
    spts = [np.c_[q[:, 0] * s + tx, q[:, 1] * s + ty] for q in strings]
    allp = np.vstack([S(q[:, 0], q[:, 1]) for q in ribs] + [S([0, 0], [dz + 12, -HANDLE - 14])] + spts)
    x0 = int(max(math.floor(allp[:, 0].min()) - 14, -60))
    y0 = int(max(math.floor(allp[:, 1].min()) - 14, -60))
    x1 = int(min(math.ceil(allp[:, 0].max()) + 14, core.W + 60))
    y1 = int(min(math.ceil(allp[:, 1].max()) + 14, core.H + 60))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    cel = core.Cel(x1 - x0, y1 - y0, ss=3, x0=x0, y0=y0)
    ink = T["ink"]
    lw = max(s, 0.75)
    R = palette.ROCKET
    cols = [R["paper"], R["paper_dk"], R["paper"], R["patch_red"], R["paper"], R["paper_dk"], R["sail"], R["paper_dk"]]
    sgn = 1.0 if T["sun"][0] > 0 else -1.0
    warm = palette.lit(np.float32((1.0, 0.80, 0.52)), T)
    gores = []
    for i in range(NR):
        j = (i + 1) % NR
        phm = phis[i] + math.pi / NR
        a, b = ribs[i], ribs[j]
        mid = (a[-1] + b[-1]) / 2
        mid = mid + (np.array([0.0, dz * 0.55]) - mid) * 0.085
        edge = core.catmull([a[-1], mid, b[-1]], 6, closed=False)
        poly = np.vstack([a, edge, b[::-1]])
        gores.append((math.cos(phm), math.sin(phm) * sgn > -0.25, i, poly, edge))
    gores.sort(key=lambda g: g[0])
    wob = lambda p, k: core.wobble(p, 0.5 * lw, 300 + k + 17 * drawing, 0.05)
    # the far side, seen from inside under the near rim: paper glowing with the light through it
    for dep, lit, i, poly, edge in gores:
        if dep > 0:
            continue
        c = palette.mix(palette.shade(cols[i], T), warm, 0.42 if lit else 0.25)
        cel.fill(wob(S(poly[:, 0], poly[:, 1]), i), c)
        cel.stroke(S(ribs[i][:, 0], ribs[i][:, 1]), 1.1 * lw, palette.mix(ink, c, 0.35), taper=(0.05, 0.05))
    # shaft, stretchers, hook
    bam = R["bamboo"]
    shaft = S([0, 0], [dz + 2, -HANDLE])
    cel.stroke(shaft, 4.6 * s, ink, taper=(0.0, 0.0))
    cel.stroke(shaft, 2.8 * s, palette.lit(bam, T), taper=(0.0, 0.0))
    hook = S([0, 3.5, 9, 11], [-HANDLE + 3, -HANDLE - 6, -HANDLE - 5, -HANDLE + 3])
    cel.stroke(core.catmull(hook, 6, closed=False), 3.6 * s, ink, taper=(0.0, 0.2))
    cel.stroke(core.catmull(hook, 6, closed=False), 2.0 * s, palette.shade(bam, T), taper=(0.0, 0.2))
    if opn > 0.45:
        for dep, lit, i, poly, edge in gores:
            if dep <= 0:
                q = ribs[i][5]
                cel.stroke(S([0, q[0]], [dz * 0.22, q[1]]), 1.0 * lw, palette.mix(ink, warm, 0.25), taper=(0.05, 0.05))
    # the near gores, outside face: flat colour, one hard shadow side
    for dep, lit, i, poly, edge in gores:
        if dep <= 0:
            continue
        c = palette.lit(cols[i], T) if lit else palette.shade(cols[i], T)
        sp = wob(S(poly[:, 0], poly[:, 1]), i)
        cel.fill(sp, c)
        if lit and dep > 0.3:
            m = cel.mask(sp)
            cel.crescent(sp, palette.shade(cols[i], T), (2.6 * s * sgn, -1.2 * s), clip=m)
        if i == 5:      # a blue patch sewn over a tear
            a, b = ribs[i], ribs[(i + 1) % NR]
            q = np.array([a[3] + (b[3] - a[3]) * 0.22, a[3] + (b[3] - a[3]) * 0.80, a[6] + (b[6] - a[6]) * 0.76, a[6] + (b[6] - a[6]) * 0.26])
            qs = wob(S(q[:, 0], q[:, 1]), 40)
            cel.fill(qs, palette.lit(R["patch_blue"], T) if lit else palette.shade(R["patch_blue"], T))
            cel.stroke(qs, 1.0 * lw, ink, closed=True)
        if i == 1:      # a strip of tape
            a, b = ribs[i], ribs[(i + 1) % NR]
            q = np.array([a[5] + (b[5] - a[5]) * 0.30, a[5] + (b[5] - a[5]) * 0.72, a[6] + (b[6] - a[6]) * 0.70, a[6] + (b[6] - a[6]) * 0.32])
            qs = S(q[:, 0], q[:, 1])
            cel.fill(qs, palette.lit(R["tape"], T) if lit else palette.shade(R["tape"], T))
            cel.stroke(qs, 0.8 * lw, ink, closed=True)
    for dep, lit, i, poly, edge in gores:
        if dep <= 0 and math.cos(phis[i]) <= 0 and math.cos(phis[(i + 1) % NR]) <= 0:
            continue
        cel.stroke(wob(S(edge[:, 0], edge[:, 1]), 60 + i), 1.5 * lw, ink, taper=(0.08, 0.08), seed=i + drawing)
    for i in range(NR):
        if math.cos(phis[i]) > -0.12:
            cel.stroke(wob(S(ribs[i][:, 0], ribs[i][:, 1]), 80 + i), 1.7 * lw, ink, taper=(0.04, 0.12), seed=9 + i + drawing)
    top = S([0], [dz + 5])[0]
    cel.dot(top[0], top[1], 4.0 * s, ink)
    cel.dot(top[0], top[1], 2.6 * s, palette.lit(R["copper"], T))
    for q in spts:
        cel.stroke(q, 2.0 * lw, ink, taper=(0.02, 0.02))
        cel.stroke(q, 0.9 * lw, palette.lit(R["string"], T), taper=(0.02, 0.02))
    core.over(pic, cel.rgba(), x0, y0)


def draw_parachute_rest(pic, cam, t, f):
    """The folded umbrella lying in the grass just right of the rocket, its
    paper stirring a little, the loose strings curled beside it."""
    tt = core.on_n(int(round(t * FPS)), 2) / FPS
    P = dict(hub=_rest_hub(), rho=REST["rot"] + 0.012 * math.sin(tt * 1.7), open=REST["open"] + 0.012 * math.sin(tt * 2.3 + 1.0),
             X=RX, Yf=common.ground_y(RX), rot=0.0, squash=0.0, attach=0.0)
    _draw_umbrella(pic, cam, P["hub"], P["rho"], P["open"], tt, int(round(tt * 12)), REST["spin"], _strings(P, tt))
    _rest_tufts(pic, cam)


def _rest_tufts(pic, cam):
    """Grass standing in front of the lower half of the umbrella where it lies, so it sits IN the grass."""
    from .. import hillset_paint as hp
    from .. import paint
    s, tx, ty = cam.xf(1.0)
    hub = _rest_hub()
    rt, dz = _dims(REST["open"], 0.0)
    n = 60
    i = np.arange(n)
    h1, h2, h3, h4 = (hp.hashf(i, 11.0 + q) for q in range(4))
    b = -HANDLE * 0.9 + (dz + HANDLE * 0.9 + 10.0) * (i + h1) / n
    a = rt * (0.15 + 1.0 * h2)
    p = _local(hub, REST["rot"], a, b)
    xs, ys = p[:, 0] * s + tx, p[:, 1] * s + ty
    ok = (xs > 3) & (xs < core.W - 3) & (ys > 3) & (ys < core.H - 3 - (rt + 8) * s)
    if not ok.any():
        return
    xs, ys, h1, h3, h4 = xs[ok], ys[ok], h1[ok], h3[ok], h4[ok]
    base = pic[np.minimum((ys + (rt * 1.3 + 8) * s).astype(int), core.H - 1), xs.astype(int)]
    cols = np.clip(base * np.where(h4 < 0.55, 1.30, 0.78)[:, None], 0, 1)
    lay = np.zeros((core.H, core.W, 4), np.float32)
    o = np.argsort(ys)
    paint.strokes(lay, xs[o], ys[o], -np.pi / 2 + (h1[o] - 0.5) * 0.9, (2.0 + 2.2 * h3[o]) * s, (16.0 + 26.0 * h3[o]) * s, cols[o], op=0.96,
                  bend=(h4[o] - 0.5) * 1.2, seed=17, dry=0.1, taper=0.9, streak=0.04, paper_k=0.0)
    core.over(pic, lay)


# ------------------------------------------------------------- smoke and puffs
def _box_cel(pts, margin, ss=2):
    x0 = int(max(math.floor(pts[:, 0].min()) - margin, 0))
    y0 = int(max(math.floor(pts[:, 1].min()) - margin, 0))
    x1 = int(min(math.ceil(pts[:, 0].max()) + margin, core.W))
    y1 = int(min(math.ceil(pts[:, 1].max()) + margin, core.H))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, 0, 0
    return core.Cel(x1 - x0, y1 - y0, ss=ss, x0=x0, y0=y0), x0, y0


def _wisps(pic, cam, P, t, td):
    """Thin threads of smoke from the scorched nozzle: trailing above it while
    it falls, rising and curling once it stands."""
    T = palette.TOD[TOD]
    s, tx, ty = cam.xf(1.0)
    nz = np.array([P["X"] + 56 * math.sin(P["rot"]), P["Yf"] - 56 * math.cos(P["rot"])])
    tt = core.on_n(int(round(t * FPS)), 2) / FPS
    landed = core.smooth(core.span(t, td, td + 0.5))
    length = 250.0 - 70.0 * landed
    paths = []
    for k, (side, ph) in enumerate(((-1.0, 0.0), (1.0, 2.1))):
        u = np.linspace(0, 1, 16)
        x = nz[0] + side * (44 * u + 120 * u * u) + 13 * u * np.sin(6.0 * u - 2.4 * tt + ph) + 30 * landed * u * u
        y = nz[1] - 6 - length * u * (0.8 + 0.2 * k)
        paths.append(np.c_[x * s + tx, y * s + ty])
    cel, x0, y0 = _box_cel(np.vstack(paths), 30 * s)
    if cel is None:
        return
    col = palette.mix(palette.lit((0.66, 0.66, 0.74), T), palette.shade((0.66, 0.66, 0.74), T), 0.45)
    for k, p in enumerate(paths):
        cel.stroke(p, (14.0 + 4 * k) * (1 + 0.7 * landed) * s, col, taper=(0.10, 0.45), pressure=[0.35, 0.8, 1.0, 0.85, 0.5, 0.2], seed=k + int(tt * 12))
    core.over(pic, cel.rgba(), x0, y0, opacity=0.58 + 0.20 * landed)


def _puffs(pic, cam, u, drawing):
    """Dust and torn grass thrown outward when the legs hit (u = seconds after touchdown)."""
    if u < 0.0 or u > 1.1:
        return
    T = palette.TOD[TOD]
    s, tx, ty = cam.xf(1.0)
    gy = common.ground_y(RX)
    r = core.rng("touchdown puffs", 3)
    k = core.ease_out(u / 0.6, 2.2)
    fade = max(1 - u / 0.85, 0.0) ** 1.4
    pts = np.array([[(RX - 330) * s + tx, (gy - 190) * s + ty], [(RX + 330) * s + tx, (gy + 40) * s + ty]])
    cel, x0, y0 = _box_cel(pts, 4)
    if cel is None:
        return
    dust_l = palette.lit((0.93, 0.90, 0.80), T)
    dust_s = palette.shade((0.86, 0.84, 0.86), T)
    for i in range(11):
        d0 = r.uniform(-95, 95)
        sg = 1.0 if d0 > 0 else -1.0
        cx = RX + d0 + sg * r.uniform(70, 170) * k
        cy = gy + 2 - r.uniform(10, 44) * k
        rad = (9 + r.uniform(20, 38) * k) * (0.8 + 0.4 * r.random())
        ph = r.uniform(0, 6.28)
        lobes = []
        for q in range(3):
            lobes.append(core.ellipse((cx + math.cos(ph + q * 2.1) * rad * 0.55) * s + tx, (cy + math.sin(ph + q * 2.1) * rad * 0.3) * s + ty,
                                      rad * (0.7 + 0.12 * q) * s, rad * (0.55 + 0.1 * q) * s, 18))
        for e in lobes:
            cel.fill(core.wobble(e, 0.8 * s, 500 + i + drawing * 7, 0.06), dust_s, alpha=0.75 * fade)
        for e in lobes:
            cel.fill(core.wobble(e - (0, 2.2 * s), 0.8 * s, 600 + i + drawing * 7, 0.06) * (1, 1) - (0, rad * 0.14 * s), dust_l, alpha=0.8 * fade)
    grass_l = palette.lit((0.52, 0.66, 0.24), T)
    grass_s = palette.shade((0.30, 0.48, 0.30), T)
    for i in range(14):
        d0 = r.uniform(-80, 80)
        sg = 1.0 if d0 > 0 else -1.0
        vx, vy = sg * r.uniform(120, 330), -r.uniform(180, 420)
        x = RX + d0 + vx * u * (1 - 0.5 * u)
        y = gy - 4 + vy * u + 620 * u * u
        if y > gy + 6:
            continue
        a = r.uniform(0, 6.28) + (6.0 + 5 * r.random()) * u * sg
        ln = r.uniform(7, 15)
        p = np.array([[x - math.cos(a) * ln, y - math.sin(a) * ln], [x + math.cos(a) * ln, y + math.sin(a) * ln]])
        cel.stroke(np.c_[p[:, 0] * s + tx, p[:, 1] * s + ty], 2.6 * s, grass_l if i % 3 else grass_s, taper=(0.3, 0.5))
    # seeds knocked loose, floating up
    seed_c = palette.lit((1.0, 0.98, 0.90), T)
    sf = min(1.0, (1.1 - u) / 0.3)
    r = core.rng("touchdown seeds", 4)
    for i in range(20):
        d0 = r.uniform(-120, 120)
        sg = 1.0 if d0 > 0 else -1.0
        x = RX + d0 + sg * r.uniform(20, 80) * u + 7 * math.sin(u * 8 + i)
        y = gy - 8 - r.uniform(40, 130) * core.ease_out(u / 0.9, 1.8)
        rr = r.uniform(1.5, 2.6) * s
        cel.fill(core.ellipse(x * s + tx, y * s + ty, rr, rr, 10), seed_c, alpha=0.9 * sf)
        cel.stroke(np.array([[x * s + tx, y * s + ty], [x * s + tx - 1.5 * s * sg, y * s + ty + 6.5 * s]]), 0.8 * max(s, 1), seed_c, taper=(0.0, 0.6))
    core.over(pic, cel.rgba(), x0, y0)


# ----------------------------------------------------------------------- render
def render(fl, n, f):
    from .. import hillset as hs
    from .. import rocket
    t = fl / FPS
    td = _td()
    cam = camera(t)
    # positions move on ones (the camera does too, so nothing strobes); the
    # drawings boil on twos, and on ones through the landing
    drawing = fl if td - 0.15 < t < td + 0.75 else common.twos(fl)
    td_t = drawing / FPS
    P = pose(t)
    gy = common.ground_y(RX)
    pic = common.sky_pic(TOD, cam, f, t)
    flat = (RX, 300.0, P["flat"]) if P["flat"] > 0.01 else None
    hs.render_back(pic, cam, TOD, f, flatten=flat, rocket_shadow=P["shadow"] if t >= td - 0.45 else 0.0)
    common.draw_map_prop(pic, cam, TOD, t, wind=0.8)
    _wisps(pic, cam, P, t, td)
    if P["contact"] > 0.02:
        common.contact_shadow(pic, cam, RX, gy + 6, 190, TOD, 0.42 * P["contact"])
    st = rocket.State(soot=1.0, squash=P["squash"], fin_rattle=P["rattle"])
    common.draw_rocket(pic, cam, TOD, drawing, state=st, X=P["X"], dY=P["Yf"] - common.ground_y(P["X"]), rot=P["rot"], shadow=False)
    _draw_umbrella(pic, cam, P["hub"], P["rho"], P["open"], td_t, drawing, P["spin"], _strings(P, td_t))
    if t - td > FALL[1] + 0.02:
        _rest_tufts(pic, cam)
    _puffs(pic, cam, t - td, drawing)
    core.over(pic, hs.grass_band(cam, TOD, f, 1100, 2100, flatten=flat))
    core.over(pic, hs.render_front(cam, TOD, f, flatten=flat))
    return pic, dict(bloom_amt=0.45)
