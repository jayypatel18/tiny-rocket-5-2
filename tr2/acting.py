"""Acting for the small panda in the wide cuts (written by the director on top
of the rig in panda.py). Each cut has key poses timed from the cue sheet; in
betweens are blended poses, so every drawing is rebuilt, never slid. Walks are
built from foot plants: the planted foot stays where it is in the world while
the body travels over it.

    act = acting.wide("s2_hill", fl, n)
    -> dict(pose, X (set x of the ground point), dY (set px, negative is up), s, drawing,
            layer ("front" or "behind_rocket"), visible)
"""
import math
from dataclasses import replace

import numpy as np

from . import core, panda
from .timeline import CUES, FPS, SHOTS

P = panda.POSES


def _start(shot):
    for n, a, b, tod in SHOTS:
        if n == shot:
            return a
    raise KeyError(shot)


def _cue(name, shot):
    v = CUES[name]
    a = _start(shot)
    return [x - a for x in v] if isinstance(v, (list, tuple)) else v - a


def mk(base, **kw):
    return replace(base, **kw)


def seq(t, ks):
    """Pose at time t from keys [(time, pose), (time, pose, ease), ...]. The
    ease shapes the move INTO that key. Holds outside the keys."""
    if t <= ks[0][0]:
        return ks[0][1]
    for i in range(1, len(ks)):
        if t <= ks[i][0]:
            t0, a = ks[i - 1][0], ks[i - 1][1]
            t1, b = ks[i][0], ks[i][1]
            e = ks[i][2] if len(ks[i]) > 2 else core.smooth
            return panda.blend(a, b, e((t - t0) / max(t1 - t0, 1e-6)))
    return ks[-1][1]


def val(t, ks, ease=core.smooth):
    return core.keys(t, ks, ease)


def over(u):
    return core.overshoot(u, 2.2)


def tail_of(pose, t, amp=7.0, speed=2.6, lag=0.0, lift=0.0, seed=0):
    """The rig's relaxed tail for this pose, swaying, with drag from motion
    (lag in units: positive drags the tip toward screen right) and lift."""
    base = np.array(pose.tail if pose.tail is not None else panda._default_tail(pose), np.float64)
    n = len(base)
    out = []
    for i, (x, y) in enumerate(base):
        w = (i / (n - 1)) ** 1.4
        sway = amp * w * math.sin(speed * t - i * 0.75 + seed)
        out.append((x + sway + lag * w, max(y + lift * w + 0.35 * amp * w * math.cos(speed * 0.8 * t - i * 0.6), 4.0)))
    return tuple(out)


def gait(t, lands, x0, x1, dur=0.36, half=7.0, lift=11.0):
    """A walk made of foot plants. lands = times each step touches down.
    Returns (X, foot_l, foot_r, bob, moving): body x, the two foot targets
    relative to the body, a vertical bob and 0..1 how much it is mid stride."""
    n = len(lands)
    D = x1 - x0
    w = [x0 - half, x0 + half]                      # world x of left and right foot
    lead = 0 if D < 0 else 1                         # the foot on the travel side goes first
    counts = [(n + 1) // 2, n // 2]
    done = [0, 0]
    lifts = [0.0, 0.0]
    moving = 0.0
    for k in range(n):
        i = (lead + k) % 2 if lead == 0 else (lead - k) % 2
        j = 0 if (k % 2 == 0) else 1                 # which of the two step counters
        idx = lead if k % 2 == 0 else 1 - lead
        m = counts[j]
        a = w[idx]
        done[j] += 1
        b = (x0 + (-half if idx == 0 else half)) + D * done[j] / m
        u = core.span(t, lands[k] - dur, lands[k])
        if u >= 1.0:
            w[idx] = b
        elif u > 0.0:
            e = core.ease_io(u, 1.6)
            w[idx] = a + (b - a) * e
            lifts[idx] = lift * math.sin(math.pi * u) ** 0.8
            moving = max(moving, math.sin(math.pi * u))
            break
        else:
            break
    X = 0.5 * (w[0] + w[1])
    # a small dip right after each plant
    dip = 0.0
    for tl in lands:
        if 0.0 <= t - tl < 0.14:
            dip = max(dip, math.sin(math.pi * (t - tl) / 0.14))
    bob = 3.2 * moving - 2.6 * dip
    return X, (w[0] - X, lifts[0]), (w[1] - X, lifts[1]), bob, moving


def _blink(t, times, d=0.14):
    for tb in times:
        if tb <= t < tb + d:
            return 1.0 - math.sin(math.pi * (t - tb) / d)
    return 1.0


# ------------------------------------------------------------------ shot 2
def s2(t):
    back = _cue("steps_back", "s2_hill")
    tomap = _cue("steps_to_map", "s2_hill")
    t_adm = _cue("admire", "s2_hill")
    t_tap = _cue("map_rustle", "s2_hill")
    XA, XB, XC = 1582.0, 1452.0, 1268.0
    # body x and feet
    if t < back[-1] + 0.2:
        X, fl_, fr_, bob, mv = gait(t, back, XA, XB)
    else:
        X, fl_, fr_, bob, mv = gait(t, tomap, XB, XC, dur=0.38)
    facing_rocket = mk(P["stand_34back"], yaw=1.28, head_yaw=1.12, pitch=0.5, smile=0.5, brow_l=0.4, brow_r=0.4,
                       paw_r=(34, 84), paw_l=(-16, 52), ear_l=-0.2, ear_r=-0.2)
    pat = mk(facing_rocket, paw_r=(40, 92), lean=4)
    step_back = mk(facing_rocket, paw_r=(20, 58), paw_l=(-20, 56), pitch=0.62, lean=-5, head_yaw=1.05)
    proud = mk(P["proud"], yaw=0.62, head_yaw=0.78, pitch=0.42, lean=-4, breath=1.0, smile=0.9)
    proud2 = mk(proud, squash=1.04, pitch=0.5, tilt=5, lean=-6)
    turn = mk(P["stand_34"], yaw=-0.25, head_yaw=-0.7, pitch=0.05, paw_l=None, paw_r=None, smile=0.5, brow_l=0.5,
              brow_r=0.5, squash=0.95)
    walk = mk(P["stand_profile"], yaw=-0.92, head_yaw=-0.9, pitch=0.0, lean=-5, smile=0.5, paw_l=(-14, 60), paw_r=(12, 56))
    read = mk(walk, lean=-9, pitch=-0.15, paw_l=(-34, 96), brow_l=0.3, brow_r=0.3, head_dx=-3, ear_l=-0.25, ear_r=-0.25)
    tap = mk(read, paw_l=(-44, 99), lean=-11)
    look = mk(walk, lean=-2, pitch=0.72, head_yaw=-0.62, paw_l=(-26, 78), smile=0.7, mouth_open=0.2, brow_l=0.9,
              brow_r=0.9, eye_l=1.2, eye_r=1.2, ear_l=-0.4, ear_r=-0.4, pupil=(-0.2, 0.7), hl=1.25)
    ks = [
        (0.0, pat), (0.45, facing_rocket), (back[0] - 0.4, facing_rocket), (back[0] - 0.1, step_back),
        (back[-1] + 0.1, step_back), (t_adm - 0.25, mk(proud, squash=0.93), core.ease_out), (t_adm, proud2, over),
        (t_adm + 0.35, proud), (tomap[0] - 0.62, proud), (tomap[0] - 0.36, turn, core.ease_io),
        (tomap[0] - 0.12, walk), (tomap[-1] + 0.05, walk), (t_tap - 0.28, read, core.ease_out),
        (t_tap - 0.03, tap, core.ease_in), (t_tap + 0.14, read, core.ease_out), (t_tap + 0.45, read),
        (t_tap + 0.85, look, over), (t_tap + 2.4, mk(look, pitch=0.8)),
    ]
    pose = seq(t, ks)
    walking = mv > 0.02
    lag = 0.0
    if t < back[-1] + 0.3:
        lag = 9.0 * mv                      # stepping left: the tail drags right
    elif tomap[0] - 0.4 < t < tomap[-1] + 0.3:
        lag = 12.0 * mv
    breath = 0.5 + 0.5 * math.sin(t * 2.1)
    pose = mk(pose, foot_l=fl_, foot_r=fr_, dy=pose.dy + bob, breath=max(pose.breath, breath * 0.6))
    pose = mk(pose, tail=tail_of(pose, t, amp=6.0 + 5.0 * mv, lag=lag, speed=2.4),
              eye_l=pose.eye_l * _blink(t, [0.8, 3.7, 5.95]), eye_r=pose.eye_r * _blink(t, [0.8, 3.7, 5.95]))
    ones = walking or abs(t - t_tap) < 0.3 or abs(t - t_adm) < 0.3
    return dict(pose=pose, X=X, ones=ones)


# ------------------------------------------------------------------ shot 3
SLEEP_X = 1646.0


def s3(t):
    tw = _cue("ear_twitch_sleep", "s3_night")
    br = 0.5 + 0.5 * math.sin(2 * math.pi * t / 3.4 - 1.0)
    ear = val(t, [(tw - 0.01, -0.05), (tw + 0.07, 0.75), (tw + 0.16, -0.3), (tw + 0.27, 0.35), (tw + 0.42, -0.05)], core.smooth)
    pose = mk(P["sleep"], breath=br, ear_r=ear, blanket=1.0, squash=1.0 + 0.012 * br)
    return dict(pose=pose, X=SLEEP_X, ones=abs(t - tw - 0.2) < 0.3, shadow=False)


# ----------------------------------------------------------------- shot 4a
def s4a(t):
    t_wake = _cue("wake", "s4a_wake")
    t_yawn = _cue("yawn_stretch", "s4a_wake")
    c1, c2 = _cue("climb_paws", "s4a_wake")
    t_in = _cue("into_cup", "s4a_wake")
    if t < t_wake + 0.22:
        br = 0.5 + 0.5 * math.sin(2 * math.pi * (t + 7.5) / 3.4 - 1.0)
        eo = core.smooth(core.span(t, t_wake, t_wake + 0.2))
        pose = mk(P["sleep"], breath=br, eye_l=0.9 * eo, eye_r=0.9 * eo, ear_r=-0.05 - 0.5 * eo, ear_l=0.7 - 0.9 * eo,
                  blanket=1.0 - 0.3 * eo)
        return dict(pose=pose, X=SLEEP_X, ones=t > t_wake - 0.05, shadow=False)
    sit = mk(P["stand"], squash=0.72, yaw=0.25, head_yaw=0.2, eye_l=0.55, eye_r=0.5, eye_smile=0.2, pitch=-0.15,
             paw_l=(-20, 40), paw_r=(20, 40), smile=0.0, ear_l=0.5, ear_r=0.4, lean=-3)
    rise = mk(sit, squash=0.9, pitch=0.1, eye_l=0.35, eye_r=0.35, mouth_open=0.3, paw_l=(-30, 80), paw_r=(30, 80))
    yawn = mk(P["stand"], squash=1.16, yaw=0.15, head_yaw=0.1, pitch=0.62, eye_l=0.0, eye_r=0.0, squeeze=0.8,
              mouth_open=1.0, mouth_wide=0.2, tongue=0.6, paw_l=(-34, 168), paw_r=(36, 172), ear_l=0.8, ear_r=0.8,
              brow_l=0.6, brow_r=0.6, lean=-3, breath=1.0)
    yawn2 = mk(yawn, squash=1.20, paw_l=(-40, 172), paw_r=(42, 176), lean=3, tilt=6)
    relax = mk(P["stand"], squash=0.94, yaw=0.3, head_yaw=0.35, eye_l=0.8, eye_r=0.8, smile=0.4, paw_l=(-24, 56),
               paw_r=(24, 56), pitch=0.0)
    shake1 = mk(relax, head_yaw=-0.25, tilt=-8, eye_l=0.1, eye_r=0.1, squeeze=0.5, ear_l=0.5, ear_r=-0.3)
    shake2 = mk(relax, head_yaw=0.55, tilt=8, eye_l=0.1, eye_r=0.1, squeeze=0.5, ear_l=-0.3, ear_r=0.5)
    look = mk(P["stand_profile"], yaw=0.8, head_yaw=0.85, pitch=0.6, smile=0.6, brow_l=0.6, brow_r=0.6, eye_l=1.15,
              eye_r=1.15, ear_l=-0.3, ear_r=-0.3, paw_l=(-10, 62), paw_r=(22, 66))
    crouch = mk(look, squash=0.74, pitch=0.5, brow_tilt=-0.6, smile=0.2, paw_l=(-6, 44), paw_r=(26, 50), lean=6)
    reach1 = mk(P["stand_34back"], yaw=1.42, head_yaw=1.3, pitch=0.5, squash=1.2, paw_r=(26, 176), paw_l=(-14, 120),
                foot_l=(-12, 6), foot_r=(14, 34), lean=4, ear_l=0.4, ear_r=0.4, brow_tilt=-0.5)
    reach2 = mk(reach1, squash=0.95, paw_l=(-20, 178), paw_r=(18, 126), foot_l=(-14, 38), foot_r=(12, 8), lean=-5,
                head_yaw=1.6)
    reach3 = mk(reach1, squash=1.15, paw_r=(24, 180), paw_l=(-18, 150), lean=3)
    over_rim = mk(P["stand_34back"], yaw=1.5, head_yaw=1.25, pitch=-0.3, squash=0.86, paw_l=(-26, 90), paw_r=(26, 92),
                  lean=10, ear_l=0.5, ear_r=0.5)
    inside = mk(P["stand"], yaw=0.3, head_yaw=0.4, pitch=0.1, smile=0.7, paw_l=(-26, 96), paw_r=(26, 96), squash=1.0,
                brow_l=0.5, brow_r=0.5)
    ks = [
        (t_wake + 0.22, sit), (t_wake + 0.45, rise, core.ease_out), (t_yawn, yawn, over), (t_yawn + 0.34, yawn2),
        (t_yawn + 0.52, relax, core.ease_in), (t_yawn + 0.62, shake1), (t_yawn + 0.72, shake2), (t_yawn + 0.82, shake1),
        (t_yawn + 0.94, relax), (t_yawn + 1.1, look, over), (c1 - 0.32, look), (c1 - 0.2, crouch, core.ease_out),
        (c1 - 0.06, reach1, core.ease_out), (c1 + 0.18, reach2), (c2 - 0.1, reach3), (c2 + 0.14, reach2),
        (t_in - 0.2, reach3), (t_in - 0.04, over_rim, core.ease_out), (t_in + 0.2, inside), (t_in + 1.5, inside),
    ]
    pose = seq(t, ks)
    X = val(t, [(c1 - 0.3, SLEEP_X + 4), (c1 - 0.02, 1694.0), (t_in - 0.1, 1708.0), (t_in + 0.12, 1722.0)], core.ease_io)
    # height on the rocket: skirt, body, cup rim, then down inside
    dY = val(t, [(c1 - 0.2, 0.0), (c1 - 0.02, -96.0), (c1 + 0.2, -150.0), (c2 - 0.05, -232.0), (c2 + 0.2, -280.0),
                 (t_in - 0.16, -352.0), (t_in - 0.02, -392.0), (t_in + 0.16, -300.0), (t_in + 0.3, -268.0)], core.smooth)
    layer = "behind_rocket" if t >= t_in - 0.05 else "front"
    on_rocket = t > c1 - 0.1
    if t >= t_in - 0.05:
        # the tail goes last: it whips up over the rim and flicks out of sight
        u = core.span(t, t_in - 0.05, t_in + 0.42)
        up = math.sin(math.pi * min(u * 1.25, 1.0))
        pose = mk(pose, tail=((6, 60), (14, 96 + 30 * up), (4 + 18 * math.sin(u * 9), 130 + 50 * up),
                              (-8 + 26 * math.sin(u * 9 - 1), 150 + 70 * up), (-2 + 30 * math.sin(u * 9 - 2), 160 + 84 * up)))
    else:
        pose = mk(pose, tail=tail_of(pose, t, amp=8.0, speed=3.0, lag=-10.0 if on_rocket else 0.0,
                                     lift=-20.0 if on_rocket else 0.0))
    return dict(pose=pose, X=X, dY=dY, layer=layer, ones=t > c1 - 0.35 or abs(t - t_yawn) < 0.25 or t_yawn + 0.5 < t < t_yawn + 1.0,
                shadow=not on_rocket)


# ----------------------------------------------------------------- shot 9b
def s9b(t):
    t_out = _cue("climb_out", "s9b_out")
    p1, p2 = _cue("paws_grass_home", "s9b_out")
    t_up = _cue("looks_up", "s9b_out")
    sooty = dict(soot=1.0, goggles=0.0, goggles_askew=14.0)
    peek = mk(P["stand"], yaw=-0.2, head_yaw=-0.3, pitch=0.1, eye_l=1.3, eye_r=1.3, brow_l=0.9, brow_r=0.9, smile=0.2,
              mouth_open=0.25, paw_l=(-28, 100), paw_r=(26, 100), ear_l=-0.3, ear_r=0.3, pupil_size=0.8, **sooty)
    shake1 = mk(peek, head_yaw=-0.75, tilt=-10, eye_l=0.0, eye_r=0.0, squeeze=0.7, ear_l=0.6, ear_r=-0.4, wind=0.6)
    shake2 = mk(peek, head_yaw=0.45, tilt=10, eye_l=0.0, eye_r=0.0, squeeze=0.7, ear_l=-0.4, ear_r=0.6, wind=0.6)
    grin = mk(peek, head_yaw=-0.35, smile=1.0, mouth_open=0.5, eye_smile=0.5, eye_l=1.0, eye_r=1.0, pupil_size=1.0)
    climb = mk(P["stand_34"], yaw=-0.6, head_yaw=-0.5, pitch=-0.2, squash=0.9, lean=-12, paw_l=(-30, 70), paw_r=(8, 96),
               foot_l=(-18, 20), foot_r=(10, 2), smile=0.7, **sooty)
    drop = mk(P["stand_34"], yaw=-0.5, head_yaw=-0.45, pitch=-0.25, squash=1.2, paw_l=(-36, 120), paw_r=(34, 124),
              foot_l=(-14, 14), foot_r=(14, 6), smile=0.6, mouth_open=0.5, ear_l=0.6, ear_r=0.6, wind=0.5, **sooty)
    land = mk(P["stand_34"], yaw=-0.4, head_yaw=-0.4, pitch=-0.3, squash=0.7, paw_l=(-34, 40), paw_r=(34, 40),
              eye_l=0.3, eye_r=0.3, squeeze=0.4, **sooty)
    stand = mk(P["stand_34"], yaw=-0.3, head_yaw=-0.35, squash=1.0, smile=0.8, paw_l=(-24, 56), paw_r=(24, 58), **sooty)
    hop = mk(stand, squash=1.1, dy=16, paw_l=(-30, 84), paw_r=(30, 86), smile=0.9, mouth_open=0.3)
    up = mk(P["look_up"], yaw=-0.2, head_yaw=-0.3, pitch=0.68, smile=0.6, mouth_open=0.12, paw_l=(-22, 56), paw_r=(20, 60),
            brow_l=0.6, brow_r=0.6, eye_l=1.15, eye_r=1.15, pupil=(-0.25, 0.65), hl=1.3, **sooty)
    ks = [
        (t_out - 0.02, mk(peek, squash=1.12)), (t_out + 0.1, peek, over), (t_out + 0.2, shake1), (t_out + 0.3, shake2),
        (t_out + 0.4, shake1), (t_out + 0.52, grin, over), (t_out + 0.68, grin), (t_out + 0.84, climb, core.ease_io),
        (p1 - 0.14, drop, core.ease_in), (p1 + 0.04, land, core.ease_out), (p1 + 0.22, stand, over),
        (p2 - 0.2, hop, core.ease_out), (p2 + 0.02, mk(stand, squash=0.84), core.ease_in), (p2 + 0.2, stand, over),
        (t_up - 0.1, stand), (t_up + 0.28, up, over), (t_up + 2.0, mk(up, pitch=0.74)),
    ]
    pose = seq(t, ks)
    X = val(t, [(t_out + 0.68, 1721.0), (t_out + 0.9, 1688.0), (p1, 1652.0), (p2 - 0.2, 1650.0), (p2, 1618.0)], core.smooth)
    dY = val(t, [(t_out - 0.1, -262.0), (t_out + 0.06, -346.0), (t_out + 0.14, -332.0), (t_out + 0.68, -334.0),
                 (t_out + 0.86, -372.0), (p1 - 0.12, -150.0), (p1, 0.0)], core.smooth)
    if p1 <= t:
        dY = -22.0 * math.sin(math.pi * core.span(t, p2 - 0.22, p2)) if t < p2 else 0.0
    layer = "behind_rocket" if t < t_out + 0.74 else "front"
    in_air = t_out + 0.84 < t < p1
    pose = mk(pose, tail=tail_of(pose, t, amp=7.0 + (10.0 if in_air else 0.0), speed=2.8, lift=40.0 if in_air else 0.0),
              breath=0.5 + 0.5 * math.sin(t * 2.4), eye_l=pose.eye_l * _blink(t, [t_up - 0.25]),
              eye_r=pose.eye_r * _blink(t, [t_up - 0.25]))
    return dict(pose=pose, X=X, dY=dY, layer=layer, visible=t > t_out - 0.1, ones=t < p2 + 0.3,
                shadow=t >= p1 - 0.1)


# ----------------------------------------------------------------- shot 9d
MAP_SIDE_X = 1268.0


def s9d(t):
    t_look = _cue("looks_at_us", "s9d_title")
    t_curl = _cue("tail_curl", "s9d_title")
    ta, tb = _cue("title_brush", "s9d_title")
    sooty = dict(soot=0.8, goggles=0.0, goggles_askew=10.0)
    at_map = mk(P["stand_profile"], yaw=-0.9, head_yaw=-0.88, pitch=0.05, lean=-6, paw_l=(-34, 92), paw_r=(10, 56),
                smile=0.7, eye_smile=0.3, **sooty)
    lower = mk(at_map, paw_l=(-20, 62), lean=-2, pitch=0.1)
    to_us = mk(P["camera_smile"], yaw=-0.22, head_yaw=0.02, tilt=-6, smile=0.95, eye_smile=0.3, paw_l=(-24, 56),
               paw_r=(24, 58), brow_l=0.4, brow_r=0.4, **sooty)
    glance = mk(to_us, head_yaw=0.38, pitch=0.42, pupil=(0.45, 0.55), tilt=2, smile=0.6, mouth_open=0.2, brow_l=0.8,
                brow_r=0.8, eye_l=1.2, eye_r=1.2, ear_l=-0.3, ear_r=-0.3)
    happy = mk(to_us, eye_l=0.0, eye_r=0.0, eye_smile=1.0, smile=1.0, tilt=7, squash=1.03)
    ks = [
        (0.0, at_map), (t_look - 0.22, lower), (t_look, mk(to_us, squash=0.95), core.ease_out), (t_look + 0.2, to_us, over),
        (ta + 0.15, to_us), (ta + 0.55, glance, over), (tb + 0.25, glance), (tb + 0.7, to_us, over),
        (tb + 1.05, to_us), (tb + 1.3, happy, over), (tb + 3.5, happy),
    ]
    pose = seq(t, ks)
    # the tail curls once: it rolls up into a spiral, holds a beat, and unrolls with a settle
    u = core.span(t, t_curl, t_curl + 1.15)
    k = math.sin(math.pi * u) ** 0.7 if 0 < u < 1 else 0.0
    base = np.array(panda._default_tail(to_us), np.float64)
    d = 1.0 if base[-1][0] >= 0 else -1.0
    curl = np.array([(d * 14, 32), (d * 38, 26), (d * 58, 40), (d * 60, 66), (d * 44, 78), (d * 34, 62)], np.float64)
    tl = base + (curl - base) * k
    sw = tail_of(mk(pose, tail=tuple(map(tuple, tl))), t, amp=4.0 * (1 - k) + 1.0, speed=2.2)
    blink_t = [t_look + 0.75, ta - 0.3, tb + 0.95]
    pose = mk(pose, tail=sw, breath=0.5 + 0.5 * math.sin(t * 2.0),
              eye_l=pose.eye_l * _blink(t, blink_t), eye_r=pose.eye_r * _blink(t, blink_t))
    return dict(pose=pose, X=MAP_SIDE_X, ones=abs(t - t_look) < 0.3 or (t_curl < t < t_curl + 1.2))


_TABLE = {"s2_hill": s2, "s3_night": s3, "s4a_wake": s4a, "s9b_out": s9b, "s9d_title": s9d}


def wide(shot, fl, n):
    fn = _TABLE[shot]
    t = fl / FPS
    a = fn(t)
    if not a.get("ones", False):
        # on twos: sample the pose on the even frame, so the drawing holds for two frames
        f2 = core.on_n(fl, 2)
        a2 = fn(f2 / FPS)
        if not a2.get("ones", False):
            a = a2
            a["drawing"] = f2
        else:
            a["drawing"] = fl
    else:
        a["drawing"] = fl
    a.setdefault("dY", 0.0)
    a.setdefault("s", 1.0)
    a.setdefault("layer", "front")
    a.setdefault("visible", True)
    return a
