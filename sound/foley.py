"""foley.py: the ambience beds and every spot effect of TINY ROCKET, placed from cues.json.
Everything is filtered noise, impulse trains through resonators, and sines."""
import numpy as np
from synth import (SR, TAU, N, m2f, db, rng, rms, nrm, nrms, lp, hp, bp, white, pink, hann, perc, fade,
                   slow, svf, k_partials, pan_gains, musicbox)


# ---------------------------------------------------------------- generators (all return peak 1)
def burst(r, dur, lo, hi, att, dec):
    n = int(dur * SR)
    return nrm(bp(white(n, r), lo, hi, 1) * perc(n, att, dec))


def tone(dur, f0, f1, tau, att=0.002):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / (tau * 0.6))
    return np.sin(np.cumsum(TAU * f / SR)) * perc(n, att, tau)


def ring(r, dur, freqs, taus, amps=None):
    """A small struck object: a few decaying partials."""
    n = int(dur * SR)
    y = np.zeros(n)
    f = np.asarray(freqs, dtype=np.float64)
    a = np.ones(f.size) if amps is None else np.asarray(amps, dtype=np.float64)
    k_partials(y, f, a, np.asarray(taus, dtype=np.float64), r.uniform(0, TAU, f.size), float(SR))
    y[:24] *= np.linspace(0, 1, 24)
    return nrm(y)


def creak(r, dur, rate0, rate1, res, wob=0.35):
    """Stick and slip: an irregular impulse train through a few resonances (fibre, bamboo or tin)."""
    n = int(dur * SR)
    rate = np.linspace(rate0, rate1, n) * (1.0 + wob * (2.0 * slow(n, 9.0, r) - 1.0))
    ph = np.cumsum(rate / SR) + 0.97          # the first slip comes at once, so a creak starts on its cue
    k = np.flatnonzero(np.diff(np.floor(ph)) > 0) + 1
    imp = np.zeros(n)
    imp[k] = r.uniform(0.35, 1.0, k.size)
    y = np.zeros(n)
    for f, q, g in res:
        y += g * svf(imp, f, q, 1)
    return nrm(y)


def paper(r, dur, dens=300.0, lo=1200.0, hi=7500.0, shape=None):
    """Paper, leaves, grass: sparse random clicks of very uneven size through a band."""
    n = int(dur * SR)
    env = perc(n, 0.02, dur / 3.0) if shape is None else shape
    env = env / env.max()
    imp = (r.random(n) < dens / SR * env) * r.exponential(1.0, n)
    return nrm(fade(bp(imp + 0.03 * white(n, r) * env, lo, hi, 1), 0.002, 0.02))


def swish(r, dur, f0, f1, f2, q=0.9):
    n = int(dur * SR)
    x = np.linspace(0, 1, n)
    fc = np.interp(x, [0, 0.4, 1], [f0, f1, f2])
    return nrm(svf(white(n, r), fc, q, 1) * np.sin(np.pi * x ** 0.6) ** 1.5)


def breath(r, dur, f0, f1, q=0.9, skew=0.35):
    """Air through a nose or mouth, no voice: band noise with a moving centre."""
    n = int(dur * SR)
    x = np.linspace(0, 1, n)
    env = np.where(x < skew, 0.5 - 0.5 * np.cos(np.pi * x / skew), 0.5 + 0.5 * np.cos(np.pi * (x - skew) / (1 - skew)))
    y = svf(white(n, r), np.linspace(f0, f1, n), q, 1) + 0.25 * bp(white(n, r), 3000.0, 7000.0, 1)
    return nrm(y * env * (0.8 + 0.2 * slow(n, 30.0, r)))


def paw_grass(r):
    n = int(0.2 * SR)
    y = np.zeros(n)
    y += 0.6 * np.pad(tone(0.12, r.uniform(95, 130), r.uniform(70, 90), 0.03), (0, n - int(0.12 * SR)))
    y += 0.5 * nrm(lp(white(n, r), 500.0, 1) * perc(n, 0.002, 0.025))
    ru = paper(r, 0.16, dens=r.uniform(350, 700), lo=r.uniform(1300, 2200), hi=6000.0)
    y[480:480 + ru.size] += 0.55 * ru
    return nrm(y)


def flick(r):
    n = int(r.uniform(0.035, 0.05) * SR)
    return nrm(svf(white(n, r), np.linspace(r.uniform(900, 1400), r.uniform(2500, 3800), n), 1.0, 1) * hann(n) ** 0.7)


def blink(r):
    y = np.zeros(int(0.11 * SR))
    a = burst(r, 0.02, 2200, 5500, 0.0004, 0.004)
    y[:a.size] += a
    y[int(0.07 * SR):int(0.07 * SR) + a.size] += 0.6 * burst(r, 0.02, 1800, 4500, 0.0004, 0.004)
    return y


def brush(r, dur):
    """A wet brush on paper: soft band noise with bristle grain, the band opening then closing."""
    n = int(dur * SR)
    x = np.linspace(0, 1, n)
    fc = np.interp(x, [0, 0.35, 1], [r.uniform(1300, 1900), r.uniform(2600, 3600), r.uniform(1100, 1600)])
    env = np.sin(np.pi * x ** 0.55) ** 1.2
    grain = 0.65 + 0.35 * nrm(np.abs(lp(white(n, r), 220.0, 1)))
    return nrm(svf(white(n, r), fc, 0.8, 1) * env * grain)


def bird_call(r, kind):
    """One bird call. Six families, every call drawn with its own pitch, speed and length."""
    syl = []  # (start, dur, f0, f1, amp)
    if kind == 0:      # falling whistle, once or twice
        t = 0.0
        for _ in range(int(r.integers(1, 3))):
            d = r.uniform(0.13, 0.22)
            syl.append((t, d, r.uniform(4800, 5600), r.uniform(3300, 3900), 1.0))
            t += d + r.uniform(0.05, 0.12)
    elif kind == 1:    # rising trill
        f = r.uniform(2800, 3600)
        rate = r.uniform(12, 18)
        for i in range(int(r.integers(6, 13))):
            syl.append((i / rate, 0.6 / rate, f, f * r.uniform(1.25, 1.45), 0.6 + 0.4 * np.sin(np.pi * i / 12)))
    elif kind == 2:    # two tones, high then low
        f = r.uniform(3700, 4400)
        syl += [(0.0, 0.16, f, f * 0.99, 1.0), (0.22, 0.28, f * 0.78, f * 0.76, 0.9)]
    elif kind == 3:    # warble
        t = 0.0
        for i in range(int(r.integers(4, 8))):
            d = r.uniform(0.05, 0.11)
            f = r.uniform(2600, 4600)
            syl.append((t, d, f, f * r.uniform(0.8, 1.3), r.uniform(0.6, 1.0)))
            t += d + r.uniform(0.01, 0.05)
    elif kind == 4:    # chips
        f = r.uniform(4800, 6200)
        gap = r.uniform(0.09, 0.16)
        for i in range(int(r.integers(2, 6))):
            syl.append((i * gap, 0.035, f * 1.15, f * 0.85, 1.0))
    else:              # dove like coo, low
        f = r.uniform(430, 520)
        syl += [(0.0, 0.22, f, f * 1.03, 0.7), (0.3, 0.42, f * 1.06, f * 0.97, 1.0), (0.82, 0.2, f * 0.98, f * 0.95, 0.6)]
    n = int((max(s[0] + s[1] for s in syl) + 0.02) * SR)
    y = np.zeros(n)
    for t, d, f0, f1, a in syl:
        m = int(d * SR)
        f = np.linspace(f0, f1, m) * (1.0 + 0.012 * np.sin(TAU * r.uniform(25, 45) * np.arange(m) / SR))
        ph = np.cumsum(TAU * f / SR)
        s = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * hann(m) ** 0.8 * a
        i = int(t * SR)
        y[i:i + m] += s
    return nrm(y)


def crickets(dur, r, count, f_lo=4100.0, f_hi=5300.0, rate=(1.3, 2.6)):
    """A field of crickets, each with its own pitch, pulse count, chirp rate, bouts and place."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for k in range(count):
        fc = r.uniform(f_lo, f_hi)
        pulses = int(r.integers(2, 5))
        prate = r.uniform(48, 72)
        crate = r.uniform(*rate)
        amp = db(r.uniform(-15, 0))
        gl, gr = pan_gains(r.uniform(-1, 1))
        env = np.zeros(n)
        pw = int(0.62 / prate * SR)
        w = hann(pw)
        tc = r.uniform(0, 1.0 / crate)
        rest_at = tc + r.uniform(3, 9)
        while tc < dur:
            for p in range(pulses):
                i = int((tc + p / prate) * SR)
                if i + pw < n:
                    env[i:i + pw] += w * (0.7 + 0.3 * p / pulses)
            tc += (1.0 / crate) * r.uniform(0.96, 1.05)
            if tc > rest_at:
                tc += r.uniform(0.8, 2.8)
                rest_at = tc + r.uniform(3, 9)
        s = np.sin(TAU * fc * t + r.uniform(0, TAU)) * env * amp
        out[0] += s * gl
        out[1] += s * gr
    return nrms(out)


def breeze(dur, r, lo=250.0, hi=2200.0, gust=0.3, hiss=0.3):
    """Air through grass: pink noise in a band with slow gusts, a little high hiss riding the gusts."""
    n = int(dur * SR)
    common = slow(n, gust, r)
    out = np.zeros((2, n))
    for c in range(2):
        g = (0.3 + 0.7 * (0.6 * common + 0.4 * slow(n, gust * 1.7, r))) ** 1.6
        out[c] = nrms(bp(pink(n, r), lo, hi, 1)) * g + hiss * nrms(bp(white(n, r), 3000.0, 6500.0, 1)) * g ** 2
    return nrms(out)


def night_bed(dur, r):
    """Night insects: one steady trill (a tree cricket), a few slow crickets."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = 0.8 * crickets(dur, r, 6, 3600.0, 4700.0, (0.7, 1.4))
    for c, f in enumerate([3150.0, 3230.0]):
        am = (0.5 + 0.5 * np.sin(TAU * (41.0 + c) * t)) ** 2
        out[c] += 0.5 * nrms(np.sin(TAU * f * t) * am) * (0.4 + 0.6 * slow(n, 0.4, r))
    return nrms(out)


def bed_env(n, fin, fout):
    e = np.ones(n)
    a, b = int(fin * SR), int(fout * SR)
    e[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    e[n - b:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(b) / b)
    return e


# ---------------------------------------------------------------- the build
def build(M, C, shots):
    S = {s["name"]: s for s in shots}
    DUR = C["end"]
    R = lambda *k: rng("foley", *k)

    # ---------------- beds
    t1 = S["s3_night"]["start"]
    n = int((t1 + 0.7) * SR)
    e = bed_env(n, 0.02, 1.3)
    # the field steps back a little when the piano enters
    e *= np.interp(np.arange(n) / SR, [0, C["motif_first"] - 0.7, C["motif_first"] + 0.6], [1.0, 1.0, 0.6])
    M.add("amb", 0.0, crickets(t1 + 0.7, R("crk"), 16) * e, gain=db(-35))
    M.add("amb", 0.0, breeze(t1 + 0.7, R("brz")) * e, gain=db(-37))
    t2 = S["s4a_wake"]["start"]
    n = int((t2 + 0.7 - (t1 - 0.6)) * SR)
    M.add("amb", t1 - 0.6, night_bed(n / SR, R("night")) * bed_env(n, 1.2, 1.3), gain=db(-39))
    M.add("amb", t1 - 0.6, breeze(n / SR, R("nbrz"), 200, 1200, 0.2, 0.1) * bed_env(n, 1.2, 1.3), gain=db(-44))
    t3 = C["stillness"][0]
    n = int((t3 - (t2 - 0.6)) * SR)
    M.add("amb", t2 - 0.6, breeze(n / SR, R("pbrz"), 220, 1800, 0.25, 0.2) * bed_env(n, 1.2, 0.01), gain=db(-41))
    M.add("amb", t2 - 0.6, crickets(n / SR, R("pcrk"), 4, 3900, 4800, (0.8, 1.5)) * bed_env(n, 1.2, 0.01), gain=db(-44))
    r = R("firstbirds")
    for i in range(5):
        t = t2 + 0.8 + (S["s5a_match"]["start"] - t2 - 1.5) * (i + r.uniform(0.1, 0.9)) / 5.0
        M.add("sky", t, lp(bird_call(r, int(r.integers(0, 5))), 5000.0, 1), pan=r.uniform(-0.9, 0.9), gain=db(-37))

    # ---------------- shot 1: the last knot (close)
    c0, c1 = C["knot_creak"]
    r = R("knot")
    y = creak(r, c1 - c0, 28, 78, [(900, 10, 1.0), (1750, 12, 0.7), (2900, 10, 0.4)])
    nn = y.size
    y = fade(y * (0.55 + 0.45 * slow(nn, 7, r)) * np.linspace(0.6, 1.0, nn), 0.008, 0.03)
    M.add("close", c0, y, pan=-0.2, gain=db(-21), cue=("knot_creak", c0))
    t = C["knot_tug"]
    y = np.zeros(int(0.3 * SR))
    a = burst(r, 0.1, 1100, 5200, 0.0005, 0.02)
    y[:a.size] += a
    y += 0.6 * ring(r, 0.3, [123, 247, 372, 498], [0.09, 0.06, 0.04, 0.03], [1, 0.6, 0.35, 0.2])
    M.add("close", t, nrm(y), pan=-0.15, gain=db(-13), cue=("knot_tug", t))
    M.add("close", t + 0.35, burst(r, 0.05, 600, 2500, 0.001, 0.012), pan=-0.1, gain=db(-34))
    for i, t in enumerate(C["ear_flicks"]):
        M.add("close", t, flick(R("ear", i)), pan=0.2 + 0.2 * i, gain=db(-28), cue=("ear_flicks", t))
    t = C["tail_swish"]
    y = swish(r, 0.38, 350, 1500, 500)
    M.add("close", t, y, pan=np.linspace(-0.5, 0.6, y.size), gain=db(-23), cue=("tail_swish", t), align=True)
    M.add("close", 0.5 * (t + C["blinks"][0]) + 0.1, burst(r, 0.06, 300, 1500, 0.002, 0.014), pan=-0.2, gain=db(-33))
    for i, t in enumerate(C["blinks"]):
        M.add("close", t, blink(R("blink", i)), pan=0.1, gain=db(-34), cue=("blinks", t))
    t = C["pleased_breath"]
    M.add("close", t, breath(r, 0.75, 950, 600, 0.8), pan=0.1, gain=db(-27), cue=("pleased_breath", t), align=True)

    # ---------------- shot 2: the hill (wide)
    for i, t in enumerate(C["steps_back"]):
        M.add("foley", t, paw_grass(R("sb", i)), pan=-0.05 - 0.07 * i, gain=db(-25), cue=("steps_back", t))
    for i, t in enumerate(C["steps_to_map"]):
        M.add("foley", t, paw_grass(R("sm", i)), pan=-0.25 - 0.07 * i, gain=db(-25), cue=("steps_to_map", t))
    v0, v1 = C["village_lights"]
    r = R("village")
    penta = ["D7", "E7", "F#7", "A7", "B7", "D8"]
    for i in range(9):
        t = v0 + (v1 - v0) * i / 8.0 + (r.uniform(-0.15, 0.15) if 0 < i < 8 else 0.0)
        y = nrm(musicbox(penta[int(r.integers(0, 6))], 0.5, r))
        M.add("sky", t, y, pan=r.uniform(0.2, 0.85), gain=db(-41), cue=("village_lights", t) if i in (0, 8) else None)
    t = C["map_rustle"]
    M.add("foley", t, paper(R("map"), 0.42, dens=420), pan=-0.4, gain=db(-23), cue=("map_rustle", t), align=True)

    # ---------------- shot 3: night
    r = R("owl")

    def hoot(d):
        n_ = int(d * SR)
        x = np.linspace(0, 1, n_)
        f = float(m2f(N("F#4"))) * (1.0 + 0.02 * np.sin(np.pi * x) - 0.015 * x)
        ph = np.cumsum(TAU * f / SR)
        y_ = (np.sin(ph) + 0.15 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)) * np.sin(np.pi * x ** 0.7) ** 1.3
        return nrm(y_ + 0.05 * nrm(bp(white(n_, r), 300, 900, 1)) * hann(n_))

    tn = C["night_starts"]
    for t, d in ((tn + 0.3, 0.3), (tn + 0.72, 0.42), (C["ear_twitch_sleep"] + 0.5, 0.3), (C["ear_twitch_sleep"] + 0.9, 0.4)):
        M.add("sky", t, hoot(d), pan=-0.7, gain=db(-31))
    s0, s1 = C["shooting_star_night"]
    n = int((s1 - s0 + 0.7) * SR)
    tt = np.arange(n) / SR
    y = np.zeros(n)
    r = R("nstar")
    for nn_ in ("A6", "D7", "E7", "A7", "D8"):
        y += np.sin(TAU * float(m2f(N(nn_))) * tt + r.uniform(0, TAU)) * (0.5 + 0.5 * slow(n, 9, r)) / (1 + 0.3 * (N(nn_) - 93) / 12)
    env = np.minimum(tt / 0.06, 1.0) * np.exp(-np.maximum(tt - (s1 - s0) * 0.6, 0) / 0.35)
    y = nrm(y * env + 0.2 * nrm(bp(white(n, r), 4000, 7500, 1)) * env ** 2)
    M.add("sky", s0, y, pan=np.linspace(-0.7, 0.75, n), gain=db(-29), cue=("shooting_star_night", s0))
    t = C["ear_twitch_sleep"]
    M.add("foley", t, flick(R("twitch")), pan=0.1, gain=db(-30), cue=("ear_twitch_sleep", t))

    # ---------------- shot 4a: first light (mid wide)
    r = R("wake")
    t = C["wake"]
    M.add("foley", t, paper(r, 0.38, dens=260, lo=400, hi=3000), pan=-0.1, gain=db(-28), cue=("wake", t), align=True)
    t = C["yawn_stretch"]
    y = np.concatenate([breath(r, 0.42, 700, 1500, 0.9, 0.6), 0.9 * breath(r, 0.62, 1300, 600, 0.8, 0.2)])
    M.add("foley", t, nrm(y), pan=-0.1, gain=db(-24), cue=("yawn_stretch", t), align=True)
    for i, t in enumerate(C["climb_paws"]):
        f = 640.0 + 130.0 * i
        y = ring(r, 0.15, f * np.array([1, 1.58, 2.24, 2.92]), [0.03, 0.022, 0.016, 0.012], [1, 0.7, 0.5, 0.3])
        y[:int(0.06 * SR)] += 0.8 * nrm(lp(white(int(0.06 * SR), r), 400, 1) * perc(int(0.06 * SR), 0.002, 0.015))
        M.add("foley", t, nrm(y), pan=0.1, gain=db(-25), cue=("climb_paws", t))
    t = C["into_cup"]
    y = ring(r, 0.25, [1240, 2850, 4830], [0.035, 0.02, 0.01], [1, 0.5, 0.25])
    y[:int(0.12 * SR)] += 0.9 * tone(0.12, 120, 85, 0.035)
    ru = paper(r, 0.22, dens=300, lo=500, hi=3500)
    y[:ru.size] += 0.4 * ru
    M.add("foley", t, nrm(y), pan=0.12, gain=db(-21), cue=("into_cup", t))
    t = C["hatch_close"]
    y = ring(r, 0.2, [420, 1130, 1790, 2600], [0.03, 0.02, 0.014, 0.01], [1, 0.6, 0.4, 0.25])
    a = burst(r, 0.02, 1500, 4500, 0.0003, 0.004)
    y[:a.size] += 0.7 * a
    M.add("foley", t, nrm(y), pan=0.12, gain=db(-21), cue=("hatch_close", t))

    # ---------------- shot 4b: goggles (close)
    r = R("goggles")
    t = C["goggles_snap"]
    pre = 0.14
    y = np.zeros(int((pre + 0.2) * SR))
    st = creak(r, pre, 55, 150, [(1250, 9, 1.0), (2150, 9, 0.6)]) * np.linspace(0.3, 1, int(pre * SR))
    y[:st.size] += 0.22 * st
    i = int(pre * SR)
    sn = 1.0 * burst(r, 0.03, 1800, 6500, 0.0002, 0.005)
    y[i:i + sn.size] += sn
    th = ring(r, 0.18, [310, 620, 980], [0.035, 0.018, 0.01], [1, 0.5, 0.3])
    y[i:i + th.size] += 0.8 * th
    M.add("close", t - pre, nrm(y), pan=0.0, gain=db(-12), cue=("goggles_snap", t), mode="peak")
    t = C["swallow"]
    y = np.concatenate([tone(0.07, 190, 120, 0.03), np.zeros(int(0.025 * SR)), 0.8 * tone(0.1, 130, 270, 0.04, 0.01)])
    y = nrm(lp(y, 900, 1))
    ck_ = burst(r, 0.03, 700, 2600, 0.001, 0.008)       # a small wet click so the gulp reads on a phone speaker
    y[:ck_.size] += 0.3 * ck_
    y[int(0.095 * SR):int(0.095 * SR) + ck_.size] += 0.22 * ck_
    M.add("close", t, nrm(y), pan=0.0, gain=db(-20), cue=("swallow", t))
    t = C["nod"]
    M.add("close", t - 0.42, breath(r, 0.34, 800, 1700, 0.9, 0.7), pan=0.0, gain=db(-29))
    M.add("close", t, paper(r, 0.16, dens=250, lo=500, hi=2500), pan=0.0, gain=db(-32), cue=("nod", t), align=True)

    # ---------------- shot 5: the spark
    r = R("match")
    t = C["match_strike"]
    n = int(0.17 * SR)
    grain = 0.3 + 0.7 * nrm(np.abs(lp(white(n, r), 350, 1)))
    sc = nrm(bp(white(n, r), 2300, 7800, 1) * grain * np.linspace(0.5, 1, n) ** 2)
    M.add("close", t, fade(sc, 0.003, 0.01), pan=0.1, gain=db(-15), cue=("match_strike", t))
    n = int(0.7 * SR)
    x = np.linspace(0, 1, n)
    fl = svf(white(n, r), np.interp(x, [0, 0.12, 1], [500, 2600, 800]), 0.7, 1) * perc(n, 0.025, 0.16)
    fl = nrm(fl + 0.7 * nrm(lp(white(n, r), 380, 1) * perc(n, 0.02, 0.12)))
    M.add("close", t + 0.15, fl, pan=0.1, gain=db(-11))
    tb = C["fuse_lit"] + 0.5 - (t + 0.3)
    n = int(tb * SR)
    burn = nrm(lp(white(n, r), 900, 1) * (0.4 + 0.6 * slow(n, 14, r)) ** 2 * bed_env(n, 0.2, 0.4))
    M.add("close", t + 0.3, burn, pan=0.1, gain=db(-27))

    # the fuse: one fizz from fuse_lit to fuse_end, panned across the field in 5b, muffled in 5c
    r = R("fuse")
    f0, f1 = C["fuse_run"][0], C["fuse_end"]
    b0, b1 = S["s5b_fuse"]["start"], S["s5b_fuse"]["end"]
    n = int((f1 - f0) * SR)
    tt = f0 + np.arange(n) / SR
    near = np.clip((tt - b0) / (b1 - b0), 0, 1)
    fc = 3000.0 * (6800.0 / 3000.0) ** near
    hiss = svf(white(n, r), fc, 1.3, 1) * (0.55 + 0.45 * slow(n, 30, r))
    imp = (r.random(n) < 75.0 / SR) * r.exponential(1.0, n)
    fz = nrms(hiss) + 0.9 * nrms(bp(imp, 1500, 7500, 1))
    muff = lp(fz, 1500.0, 2) * 1.6
    w = np.clip((tt - b1) / 0.04, 0, 1)
    fz = fz * (1 - w) + muff * w
    lev = np.interp(tt, [f0, f0 + 0.006, b0, b0 + 0.02, b1, b1 + 0.05, f1 - 0.28, f1],
                    [0.0, 0.85, 0.5, 0.55, 1.0, 0.5, 0.45, 0.0])
    flick_ = np.where(tt > b1, 0.5 + 0.5 * slow(n, 12, r), 1.0)
    pan = np.interp(tt, [f0, b0, b0 + 0.001, b1, b1 + 0.001, f1], [-0.3, -0.8, -0.88, 0.75, 0.3, 0.3])
    fz = fz * lev * flick_
    fz *= 1.0 / np.max(np.abs(fz))
    M.add("foley", f0, fz, pan=pan, gain=db(-12), cue=("fuse_run", f0))
    M.add("close", C["fuse_lit"], nrm(hp(white(int(0.25 * SR), r), 3000, 1) * perc(int(0.25 * SR), 0.002, 0.07)),
          pan=-0.3, gain=db(-15), cue=("fuse_lit", C["fuse_lit"]))
    t = C["eyes_shut"]
    M.add("close", t, breath(r, 0.24, 900, 2300, 0.9, 0.75), pan=0.0, gain=db(-24), cue=("eyes_shut", t), align=True)
    M.add("close", t + 0.1, creak(r, 0.1, 50, 90, [(700, 8, 1.0), (1500, 8, 0.5)]), pan=0.0, gain=db(-32))
    t = C["fuse_end"]
    M.add("foley", t - 0.03, burst(r, 0.06, 1200, 5000, 0.001, 0.012), pan=0.3, gain=db(-22), cue=("fuse_end", t), mode="peak")

    # the stillness: one eyelid, bone dry, on its own bus
    t = C["one_eye_opens"]
    y = np.zeros(int(0.09 * SR))
    a = burst(R("lid"), 0.03, 1500, 4500, 0.001, 0.007)
    y[:a.size] += a
    y[int(0.045 * SR):int(0.045 * SR) + a.size] += 0.5 * burst(R("lid2"), 0.03, 1100, 3500, 0.001, 0.008)
    M.add("dry", t, y, pan=0.05, gain=db(-25), cue=("one_eye_opens", t))

    # ---------------- shot 6: ignition and lift off
    r = R("ign")
    ti, tlo = C["ignition"], C["liftoff"]
    tcs, tbr, tvi = C["clouds_start"], C["break_into_open_sky"], C["vista"]
    tp = S["s6_lift"]["start"]
    n = int((ti - 0.014 - tp) * SR)
    if n > 2000:
        x = np.linspace(0, 1, n)
        y = svf(white(n, r), 2000.0 * (7000.0 / 2000.0) ** x, 1.5, 1) * x ** 3
        M.add("foley", tp, nrm(fade(y, 0.01, 0.002)), pan=0.05, gain=db(-22))
    # the crack and the boom
    n = int(1.6 * SR)
    ck = nrm(hp(white(n, r), 180, 1) * perc(n, 0.0002, 0.04))
    ck += 0.9 * nrm(lp(white(n, r), 160, 2) * perc(n, 0.001, 0.14))
    ck += 1.0 * tone(1.6, 62, 31, 0.6, 0.004)
    crack = np.stack([nrm(ck), nrm(ck)])
    crack[1, 30:] = crack[1, :-30] * 0.97
    M.add("big", ti, crack, gain=db(-5.5), cue=("ignition", ti))
    # the roar: sub, body, tearing mids, crackle. Envelope from the cues.
    t_end = tvi + 0.4
    n = int((t_end - ti) * SR)
    tt = ti + np.arange(n) / SR
    env = np.interp(tt, [ti, ti + 0.01, ti + 0.3, tlo, tlo + 0.3], [0.0, 1.0, 0.72, 1.0, 1.0])
    after = tt > tlo + 0.3
    env[after] = 0.2 + 0.8 * np.exp(-(tt[after] - tlo - 0.3) / 1.5)
    env *= np.interp(tt, [tcs, tbr, tbr + 0.4, C["wonder"], t_end], [0.8, 0.5, 0.25, 0.2, 0.0])
    inside = np.clip((tt - tbr) / 0.4, 0, 1)
    sub = nrms(lp(white(n, r), 85.0, 2)) * (1.0 + 0.5 * np.sin(TAU * 11.0 * (tt - ti)) * np.clip((tt - ti) / (tlo - ti), 0, 1) * (tt < tlo + 0.4))
    roar = np.zeros((2, n))
    for c in range(2):
        body = nrms(bp(pink(n, r), 90, 700, 1)) * (0.55 + 0.45 * slow(n, 22, r))
        tear = nrms(bp(white(n, r), 700, 3200, 1)) * (0.25 + 0.75 * nrm(np.abs(lp(white(n, r), 45, 1))))
        imp = (r.random(n) < 160.0 / SR) * r.exponential(1.0, n)
        crk = nrms(bp(imp, 1000, 6000, 1))
        roar[c] = 0.7 * sub +(1.0 * body + (0.55 * tear + 0.4 * crk) * (1 - 0.85 * inside)) * (1 - 0.4 * inside)
    roar = nrms(roar) * env
    M.add("big", ti, roar, gain=db(-13.5))
    # rattling tin and fluttering paper through the shudder, then the fins in flight
    tin = [(1150, 30, 1.0), (1830, 30, 0.8), (2470, 25, 0.6), (3320, 25, 0.5), (4480, 20, 0.3)]
    y = creak(r, tlo - ti + 0.5, 22, 50, tin, 0.5)
    y *= np.interp(np.arange(y.size) / SR, [0, tlo - ti, tlo - ti + 0.5], [0.35, 1.0, 0.0])
    M.add("big", ti, y, pan=0.15, gain=db(-15), cue=("shudder", ti))
    n = int((tlo - ti + 0.9) * SR)
    k = np.arange(n) / SR
    am = (np.sin(TAU * np.cumsum(22.0 * (1 + 0.3 * (slow(n, 5, r) - 0.5))) / SR) > 0.1).astype(np.float64)
    am = lp(am, 300, 1)
    pf = nrm(bp(white(n, r), 900, 5200, 1) * am * np.interp(k, [0, tlo - ti, n / SR], [0.4, 1.0, 0.0]))
    M.add("big", ti, pf, pan=-0.2, gain=db(-17))
    y = creak(r, tbr - tlo, 30, 38, tin, 0.6)
    y *= (0.3 + 0.7 * slow(y.size, 2.5, r)) * bed_env(y.size, 0.5, 0.3)
    M.add("big", tlo, y, pan=0.1, gain=db(-25))
    n = int(1.7 * SR)
    x = np.linspace(0, 1, n)
    wh = nrm(svf(white(n, r), np.interp(x, [0, 0.3, 1], [150, 1100, 450]), 0.7, 1) * perc(n, 0.12, 0.5))
    M.add("big", tlo, wh, pan=0.0, gain=db(-12), cue=("liftoff", tlo), align=True)
    M.add("big", tlo, nrm(tone(0.9, 75, 40, 0.3, 0.005)), pan=0.0, gain=db(-12))
    t = C["passes_tree"]
    y = paper(r, 1.1, dens=950, lo=700, hi=6000, shape=perc(int(1.1 * SR), 0.06, 0.4))
    M.add("foley", t, y, pan=np.linspace(-0.2, -0.7, y.size), gain=db(-14), cue=("passes_tree", t), align=True)
    # the birds: every bird its own wing beat, path and start
    t = C["birds_burst"]
    r = R("birds")
    n = int(3.0 * SR)
    fl = np.zeros((2, n))
    for b in range(16):
        st = r.uniform(0.0, 0.5) if b > 2 else 0.0
        rate = r.uniform(9, 14)
        life = r.uniform(1.2, 2.4)
        p0, p1 = r.uniform(-0.6, 0.0), r.uniform(-1, 1)
        lo, hi = r.uniform(500, 900), r.uniform(2600, 4200)
        tw = st
        j = 0
        while tw < st + life and tw < 2.9:
            d = r.uniform(0.022, 0.034)
            a = burst(r, d + 0.02, lo, hi, 0.004, d / 2.5) * (1.0 if j % 2 == 0 else 0.5)
            x = (tw - st) / life
            g = (1 - x) ** 1.3 * r.uniform(0.6, 1.0)
            gl, gr = pan_gains(p0 + (p1 - p0) * x)
            i = int(tw * SR)
            fl[0, i:i + a.size] += a * g * gl
            fl[1, i:i + a.size] += a * g * gr
            tw += (0.5 / rate) * r.uniform(0.9, 1.1)
            j += 1
    M.add("foley", t, nrm(fl), gain=db(-12), cue=("birds_burst", t))
    for i in range(5):
        M.add("foley", t + 0.05 + r.uniform(0, 0.9), bird_call(r, 4 if i % 2 else 0), pan=r.uniform(-0.9, 0.5), gain=db(-20))
    t = C["into_first_clouds"]
    y = swish(r, 1.0, 200, 750, 300, 0.7)
    M.add("big", t, y, pan=0.0, gain=db(-17), cue=("into_first_clouds", t), align=True)

    # ---------------- shot 7: wind, then thin high air
    r = R("wind")
    n = int((tbr + 0.25 - (tcs - 0.4)) * SR)
    w = np.zeros((2, n))
    for c in range(2):
        fcw = 380.0 + 1100.0 * slow(n, 1.6, r)
        w[c] = svf(pink(n, r), fcw, 0.9, 1) * (0.5 + 0.5 * slow(n, 2.2, r)) + 0.2 * nrms(bp(white(n, r), 2500, 6000, 1)) * slow(n, 3, r) * 0.1
    M.add("big", tcs - 0.4, nrms(w) * bed_env(n, 0.5, 0.25), gain=db(-24))
    for i in range(3):
        t = tcs + (tbr - tcs) * (i + 0.45) / 3.2
        y = swish(R("wisp", i), 0.45, 500, 2600, 900, 1.1)
        sgn = 1 if i % 2 == 0 else -1
        M.add("big", t, y, pan=np.linspace(-0.9 * sgn, 0.9 * sgn, y.size), gain=db(-17))
    n = int(0.9 * SR)
    mist = nrm(hp(white(n, r), 1700, 1) * perc(n, 0.012, 0.2))
    M.add("big", tbr, np.stack([mist, nrm(hp(white(n, r), 1700, 1) * perc(n, 0.012, 0.2))]), gain=db(-13),
          cue=("break_into_open_sky", tbr))
    t_air_end = S["s8_star"]["end"] + 0.6
    n = int((t_air_end - tbr) * SR)
    air = np.zeros((2, n))
    for c in range(2):
        air[c] = svf(white(n, r), 4200.0 + 2200.0 * slow(n, 0.7, r), 2.2, 1) * (0.5 + 0.5 * slow(n, 0.9, r))
        air[c] += 0.5 * nrms(bp(pink(n, r), 500, 2500, 1)) * rms(air[c]) * (0.4 + 0.6 * slow(n, 0.5, r))
    M.add("amb", tbr, nrms(air) * bed_env(n, 0.3, 1.2), gain=db(-36))
    twf, two = C["wind_face"], C["wonder"]
    n = int((two + 0.5 - twf) * SR)
    k = twf + np.arange(n) / SR
    buf = np.zeros((2, n))
    ease = np.interp(k, [twf, twf + 0.05, two - 0.4, two + 0.5], [0.0, 1.0, 1.0, 0.0])
    for c in range(2):
        am = 0.35 + 0.65 * nrm(np.abs(np.sin(TAU * np.cumsum(19.0 * (1 + 0.5 * (slow(n, 6, r) - 0.5))) / SR)))
        buf[c] = nrms(bp(pink(n, r), 120, 900, 1)) * am + 0.5 * nrms(bp(white(n, r), 900, 3500, 1)) * am ** 3
    M.add("big", twf, nrms(buf) * ease, gain=db(-20), cue=("wind_face", twf))
    M.add("close", two, breath(R("gasp"), 0.32, 900, 2100, 0.9, 0.65), pan=0.0, gain=db(-25), cue=("wonder", two), align=True)

    # ---------------- shot 8: the star
    r = R("star")
    a0, a1 = C["star_arc"]
    ts = C["star_ignites"]
    y = swish(r, 1.3, 300, 2600, 1000, 0.8)
    M.add("sky", ts, y, pan=-0.6, gain=db(-15), cue=("star_ignites", ts), align=True)
    n = int((a1 - a0 + 1.0) * SR)
    k = np.arange(n) / SR
    x = np.clip(k / (a1 - a0), 0, 1)
    arc = 1.0 - (1.0 - x) ** 2.2
    sh = np.zeros(n)
    notes = ["A6", "D7", "F#7", "A7", "D8"]
    for i, nn_ in enumerate(notes):
        centre = (i + 0.5) / len(notes)
        wgt = np.exp(-((x - centre * 0.8) / 0.38) ** 2)
        f = float(m2f(N(nn_)))
        sh += wgt * (0.5 + 0.5 * np.sin(TAU * r.uniform(6, 11) * k + r.uniform(0, TAU))) * (
            np.sin(TAU * f * k) + 0.5 * np.sin(TAU * f * 1.003 * k + 1.0)) / (1.0 + 0.5 * i)
    sh = nrm(sh * np.minimum(k / 0.15, 1.0) * np.interp(k, [0, a1 - a0, n / SR], [1.0, 0.45, 0.0]))
    M.add("sky", a0, sh, pan=-0.8 + 1.6 * arc, gain=db(-25))
    for i in range(70):
        tg = (a1 - a0) * r.uniform(0, 1) ** 1.3
        xg = 1.0 - (1.0 - min(tg / (a1 - a0), 1.0)) ** 2.2
        f = r.uniform(3500, 7500)
        g = ring(r, 0.12, [f, f * 2.76], [r.uniform(0.012, 0.035), 0.008], [1, 0.2])
        M.add("sky", a0 + tg + r.uniform(0.05, 0.5), g, pan=np.clip(-0.8 + 1.6 * xg - r.uniform(0, 0.4), -1, 1), gain=db(r.uniform(-40, -31)))

    # ---------------- shot 9: home
    t9 = S["s9a_home"]["start"]
    tfp, tfb = C["fade_to_paper"], C["fade_to_black"]
    n = int((DUR - (t9 - 0.7)) * SR)
    k = t9 - 0.7 + np.arange(n) / SR
    e = bed_env(n, 1.3, 0.01) * np.interp(k, [0, tfp, tfb, DUR - 0.08, DUR], [1, 1, 0.3, 0.0, 0.0])
    M.add("amb", t9 - 0.7, breeze(n / SR, R("mbrz"), 260, 2400, 0.3, 0.3) * e, gain=db(-37))
    r = R("dawn")
    kinds = [0, 1, 2, 3, 4, 5, 3, 1]
    for i in range(30):
        t = t9 - 0.2 + (tfb - t9) * (i + r.uniform(0, 1)) / 30.0
        if abs(t - C["brush_stroke"]) < 0.5 and t < C["brush_stroke"] + 0.5:
            continue
        dist = r.uniform(0, 1)
        y = bird_call(r, kinds[int(r.integers(0, 8))])
        if dist > 0.5:
            y = lp(y, 4500.0, 1)
        g = db(-25 - 11 * dist) * float(np.interp(t, [0, tfp, tfb], [1, 1, 0.3]))
        M.add("sky", t, y, pan=r.uniform(-0.95, 0.95), gain=g)
    r = R("chute")
    t = C["parachute"]
    td = C["touchdown"]
    bamboo = [(380, 9, 1.0), (820, 10, 0.7), (1500, 9, 0.4)]
    for i in range(4):
        tc = t + 0.25 + (td - t - 0.5) * i / 4.0
        y = creak(R("ck", i), r.uniform(0.2, 0.32), r.uniform(18, 30), r.uniform(30, 55), bamboo)
        y *= hann(y.size) ** 0.6
        M.add("foley", tc, y, pan=0.25 * (-1) ** i, gain=db(-23))
    n = int((td - t + 0.2) * SR)
    k = np.arange(n) / SR
    swing = 0.45 + 0.55 * np.abs(np.sin(np.pi * k / ((td - t - 0.5) / 4.0) + 0.6))
    fl = paper(r, n / SR, dens=260, lo=900, hi=5500, shape=swing * bed_env(n, 0.03, 0.2))
    M.add("foley", t, fl, pan=0.25 * np.sin(np.pi * k / ((td - t - 0.5) / 4.0)), gain=db(-25), cue=("parachute", t), align=True)
    y = np.zeros(int(0.5 * SR))
    a = tone(0.3, 110, 68, 0.07, 0.004)
    y[:a.size] += a
    y += 0.6 * nrm(lp(white(y.size, r), 350, 1) * perc(y.size, 0.003, 0.05))
    M.add("foley", td, nrm(y), pan=0.15, gain=db(-12), cue=("touchdown", td))
    M.add("foley", td + 0.01, paper(r, 0.35, dens=600, lo=1400, hi=6000), pan=0.15, gain=db(-21))
    M.add("foley", td + 0.07, creak(r, 0.24, 40, 22, bamboo) * hann(int(0.24 * SR)) ** 0.5, pan=0.15, gain=db(-23))
    M.add("foley", td + 0.3, nrm(tone(0.2, 95, 70, 0.04)), pan=0.15, gain=db(-21))
    M.add("foley", td + 0.38, paper(r, 0.6, dens=380, lo=900, hi=5500), pan=0.3, gain=db(-23))
    r = R("out")
    t = C["hatch_open_home"]
    y = np.zeros(int(0.3 * SR))
    a = creak(r, 0.13, 45, 80, [(520, 9, 1.0), (1300, 10, 0.6)])
    y[:a.size] += 0.6 * a
    b = ring(r, 0.15, [420, 1130, 1790], [0.03, 0.02, 0.012], [1, 0.6, 0.4])
    y[a.size:a.size + b.size] += b
    M.add("foley", t, nrm(y), pan=0.1, gain=db(-22), cue=("hatch_open_home", t))
    t = C["climb_out"]
    for i in range(2):
        y = ring(R("tok", i), 0.2, [1240 + 90 * i, 2850, 4830], [0.03, 0.018, 0.01], [1, 0.45, 0.2])
        M.add("foley", t + 0.11 * i, y, pan=0.1, gain=db(-24), cue=("climb_out", t) if i == 0 else None)
    t = C["soot_shake"]
    n = int(0.45 * SR)
    k = np.arange(n) / SR
    y = nrm(bp(white(n, r), 700, 4000, 1) * np.abs(np.sin(TAU * 8.0 * k)) ** 1.5 * perc(n, 0.02, 0.2))
    M.add("foley", t, y, pan=0.1, gain=db(-22), cue=("soot_shake", t), align=True)
    for i, t in enumerate(C["paws_grass_home"]):
        M.add("foley", t, paw_grass(R("ph", i)), pan=-0.05 - 0.1 * i, gain=db(-21), cue=("paws_grass_home", t))
    t = C["looks_up"]
    M.add("foley", t, paper(r, 0.15, dens=200, lo=500, hi=2500), pan=-0.1, gain=db(-34), cue=("looks_up", t), align=True)
    t = C["brush_stroke"]
    r = R("brush")
    M.add("close", t, burst(r, 0.04, 400, 1800, 0.001, 0.008), pan=0.1, gain=db(-30))
    M.add("close", t, brush(r, 0.42), pan=np.linspace(0.05, 0.2, int(0.42 * SR)), gain=db(-23), cue=("brush_stroke", t), align=True)
    t = C["looks_at_us"]
    M.add("foley", t, flick(R("look")), pan=-0.1, gain=db(-32), cue=("looks_at_us", t))
    t = C["tail_curl"]
    y = swish(r, 0.42, 300, 950, 400)
    M.add("foley", t, y, pan=np.linspace(-0.3, 0.0, y.size), gain=db(-28), cue=("tail_curl", t), align=True)
    b0, b1 = C["title_brush"]
    lens = r.uniform(0.14, 0.3, 6)
    gap = (b1 - b0 - lens[-1]) / 5.0
    for i in range(6):
        ts_ = b0 + i * gap
        y = brush(R("tb", i), float(min(lens[i], gap * 0.95 if i < 5 else lens[i])))
        M.add("close", ts_, y, pan=np.linspace(-0.35 + 0.12 * i, -0.3 + 0.12 * i, y.size), gain=db(-21),
              cue=("title_brush", b0) if i == 0 else None)
    t = C["hanko"]
    y = ring(r, 0.3, [150, 330, 720, 1180], [0.05, 0.028, 0.013, 0.008], [1, 0.7, 0.4, 0.2])
    y[:int(0.08 * SR)] += 0.6 * nrm(lp(white(int(0.08 * SR), r), 1200, 1) * perc(int(0.08 * SR), 0.0005, 0.012))
    M.add("close", t, nrm(y), pan=0.3, gain=db(-13), cue=("hanko", t))
    b0, b1 = C["credit_brush"]
    for i in range(3):
        y = brush(R("cb", i), 0.16)
        M.add("close", b0 + i * (b1 - b0 - 0.16) / 2.0, y, pan=0.1 + 0.08 * i, gain=db(-25),
              cue=("credit_brush", b0) if i == 0 else None)
