"""music.py: the score of TINY ROCKET. Key of D major. Every time comes from cues.json.

The motif (three reaches, each one higher, the last one arrives):
    A4 B4 | A4 D5 | A4 E5 F#5        degrees 5 6, 5 1, 5 2 3
    s  L  | s  L  | s  s  LL         s = half a unit, L = one unit, LL = two units
"""
import numpy as np
from scipy import signal
from synth import (SR, N, m2f, db, rng, rms, nrm, piano, pluck, musicbox, timpani, bassdrum, cymbal_swell,
                   ens, flute, white, bp, perc, k_partials, TAU)

BUSMAP = {"vln1": "strings", "vln2": "strings", "vla": "strings", "vc": "strings", "cb": "low",
          "horn": "brass", "ch_wa": "choir", "ch_wo": "choir", "ch_m": "choir"}
MOTIF = [(0.0, "A4", 0.5), (0.5, "B4", 1.0), (1.5, "A4", 0.5), (2.0, "D5", 1.0),
         (3.0, "A4", 0.5), (3.5, "E5", 0.5), (4.0, "F#5", 2.2)]


def sympathetic(sec, level_db=-25.0):
    """Sympathetic resonance: the undamped strings of the D major chord ring along with the piano."""
    if "piano" not in sec.b:
        return
    x = sec.b["piano"].astype(np.float64)
    mono = x[0] + x[1]
    fs = []
    for nn in ("D2", "A2", "D3", "F#3", "A3", "D4"):
        for h in range(1, 7):
            f = float(m2f(N(nn))) * h
            if all(abs(f / g - 1.0) > 0.012 for g in fs):
                fs.append(f)
    out = np.zeros_like(x)
    for i, f in enumerate(fs):
        b, a = signal.iirpeak(f, 220.0, fs=SR)
        out[i % 2] += signal.lfilter(b, a, mono) / (1.0 + f / 900.0)
    out += 0.4 * out[::-1]
    g = db(level_db) * rms(x) / max(rms(out), 1e-12)
    sec.b["piano"] += (out * g).astype(np.float32)


def tick(r, hi):
    """The clock made from the fuse: a tiny struck spark, two pitches for tick and tock."""
    f = [2350.0, 3720.0, 5100.0] if hi else [1760.0, 2930.0, 4300.0]
    n = int(0.06 * SR)
    y = np.zeros(n)
    k_partials(y, np.array(f), np.array([1.0, 0.6, 0.3]), np.array([0.007, 0.005, 0.003]), r.uniform(0, TAU, 3), float(SR))
    y = nrm(y)
    c = bp(white(480, r), 3000.0, 8000.0, 1) * perc(480, 0.0002, 0.0015)
    y[:480] += 0.5 * nrm(c)
    return nrm(y)


def build(M, C, shots):
    S = {s["name"]: s for s in shots}
    R = lambda *k: rng("music", *k)
    h = R("human")

    def jit(s=0.007):
        return float(h.normal(0, s))

    def pn(sec, t, note, vel, dur, felt=1.0, bright=1.0, gain=1.0):
        m = N(note)
        y = piano(m, vel, dur, R("pn", m, round(t, 3)), felt, bright)
        sec.add("piano", t, y, pan=float(np.clip((m - 64.0) / 55.0, -0.45, 0.45)), gain=gain)
        return y

    def en(sec, name, t, note, dur, vel, **kw):
        m = N(note)
        y = ens(name, m, max(dur, 0.03), vel, R(name, m, round(t, 3)), **kw)
        sec.add(BUSMAP[name], t, y)
        return y

    def roll(sec, t0, t1, note, v0, v1, rate=14.0):
        r = R("roll", round(t0, 3))
        t = t0
        while t < t1:
            x = (t - t0) / max(t1 - t0, 1e-6)
            sec.add("perc", t, timpani(note, v0 + (v1 - v0) * x ** 1.5 + r.uniform(-0.03, 0.03), r), pan=0.2)
            t += (1.0 / rate) * r.uniform(0.93, 1.07)

    def harp(sec, t, note, vel, pan=-0.3, kind="harp"):
        m = N(note)
        sec.add("harp", t, pluck(m, vel, R("hp", m, round(t, 3)), kind), pan=pan)

    # ------------------------------------------------ shot 1 end and shot 2: the motif, lone felt piano
    t0 = C["motif_first"]
    u = (C["admire"] - t0) / 4.0
    tpb = C["pleased_breath"]
    sec = M.section(tpb - 0.1, S["s3_night"]["start"] + 4.0)
    en(sec, "vln1", tpb, "A5", t0 - tpb + 0.7, 0.22, nv=3, g0=0.15, g1=1.0, att=0.8, rel=1.8, vib=0.4, bright=-0.3)
    for b, nn, d in MOTIF:
        lng = d >= 1.0
        t = t0 + b * u + (0.0 if b in (0.0, 4.0) else jit())
        y = pn(sec, t, nn, (0.56 if lng else 0.46) + h.uniform(-0.03, 0.03), d * u + (2.4 if lng else 1.3))
        if b == 0.0:
            M.mark("motif_first", t0, t, y)
        if b == 4.0:
            M.mark("admire", C["admire"], t, y)
    LH = [(0.5, "D3", 0.36), (0.5, "A3", 0.30), (2.0, "G2", 0.36), (2.0, "D3", 0.30), (3.0, "A2", 0.32),
          (3.0, "E3", 0.27), (4.0, "B2", 0.36), (4.0, "F#3", 0.32), (5.0, "G2", 0.32), (5.0, "D3", 0.28)]
    for i, (b, nn, v) in enumerate(LH):
        pn(sec, t0 + b * u + 0.024 * (i % 2) + jit(0.004), nn, v, 2.8)
    ta = t0 + 6.5 * u
    ANS = [(0.0, "F#5", 0.5), (0.5, "E5", 1.0), (1.5, "D5", 0.5), (2.0, "B4", 1.0), (3.0, "A4", 0.5),
           (3.5, "D5", 0.5), (4.0, "E5", 2.5)]
    for b, nn, d in ANS:
        lng = d >= 1.0
        pn(sec, ta + b * u + jit(), nn, (0.52 if lng else 0.43) + h.uniform(-0.03, 0.03), d * u + (2.4 if lng else 1.3))
    LH2 = [(0.5, "A2", 0.34), (0.5, "E3", 0.30), (2.0, "G2", 0.34), (2.0, "D3", 0.30), (4.0, "A2", 0.34),
           (4.0, "E3", 0.30), (4.5, "C#4", 0.24)]
    for i, (b, nn, v) in enumerate(LH2):
        pn(sec, ta + b * u + 0.024 * (i % 2) + jit(0.004), nn, v, 2.8)
    sympathetic(sec)
    M.merge(sec, -29.0, (t0, S["s2_hill"]["end"]))

    # ------------------------------------------------ shot 3: music box over a muted string pad
    tn = C["night_starts"]
    s0, s1 = C["shooting_star_night"]
    tw = C["wake"]
    sec = M.section(tn - 0.2, tw + 3.5)
    for name, nn in (("vla", "B3"), ("vln2", "F#4"), ("vln2", "A4"), ("vln1", "D5")):      # B minor 9
        en(sec, name, tn + 0.3, nn, s1 - tn + 0.1, 0.3, att=1.6, rel=2.2, vib=0.6, bright=-0.3, g0=0.7, g1=0.8, gm=1.0)
    for name, nn in (("vla", "G3"), ("vla", "D4"), ("vln2", "F#4"), ("vln1", "B4")):       # G major 7
        en(sec, name, s1 - 0.5, nn, tw - s1 + 0.2, 0.3, att=1.6, rel=2.6, vib=0.6, bright=-0.3, g0=0.9, g1=0.6)
    um = 0.7
    t1 = tn + 1.2
    t2 = s1 + 0.45
    mb = [(t1, "A5", 0.55), (t1 + 0.5 * um, "B5", 0.7), (t1 + 1.5 * um, "A5", 0.55), (t1 + 2.0 * um, "D6", 0.75),
          (t2, "A5", 0.55), (t2 + 0.52 * um, "E6", 0.65), (t2 + 1.15 * um, "F#6", 0.8), (t2 + 1.15 * um + 0.025, "D6", 0.45)]
    for t, nn, v in mb:
        sec.add("mbox", t + jit(0.01), musicbox(nn, v, R("mb", nn, round(t, 2))), pan=0.15)
    M.merge(sec, -32.0, (tn + 1.0, S["s3_night"]["end"]))

    # ------------------------------------------------ shot 4: solo flute variation over a drone and pizzicato
    t4 = S["s4a_wake"]["start"]
    t5 = S["s5a_match"]["start"]
    sec = M.section(t4 - 0.1, t5 + 3.0)
    en(sec, "vc", t4 + 0.2, "D3", t5 - t4, 0.3, att=1.5, rel=1.3, vib=0.3, bright=-0.2)
    en(sec, "vla", t4 + 0.2, "A3", t5 - t4, 0.24, att=1.5, rel=1.3, vib=0.3, bright=-0.2)
    en(sec, "cb", t4 + 0.2, "D2", t5 - t4, 0.25, att=1.5, rel=1.3, vib=0.3)
    tf = C["yawn_stretch"] + 0.2
    uf = (C["into_cup"] - tf) / 4.0
    FL = [(0.0, 0.5, "D5", 0.6), (0.5, 1.0, "E5", 0.7), (1.5, 0.5, "D5", 0.6), (2.0, 1.0, "G5", 0.75),
          (3.0, 0.5, "D5", 0.6), (3.5, 0.5, "A5", 0.75), (4.0, 1.6, "B5", 0.85), (5.6, 0.4, "A5", 0.65), (6.0, 1.0, "F#5", 0.6)]
    y = flute([(b * uf, d * uf, nn, v) for b, d, nn, v in FL], R("flute"))
    sec.add("wind", tf, y, pan=-0.15)
    M.mark("into_cup (flute top)", C["into_cup"], tf, np.where(np.arange(y.size) > int(3.99 * uf * SR), 1.0, 0.0))
    for b, nn in ((0.5, "D3"), (2.0, "G3"), (2.0, "B3"), (4.0, "G2"), (4.0, "D3"), (5.6, "D3")):
        harp(sec, tf + b * uf + jit(0.004), nn, 0.5, pan=0.3, kind="pizz")
    harp(sec, C["swallow"] - 0.6, "D3", 0.42, pan=0.3, kind="pizz")
    tnod = C["nod"]
    harp(sec, tnod - 0.55, "A2", 0.42, pan=0.3, kind="pizz")
    y = flute([(0.0, 0.2, "A5", 0.6), (0.2, 0.85, "D6", 0.75)], R("flute2"))
    sec.add("wind", tnod - 0.02, 0.8 * y, pan=-0.15)
    for i, nn in enumerate(("D4", "F#4", "A4")):
        harp(sec, tnod + 0.07 * i, nn, 0.45, pan=-0.2)
    M.merge(sec, -30.0, (tf, t5))

    # ------------------------------------------------ shot 5: ticking, an ostinato climbing by half steps
    tl, te = C["fuse_lit"], C["fuse_end"]
    T = (te - tl) / 16.0
    tcut = C["stillness"][0]
    sec = M.section(t5 - 0.05, tcut + 0.5)
    kmin = int(np.ceil((t5 + 0.12 - tl) / T))
    r = R("tick")
    for k in range(kmin, 16):
        x = (k - kmin) / float(15 - kmin)
        y = tick(r, k % 2)
        sec.add("tick", tl + k * T, y, pan=0.15 * (1 if k % 2 else -1), gain=0.25 * db(-9 + 9 * x))
        if k == 0:
            M.mark("fuse_lit (tick)", tl, tl, y)
    steps = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4]
    for k in range(16):
        x = k / 15.0
        t = tl + k * T
        m = N("E3") + steps[k]
        en(sec, "vla", t, m, 0.09, 0.4 + 0.45 * x, att=0.012, rel=0.08, vib=0.0, nv=4, bright=-0.25)
        if k >= 4:
            en(sec, "vln2", t, m + 12, 0.09, 0.3 + 0.5 * x, att=0.012, rel=0.08, vib=0.0, nv=4, bright=-0.25)
        if k % 4 == 0 or k == 14:
            harp(sec, t, m - 12, 0.6 + 0.3 * x, pan=0.35, kind="pizz")
            sec.add("low", t, bassdrum(0.35 + 0.4 * x, R("hb", k)), gain=0.8)
    en(sec, "vln1", t5 + 0.3, "A5", te - t5 - 0.2, 0.36, nv=4, g0=0.12, g1=1.0, att=1.0, rel=0.1, trem=0.6, vib=0.3, bright=-0.1)
    en(sec, "vc", t5, "D3", tcut - t5, 0.4, g0=0.55, g1=1.0, att=0.7, rel=0.1, vib=0.3, bright=-0.2)
    en(sec, "cb", t5, "D2", tcut - t5, 0.35, g0=0.55, g1=1.0, att=0.7, rel=0.1, vib=0.3)
    M.merge(sec, -24.0, (tl, te))

    # ------------------------------------------------ shot 6: the orchestra enters under the roar
    ti, tlo, tt, tc, tcs = C["ignition"], C["liftoff"], C["passes_tree"], C["into_first_clouds"], C["clouds_start"]
    sec = M.section(ti - 0.02, tcs + 4.0)
    d = tlo - ti
    kw = dict(trem=0.75, g0=0.45, g1=1.0, att=0.12, rel=0.25)
    en(sec, "cb", ti, "B1", d, 0.7, **kw)
    en(sec, "vc", ti, "B2", d, 0.65, **kw)
    en(sec, "vc", ti, "F#3", d, 0.6, **kw)
    en(sec, "vla", ti, "D4", d, 0.5, **kw)
    en(sec, "vln2", ti, "F#4", d, 0.45, **kw)
    roll(sec, ti + 0.2, tlo - 0.05, "B2", 0.2, 0.75)
    y = timpani("E2", 1.0, R("tlo"))
    sec.add("perc", tlo, y, pan=0.2)
    M.mark("liftoff (timpani)", tlo, tlo, y)
    sec.add("low", tlo, bassdrum(1.0, R("bd", 0)))
    d = tt - tlo
    kw = dict(g0=0.7, g1=1.0, att=0.06, rel=0.2)
    en(sec, "cb", tlo, "E2", d, 0.8, **kw)
    for nn in ("B2", "E3"):
        en(sec, "vc", tlo, nn, d, 0.65, **kw)
    for nn in ("E3", "G3", "B3"):
        en(sec, "horn", tlo, nn, d, 0.6, g0=0.35, g1=1.0, att=0.15, rel=0.15)
    scale = ["B3", "C#4", "D4", "E4", "F#4", "G4", "A4", "B4", "C#5", "D5", "E5", "F#5", "G5"]
    tk = [tlo + 0.12 + (d - 0.12) * (k / 13.0) ** 0.8 for k in range(14)]
    for k, nn in enumerate(scale):
        dd = (tk[k + 1] - tk[k]) * 1.05
        v = 0.5 + 0.45 * k / 12.0
        en(sec, "vln1", tk[k], nn, dd, v, att=0.015, rel=0.06, vib=0.0, nv=5)
        en(sec, "vln2", tk[k], nn, dd, v, att=0.015, rel=0.06, vib=0.0, nv=5)
        en(sec, "vla", tk[k], N(nn) - 12, dd, v * 0.9, att=0.015, rel=0.06, vib=0.0, nv=4)
    uh = (tc - tt) / 5.0
    HM = [(0.0, "A3", 0.5), (0.5, "B3", 1.0), (1.5, "A3", 0.5), (2.0, "D4", 1.0), (3.0, "A3", 0.5), (3.5, "E4", 0.5), (4.0, "F#4", 2.6)]
    for b, nn, dd in HM:
        y = en(sec, "horn", tt + b * uh, nn, dd * uh * 0.96, 0.95, nv=5, bright=0.05, att=0.05, rel=0.12)
        en(sec, "vla", tt + b * uh, nn, dd * uh * 0.96, 0.7, att=0.04, rel=0.12)
        if b == 0.0:
            M.mark("passes_tree (horns)", tt, tt, y)
    CH = [(0.0, "F#2", ["D3", "A3"], ["A4", "D5"]), (2.0, "G2", ["D3", "B3"], ["G4", "B4"]),
          (3.0, "A2", ["E3", "A3"], ["A4", "C#5"]), (4.0, "B2", ["F#3", "B3"], ["B4", "D5"]),
          (5.0, "G2", ["D3", "G3"], ["G4", "B4"])]
    for i, (b, bass, mid, top) in enumerate(CH):
        t = tt + b * uh
        e = tt + CH[i + 1][0] * uh if i + 1 < len(CH) else tcs + 0.1
        en(sec, "cb", t, bass, e - t, 0.78, att=0.05, rel=0.2)
        for nn in mid:
            en(sec, "vc", t, nn, e - t, 0.6, att=0.06, rel=0.2)
        for nn in top:
            en(sec, "vln2", t, nn, e - t, 0.5, att=0.08, rel=0.2)
        sec.add("perc", t, timpani(bass if N(bass) >= N("F2") else N(bass) + 12, 0.8, R("tch", i)), pan=0.2)
    VL = [(0.0, "A5"), (2.0, "B5"), (3.0, "C#6"), (4.0, "D6")]
    for i, (b, nn) in enumerate(VL):
        t = tt + b * uh
        e = tt + VL[i + 1][0] * uh if i + 1 < len(VL) else tcs + 0.1
        en(sec, "vln1", t, nn, e - t, 0.75, att=0.07, rel=0.2, g0=0.85, g1=1.0)
    sec.add("low", tt, bassdrum(0.8, R("bd", 1)))
    sec.add("low", tc, bassdrum(0.8, R("bd", 2)))
    M.merge(sec, -14.5, (tlo, tcs))

    # ------------------------------------------------ shot 7: the swell, then the open sky
    tb, two, tv, ts8 = C["break_into_open_sky"], C["wonder"], C["vista"], C["star_ignites"]
    sec = M.section(tcs - 0.05, ts8 + 5.0)
    q = (tb - tcs) / 4.0
    PR = [("E2", ["E3", "G3", "B3", "D4"]), ("F#2", ["F#3", "A3", "C#4", "E4"]),
          ("G2", ["G3", "B3", "D4", "A4"]), ("A2", ["A3", "C#4", "E4", "A4"])]
    for i, (bass, tones) in enumerate(PR):
        t = tcs + i * q
        x = i / 3.0
        en(sec, "cb", t, bass, q, 0.7 + 0.2 * x, att=0.06, rel=0.2)
        en(sec, "vc", t, N(bass) + 12, q, 0.6 + 0.2 * x, att=0.06, rel=0.2)
        for nn in tones[:3]:
            en(sec, "horn", t, nn, q, 0.55 + 0.3 * x, att=0.08, rel=0.15)
        seq = [N(nn) for nn in tones] + [N(nn) + 12 for nn in tones]
        for k, m in enumerate(seq):
            harp(sec, t + k * q / 8.0 + jit(0.004), m, 0.5 + 0.2 * x)
    sec.add("perc", tcs, timpani("E2", 0.85, R("t7")), pan=0.2)
    sec.add("low", tcs, bassdrum(0.9, R("bd", 3)))
    MEL = [(0.0, 0.5, "B4"), (0.5, 0.5, "D5"), (1.0, 0.5, "C#5"), (1.5, 0.5, "E5"), (2.0, 0.5, "D5"), (2.5, 0.5, "G5"), (3.0, 1.0, "A5")]
    for b, dd, nn in MEL:
        for name, off, v in (("vln1", 12, 0.8), ("vln2", 0, 0.75), ("vla", -12, 0.65)):
            y = en(sec, name, tcs + b * q, N(nn) + off, dd * q * 1.0, v * (0.8 + 0.2 * b / 3.0), att=0.05, rel=0.14)
        if b == 0.0:
            M.mark("clouds_start (strings)", tcs, tcs, y)
    for name, nn in (("ch_wa", "D5"), ("ch_wa", "B4"), ("ch_wo", "G4"), ("ch_m", "G3"), ("ch_m", "D4")):
        en(sec, name, tcs + 2 * q, nn, q, 0.5, g0=0.3, g1=1.0, att=0.3, rel=0.35)
    for name, nn in (("ch_wa", "E5"), ("ch_wa", "C#5"), ("ch_wo", "A4"), ("ch_m", "A3"), ("ch_m", "E4")):
        en(sec, name, tcs + 3 * q, nn, q, 0.7, g0=0.7, g1=1.0, att=0.2, rel=0.5)
    roll(sec, tcs + 3 * q, tb - 0.03, "A2", 0.3, 0.9)
    sec.add("perc", tcs + 2 * q, cymbal_swell(tb - (tcs + 2 * q), 2.5, R("cym", 1)), gain=0.26)
    # open sky: B flat major 7, the high A enters and stays to the climax
    for nn, v in (("A6", 0.34), ("A5", 0.42)):
        y = en(sec, "vln1", tb, nn, ts8 - tb, v, nv=4, att=0.35, rel=0.3, g0=0.7, g1=1.0, vib=0.8)
    M.mark("break_into_open_sky (high A)", tb, tb, np.ones(8))
    d = two - tb
    kw = dict(att=0.45, rel=0.9, g0=0.8, g1=0.8, gm=1.0)
    for name, nn, v in (("vln2", "D5", 0.38), ("vln2", "F5", 0.38), ("vla", "Bb3", 0.36), ("vla", "F4", 0.36), ("vc", "Bb2", 0.36),
                        ("ch_wa", "D5", 0.45), ("ch_wa", "A4", 0.45), ("ch_wo", "F4", 0.4), ("ch_m", "Bb3", 0.4), ("ch_m", "F3", 0.4)):
        en(sec, name, tb, nn, d, v, **kw)
    for k, nn in enumerate(("Bb3", "F4", "A4", "D5", "F5", "A5")):
        harp(sec, tb + 0.2 + 0.3 * k, nn, 0.45)
    # wonder: C major with the sixth and ninth, a single horn reaches the fourth
    d = tv - two
    for name, nn, v in (("vln2", "D5", 0.4), ("vln2", "E5", 0.4), ("vla", "E4", 0.4), ("vla", "G4", 0.4), ("vc", "C3", 0.4),
                        ("vc", "G3", 0.4), ("ch_wa", "D5", 0.45), ("ch_wa", "G4", 0.45), ("ch_wo", "E4", 0.42),
                        ("ch_m", "C3", 0.42), ("ch_m", "G3", 0.42)):
        en(sec, name, two - 0.1, nn, d, v, **kw)
    y = en(sec, "horn", two, "A4", 0.45, 0.55, nv=1, att=0.06, rel=0.1, pan=-0.1)
    M.mark("wonder (horn)", two, two, y)
    en(sec, "horn", two + 0.45, "D5", d - 0.6, 0.62, nv=1, att=0.06, rel=0.4, pan=-0.1, gm=1.0, g0=0.8, g1=0.6)
    for k, nn in enumerate(("C3", "G3", "D4", "E4", "G4", "D5", "E5", "G5")):
        harp(sec, two + 0.18 * k, nn, 0.5)
    # vista: C over D, everything leans on the dominant
    d = ts8 - tv
    kw = dict(att=0.3, rel=0.25, g0=0.5, g1=1.0)
    for name, nn, v in (("cb", "D2", 0.55), ("vc", "D3", 0.5), ("vc", "A3", 0.5), ("vla", "C4", 0.5), ("vla", "E4", 0.5),
                        ("vla", "G4", 0.45), ("vln2", "E5", 0.5), ("vln2", "G5", 0.5), ("ch_wa", "E5", 0.55),
                        ("ch_wa", "C5", 0.5), ("ch_wo", "G4", 0.5), ("ch_m", "D3", 0.5), ("ch_m", "A3", 0.5)):
        en(sec, name, tv - 0.05, nn, d, v, **kw)
    y = en(sec, "horn", tv, "A3", 0.3, 0.7, att=0.05, rel=0.08)
    M.mark("vista (horns)", tv, tv, y)
    en(sec, "horn", tv + 0.3, "E4", d - 0.3, 0.72, att=0.05, rel=0.2, g0=0.6, g1=1.0)
    roll(sec, tv + 0.15, ts8 - 0.03, "A2", 0.25, 0.7)
    # one breath before the downbeat: the whole build drops away in the last tenth of a second
    tt7 = sec.t0 + np.arange(sec.n) / SR
    duck = np.interp(tt7, [ts8 - 0.16, ts8 - 0.04, ts8 + 0.25], [1.0, 0.35, 0.0]).astype(np.float32)
    for k_ in sec.b:
        sec.b[k_] *= duck
    g7 = M.merge(sec, -17.0, (tcs, tb))
    gl = [N(a) + 12 * o for o in range(3) for a in ("D4", "E4", "F#4", "A4", "B4")][:13]

    # ------------------------------------------------ shot 8: the motif at full scale, D major at last
    a1 = C["star_arc"][1]
    uc = (a1 - ts8) / 9.5
    t8e = S["s8_star"]["end"]
    sec = M.section(tv - 0.02, t8e + 5.0)
    # the cymbal roll and the harp run that lead in belong to this section so they ring through the downbeat
    sec.add("perc", tv + 0.1, cymbal_swell(ts8 - tv - 0.1, 3.2, R("cym", 2)), gain=0.2)
    for k, m in enumerate(gl):
        harp(sec, ts8 - 0.55 + 0.52 * k / 13.0, m, 0.5 + 0.2 * k / 13.0)
    MEL8 = [(0.0, 0.5, "A4"), (0.5, 1.0, "B4"), (1.5, 0.5, "A4"), (2.0, 1.0, "D5"), (3.0, 0.5, "A4"), (3.5, 0.5, "E5"),
            (4.0, 2.0, "F#5"), (6.0, 1.0, "A5"), (7.0, 1.0, "B5"), (8.0, None, "A5")]
    for b, dd, nn in MEL8:
        t = ts8 + b * uc
        last = dd is None
        dur = (t8e - t + 0.3) if last else dd * uc * 1.03
        kw = dict(g0=1.0, g1=0.12, att=0.06, rel=0.9) if last else dict(att=0.05, rel=0.16)
        y = en(sec, "vln1", t, N(nn) + 12, dur, 1.0, **kw)
        en(sec, "vln2", t, nn, dur, 0.9, **kw)
        kw2 = dict(kw, att=0.1)
        en(sec, "ch_wa", t, nn, dur, 0.85, **kw2)
        if b < 6.0:
            en(sec, "vla", t, N(nn) - 12, dur, 0.75, **kw)
            en(sec, "horn", t, N(nn) - 12, dur * 0.97, 0.9, nv=5, att=0.05, rel=0.14)
        if not last:
            pn(sec, t, nn, 0.95, dd * uc + 1.6, felt=0.3, bright=1.5, gain=1.5)
            p = pn(sec, t + 0.004, N(nn) + 12, 0.9, dd * uc + 1.6, felt=0.3, bright=1.5, gain=1.5)
        if b == 0.0:
            M.mark("star_ignites (tutti)", ts8, t, p)
    CH8 = [(0.0, 1.5, "D"), (1.5, 3.0, "G"), (3.0, 4.0, "A"), (4.0, 6.0, "D"), (6.0, 8.0, "G"), (8.0, None, "D")]
    VO = {"D": ("D2", ["D3", "A3"], ["F#4", "A4"], ["D3", "F#3", "A3"], ["F#3", "A3", "D4"], ["D3", "A3", "D4", "F#4", "A4", "D5"]),
          "G": ("G2", ["D3", "G3"], ["G4", "B4"], ["G3", "B3", "D3"], ["G3", "B3", "D4"], ["G3", "B3", "D4", "G4", "B4", "D5"]),
          "A": ("A2", ["E3", "A3"], ["E4", "A4"], ["A3", "C#4", "E3"], ["A3", "C#4", "E4"], ["A3", "C#4", "E4", "A4", "C#5", "E5"])}
    for i, (b0, b1, ch) in enumerate(CH8):
        t = ts8 + b0 * uc
        last = b1 is None
        dur = (t8e - t + 0.3) if last else (b1 - b0) * uc
        kw = dict(g0=0.9, g1=0.12, att=0.08, rel=0.9) if last else dict(att=0.06, rel=0.2)
        bass, cel, wom, men, hrn, arp = VO[ch]
        v = 0.55 if last else 0.8
        en(sec, "cb", t, bass, dur, v, **kw)
        for nn in cel:
            en(sec, "vc", t, nn, dur, v * 0.85, **kw)
        for nn in wom:
            en(sec, "ch_wo", t, nn, dur, v * 0.8, **dict(kw, att=0.12))
        for nn in men:
            en(sec, "ch_m", t, nn, dur, v * 0.8, **dict(kw, att=0.12))
        if b0 >= 6.0:
            for nn in hrn:
                en(sec, "horn", t, nn, dur, 0.6 if not last else 0.4, **kw)
        if not last:
            pn(sec, t, bass, 0.9, dur + 1.5, felt=0.3, bright=1.4)
            pn(sec, t + 0.006, N(bass) + 12, 0.85, dur + 1.5, felt=0.3, bright=1.4)
            nar = int(round((b1 - b0) * 2))
            for k in range(1, nar):
                pn(sec, t + k * 0.5 * uc + jit(0.005), arp[(k - 1) % len(arp)], 0.5, 1.6, felt=0.5, bright=1.2)
            sec.add("perc", t, timpani({"D": "D3", "G": "G2", "A": "A2"}[ch], 1.0 if b0 in (0.0, 4.0) else 0.8, R("t8", i)), pan=0.2, gain=1.7)
        else:
            sec.add("perc", t, timpani("D3", 0.55, R("t8", i)), pan=0.2)
    for b in (0.0, 4.0):
        sec.add("low", ts8 + b * uc, bassdrum(1.0, R("bd8", b)))
    roll(sec, ts8 + 3.35 * uc, ts8 + 4.0 * uc - 0.03, "A2", 0.4, 0.9)
    sec.add("perc", ts8 + 3.0 * uc, cymbal_swell(uc, 3.5, R("cym", 3)), gain=0.3)
    for k, m in enumerate(gl):
        harp(sec, ts8 + 4.0 * uc - 0.5 + 0.47 * k / 13.0, m, 0.5 + 0.2 * k / 13.0)
    for k, nn in enumerate(("D5", "A5", "D6", "F#6", "A6")):
        pn(sec, ts8 + (8.4 + 0.45 * k) * uc + jit(), nn, 0.42 - 0.03 * k, 3.0)
    sympathetic(sec, -28.0)
    M.merge(sec, -10.5, (ts8, ts8 + 6.0 * uc))

    # ------------------------------------------------ shot 9: the motif once more, slower, and the one note
    tp = C["climb_out"]
    tbs = C["brush_stroke"]
    u9 = (tbs - 1.25 - tp) / 3.5
    sec = M.section(tp - 0.05, tbs + 6.5)
    for b, nn, dd in MOTIF[:6]:
        pn(sec, tp + b * u9 + (0.0 if b == 0.0 else jit()), nn, 0.5 if dd >= 1.0 else 0.42, dd * u9 + 2.2)
    for i, (b, nn, v) in enumerate([(0.5, "D3", 0.3), (2.0, "G2", 0.3), (2.0, "D3", 0.26), (3.5, "A2", 0.3), (3.5, "E3", 0.26)]):
        pn(sec, tp + b * u9 + 0.024 * (i % 2), nn, v, 3.0)
    y = pn(sec, tbs, "F#5", 0.52, 5.5)
    M.mark("brush_stroke (the one piano note)", tbs, tbs, y)
    sympathetic(sec)
    M.merge(sec, -30.0, (tp, tbs + 1.5))

    # the last chord: strings only, G with the ninth falling home to D, gone before the fade to black
    ttc = C["tail_curl"]
    cb0 = C["credit_brush"][0]
    tfb = C["fade_to_black"]
    sec = M.section(ttc - 0.05, tfb + 0.6)
    for name, nn in (("vc", "G3"), ("vla", "D4"), ("vln2", "A4"), ("vln1", "B4")):
        en(sec, name, ttc + 0.1, nn, cb0 - ttc, 0.3, att=1.4, rel=1.2, vib=0.6, bright=-0.25, g0=0.6, g1=1.0)
    for name, nn in (("vc", "D3"), ("vc", "A3"), ("vla", "F#4"), ("vln2", "A4"), ("vln1", "D5")):
        en(sec, name, cb0 - 0.1, nn, tfb - 1.25 - (cb0 - 0.1), 0.3, att=0.9, rel=1.2, vib=0.6, bright=-0.25, g0=1.0, g1=0.5)
    M.merge(sec, -33.0, (C["title_brush"][0], C["fade_to_paper"]))
