"""synth.py: every instrument, every noise tool and the reverb of TINY ROCKET.
All sound is computed from sines and random numbers. No samples, no impulse files.
numpy, scipy and numba only."""
import zlib
import numpy as np
import numba as nb
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

SR = 48000
TAU = 2.0 * np.pi


# ---------------------------------------------------------------- small tools
def rng(*key):
    """Deterministic generator from any key."""
    return np.random.default_rng(zlib.crc32(repr(key).encode()) & 0xFFFFFFFF)


_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def N(name):
    """'F#5' to midi number (A4 = 69)."""
    if isinstance(name, (int, float, np.integer, np.floating)):
        return float(name)
    p = _NOTE[name[0]]
    i = 1
    while name[i] in "#b":
        p += 1 if name[i] == "#" else -1
        i += 1
    return float(12 * (int(name[i:]) + 1) + p)


def m2f(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def db(d):
    return 10.0 ** (d / 20.0)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-30))


def nrm(x, peak=1.0):
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > 0 else x


def nrms(x, target=1.0):
    return x * (target / rms(x))


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def white(n, r):
    return r.standard_normal(int(n))


def pink(n, r):
    n = int(n)
    X = np.fft.rfft(r.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    x = np.fft.irfft(X / np.sqrt(f), n)
    return x / np.std(x)


def hann(n):
    n = int(n)
    return 0.5 - 0.5 * np.cos(TAU * np.arange(n) / max(n - 1, 1))


def perc(n, att, dec):
    """Linear attack over att seconds, then exponential decay with time constant dec."""
    t = np.arange(int(n)) / SR
    return np.minimum(t / max(att, 1e-5), 1.0) * np.exp(-np.maximum(t - att, 0.0) / dec)


def fade(x, fin=0.004, fout=0.01):
    x = np.array(x, dtype=np.float64)
    n = x.shape[-1]
    a = min(int(fin * SR), n // 2)
    b = min(int(fout * SR), n // 2)
    if a > 0:
        x[..., :a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    if b > 0:
        x[..., n - b:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(b) / b)
    return x


def slow(n, rate, r):
    """Smooth random curve in 0..1 that changes about `rate` times a second."""
    n = int(n)
    k = int(n / SR * rate) + 4
    pts = r.random(k)
    xi = np.linspace(0, k - 1.001, n)
    i = xi.astype(np.int64)
    fr = 0.5 - 0.5 * np.cos(np.pi * (xi - i))
    return pts[i] * (1 - fr) + pts[i + 1] * fr


def sus_env(dur, att, rel, g0=1.0, g1=1.0, gm=None):
    """Envelope of a sustained note: eased attack, a dynamic shape (g0 to g1, or through gm), eased release."""
    n = max(8, int((dur + rel) * SR))
    nr = max(2, min(int(rel * SR), n - 4))
    ns = n - nr
    na = max(2, min(int(att * SR), ns))
    x = np.linspace(0.0, 1.0, ns)
    if gm is None:
        e = g0 + (g1 - g0) * x
    else:
        s = np.sin(np.pi * x) ** 1.2
        e = (g0 + (g1 - g0) * x) * (1 - s) + gm * s
    e[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    tail = e[-1] * (0.5 + 0.5 * np.cos(np.pi * np.arange(nr) / nr))
    return np.concatenate([e, tail])


def pan_gains(p):
    p = np.clip(p, -1.0, 1.0)
    a = (p + 1.0) * np.pi / 4.0
    return np.cos(a), np.sin(a)


# ---------------------------------------------------------------- numba kernels
@nb.njit(cache=True, fastmath=True)
def k_partials(out, f, a, tau, ph, sr):
    """Sum of exponentially decaying sines (recursive oscillators)."""
    n = out.shape[0]
    for p in range(f.shape[0]):
        if f[p] >= 0.45 * sr or a[p] == 0.0:
            continue
        w = TAU * f[p] / sr
        c = 2.0 * np.cos(w)
        y1 = np.sin(ph[p] - w)
        y2 = np.sin(ph[p] - 2.0 * w)
        d = np.exp(-1.0 / (tau[p] * sr))
        g = a[p]
        L = min(n, int(tau[p] * sr * 10.0) + 1)
        for i in range(L):
            y = c * y1 - y2
            y2 = y1
            y1 = y
            out[i] += g * y
            g *= d


@nb.njit(cache=True, fastmath=True)
def k_ensemble(outL, outR, f0, det, vrate, vdep, vdel, vph, ph0, gL, gR, drate, trate, trem,
               env, dmul, table, tdf, hw, hmax, sr):
    """A section of voices. Each voice is a harmonic series whose harmonic gains are read from a
    fixed body or formant table at the instantaneous frequency, so vibrato moves the harmonics
    through the resonances the way a real body or vocal tract does."""
    n = env.shape[0]
    nt = table.shape[0]
    vramp = 0.4 * sr
    for v in range(det.shape[0]):
        ph = ph0[v]
        fb = f0 * det[v]
        wr = TAU * vrate[v] / sr
        wd = TAU * drate[v] / sr
        wt = TAU * trate[v] / sr
        vd = vdel[v] * sr
        for i in range(n):
            x = (i - vd) / vramp
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0
            f = fb * (1.0 + vdep[v] * x * np.sin(wr * i + vph[v]) + 0.0018 * np.sin(wd * i + 1.7 * vph[v]))
            ph += TAU * f / sr
            if ph > TAU:
                ph -= TAU
            e = env[i]
            if e < 1e-5:
                continue
            if trem > 0.0:
                e *= (1.0 - trem) + trem * abs(np.sin(wt * i + 2.3 * vph[v]))
            sc = np.sin(ph)
            c2 = 2.0 * np.cos(ph)
            sp = 0.0
            acc = 0.0
            dh = 1.0
            dm = dmul[i]
            hm = int(0.42 * sr / f)
            if hm > hmax:
                hm = hmax
            fi = f / tdf
            for h in range(1, hm + 1):
                idx = int(h * fi)
                if idx >= nt:
                    break
                acc += sc * table[idx] * hw[h] * dh
                sn = c2 * sc - sp
                sp = sc
                sc = sn
                dh *= dm
            acc *= e
            outL[i] += acc * gL[v]
            outR[i] += acc * gR[v]


@nb.njit(cache=True, fastmath=True)
def k_svf(x, fc, q, mode, sr):
    """State variable filter with a cutoff that moves every sample. mode 0 low, 1 band, 2 high."""
    n = x.shape[0]
    y = np.empty(n)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for i in range(n):
        f = fc[i]
        if f > 0.45 * sr:
            f = 0.45 * sr
        g = np.tan(np.pi * f / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1 * k
        else:
            y[i] = x[i] - k * v1 - v2
    return y


@nb.njit(cache=True, fastmath=True)
def k_fdn(xL, xR, dl, fb, damp, apd, apg):
    """Feedback delay network: 4 allpass diffusers per channel into 8 delay lines mixed by a
    Hadamard matrix, a one pole damper in every line, two decorrelated outputs."""
    n = xL.shape[0]
    yL = np.zeros(n)
    yR = np.zeros(n)
    md = 0
    for j in range(8):
        if dl[j] > md:
            md = dl[j]
    buf = np.zeros((8, md + 1))
    pos = np.zeros(8, np.int64)
    lpz = np.zeros(8)
    v = np.zeros(8)
    K = apd.shape[1]
    ma = 0
    for c in range(2):
        for k in range(K):
            if apd[c, k] > ma:
                ma = apd[c, k]
    ab = np.zeros((2, K, ma + 1))
    ap = np.zeros((2, K), np.int64)
    for i in range(n):
        a = xL[i]
        b = xR[i]
        for k in range(K):
            z = ab[0, k, ap[0, k]]
            w = a - apg * z
            a = apg * w + z
            ab[0, k, ap[0, k]] = w
            ap[0, k] += 1
            if ap[0, k] >= apd[0, k]:
                ap[0, k] = 0
            z = ab[1, k, ap[1, k]]
            w = b - apg * z
            b = apg * w + z
            ab[1, k, ap[1, k]] = w
            ap[1, k] += 1
            if ap[1, k] >= apd[1, k]:
                ap[1, k] = 0
        for j in range(8):
            o = buf[j, pos[j]]
            lpz[j] = o * (1.0 - damp) + lpz[j] * damp
            v[j] = lpz[j] * fb[j]
        yL[i] = v[0] - v[1] + v[2] - v[3] + v[4] - v[5] + v[6] - v[7]
        yR[i] = v[0] + v[1] - v[2] - v[3] + v[4] + v[5] - v[6] - v[7]
        h = 1
        while h < 8:
            j = 0
            while j < 8:
                for k in range(j, j + h):
                    x0 = v[k]
                    x1 = v[k + h]
                    v[k] = x0 + x1
                    v[k + h] = x0 - x1
                j += 2 * h
            h *= 2
        for j in range(8):
            inp = a if (j & 1) == 0 else b
            buf[j, pos[j]] = v[j] * 0.35355339 + inp
            pos[j] += 1
            if pos[j] >= dl[j]:
                pos[j] = 0
    return yL, yR


@nb.njit(cache=True)
def k_release(g, k):
    """Gain may fall at once but recovers slowly (limiter release)."""
    out = np.empty_like(g)
    s = 1.0
    for i in range(g.shape[0]):
        s = s + (1.0 - s) * k
        if g[i] < s:
            s = g[i]
        out[i] = s
    return out


def svf(x, fc, q=1.0, mode=1):
    fc = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape).copy()
    return k_svf(np.ascontiguousarray(x, dtype=np.float64), fc, float(q), int(mode), float(SR))


# ---------------------------------------------------------------- the reverb
class Reverb:
    def __init__(self, rt60=2.3, damp=0.35, size=1.0, pre=0.018, er=0.6):
        base = np.array([1523, 1783, 1987, 2293, 2591, 2927, 3229, 3539], dtype=np.float64)
        self.dl = (np.maximum(64, base * size).astype(np.int64)) | 1
        self.fb = 10.0 ** (-3.0 * self.dl / (rt60 * SR))
        ap = np.array([[241, 373, 541, 709], [263, 397, 563, 733]], dtype=np.float64)
        self.apd = np.maximum(16, ap * min(1.0, size ** 0.5)).astype(np.int64)
        self.damp = float(damp)
        self.pre = int(pre * SR)
        self.er = er
        ms = np.array([[11.3, 17.9, 23.1, 29.7, 37.3, 43.9, 52.1, 61.7],
                       [13.1, 19.3, 25.9, 31.1, 39.7, 47.3, 55.7, 64.9]]) * size
        self.taps = (ms * SR / 1000.0).astype(np.int64)
        g = 0.82 ** np.arange(8)
        self.tg = g / np.sqrt(np.sum(g * g))
        imp = np.zeros(int(SR * min(rt60 * 1.6, 5.0)))
        imp[0] = 1.0
        l, r_ = k_fdn(imp, np.zeros_like(imp), self.dl, self.fb, self.damp, self.apd, 0.6)
        self.norm = 1.0 / np.sqrt(np.sum(l * l) + np.sum(r_ * r_))

    def __call__(self, x):
        x = hp(np.asarray(x, dtype=np.float64), 130.0, 1)
        n = x.shape[1]
        xp = np.zeros_like(x)
        if self.pre < n:
            xp[:, self.pre:] = x[:, :n - self.pre]
        l, r_ = k_fdn(np.ascontiguousarray(xp[0]), np.ascontiguousarray(xp[1]),
                      self.dl, self.fb, self.damp, self.apd, 0.6)
        out = np.stack([l, r_]) * self.norm
        for c in range(2):
            for k in range(8):
                d = int(self.taps[c, k])
                if d >= n:
                    continue
                src = c if k % 3 != 2 else 1 - c
                out[c, d:] += self.er * self.tg[k] * x[src, :n - d]
        return lp(out, 9500.0, 1)


# ---------------------------------------------------------------- struck and plucked
def piano(midi, vel, dur, r, felt=1.0, bright=1.0):
    """Felt piano. Stiff string partials (inharmonic), each with its own decay, a prompt sound and a
    slower aftersound from slightly detuned unison strings, a soft hammer thump, key and damper noise."""
    midi = N(midi)
    f0 = float(m2f(midi))
    B = 1.0e-4 + 2.5e-4 * 2.0 ** ((midi - 60.0) / 10.0)
    nmax = int(max(3, min(48, 9000.0 / f0)))
    n = np.arange(1, nmax + 1, dtype=np.float64)
    fn = n * f0 * np.sqrt(1.0 + B * n * n)
    fc = (650.0 + 2700.0 * vel ** 1.3) * bright * (1.0 + (midi - 60.0) / 40.0) / (0.6 + 0.4 * felt)
    fc = max(fc, 350.0)
    spec = n ** -0.7 * (0.25 + np.abs(np.sin(np.pi * n / 7.3))) * np.exp(-fn / fc)
    tau1 = min(4.5, 2.3 * 2.0 ** (-(midi - 60.0) / 18.0))
    taun = tau1 / (1.0 + 0.12 * (n - 1.0) ** 1.3)
    cents = r.uniform(0.3, 1.0, nmax) * r.choice([-1.0, 1.0], nmax)
    de = 2.0 ** (cents / 1200.0)
    f = np.concatenate([fn / de, fn * de])
    a = np.concatenate([spec * 0.75, spec * 0.33])
    tau = np.concatenate([taun * 0.55, taun * 1.9])
    ph = r.uniform(0, TAU, 2 * nmax)
    L = int(min(dur + 0.45, tau1 * 8.0 + 0.5) * SR)
    y = np.zeros(L)
    k_partials(y, f, a, tau, ph, float(SR))
    na = int((0.004 + 0.007 * felt * (1.0 - vel)) * SR)
    y[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
    y = nrm(y)
    nn = min(L, int(0.25 * SR))
    thump = bp(white(nn, r), 70.0, 380.0 + 500.0 * vel, 1) * perc(nn, 0.002, 0.03)
    key = bp(white(nn, r), 1700.0, 4200.0, 1) * perc(nn, 0.0004, 0.006)
    y[:nn] += nrm(thump) * 0.16 * felt + nrm(key) * 0.035 * felt
    t = np.arange(L) / SR
    y *= np.exp(-np.maximum(t - dur, 0.0) / 0.11)
    i = int(dur * SR)
    if i + 2400 < L:
        y[i:i + 2400] += nrm(bp(white(2400, r), 900.0, 2600.0, 1) * perc(2400, 0.004, 0.012)) * 0.012 * felt
    return nrm(y) * 0.3 * vel ** 1.5


def pluck(midi, vel, r, kind="harp"):
    """Plucked string: harp (long) or pizzicato (short, dark)."""
    midi = N(midi)
    f0 = float(m2f(midi))
    nmax = int(max(3, min(40, 7500.0 / f0)))
    n = np.arange(1, nmax + 1, dtype=np.float64)
    if kind == "harp":
        pos, fc = 0.21, 1300.0 + 2600.0 * vel
        tau1 = min(3.0, 1.7 * 2.0 ** (-(midi - 60.0) / 16.0))
        k = 0.22
    else:
        pos, fc = 0.13, 800.0 + 1500.0 * vel
        tau1 = 0.34 * 2.0 ** (-(midi - 48.0) / 26.0)
        k = 0.3
    fn = n * f0 * np.sqrt(1.0 + 1.2e-4 * n * n)
    a = np.abs(np.sin(np.pi * n * pos)) / n * np.exp(-fn / fc)
    tau = tau1 / (1.0 + k * (n - 1.0))
    L = int((tau1 * 7.0 + 0.2) * SR)
    y = np.zeros(L)
    k_partials(y, fn, a, tau, r.uniform(0, TAU, nmax), float(SR))
    y[:96] *= np.linspace(0, 1, 96)
    y = nrm(y)
    nn = min(L, 2400)
    y[:nn] += nrm(bp(white(nn, r), 500.0, 2600.0, 1) * perc(nn, 0.0005, 0.008)) * 0.09
    return nrm(y) * 0.3 * vel ** 1.5


def musicbox(midi, vel, r):
    """Music box: a clamped steel tooth (partials near 1, 6.27, 17.5) on a small resonant box."""
    f0 = float(m2f(N(midi)))
    sc = (880.0 / f0) ** 0.5
    f = f0 * np.array([1.0, 2.0, 3.01, 6.27, 12.4])
    a = np.array([1.0, 0.10, 0.04, 0.24, 0.05])
    tau = np.array([1.15, 0.7, 0.45, 0.11, 0.04]) * sc
    L = int(tau[0] * 8 * SR)
    y = np.zeros(L)
    k_partials(y, f, a, tau, r.uniform(0, TAU, 5), float(SR))
    y[:48] *= np.linspace(0, 1, 48)
    y = nrm(y)
    y[:960] += nrm(bp(white(960, r), 2500.0, 6500.0, 1) * perc(960, 0.0003, 0.003)) * 0.12
    return nrm(y) * 0.3 * vel ** 1.5


def _drum(f1, ratios, amps, taus, glide, gt, nlp, ndec, vel, r, dur):
    L = int(dur * SR)
    t = np.arange(L) / SR
    y = np.zeros(L)
    warp = t + glide * gt * (1.0 - np.exp(-t / gt))
    for q, a, ta in zip(ratios, amps, taus):
        y += a * np.exp(-t / ta) * np.sin(TAU * f1 * q * warp + r.uniform(0, 0.6))
    y = nrm(y)
    y += nrm(lp(white(L, r), nlp, 1) * perc(L, 0.001, ndec)) * (0.35 + 0.3 * vel)
    y[:48] *= np.linspace(0, 1, 48)
    return nrm(y)


def timpani(midi, vel, r):
    """Kettle drum: the near harmonic membrane modes (1, 1.5, 2, 2.44, 2.9), pitch settling, mallet noise."""
    f1 = float(m2f(N(midi)))
    y = _drum(f1, [0.5, 1.0, 1.504, 1.99, 2.44, 2.89], [0.25, 1.0, 0.7, 0.45, 0.28, 0.18],
              [0.12, 1.5, 1.1, 0.7, 0.45, 0.3], 0.03 * vel, 0.05, 700.0 + 1600.0 * vel, 0.02, vel, r, 2.6)
    return y * 0.3 * vel ** 1.5


def bassdrum(vel, r, f1=47.0):
    """Low drum: ideal membrane modes (1, 1.59, 2.14, 2.30, 2.65), a falling pitch, a soft noise thud."""
    y = _drum(f1, [1.0, 1.59, 2.14, 2.30, 2.65], [1.0, 0.55, 0.3, 0.25, 0.15],
              [0.5, 0.3, 0.2, 0.16, 0.1], 0.14, 0.04, 380.0, 0.035, vel, r, 1.6)
    return y * 0.3 * vel ** 1.5


def cymbal_swell(rise, fall, r):
    """Suspended cymbal roll: noise bands with independent slow shimmer plus about 70 inharmonic
    partials; the high bands arrive later in the swell. Returns stereo, peak near 1 at the crest."""
    n = int((rise + fall) * SR)
    t = np.arange(n) / SR
    env = np.where(t < rise, (t / rise) ** 2.6, np.exp(-(t - rise) / (fall / 4.5)))
    bands = [(350, 700, 0.5), (700, 1300, 0.8), (1300, 2400, 1.0), (2400, 4000, 1.0),
             (4000, 6500, 0.7), (6500, 9500, 0.35)]
    out = np.zeros((2, n))
    for c in range(2):
        for i, (lo, hi, w) in enumerate(bands):
            nz = bp(white(n, r), lo, hi, 1)
            out[c] += nrms(nz) * w * (0.55 + 0.45 * slow(n, 4.0, r)) * env ** (1.0 + 0.3 * i)
    met = np.zeros(n)
    for f in np.exp(r.uniform(np.log(450.0), np.log(7000.0), 70)):
        met += np.sin(TAU * f * t + r.uniform(0, TAU)) * (0.4 + 0.6 * slow(n, 2.5, r)) / np.sqrt(f / 450.0)
    met = nrms(met) * env ** 1.4 * 1.2
    out += met
    return fade(nrm(out), 0.01, 0.2)


# ---------------------------------------------------------------- sections of voices
NT = 3200
TDF = 5.0


def formant_table(peaks, base=0.3, lowcut=0.0, hicut=6000.0, hiord=2.0):
    f = np.arange(NT) * TDF
    g = np.full(NT, base, dtype=np.float64)
    for fc, bw, gain in peaks:
        g += gain / (1.0 + ((f - fc) / bw) ** 2)
    if lowcut > 0:
        x = (np.maximum(f, 1.0) / lowcut) ** 4
        g *= x / (1.0 + x)
    g *= 1.0 / (1.0 + (f / hicut) ** (2 * hiord))
    return g


def _body(s):
    """String body: air and wood resonances and the bridge hill, scaled by instrument size."""
    pk = [(275 * s, 35 * s, 1.2), (460 * s, 45 * s, 2.0), (555 * s, 45 * s, 1.5), (820 * s, 110 * s, 0.7),
          (1250 * s, 200 * s, 0.6), (2600 * s ** 0.7, 600, 1.5), (3500 * s ** 0.7, 700, 0.6)]
    return formant_table(pk, base=0.45, lowcut=165 * s, hicut=5000 * s ** 0.5)


_STR = dict(det=9.0, vdep=20.0, vrate=5.9, vdel=0.22, att=0.12, rel=0.25, tilt=0.85, dynk=0.10, dk2=0.16,
            hmax=30, noise=0.035, nband=(1800.0, 6000.0))
INST = {
    "vln1": dict(_STR, tab=_body(1.0), nv=7, pan=-0.45, spread=0.3),
    "vln2": dict(_STR, tab=_body(1.0), nv=6, pan=-0.12, spread=0.3),
    "vla": dict(_STR, tab=_body(0.72), nv=5, pan=0.2, spread=0.25, vrate=5.6),
    "vc": dict(_STR, tab=_body(0.42), nv=5, pan=0.4, spread=0.25, vrate=5.3, hmax=40, nband=(900.0, 4000.0)),
    "cb": dict(_STR, tab=_body(0.27), nv=3, pan=0.0, spread=0.12, vrate=5.0, vdep=12.0, hmax=44,
               att=0.16, nband=(500.0, 2500.0)),
    # horn: the spectrum opens with loudness (dynk, dk2), a broad resonance near 450 Hz
    "horn": dict(tab=formant_table([(430, 230, 3.0), (900, 300, 0.6)], base=0.6, hicut=2600, hiord=1.5),
                 nv=4, det=4.0, vdep=2.5, vrate=5.0, vdel=0.4, att=0.08, rel=0.2, tilt=0.35, dynk=0.35,
                 dk2=0.6, hmax=26, noise=0.006, nband=(600.0, 2500.0), pan=-0.25, spread=0.25),
    # choir: the vowel is the table. women "ah", women "oh", men "ah"
    "ch_wa": dict(tab=formant_table([(850, 75, 1.0), (1220, 85, 0.6), (2810, 120, 0.28), (3800, 160, 0.12)],
                                    base=0.012, hicut=5500),
                  nv=7, det=13.0, vdep=26.0, vrate=5.5, vdel=0.3, att=0.3, rel=0.45, tilt=1.55, dynk=0.08,
                  dk2=0.1, hmax=30, noise=0.03, nband=(700.0, 3500.0), pan=-0.2, spread=0.6),
    "ch_wo": dict(tab=formant_table([(470, 60, 1.0), (820, 80, 0.5), (2830, 120, 0.12), (3600, 160, 0.05)],
                                    base=0.01, hicut=5000),
                  nv=6, det=13.0, vdep=24.0, vrate=5.3, vdel=0.3, att=0.3, rel=0.45, tilt=1.5, dynk=0.08,
                  dk2=0.1, hmax=30, noise=0.025, nband=(500.0, 3000.0), pan=0.25, spread=0.6),
    "ch_m": dict(tab=formant_table([(650, 65, 1.0), (1080, 80, 0.55), (2650, 110, 0.3), (3300, 150, 0.12)],
                                   base=0.012, hicut=5000),
                 nv=7, det=11.0, vdep=22.0, vrate=5.1, vdel=0.3, att=0.3, rel=0.45, tilt=1.45, dynk=0.08,
                 dk2=0.1, hmax=36, noise=0.02, nband=(400.0, 3000.0), pan=0.1, spread=0.6),
}


def ens(name, midi, dur, vel, r, att=None, rel=None, g0=1.0, g1=1.0, gm=None, trem=0.0, vib=1.0,
        nv=None, bright=0.0, pan=None):
    """One note of a section. Returns stereo; the loud part of the note has RMS 0.1 * vel^1.4."""
    I = INST[name]
    att = I["att"] if att is None else att
    rel = I["rel"] if rel is None else rel
    env = sus_env(dur, att, rel, g0, g1, gm)
    env /= env.max()
    n = env.size
    nv = I["nv"] if nv is None else nv
    f0 = float(m2f(N(midi)))
    det = 2.0 ** (r.normal(0, I["det"], nv) / 1200.0)
    vrate = I["vrate"] * r.uniform(0.9, 1.1, nv)
    vdep = 2.0 ** (I["vdep"] * vib * r.uniform(0.6, 1.2, nv) / 1200.0) - 1.0
    vdel = I["vdel"] * r.uniform(0.6, 1.6, nv)
    pc = I["pan"] if pan is None else pan
    pans = pc + I["spread"] * r.permutation(np.linspace(-1, 1, nv))
    gL, gR = pan_gains(pans)
    dmul = np.minimum(np.exp(-(I["dynk"] * (1.0 - env) + I["dk2"] * (1.0 - vel) - bright)), 1.03)
    hmax = int(I["hmax"])
    hw = np.zeros(hmax + 1)
    hw[1:] = np.arange(1, hmax + 1, dtype=np.float64) ** -I["tilt"]
    L = np.zeros(n)
    R = np.zeros(n)
    k_ensemble(L, R, f0, det, vrate, vdep, vdel, r.uniform(0, TAU, nv), r.uniform(0, TAU, nv),
               np.ascontiguousarray(gL), np.ascontiguousarray(gR), r.uniform(0.2, 0.8, nv),
               r.uniform(10.5, 14.5, nv), float(trem), env, dmul, I["tab"], TDF, hw, hmax, float(SR))
    y = np.stack([L, R])
    target = 0.1 * vel ** 1.4
    y *= target / rms(y[:, env > 0.5])
    if I["noise"] > 0:
        lo, hi = I["nband"]
        g = I["noise"] * target * (1.0 + 1.5 * trem)
        cL, cR = pan_gains(pc)
        y[0] += nrms(bp(white(n, r), lo, hi, 1)) * env * g * cL * 1.4
        y[1] += nrms(bp(white(n, r), lo, hi, 1)) * env * g * cR * 1.4
    return y


def flute(notes, r, vib=1.0):
    """A breathy solo flute line. notes: (t, dur, midi, vel) in seconds from 0. Legato when notes touch.
    Weak upper harmonics, breath noise that follows the pitch, a small scoop on tongued notes, late vibrato."""
    end = max(t + d for t, d, m, v in notes) + 0.5
    n = int(end * SR)
    tt = np.arange(n) / SR
    pitch = np.full(n, N(notes[0][2]))
    amp = np.zeros(n)
    vibd = np.zeros(n)
    scoop = np.zeros(n)
    chiff = np.zeros(n)
    prev = -9.0
    for t, d, m, v in notes:
        i0 = int(t * SR)
        i1 = min(n, int((t + d) * SR))
        pitch[i0:] = N(m)
        amp[i0:i1] = v ** 1.2
        k = np.arange(i1 - i0)
        vibd[i0:i1] = np.clip((k - 0.22 * SR) / (0.3 * SR), 0, 1)
        if t - prev > 0.04:
            m_ = min(n - i0, int(0.3 * SR))
            scoop[i0:i0 + m_] -= 0.4 * np.exp(-np.arange(m_) / (0.04 * SR))
            chiff[i0:i0 + m_] += 2.5 * np.exp(-np.arange(m_) / (0.03 * SR))
        prev = t + d
    pitch = uniform_filter1d(pitch, int(0.035 * SR))
    amp = gaussian_filter1d(amp, 0.022 * SR)
    vibd = gaussian_filter1d(vibd, 0.03 * SR)
    lfo = np.sin(TAU * 5.1 * tt + 0.6 * np.sin(TAU * 0.31 * tt))
    f = m2f(pitch + scoop + 0.13 * vib * vibd * lfo + 0.03 * (slow(n, 3.0, r) - 0.5))
    ph = np.cumsum(TAU * f / SR)
    an = amp / max(amp.max(), 1e-9)
    y = np.zeros(n)
    for h, c in enumerate([1.0, 0.42, 0.2, 0.09, 0.045, 0.02], start=1):
        y += c * an ** (0.55 * (h - 1)) * np.sin(h * ph + r.uniform(0, TAU))
    y *= 1.0 + 0.09 * vibd * lfo
    br = nrms(lp(white(n, r), 280.0, 1)) * np.sin(ph) * 0.3 + nrms(lp(white(n, r), 400.0, 1)) * np.sin(2 * ph) * 0.16
    br += nrms(bp(white(n, r), 2200.0, 7500.0, 1)) * 0.035
    y = (y + br * (0.5 + chiff)) * amp
    return y * (0.1 / rms(y[an > 0.3]))
