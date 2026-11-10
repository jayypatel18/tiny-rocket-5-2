"""mixer.py: stems, sends, the two reverbs, the master chain, loudness and true peak
measurement, the 24 bit wav writer and the measurement tables."""
import os
import re
import struct
import subprocess
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d, uniform_filter1d
from synth import SR, Reverb, db, hp, lp, pan_gains, rms, k_release

# bus: (gain, hall send, outdoor send)
BUSES = {
    "piano": (1.0, 0.38, 0.0),
    "mbox": (1.0, 0.5, 0.0),
    "strings": (1.0, 0.42, 0.0),
    "low": (1.0, 0.22, 0.0),      # basses, cellos' roots, low drum: kept centred and drier
    "wind": (1.0, 0.45, 0.0),
    "brass": (1.0, 0.5, 0.0),
    "choir": (1.0, 0.65, 0.0),
    "perc": (1.0, 0.4, 0.0),
    "harp": (1.0, 0.4, 0.0),
    "tick": (1.0, 0.1, 0.1),
    "amb": (1.0, 0.0, 0.0),       # beds: already diffuse
    "foley": (1.0, 0.0, 0.3),     # spot effects out on the hill
    "close": (1.0, 0.0, 0.09),    # intimate close up sounds
    "big": (1.0, 0.0, 0.12),      # roar, wind
    "sky": (1.0, 0.3, 0.0),       # star shimmer, glitter
    "dry": (1.0, 0.0, 0.0),
}


class Stems:
    """A set of stereo buses covering [t0, t0 + dur)."""

    def __init__(self, t0, dur):
        self.t0 = float(t0)
        self.n = int(round(dur * SR))
        self.b = {}

    def bus(self, name):
        if name not in BUSES:
            raise KeyError(name)
        if name not in self.b:
            self.b[name] = np.zeros((2, self.n), dtype=np.float32)
        return self.b[name]

    def add(self, bus, t, sig, pan=0.0, gain=1.0):
        """Place a mono (panned, pan may be a curve) or stereo signal at absolute time t."""
        sig = np.asarray(sig)
        if sig.ndim == 1:
            gl, gr = pan_gains(np.asarray(pan, dtype=np.float64))
            sig = np.stack([sig * gl, sig * gr])
        i0 = int(round((t - self.t0) * SR))
        s0 = 0
        if i0 < 0:
            s0 = -i0
            i0 = 0
        m = min(sig.shape[1] - s0, self.n - i0)
        if m <= 0:
            return
        self.bus(bus)[:, i0:i0 + m] += (gain * sig[:, s0:s0 + m]).astype(np.float32)

    def total(self):
        out = np.zeros((2, self.n), dtype=np.float64)
        for k, v in self.b.items():
            out += v * BUSES[k][0]
        return out

    def window_rms(self, ta, tb):
        a = max(0, int((ta - self.t0) * SR))
        b = min(self.n, int((tb - self.t0) * SR))
        return rms(self.total()[:, a:b])


class Mixer:
    """Two stem sets split at the start of the stillness beat. Anything that starts before the cut
    is truncated at the cut and gets its own reverb pass, so no tail can leak into the silence."""

    def __init__(self, dur, t_cut):
        self.dur = dur
        self.t_cut = t_cut
        self.pre = Stems(0.0, t_cut)
        self.post = Stems(t_cut, dur - t_cut)
        self.log = []

    def _side(self, t):
        return self.pre if t < self.t_cut - 1e-9 else self.post

    def add(self, bus, t, sig, pan=0.0, gain=1.0, cue=None, mode="onset", align=False):
        """cue = (name, cue time). The landing time of the sound is logged against it.
        align=True starts a soft attack sound early so that its audible onset sits on t."""
        if align:
            a = np.abs(np.asarray(sig))
            a = a.max(axis=0) if a.ndim == 2 else a
            t = max(0.0, t - int(np.argmax(a >= 0.2 * a.max())) / SR)
        self._side(t).add(bus, t, sig, pan, gain)
        if cue is not None:
            self.mark(cue[0], cue[1], t, sig, mode)

    def mark(self, name, t_cue, t, sig, mode="onset"):
        a = np.abs(np.asarray(sig))
        if a.ndim == 2:
            a = a.max(axis=0)
        if mode == "peak":
            k = int(np.argmax(a))
        else:
            k = int(np.argmax(a >= 0.2 * a.max()))
        self.log.append((name, float(t_cue), t + k / SR))

    def section(self, t0, t1):
        return Stems(t0, t1 - t0)

    def merge(self, sec, target_db=None, window=None, gain=1.0):
        """Add a section. With target_db the whole section is scaled so that the sum of its buses
        has that RMS (dBFS) inside window."""
        if target_db is not None:
            gain = db(target_db) / sec.window_rms(*window)
        side = self._side(sec.t0)
        for k, v in sec.b.items():
            side.add(k, sec.t0, v, gain=gain)
        return gain

    def render(self):
        hall = Reverb(rt60=2.4, damp=0.38, size=1.0, pre=0.022, er=0.5)
        outd = Reverb(rt60=0.5, damp=0.6, size=0.55, pre=0.006, er=0.9)
        parts = []
        stems = {}
        for side in (self.pre, self.post):
            dry = np.zeros((2, side.n))
            hin = np.zeros((2, side.n))
            oin = np.zeros((2, side.n))
            for k, v in side.b.items():
                g, hs, os_ = BUSES[k]
                dry += g * v
                if hs:
                    hin += g * hs * v
                if os_:
                    oin += g * os_ * v
            mix = dry + hall(hin) + outd(oin)
            parts.append(master_filters(mix))
        for k in BUSES:
            a = self.pre.b.get(k)
            b = self.post.b.get(k)
            if a is None and b is None:
                continue
            a = a if a is not None else np.zeros((2, self.pre.n), np.float32)
            b = b if b is not None else np.zeros((2, self.post.n), np.float32)
            stems[k] = np.concatenate([a, b], axis=1) * BUSES[k][0]
        # the cut itself: 4 ms fade so the silence starts without a click
        k = int(0.004 * SR)
        parts[0][:, -k:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
        return np.concatenate(parts, axis=1), stems


def master_filters(x):
    """DC and rumble out, bass to the centre below 120 Hz, a gentle air roll off."""
    x = hp(x, 27.0, 2)
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1])
    side = hp(side, 125.0, 2)
    x = np.stack([mid + side, mid - side])
    return x - 0.2 * hp(x, 9000.0, 1)


# ---------------------------------------------------------------- loudness (ITU-R BS.1770)
_K1 = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
_K2 = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def kweight(x):
    return signal.lfilter(*_K2, signal.lfilter(*_K1, x))


def block_power(x, win=0.4, hop=0.1):
    k = kweight(x)
    p = np.sum(k * k, axis=0)
    c = np.concatenate([[0.0], np.cumsum(p)])
    w = int(win * SR)
    h = int(hop * SR)
    idx = np.arange(0, x.shape[1] - w + 1, h)
    return (c[idx + w] - c[idx]) / w


def lufs(x):
    z = block_power(x)
    l = -0.691 + 10 * np.log10(z + 1e-30)
    z1 = z[l > -70.0]
    if z1.size == 0:
        return -99.0
    rel = -0.691 + 10 * np.log10(np.mean(z1)) - 10.0
    z2 = z[(l > -70.0) & (l > rel)]
    return float(-0.691 + 10 * np.log10(np.mean(z2)))


def seg_lufs(x, ta, tb):
    k = kweight(x[:, int(ta * SR):int(tb * SR)])
    return float(-0.691 + 10 * np.log10(np.sum(np.mean(k * k, axis=1)) + 1e-30))


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=1)
    return float(20 * np.log10(max(np.max(np.abs(up)), np.max(np.abs(x))) + 1e-30))


# ---------------------------------------------------------------- dynamics
def lift_quiet(x, thr_db=-25.5, ratio=0.6, max_db=8.0):
    """Gentle upward compression of the quiet scenes. The detector looks a second ahead and behind
    and takes the loudest level it sees, so the gain is already down before anything loud arrives."""
    hopn = 480
    m = hp(0.5 * (x[0] + x[1]), 120.0, 1)
    p = uniform_filter1d(m * m, int(0.4 * SR))[::hopn]
    lev = 10 * np.log10(p + 1e-12)
    lev = maximum_filter1d(lev, 200)
    g = np.clip(ratio * (thr_db - lev), 0.0, max_db)
    g = gaussian_filter1d(g, 45.0)
    gain = np.interp(np.arange(x.shape[1]), np.arange(g.size) * hopn, g)
    return x * db(gain), g


def limit(x, ceil_db=-1.3, look=0.004, rel=0.09):
    """Look ahead true peak limiter: 4x oversampled peak detection, a held minimum, slow release."""
    up = signal.resample_poly(x, 4, 1, axis=1)
    pk = np.max(np.abs(up), axis=0).reshape(-1, 4).max(axis=1)
    pk = np.maximum(pk, np.max(np.abs(x), axis=0))
    c = db(ceil_db)
    # slow stage: rides sustained overs (the envelope of the peaks, averaged over 120 ms with
    # look ahead, slow recovery), so the fast stage below only has to catch single transients
    envp = maximum_filter1d(pk, int(0.012 * SR))
    over = np.maximum(20 * np.log10(np.maximum(envp, 1e-9) / c), 0.0)
    gs = db(-0.8 * uniform_filter1d(over, int(0.12 * SR)))
    gs = uniform_filter1d(k_release(gs, 1.0 - np.exp(-1.0 / (0.45 * SR))), int(0.03 * SR))
    x = x * gs
    pk = pk * gs
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = int(look * SR)
    g = minimum_filter1d(g, 2 * L + 1)
    g = k_release(g, 1.0 - np.exp(-1.0 / (rel * SR)))
    g = minimum_filter1d(g, L // 2 + 1)
    g = uniform_filter1d(g, L // 2 + 1)
    limit.stats = (20 * np.log10(gs), 20 * np.log10(g))
    return x * g


def master(x, target=-14.0, ceil_db=-1.3):
    x, upg = lift_quiet(x)
    g = db(target - lufs(x))
    y = x
    for _ in range(10):
        y = limit(x * g, ceil_db)
        err = target - lufs(y)
        if abs(err) < 0.02:
            break
        g *= db(1.3 * err)
    tp = true_peak_db(y)
    if tp > -1.05:
        y = y * db(-1.1 - tp)
    sl, fa = limit.stats
    print("  master: gain %+.2f dB. slow stage (gain ride): over 1 dB %.1f%% of the film, over 3 dB %.1f%%, deepest %.1f dB" % (
        20 * np.log10(g), 100 * np.mean(sl < -1.0), 100 * np.mean(sl < -3.0), float(sl.min())))
    print("          fast stage (peak limiter): over 1 dB %.2f%% of the film, over 3 dB %.2f%%, deepest %.1f dB" % (
        100 * np.mean(fa < -1.0), 100 * np.mean(fa < -3.0), float(fa.min())))
    master.stats = (sl, fa)
    return y, upg


# ---------------------------------------------------------------- files and tables
def write_wav24(path, x):
    a = np.clip(np.round(x.T * 8388607.0), -8388608, 8388607).astype("<i4")
    raw = np.ascontiguousarray(a).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    ch = x.shape[0]
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, SR, SR * ch * 3, ch * 3, 24))
        f.write(b"data" + struct.pack("<I", len(raw)))
        f.write(raw)


def ffmpeg_ebur128(path):
    p = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", path, "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    s = p.stderr[p.stderr.rfind("Summary:"):]
    out = {}
    for key, pat in (("I", r"I:\s+(-?[\d.]+) LUFS"), ("LRA", r"LRA:\s+(-?[\d.]+) LU"),
                     ("TP", r"Peak:\s+(-?[\d.]+) dBFS")):
        m = re.search(pat, s)
        out[key] = float(m.group(1)) if m else None
    return out


def spectrogram(path, png, ss=None, t=None, size="1776x640"):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss), "-t", str(t)]
    cmd += ["-i", path, "-lavfi",
            "showspectrumpic=s=%s:mode=combined:color=magma:scale=log:fscale=log:legend=1:drange=100" % size,
            "-frames:v", "1", png]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("spectrogram failed:", r.stderr[-300:])


def dbf(v):
    return 20 * np.log10(v + 1e-12)


def shot_table(x, shots):
    print("\nper shot (final master)   rms dBFS  peak dBFS   LUFS   <150Hz   150-8k    >8k   side   side<120Hz")
    lo120 = lp(x, 120.0, 4)
    lo = lp(x, 150.0, 4)
    hi = hp(x, 8000.0, 4)
    for s in shots:
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        seg = x[:, a:b]
        e = np.sum(seg * seg) + 1e-20
        el = np.sum(lo[:, a:b] ** 2) / e
        eh = np.sum(hi[:, a:b] ** 2) / e
        sd = np.sum((seg[0] - seg[1]) ** 2) / (np.sum((seg[0] + seg[1]) ** 2) + np.sum((seg[0] - seg[1]) ** 2) + 1e-20)
        b = lo120[:, a:b]
        sdl = np.sum((b[0] - b[1]) ** 2) / (np.sum((b[0] + b[1]) ** 2) + np.sum((b[0] - b[1]) ** 2) + 1e-20)
        print("  %-12s %5.1f-%5.1f  %7.1f   %7.1f  %6.1f   %5.1f%%   %5.1f%%  %5.1f%%  %4.0f%%  %6.1f%%" % (
            s["name"], s["start"], s["end"], dbf(rms(seg)), dbf(np.max(np.abs(seg))),
            seg_lufs(x, s["start"], s["end"]), 100 * el, 100 * max(0.0, 1 - el - eh), 100 * eh, 100 * sd, 100 * sdl))


def stem_table(stems, shots):
    names = list(stems.keys())
    print("\nstems before the master, rms dBFS per shot")
    print("  %-12s" % "shot" + "".join("%7s" % n[:6] for n in names))
    for s in shots:
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        row = ""
        for n in names:
            v = dbf(rms(stems[n][:, a:b].astype(np.float64)))
            row += "%7s" % ("." if v < -90 else "%.0f" % v)
        print("  %-12s" % s["name"] + row)
