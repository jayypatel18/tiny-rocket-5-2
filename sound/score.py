"""score.py: renders the whole soundtrack of TINY ROCKET from cues.json.

    python3 sound/score.py            full render, measurements, spectrograms
    python3 sound/score.py --quick    skip the spectrogram images

Writes build/audio/tiny-rocket-5-2.wav (48 kHz, stereo, 24 bit, exactly the film length)."""
import json
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import foley  # noqa: E402
import music  # noqa: E402
import mixer  # noqa: E402
from synth import SR  # noqa: E402

OUT = os.path.join(ROOT, "build", "audio", "tiny-rocket-5-2.wav")
REVIEW = os.path.join(ROOT, "build", "review")


def main():
    t_start = time.time()
    with open(os.path.join(ROOT, "cues.json")) as f:
        J = json.load(f)
    C, shots, dur = J["cues"], J["shots"], float(J["duration"])
    frame = 1.0 / J["fps"]
    n_total = int(round(dur * SR))
    M = mixer.Mixer(dur, C["stillness"][0])
    foley.build(M, C, shots)
    print("foley built   %.1f s" % (time.time() - t_start))
    music.build(M, C, shots)
    print("music built   %.1f s" % (time.time() - t_start))
    mix, stems = M.render()
    print("mixed         %.1f s" % (time.time() - t_start))
    y, upg = mixer.master(mix)
    y = y[:, :n_total]
    if y.shape[1] < n_total:
        y = np.pad(y, ((0, 0), (0, n_total - y.shape[1])))
    # the stillness is gated to digital zero; only the eyelid passes (100 ms, faded)
    i0, i1 = int(C["stillness"][0] * SR), int(C["stillness"][1] * SR)
    ie = int(C["one_eye_opens"] * SR) - i0
    gate = np.zeros(i1 - i0)
    gate[ie:ie + int(0.1 * SR)] = 1.0
    gate[ie + int(0.1 * SR) - 480:ie + int(0.1 * SR)] *= np.linspace(1, 0, 480)
    y[:, i0:i1] *= gate
    k = int(0.003 * SR)
    y[:, :k] *= np.linspace(0, 1, k)
    k = int(0.05 * SR)
    y[:, -k:] *= np.linspace(1, 0, k)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mixer.write_wav24(OUT, y)
    print("mastered      %.1f s" % (time.time() - t_start))

    # ------------------------------------------------ measurements
    q = np.round(y * 8388607.0)
    print("\n%s\n  %d samples, %.3f s, 48000 Hz, 2 ch, 24 bit" % (OUT, y.shape[1], y.shape[1] / SR))
    print("  own meter : integrated %.2f LUFS, true peak %.2f dBTP, DC %.6f" % (
        mixer.lufs(y), mixer.true_peak_db(y), float(np.mean(y))))
    e = mixer.ffmpeg_ebur128(OUT)
    print("  ffmpeg ebur128 : I %s LUFS, LRA %s LU, true peak %s dBTP" % (e["I"], e["LRA"], e["TP"]))
    mixer.stem_table(stems, shots)
    mixer.shot_table(y, shots)
    print("\nlift of the quiet scenes (dB, mean per shot): " + " ".join(
        "%s %.1f" % (s["name"].split("_")[0], float(np.mean(upg[int(s["start"] * 100):int(s["end"] * 100)]))) for s in shots))

    sl, fa = mixer.master.stats
    print("gain ride / peak limiter (dB, mean per shot): " + " ".join(
        "%s %.1f/%.1f" % (s["name"].split("_")[0], float(np.mean(sl[int(s["start"] * SR):int(s["end"] * SR)])),
                          float(np.mean(fa[int(s["start"] * SR):int(s["end"] * SR)]))) for s in shots))

    # the silence
    s0, s1 = C["stillness"]
    eye = C["one_eye_opens"]
    a = q[:, int(s0 * SR) + 1:int(eye * SR) - 1]
    b = q[:, int((eye + 0.1) * SR):int(s1 * SR)]
    seg = y[:, int(eye * SR):int((eye + 0.1) * SR)]
    print("\nstillness %.2f to %.2f: max |sample| before the eyelid %d, after it %d (24 bit integers, 0 = digital silence)" % (
        s0, s1, int(np.max(np.abs(a))), int(np.max(np.abs(b)))))
    print("  the eyelid at %.2f: peak %.1f dBFS" % (eye, mixer.dbf(np.max(np.abs(seg)))))
    ig = C["ignition"]
    w = y[:, int((ig - 0.5) * SR):int((ig + 0.5) * SR)]
    on = np.argmax(np.max(np.abs(w), axis=0) > 0.2) / SR + ig - 0.5
    print("  ignition: first sample over -14 dBFS at %.4f s (cue %.3f), 100 ms before %.1f dBFS rms, 100 ms after %.1f dBFS rms" % (
        on, ig, mixer.dbf(mixer.rms(y[:, int((ig - 0.1) * SR):int(ig * SR)])), mixer.dbf(mixer.rms(y[:, int(ig * SR):int((ig + 0.1) * SR)]))))
    st = C["star_ignites"]
    print("  star_ignites: 300 ms before %.1f dBFS rms, 300 ms after %.1f dBFS rms" % (
        mixer.dbf(mixer.rms(y[:, int((st - 0.3) * SR):int(st * SR)])), mixer.dbf(mixer.rms(y[:, int(st * SR):int((st + 0.3) * SR)]))))

    # every logged effect against its cue
    print("\ncue check (sound landing against the cue, one frame = %.1f ms)" % (frame * 1000))
    bad = 0
    for name, tc, tl in sorted(M.log, key=lambda v: v[1]):
        dms = (tl - tc) * 1000.0
        flag = "" if abs(dms) <= frame * 1000 else "   LATE OR EARLY"
        bad += bool(flag)
        print("  %-36s cue %7.3f  lands %7.3f  %+6.1f ms%s" % (name, tc, tl, dms, flag))
    logged = {n.split(" ")[0] for n, _, _ in M.log}
    missing = [k for k in C if k not in logged and k not in ("end",)]
    print("  %d sounds checked, %d outside one frame. cues with no logged sound: %s" % (len(M.log), bad, ", ".join(missing) or "none"))

    if "--quick" not in sys.argv:
        os.makedirs(REVIEW, exist_ok=True)
        mixer.spectrogram(OUT, os.path.join(REVIEW, "sound-spectrogram-full.png"))
        mixer.spectrogram(OUT, os.path.join(REVIEW, "sound-spectrogram-28-to-40.png"), ss=28, t=12)
        mixer.spectrogram(OUT, os.path.join(REVIEW, "sound-spectrogram-05-to-15.png"), ss=5, t=10)
    print("\ntotal render time %.1f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
