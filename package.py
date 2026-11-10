"""Join the rendered cuts and the soundtrack into the deliverables.

  python3 package.py join      build/seg/<shot>.mp4 (in film order) -> build/master-silent.mp4 (stream copy)
  python3 package.py x         master + build/audio/tiny-rocket-5-2.wav -> out/tiny-rocket-5-2-x.mp4
  python3 package.py web       -> docs/film.mp4 (under 95 MB for GitHub Pages) and docs/poster.jpg
  python3 package.py stills    -> out/still-title.png, still-closeup.png, still-climax.png (from the master)
  python3 package.py loudness  print integrated loudness, range and true peak of the wav and the X file
  python3 package.py all       everything above in order
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from tr2.timeline import DUR, FPS, SHOTS  # noqa: E402

SEG = os.path.join(ROOT, "build", "seg")
MASTER = os.path.join(ROOT, "build", "master-silent.mp4")
WAV = os.path.join(ROOT, "build", "audio", "tiny-rocket-5-2.wav")
XFILE = os.path.join(ROOT, "out", "tiny-rocket-5-2-x.mp4")
WEB = os.path.join(ROOT, "docs", "film.mp4")
POSTER = os.path.join(ROOT, "docs", "poster.jpg")

# Frames for the three stills (same three moments as the 5.1 stills). Set when the picture is final.
STILLS = {"still-closeup.png": 56, "still-climax.png": 1308, "still-title.png": 1735}
POSTER_FRAME = 1308
# AAC encoding overshoots the wav's -1.3 dBTP peak (measured -0.9 dBFS in the first X file,
# -1.7 with a 0.6 dB trim), so the muxed audio is trimmed 0.3 dB: under -1 dBTP after encoding
# and still about -14 LUFS.
AUDIO_TRIM = "volume=-0.3dB"


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def join():
    lst = os.path.join(SEG, "list.txt")
    full = os.path.join(SEG, "full.mp4")
    names = [n for n, a, b, tod in SHOTS]
    if all(os.path.exists(os.path.join(SEG, n + ".mp4")) for n in names):
        with open(lst, "w") as fh:
            for n in names:
                fh.write("file '%s'\n" % os.path.join(SEG, n + ".mp4"))
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
             "-movflags", "+faststart", MASTER])
    elif os.path.exists(full):
        run(["ffmpeg", "-v", "error", "-y", "-i", full, "-c", "copy", "-movflags", "+faststart", MASTER])
    else:
        missing = [n for n in names if not os.path.exists(os.path.join(SEG, n + ".mp4"))]
        raise SystemExit("missing cuts: " + ", ".join(missing))
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", MASTER], capture_output=True, text=True)
    print("master frames:", out.stdout.strip(), "expected", int(DUR * FPS))


def x():
    os.makedirs(os.path.dirname(XFILE), exist_ok=True)
    # the picture has film grain, so it is expensive to encode: step the quality down until
    # the file is safely under X's 500 MB (CRF 15 came out at 747 MB)
    for crf in (15, 16, 17, 18, 19, 20, 21, 22):
        run(["ffmpeg", "-v", "error", "-y", "-i", MASTER, "-i", WAV, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-r", str(FPS), "-g", "48", "-bf", "2", "-colorspace", "bt709", "-color_primaries", "bt709",
             "-color_trc", "bt709", "-af", AUDIO_TRIM, "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest",
             "-movflags", "+faststart", XFILE])
        mb = os.path.getsize(XFILE) / 1e6
        print("X file: %.1f MB at CRF %d" % (mb, crf))
        if mb < 470:
            break


def web():
    # two pass to land just under the size limit: 95 MB minus audio and a safety margin
    total_kbps = 88e6 * 8 / DUR / 1000
    v_kbps = int(total_kbps - 200)
    log = os.path.join(ROOT, "build", "ff2pass")
    base = ["ffmpeg", "-v", "error", "-y", "-i", MASTER]
    venc = ["-c:v", "libx264", "-preset", "slow", "-b:v", "%dk" % v_kbps, "-maxrate", "%dk" % int(v_kbps * 1.6),
            "-bufsize", "%dk" % int(v_kbps * 3), "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "48",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-passlogfile", log]
    run(base + venc + ["-pass", "1", "-an", "-f", "mp4", os.devnull])
    run(base + ["-i", WAV, "-map", "0:v:0", "-map", "1:a:0"] + venc +
        ["-pass", "2", "-af", AUDIO_TRIM, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
         "-movflags", "+faststart", WEB])
    run(["ffmpeg", "-v", "error", "-y", "-i", MASTER, "-vf", "select=eq(n\\,%d)" % POSTER_FRAME, "-frames:v", "1",
         "-q:v", "3", POSTER])
    mb = os.path.getsize(WEB) / 1e6
    print("web file: %.1f MB" % mb)
    if mb > 95:
        raise SystemExit("web file is over 95 MB")


def stills():
    for name, fr in STILLS.items():
        run(["ffmpeg", "-v", "error", "-y", "-i", MASTER, "-vf", "select=eq(n\\,%d)" % fr, "-frames:v", "1",
             os.path.join(ROOT, "out", name)])


def loudness():
    for p in (WAV, XFILE):
        if not os.path.exists(p):
            continue
        r = subprocess.run(["ffmpeg", "-nostats", "-i", p, "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
                           capture_output=True, text=True)
        tail = r.stderr[r.stderr.rfind("Summary:"):]
        print(p)
        print(tail)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"join": join, "x": x, "web": web, "stills": stills, "loudness": loudness}
    if cmd == "all":
        for k in ("join", "x", "web", "stills", "loudness"):
            steps[k]()
    else:
        steps[cmd]()
