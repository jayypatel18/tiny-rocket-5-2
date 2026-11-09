"""The director's timeline: every cut and every story beat, in seconds.
The picture and the sound are both built from this one file.
Run `python3 -m tr2.timeline` from the repo root to write cues.json."""
import json
import os

FPS = 24
DUR = 74.0
NFRAMES = int(DUR * FPS)  # 1776

# (name, start, end, time of day)
SHOTS = [
    ("s1_knot", 0.0, 6.0, "dusk"),
    ("s2_hill", 6.0, 14.0, "dusk"),
    ("s3_night", 14.0, 21.5, "night"),
    ("s4a_wake", 21.5, 25.0, "predawn"),
    ("s4b_goggles", 25.0, 28.5, "predawn"),
    ("s5a_match", 28.5, 30.5, "predawn"),
    ("s5b_fuse", 30.5, 33.0, "predawn"),
    ("s5c_breath", 33.0, 36.0, "predawn"),
    ("s6_lift", 36.0, 43.0, "launch"),
    ("s7a_clouds", 43.0, 46.5, "sunrise"),
    ("s7b_wonder", 46.5, 50.0, "sunrise"),
    ("s8_star", 50.0, 58.5, "sunrise"),
    ("s9a_home", 58.5, 62.0, "morning"),
    ("s9b_out", 62.0, 65.0, "morning"),
    ("s9c_paint", 65.0, 67.5, "morning"),
    ("s9d_title", 67.5, 74.0, "morning"),
]

CUES = {
    # shot 1: the last knot (close, dusk)
    "knot_creak": [0.0, 1.2],          # the string under tension while it pulls
    "knot_tug": 1.25,                  # the quick tight tug
    "ear_flicks": [1.9, 2.3],
    "tail_swish": 2.6,
    "blinks": [3.6, 4.1],
    "pleased_breath": 4.6,
    # shot 2: the hill (wide, dusk)
    "motif_first": 6.5,                # lone felt piano motif starts
    "steps_back": [7.4, 7.9, 8.4],     # paws in grass
    "village_lights": [7.0, 13.5],     # windows come on one by one in this span
    "admire": 9.0,
    "steps_to_map": [10.6, 11.1, 11.6],
    "map_rustle": 12.2,
    # shot 3: waiting for morning (night)
    "night_starts": 14.0,              # music box or celesta colour
    "shooting_star_night": [17.2, 18.0],
    "ear_twitch_sleep": 20.3,
    # shot 4: first light (predawn)
    "wake": 21.8,
    "yawn_stretch": 22.4,
    "climb_paws": [23.4, 23.9],        # paws on tin and paper
    "into_cup": 24.5,
    "goggles_snap": 25.7,
    "swallow": 26.9,
    "nod": 27.8,
    # shot 5: the spark
    "match_strike": 29.0,
    "fuse_lit": 30.0,
    "fuse_run": [30.0, 33.9],          # fizz travelling; in 5b (30.5 to 33.0) it crosses the frame left to right
    "eyes_shut": 33.2,
    "fuse_end": 33.9,                  # the spark reaches the rocket and dies
    "stillness": [34.0, 35.6],         # true silence
    "one_eye_opens": 35.3,
    # shot 6: lift
    "ignition": 36.3,
    "shudder": [36.3, 37.6],
    "liftoff": 37.6,
    "passes_tree": 39.0,
    "birds_burst": 39.5,
    "into_first_clouds": 42.0,
    # shot 7: through the clouds
    "clouds_start": 43.0,              # full orchestral swell builds from here
    "break_into_open_sky": 46.2,
    "wind_face": 46.5,
    "wonder": 48.4,
    # shot 8: the shooting star (climax)
    "vista": 50.0,
    "star_ignites": 51.0,              # the piano motif returns at full scale
    "star_arc": [51.0, 57.5],
    # shot 9: home (morning)
    "parachute": 58.5,                 # creak and flutter of the paper umbrella
    "touchdown": 61.3,
    "climb_out": 62.4,
    "paws_grass_home": [63.4, 63.8],
    "looks_up": 64.2,
    "brush_stroke": 66.4,              # the paw paints the new star: one soft piano note
    "looks_at_us": 67.9,
    "tail_curl": 68.6,
    "title_brush": [69.0, 70.6],       # "Tiny Rocket" brushed in ink
    "hanko": 70.9,                     # the red seal presses
    "credit_brush": [71.3, 72.0],      # the small line "Fable 5.2"
    "fade_to_paper": 72.6,
    "fade_to_black": 73.4,
    "end": 74.0,
}


def shot_at(f):
    """(index, name, local frame, shot length in frames, tod) for global frame f."""
    t = f / FPS
    for i, (name, a, b, tod) in enumerate(SHOTS):
        if a - 1e-9 <= t < b - 1e-9:
            return i, name, f - int(round(a * FPS)), int(round((b - a) * FPS)), tod
    name, a, b, tod = SHOTS[-1]
    return len(SHOTS) - 1, name, f - int(round(a * FPS)), int(round((b - a) * FPS)), tod


def shot_range(name):
    for n, a, b, tod in SHOTS:
        if n == name:
            return int(round(a * FPS)), int(round(b * FPS))
    raise KeyError(name)


def write_cues(path=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = path or os.path.join(root, "cues.json")
    data = {"fps": FPS, "duration": DUR,
            "shots": [{"name": n, "start": a, "end": b, "light": tod} for n, a, b, tod in SHOTS],
            "cues": CUES}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    return path


if __name__ == "__main__":
    print(write_cues())
