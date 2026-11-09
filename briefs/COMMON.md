# TINY ROCKET (Fable 5.2): rules every agent on this film reads first

Folder: `/Users/jaypatel/Downloads/code-films-5-2/` (call it ROOT). Director: the `fable film` main session. You are one of at most four builders. All of us are Fable 5.2; the end card will say so, so every line must be written by us, from scratch.

## What we are making
A 74 second 2D anime short, 1920x1080, 24 fps, where every frame and every sound is code in this folder. It must be a clear level above the Fable 5.1 version of the same story: fuller animation, real light, parallax, simulated smoke and fire, a real score. Story and quality standard: `ROOT/PROMPT.md` (read it all, including "Fable 5.2 brief" at the bottom). What we must beat: `ROOT/build/ref51/sheet.jpg` and `still-*.jpg` there (frames of the 5.1 film). Its weak points: a stiff flat panda, a tiny cast in the wide shots, simple backgrounds, crude brush lettering, little camera movement.

## Hard rules
1. Every pixel and sound comes from code written here tonight. No image, video or music models, no downloaded or stock assets, no samples, no font files (`PIL.ImageFont` and system fonts are banned; lettering is brush strokes in code).
2. Do NOT read anything under `/Users/jaypatel/Downloads/code-films/` except what the director already copied to `ROOT/build/ref51/`. Never read the 5.1 code or its engine. Do not touch any folder outside ROOT.
3. Tools: Python 3.12 (`python3`), numpy 2.2, scipy 1.14, numba 0.65, Pillow 12, ffmpeg (libx264, aac, no drawtext). Nothing else; do not pip install.
4. Never write an em dash or an en dash anywhere (code, comments, notes, reports), and never a spaced hyphen standing in for one. Restructure the sentence.
5. No git commands. The director commits. Never post anything anywhere.
6. Absolute paths in every Bash command. Never `cd X && ...`. Create files with the Write and Edit tools, not heredocs.
7. Write only the files your brief says you own. You may read everything in ROOT.
8. The machine has 8 cores and 16 GB, shared by four agents. Keep any single job under about 3 GB RAM, cap parallelism at 2 processes, set `NUMBA_NUM_THREADS=2` for long jobs. Jobs over about 4 minutes run with `run_in_background`. Disk is tight (about 40 GB free): keep your cache under 3 GB in `ROOT/build/cache/<your name>/`, never write a PNG sequence of a whole shot.
9. Deterministic, random access rendering: frame f renders the same every time and alone. Seed randomness with `core.rng(key...)`. Simulations (smoke) are precomputed to a cache, then read by frame.
10. Speed matters: the whole film is 1776 frames. A finished frame must cost under about 6 s on one core in total, so your part of a frame should cost under about 1.5 s. Cache anything static (painted layers) to `.npy`.

## The engine (already written, read these two files before anything else)
- `ROOT/tr2/core.py`: buffers (picture = float32 HxWx3 0..1; layer = float32 HxWx4 premultiplied), `rng`, `fbm` noise, timing helpers (`keys`, `overshoot`, `on_n`), curves (`catmull`, `wobble`), the `Cel` rasteriser (flat fills, clip masks, hard crescent shadows, tapered ink strokes), `over`, `add_light`, `gblur`, `warp`, the multiplane `Camera`, `save`, `sheet`.
- `ROOT/tr2/palette.py`: the six times of day (`TOD`: dusk, night, predawn, launch, sunrise, morning) with sky stops, sun direction, key, ambient, rim, fog, ink; local colours of the panda and rocket; `lit()`, `shade()`, `rim()`. Take ALL colour from here so everything sits in one light. If a palette value is wrong for your work, tell the director in your report (do not edit these two files; add what you need in your own module).
Import with `sys.path.insert(0, ROOT)` then `from tr2 import core, palette`.

## The film (director's timeline; f = frame, t = seconds; total 74.0 s = 1776 frames)
| Shot | t | Light | What happens, and the camera |
|---|---|---|---|
| 1 The Last Knot | 0.0 to 6.0 | dusk | Close on the panda, three quarter view, golden rim light, soft defocused grass and valley behind, pollen and first fireflies. Frame 0 is already mid action: string in teeth, paws pulling the knot tight on a bamboo strut. Tug, ears flick, tail swishes, it blinks at its work, pleased. Slow push in. |
| 2 The Hill | 6.0 to 14.0 | dusk | Wide. Hilltop, leaning camphor tree, river valley, village lights come on one by one, first stars. The rocket stands on the hilltop. Panda steps back to admire it, then checks the star map pinned to a stick. Slow lateral dolly with strong parallax. |
| 3 Waiting for Morning | 14.0 to 21.5 | night | Starts on the turning star field, tilts down to the panda asleep curled against the rocket under a patchwork blanket. A shooting star crosses. One ear twitches. Fireflies. |
| 4a First Light | 21.5 to 25.0 | predawn | Mid wide. It wakes, stretches, climbs into the teacup cockpit. |
| 4b | 25.0 to 28.5 | predawn | Close up through the round window: goggles pulled down, whiskers, a nervous swallow, a small determined nod. Rack focus from the window rim to the face. |
| 5a The Spark | 28.5 to 30.5 | predawn | Close: a paw strikes a match on the cup rim; the flare lights the face warm; it touches the fuse. |
| 5b | 30.5 to 33.0 | predawn | Ground level, the spark runs along the fuse through the grass toward the rocket legs. Camera tracks with it, on ones. |
| 5c | 33.0 to 36.0 | predawn | Close up: eyes squeezed shut, breath held. The spark light dies. One beat of total stillness (34.0 to 35.6). It opens one eye. |
| 6 Lift | 36.0 to 43.0 | launch | Low wide. Ignition at 36.3: burst of fire and smoke, sparks, grass flattened in a ring, the rocket squashes, shudders, leaps at 37.6. ONE continuous camera move: tilt up with the rocket past the camphor tree (birds burst out at 39.5) into the dawn sky and the first clouds. |
| 7a Through the Clouds | 43.0 to 46.5 | sunrise | The rocket punches up through cloud layers into the first sun. Cloud layers stream down with parallax, light shafts, fins rattle. |
| 7b | 46.5 to 50.0 | sunrise | Inside the teacup, close: cheeks pushed back by wind, fear turning to wonder as gold light floods the face. |
| 8 The Shooting Star | 50.0 to 58.5 | sunrise | The climax. From very far and high: mountains, a sea of clouds, rivers catching light, the curve of the world. The trail ignites (51.0) into a brilliant shooting star that arcs across the whole sky. Slow majestic push. |
| 9a Home | 58.5 to 62.0 | morning | Hilltop, morning. The rocket drifts down under an umbrella parachute and settles in the grass (touchdown 61.3). |
| 9b | 62.0 to 65.0 | morning | The panda climbs out, sooty face, goggles askew, looks up where one star is still fading. |
| 9c | 65.0 to 67.5 | morning | Close on the star map: the paw paints one tiny new star (66.4). |
| 9d | 67.5 to 74.0 | morning | It looks at us, the tail curls once (68.6). The title "Tiny Rocket" brushes in over the sky in ink (69.0 to 70.6), the red hanko seal presses (70.9), then the small line "Fable 5.2" (71.3 to 72.0). Hold. Fade to paper (72.6), then black (73.4 to 74.0). |

## The hilltop set (shots 2, 3, 4a, 5b, 6, 9), in SET coordinates
One composition, x right, y down, default camera centre (1600, 900) so the default view is x 640..2560, y 360..1440.
- Hill crest (the ground line characters stand on): `ground_y(X)`, from (640, 1165) rising to the summit near (1500, 1050), falling to (2560, 1330). Everything below is the grassy near slope.
- Rocket: stands with the centre of its feet at (1720, ground_y). At scale 1 it is 500 px tall and about 190 px wide at the fin.
- Panda home position: feet at (1400, ground_y). At scale 1 the standing panda is 170 px tall to the ear tips (the rocket is three pandas tall).
- Star map on a stick: at (1180, ground_y), about 150 px tall.
- Camphor tree: trunk base (880, ground_y), leaning right, canopy roughly x 640..1560, y 330..820, a branch reaching toward the rocket.
- Valley behind the right half: the river winds from lower right into the distance; the village sits along it around x 1850..2500, y 930..1090; mountain ranges behind; horizon at y about 880.
- Parallax factors: sky 0.02, far mountains 0.07, near mountains 0.14, valley and village 0.25, hill, tree, rocket and panda 1.0, foreground grass 1.6.
- Key light direction per time of day is in the palette (`sun`). Dusk: low sun on the left. Morning and dawn: sun on the right, behind the valley.

## House style: what makes a frame read as theatrical 2D anime
- Contrast of two worlds: crisp, flat, inked cels (characters, props) over soft, rich, painted gouache backgrounds, tied by one light and one atmosphere.
- Cels: flat colour, ONE hard shadow tone (hue pushed toward the ambient colour), a thin rim light inside the line on the sun side, sometimes one highlight. Never soft shading on a cel. Ink lines with pressure: thicker on the shadow side and at overlaps, tapered ends, coloured (the palette `ink`), slight wobble, a tiny boil from drawing to drawing.
- Backgrounds: compose values and light first, then paint over with a stroke painter (big to small brushes, strokes following the form, colour jitter, dry brush texture, paper grain), then place crisp accents. Colourful shadows (violet, teal, rose), never grey. Atmospheric perspective. No two trees, roofs, clouds, stars or sparks alike. No computer gradients, no perfect geometry.
- Animation: nothing slides, nothing is a rotating sprite. Every drawing is rebuilt from its pose. Key action on ones, the rest on twos. Anticipation, overshoot, settle, overlap and follow through (tail, ears, paper, smoke, grass). Squash and stretch on the panda. Smears on the fastest frames. Holds are alive (a breath, a boil), never frozen.
- It must NOT look like: 3D, vector art, web animation, AI art, a western cartoon, an ad, or amateur work (see the list in PROMPT.md).

## The inspection loop (not optional)
Save stills and LOOK at them with the Read tool (downscale to about 1280 px JPEG with `sips -Z 1280 -s format jpeg in.png --out out.jpg` first: full PNGs waste your context). Use contact sheets (`core.sheet`) for motion, and tight crops for line and brush quality. For every still ask: would this pass as a frame from a finished theatrical anime film, and is it clearly better than the 5.1 frame? Fix what fails before moving on.

## Reporting
- Keep a log at `ROOT/notes/<your name>.md`: start time first (`TZ=Asia/Kolkata date`), then what you made, what you saw, what you changed, and the time of each entry from `date`. If you are interrupted, this file is how work resumes.
- Hit the checkpoint time in your brief even if the work is rough: save the stills, then end your turn with a short report (paths of stills, what works, what is still weak, your module's API in a few lines, minutes spent). The director will reply with notes and you continue. Be honest about what falls short.
