# A: the red panda (character design, rig, acting)

Read `ROOT/briefs/COMMON.md` first (ROOT = `/Users/jaypatel/Downloads/code-films-5-2`), then `ROOT/PROMPT.md`, `ROOT/tr2/core.py`, `ROOT/tr2/palette.py`. Your name for notes and cache: `panda`.

You own `ROOT/tr2/panda.py` (plus `ROOT/tr2/panda_*.py` helpers if you want) and `ROOT/notes/panda.md`. The panda is the heart of the film. The 5.1 panda (`ROOT/build/ref51/still-closeup.jpg`) is a flat sticker with a stiff face. Yours must be a character an audience falls for in the first second: a theatrical anime animal lead, original, appealing, solid, alive.

## Design
A young red panda, a small tinkerer. Big head (about 45 percent of the standing height), round chubby body, short limbs, a thick ringed tail as long as the body, big ears with white rims and fur tufts. Russet fur, white face mask (muzzle, cheek patches, brow dots), red tear marks from the eyes down the cheeks, dark chocolate legs, belly and paws with visible toes and pads, a small dark nose, big dark glossy eyes with amber irises and two highlights. It stands and walks upright like a small child, but can drop to all fours. Solid volume: the same design must read from front, three quarter (both sides), profile and three quarter back. Props it wears: oversized aviator goggles (on the forehead, over the eyes, or askew), soot smudges (shot 9).

## How to build it
- A pose is a small set of numbers and points (a dataclass `Pose`): root position, body lean and squash, head yaw (-1 left .. +1 right) and pitch and tilt, eye openness per eye, pupil direction, brows, mouth shape, cheek puff, ear angles per ear, arm and leg targets (IK to paw positions), tail control points, breath, goggles state, soot, fur ruffle (wind). The DRAWING is rebuilt from the pose every time: closed spline shapes placed on a simple 2.5D volume (features slide over the head ellipsoid as it turns, the far side foreshortens, ears and cheeks overlap correctly). No sprite is ever rotated or slid.
- Cel look per COMMON: flat fills, one hard shadow tone, thin rim light on the sun side INSIDE the ink line, pressure ink lines in the palette ink colour, fur tufts as a few confident shapes at cheeks, ear tips, chest and tail (not a sawtooth edge). Light it from `palette.TOD[tod]` (sun direction, key, amb, rim). Optional second light (match flare, fire glow, gold sunrise) passed as `light=dict(color=(r,g,b), dir=(dx,dy), amount=0..1)` adds a warm lit tone on that side.
- Line boil: the `drawing` number seeds a sub pixel wobble so held poses stay alive on twos.
- Squash and stretch, smears: the pose takes `squash` (volume preserving) and a `smear=(dx,dy)` for the fastest frames.
- Overlap and follow through: give helpers that turn key poses plus time into in between poses with arcs, overshoot and settle, and a tail and ear follow through (a simple spring chain evaluated from the motion history, deterministic for any frame).

## API the director and other agents will call
```python
from tr2 import panda
pose = panda.Pose(...)                      # all fields have sane defaults
panda.draw(cel, pose, x, y, s=1.0, tod="dusk", drawing=0, light=None)
#   cel: core.Cel. (x, y): screen position of the ground point between the feet.
#   s: scale; at s=1 the standing panda is 170 px tall to the ear tips.
#   Must look good from s=0.8 (wide shots, panda about 140 px) to s=7 (close up, head fills the frame).
panda.POSES["name"]                         # library of key poses (see list)
panda.blend(pose_a, pose_b, u)              # in between (arcs, not linear slides)
panda.bbox(pose, x, y, s)                   # screen box, so a small Cel can be used
```
Key poses and expressions needed by the film (make them as library entries, each must look good): stand neutral, stand proud (chest out, fists on hips), step back (two walk keys), look up, read map (leaning, one paw raised), tie knot (three quarter close, paws together, string end in teeth, the tug), pleased blink, sleep curled (tail wrapped over the nose), ear twitch, wake and stretch (big yawn, arms up), climb (reach up, pull up, over the rim), cockpit bust (shoulders up, paws on the rim), goggles down, nervous swallow, determined nod, strike match (arm sweep), eyes squeezed shut with held breath (cheeks puffed), one eye open, wind face (cheeks and fur pushed back, mouth pulled wide, ears flat), wonder (eyes huge, mouth open, highlights big), climb out sooty, look up soft, paint with paw (leaning, tongue out), look at camera with a small smile, tail curl.

## Checkpoint 1: 02:40 IST (hard)
`ROOT/build/review/panda-sheet.png` (1920x1080 on a warm paper colour): a model sheet with a turnaround (front, three quarter, profile, three quarter back, standing, at s about 2.5), at least six expression heads (s about 3.5), and four action poses (tie knot, sleep curled, wind face, wonder). Plus `ROOT/build/review/panda-closeup.png`: one 1920x1080 close up at s about 6.5 in dusk light on a plain dusk gradient, three quarter view, tying the knot with the string in its teeth. Inspect, fix, repeat until then. End your turn with the report at 02:40 even if rough. After the director's notes you will go on to the remaining poses, the follow through helpers, and then animate whole shots.
