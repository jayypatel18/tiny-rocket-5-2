# D: the rocket and the effects (smoke, fire, sparks, light)

Read `ROOT/briefs/COMMON.md` first (ROOT = `/Users/jaypatel/Downloads/code-films-5-2`), then `ROOT/PROMPT.md`, `ROOT/tr2/core.py`, `ROOT/tr2/palette.py`. Your name for notes and cache: `fx`.

You own `ROOT/tr2/rocket.py`, `ROOT/tr2/fx.py` (plus `fx_*.py` helpers), and `ROOT/notes/fx.md`.

## The rocket (`rocket.py`), a cel prop with handmade character
As tall as three red pandas (500 px at scale 1, feet to tip). From the ground up: three splayed bamboo legs lashed with string (with nodes), a dented tin skirt with rivets and a soot stained nozzle (an old bucket), a body of rolled paper with seams, tape, paper patches and stars painted on by paw (uneven, different sizes), a big white TEACUP as the cockpit section (blue band, a handle on one side, a round brass rimmed window in its front, large enough to show the panda's head and shoulders), and a TEAKETTLE upside down as the nose cone (spout as an antenna, lid knob, dents). One big patched sailcloth fin on a bamboo spar, plus two small paper fins. Paint drips, a wind up key. It must keep the same design in every shot and from 3 views: side (wide shots), low three quarter (launch), and from below or behind (climbing away). The 5.1 rocket (`ROOT/build/ref51/still-title.jpg`) is a plain cone on a tube: beat it with silhouette and detail.
- Cel look per COMMON: flat fills, one hard shadow tone, thin rim light, pressure ink lines, everything rebuilt per drawing (line boil on twos via `drawing`).
- State: `shake` (dx, dy, rot), `squash` (anticipation before the leap), `fin_rattle` (0..1, fins and paper edges flutter, each part its own phase), `flame` (0..1), `soot` (0..1 after the flight), `legs_fold`, `hatch_open`, `window_glow`, a second `light` (fire from below).
- The window is a hole: return its screen centre and radius (and a clip mask) so the panda cel can be drawn behind the glass, then call a second function to draw glass reflections and the rim over it.
- Cockpit close up (shots 4b, 5a, 5c, 7b): the teacup rim, the round window frame with rivets, the inside wall with pinned drawings and a tiny dial, drawn large (the window about 760 px across) with the same design. The panda agent draws the panda; you provide `draw_cockpit_back(...)` and `draw_cockpit_front(...)` layers around it.
- The umbrella parachute (shot 9a): a patched paper umbrella with ribs and four strings, swinging, then collapsing on touchdown. The star map on a stick and the match are also yours (small cel props).
```python
from tr2 import rocket
st = rocket.State(...)
info = rocket.draw(cel, x, y, s=1.0, tod="dusk", drawing=0, rot=0.0, view="side", state=st, light=None)
#   (x, y): screen point at the centre of the feet. Returns dict(window=(cx, cy, r), nozzle=(x, y), tip=(x, y), mask=...).
rocket.draw_glass(cel, info, tod, drawing)
```

## Effects (`fx.py`): simulated, painted, never a particle logo effect
1. Smoke and fire: a 2D fluid solver in numba (semi Lagrangian advection, pressure projection, buoyancy, vorticity confinement) on a grid of about 384x216 or finer, with density, temperature and a fuel field. It is precomputed for the launch (shot 6: ignition burst that rolls outward along the ground in a ring, then the column under the climbing rocket, then the hanging trail) and cached per frame (float16, compressed) so any frame reads back alone. Then RENDER it like anime effects animation, not like a physics demo: threshold the density into 3 or 4 flat tones with crisp, slightly wobbling edges (light tone on the sun side, a violet shadow tone under, a hot orange underside where the fire lights it), an ink free look, small detached puffs, and a white hot core to yellow to orange flame with a hard inner shape. Upsample with noise warping so edges are sharp at 1080p.
2. Sparks: particles with gravity and drag, each drawn as a motion blurred streak along its velocity (length from shutter), with hot white heads and orange tails, flickering sizes, a few that bounce. Used at ignition, along the fuse (shot 5b: a fizzing spark head with a short spray and a thin smoke thread), and the match flare.
3. The star trail of shot 8 (a brilliant head with a long tapering luminous trail and glitter falling from it) and its lens flare.
4. Light: `bloom(pic, threshold, radius, amount)` on highlights, `glow_at(pic, x, y, radius, color, amount)`, `light_shafts(pic, sun_xy, occlusion_alpha, amount)` (radial shafts through the gaps of an occluder such as clouds, tree, smoke), and a ground light pass (the fire lighting the grass and the underside of smoke warm).
5. Fireflies and pollen: drifting, each with its own slow wander and soft blink; in defocus they become bokeh discs.
Document the API at the top of `fx.py`. Everything takes a `core.Camera` or screen coordinates plus the frame, and composites onto a picture or returns an RGBA screen layer.

## Checkpoint 1: 02:40 IST (hard)
`ROOT/build/review/launch-look.png`, 1920x1080: one lit frame from the launch, about 0.6 s after ignition. Low wide view: the rocket (drawn large, about 620 px tall on screen) just lifting off its legs, a white hot flame under it, the simulated smoke rolling out along the ground in cel tones lit orange from inside, sparks with motion blur, warm fire light on the rocket's underside and on the grass, bloom. Until agent B's hill set exists, stand it on a simple placeholder (palette "launch" sky gradient, a dark grassy slope silhouette you paint quickly). Also `ROOT/build/review/rocket-sheet.png`: the rocket in side and low three quarter views on paper colour at s about 1.8, with labels drawn as nothing (no text). End your turn with the report at 02:40 even if rough. After the director's notes: the full launch simulation, the fuse, the parachute, the cockpit layers, the star trail.
