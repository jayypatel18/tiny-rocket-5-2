# B: the gouache painter and the hilltop set

Read `ROOT/briefs/COMMON.md` first (ROOT = `/Users/jaypatel/Downloads/code-films-5-2`), then `ROOT/PROMPT.md`, `ROOT/tr2/core.py`, `ROOT/tr2/palette.py`. Your name for notes and cache: `hill`.

You own `ROOT/tr2/paint.py` (a reusable stroke painter), `ROOT/tr2/hillset.py` (the hilltop location, plus `hillset_*.py` helpers) and `ROOT/notes/hill.md`. Agent C (sky and cloudscapes) will reuse your `paint.py`, so write it first, keep its API small and document it at the top of the file.

## The painter (`paint.py`)
Turns a composed reference (a float RGB image with the layout, values and light already right, plus optional direction field and alpha) into a gouache painting: several passes from big brushes to small, each stroke a textured, slightly opaque dab or drag oriented along a direction field (along cloud contours, grass growth, roof slopes, tree foliage clumps), colour sampled from the reference with jitter in hue and value, dry brush breakup at the end of strokes, paint pooling slightly darker at stroke edges, paper grain. Must keep crisp edges where asked (an edge mask or alpha that strokes may not cross) and must paint transparent layers correctly (premultiplied RGBA out). Write the stroke loop in numba. A 2880x1620 layer should paint in under about 40 s. Deterministic from a seed.

## The hilltop set (`hillset.py`)
Layout in SET coordinates is fixed in COMMON ("The hilltop set"). Paint it as separate depth layers, each a premultiplied RGBA float32 array with an origin, so the multiplane camera can move through it:
1. far mountains (p 0.07) and near mountains (p 0.14): ranges in haze, each ridge different, snow hints, atmospheric perspective toward the palette fog.
2. the valley (p 0.25): the river winding into the distance catching sky light, fields, tree clumps, mist pooled low, and the village along the river: about 25 houses, every roof different (thatch and tile, a temple roof, a bridge, a water wheel), with windows. Window lights are a separate animated element: `lights` (0..1) turns them on one by one in a fixed random order, each with a warm glow, reflected in the river.
3. the hill (p 1.0): grassy crest and near slope, lit along the crest by the sun, wild flowers, a worn path, stones; grass painted in clumps following the slope. The camphor tree (p 1.0, its own layer so it can sway and sit in front of or behind things): old leaning trunk with bark, roots, big rounded foliage masses painted clump by clump with sky holes, lit side and violet shadow side. Canopy sway: return the foliage as 3 or 4 sub layers that the caller offsets by a wind phase (or apply a small per frame warp yourself).
4. foreground grass (p 1.6): big soft blades and seed heads at the bottom edge, drawn per frame so they sway with wind (overlapping motion, each blade its own phase). Also a band of short grass tufts at the ground line (p 1.0) to overlap the feet of the panda and the rocket legs.
The sky is NOT yours (agent C renders it); for your own stills use the plain palette gradient.

Times of day: every layer must exist for dusk, night, predawn, launch and morning (palette.TOD). Compose the reference per time of day (light direction, long shadows of tree and rocket on the grass, fog colour, night village glow), then paint. Cache painted layers as `.npy` under `ROOT/build/cache/hill/` keyed by time of day; building all five must take under about 10 minutes.

## API
```python
from tr2 import hillset
hillset.ground_y(X)                         # hill crest height in set coords (float or array)
hillset.render_back(pic, cam, tod, f, lights=0.0, wind=None, defocus=None, flatten=None)
#   Composites every layer with p <= 1.0 (mountains, valley, village, hill, tree) onto `pic`
#   (float RGB screen picture that already holds the sky), in place, through core.Camera `cam`.
#   lights 0..1: village windows. defocus: optional function p -> blur sigma px (rack focus).
#   flatten: optional (X_centre, radius, amount) grass pressed outward in a ring (ignition).
hillset.render_front(cam, tod, f, wind=None, defocus=None, flatten=None)   # -> RGBA screen layer: foreground grass p 1.6
hillset.grass_band(cam, tod, f, x_from, x_to, flatten=None)                # -> RGBA screen layer: tufts over feet
hillset.tree_layer(cam, tod, f)             # -> RGBA screen layer, if the caller needs the tree apart (optional)
```
The camera may zoom from 0.8 to 3.0 and tilt up until the hill leaves the frame (shot 6), so paint the hill and tree layers at 1.5x set resolution (or paint the area around the rocket and panda at higher density) so a 2x zoom is still painterly, not blurry. Layers with p below 1 can be modest in size: they barely move.

## Checkpoint 1: 02:40 IST (hard)
`ROOT/build/review/hill-dusk.png`: the default wide view (camera at defaults) at dusk, 1920x1080, over the plain palette sky gradient, fully painted. This is the look test for "one painted background": it must already beat `ROOT/build/ref51/still-title.jpg` and the dusk frames in `sheet.jpg` on richness, depth and light. Also save `hill-dusk-crop.png` (a 1:1 crop of the village and of the tree foliage side by side). End your turn with the report at 02:40 even if only dusk exists. After the director's notes: the other times of day, lights, sway, foreground grass, the flatten ring, speed.
