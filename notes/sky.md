# sky (builder C) log

Start: 2026-11-10 01:55 IST. All times below are from `TZ=Asia/Kolkata date`.

## Files I own
- `tr2/sky_paint.py`: shared helpers. Numba brush stroke fields (`stroke_field`), stroke repainter (`repaint`), painted cloud sprites as FIELDS (alpha, light, underside) lit from the upper right (`cloud_fields`, cached `cloud_sprite`, kinds cumulus, tower, bar, wisp, band), five tone colouring (`tones`, `colourise`), sprite placing with mip levels (`mips`, `place`). Cache: `build/cache/sky/`.
- `tr2/sky.py`: the living sky. `render(tod, cam, f, t, stars=None, clouds=1.0, star_turn=0.0, shooting=None, sun=True, fade_star=None)`.
- `tr2/vista.py`: shot 8 set. `render(cam, f, t, star=None)`.
- `tr2/cloudset.py`: shot 7a set. `render(cam, f, t, climb)`, `render_front(...)`, `sun_xy`, `sun_amount`, `last_alpha`, `thickness`.
- `tr2/shots/s8_star.py`, `tr2/shots/s7a_clouds.py`: the two cuts (given to me by the director at 02:19).
- Test scripts and scratch stills: `build/cache/sky/test/` (not the director's scratchpad).

## How things work (for whoever resumes)
- Sky coordinates: default screen is SX 0..1920, SY 0..1080, horizon SY 520, sky goes up to SY -2800. `sky.view(cam)` maps sky to screen: dolly and crane (cam.cx, cam.cy) move it by 0.02 and 0.05; `cam.tilt` and `cam.pan` scroll it 1:1 (director's engine change at 02:14).
- Gradient: palette stops, pushed up and down by a cached brush stroke canvas so bands have brushed borders. Sun or moon glow from the palette sun direction (`sun_pos`).
- Clouds: `LOOKS` table per look (dusk, night, dawn, morning), layers far to near plus a high layer for tilts. Sprites are mirrored when the sun is on the left. Tones come from the palette key, amb, rim.
- Stars: catalogue in polar coordinates round `POLE`, about 106k stars with a milky way band; `star_turn` rotates all of it. Glints only on the brightest.
- Vista: sky canvas, then a perspective cloud floor with two gaps (land, river, lake, glint mask), then about 25 rows of cumulus sprites from the horizon to the lens (depth z, row at y = HZ + F / z), grouped into banks by `mass_at` with shadowed valleys (`shade_at`), mountains (small height fields drawn front to back) slotted between rows. All layers cached in `build/cache/sky/vista_layers_v<N>.npz`; bump `vista.VERSION` after any change to the bake.
- Cloudset: altitude A = climb * TRAVEL. Six parallax depths; the last deck is pre composed into field strips (`cloudset_deck_v3_<layer>.npz`; rename the cache key after changing `_items`). Clouds above the camera show dark undersides, below it lit tops; light climbs blue grey, rose, gold.

## Log
- 01:55 read COMMON, C-sky, PROMPT, core.py, palette.py.
- 02:00 first cloud sprites: union of spheres height field, two scale lambert plus marched transmission, brushed tone borders. First try was a flat blob with a lit rim; fixed by giving child puffs a z offset so lumps show on the face.
- 02:04 sky.py first stills (dusk, night). Night had far too many glints; star brightness now u^9, glints for the top 0.3 percent.
- 02:08 vista first pass as a voxel height field: read as purple rock. Director agreed. Rebuilt as painted sprite rows in perspective (02:14).
- 02:10 to 02:17 director notes on dusk: melting curved bases with wisps, no dark fringe, smoothed alpha, warm mauve shadows, far golden bands near the horizon (`band` sprite), bigger sun glow on the left. Night: wisps reworked into combed cirrus.
- 02:16 engine change: cam.tilt, cam.pan in `sky.view`; sky extended to SY -2800; dawn look has high rose cirrus.
- 02:19 director approved the vista direction; phase 2: I own s8_star and s7a_clouds.
- 02:20 to 02:27 vista: banks and valleys, shadows raking left, calm far field, soft near tops, deeper upper sky, gaps moved into valleys, river glints under the rows, disk cache of layers.
- 02:24 s8_star.py: climb thread, ignition flash, star head (glow stack, glints, streak, lens ghosts), aging trail, falling glitter, light pool on the clouds, push in and pan.
- 02:28 to 02:33 cloudset.py: first version was mush (near layers too dense and too big); thinned, front layers only wisps, deck strips for speed (about 1.3 s a frame).
- 02:34 s7a_clouds.py: rocket (D's rocket.draw, side view), cel flame, cel smoke lobes left behind in the stream, spray at the break, god rays through fx.light_shafts.
- 02:36 contact sheets of both cuts in build/review/.

## Known weak points (update as fixed)
- Cumulus sprites are still a little chunky and popcorn like in close view; tone shapes are blobs at scales above 1.
- sky.render costs about 0.8 s (budget 0.8).
- s8: trail is thin and even; light pool on the clouds is too faint; cloud sea does not move.
- s7a: rocket sits low so the smoke column is off the bottom; whiteout veils ghost the rocket.
