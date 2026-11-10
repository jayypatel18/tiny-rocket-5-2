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

- 02:37 to 02:40 polish: trail and glitter, cloud rows drift left on the wind, small cel boxes for rocket, flame and smoke (s7a about 3 s a frame on one thread), smoke redrawn as one cel silhouette with one hard shadow.
- 02:40 to 05:53 cut off by a usage limit. No work in this gap.
- 05:53 to 05:56 director's polish list. s8: head with a 26 px white core, wide bloom, four breathing spikes; trail with white core and gold to rose edge tapering over 60 percent of the arc; stronger light pool, river glint that runs with the star; camera cx travel 120 px, pan 34 px after ignition; ignition is one white frame then two heavy bloom frames. s7a: rocket s 0.96 (about 480 px), lean 0.075 rad, hard shake on ones; climb accelerates and its rate drives the vertical smear; larger spray puffs.
- next round: s8 opens tighter (zoom 1.28, pan and tilt toward the rocket) and eases out by ignition; `_rocket` draws D's rocket tiny as a dark silhouette with a gold rim and a small flame at the head of the thread.

- 06:25 to 06:30 s8: opening tightened on the tiny rocket silhouette; ignition frames 1224 to 1226 checked.
- 08:35 to 08:50 s6_lift.py (D's cut, mine for this round only): `cloud_bank()` now returns painted sprite clouds at three depths (0.55, 1.0, 1.5), relit for the dawn from below with `_dawn_sprite` (cool tops, rose then gold underside; registered through the new `sky_paint.add_variant`); `ascent_clouds()` draws them where D's inline block was, with flame light on the near faces and D's two glow calls for the punch into the bank. The bank is pre composed into one field strip (`_bank_strip`) for speed. Birds: n 22, base_size 32, lighter body, pale rose light. Leaves: 7, larger, slower spin, dark green with a lit edge.
- Timing could NOT be verified: load average was 125 to 160 during the tests (a warm s6 frame took 18 to 24 s, everything was slow).
- Found, not fixed: a dark triangular notch in the high dawn sky (visible near the top of frame 1020). It is a gap between two `wisp` sprites whose streaks are cut by the sprite box. Fix when no render is running: in `sky_paint.cloud_fields`, wisp branch, halve `bend` (0.16 to 0.08) and the tilt factor (0.4 to 0.2), and give wisps a new cache key.

- 08:55 to 09:05 the notch in frame 1020: my first diagnosis (wisp sprites cut by their box, in sky.py) was WRONG. Rendering the sky alone showed no notch. I had added an optional soft wisp variant; it is fully removed again, sky.py and the sky sprites are exactly as before, so no other cut changes. The real cause: a gap of dark sky between two heaps of the upper tier of the s6 cloud bank. Fixed inside `s6_lift.ascent_clouds` only, with a soft mist layer behind the bank (blurred bank alpha). Bank uses cumulus only now. Frame 1020 renders in about 7 s including process start. Birds call left as set (director is redrawing wings in birds.py).

## Known weak points (update as fixed)
- Cumulus sprites are still a little chunky and popcorn like in close view; tone shapes are blobs at scales above 1.
- sky.render costs about 0.8 s (budget 0.8).
- s8: trail is thin and even; light pool on the clouds is too faint; cloud sea does not move.
- s7a: rocket sits low so the smoke column is off the bottom; whiteout veils ghost the rocket.
