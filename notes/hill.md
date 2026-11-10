# hill (builder B): log

Start: Tue Nov 10 01:55 IST 2026.

## Files
- `tr2/paint.py`: the gouache stroke painter (numba kernel). API at the top of the file.
- `tr2/hillset_paint.py`: composes and paints each layer (mountains, valley and village, hill, tree).
- `tr2/hillset.py`: public API, cache, per frame rendering (lights, sway, front grass, band, flatten, fire).
- Cache: `build/cache/hill/v1_<tod>_<name>.npy` (float16), about 200 MB per time of day, 1.0 GB for five.
- Rebuild: `python3 tr2/hillset.py build all` (about 22 s per time of day, 2 threads).
- Stills: `python3 tr2/hillset.py still <tod> [force]` writes `build/review/hill-<tod>.png` and `-crop.png`.

## 01:58 Read briefs
COMMON, B-hill, PROMPT, core.py, palette.py read. Looked at ref51/still-title.jpg.

## 02:00 paint.py
Stroke kernel in numba: curved 4 segment strokes along a direction field, bristle profile, dry brush
breakup toward the stroke end, paper tooth, pooled rim, colour jitter, `keep` (a stroke does not cross
a colour edge), optional region labels, brushy alpha (`spill`). `strokes()` paints explicit strokes
(blades, leaves, flowers). 2880x1620 paints in about 2.7 s.

## 02:09 First dusk still
All layers painted in 20 s. Saw: spiky foliage, flat tan valley, purple blob stones, river invisible.

## 02:11 to 02:20 Dusk iterations
- Foliage: rounder clumps, smaller round leaf dabs, softer tone steps, more mid tone, greener light.
- Valley: wider river with sky reflection, darker banks, less haze, field patchwork, long clump shadows.
- Mountains: spurs running down from ridges, base haze so ranges dissolve at the horizon.
- Foreground grass (p 1.6) drawn per frame, grass_band at the ground line.
- Rocket shadow split out of the painted hill into its own mask (`rocket_shadow=` in render_back).
- Night, predawn, morning built and checked: LOOK table (direct light, saturation) per time of day.

## 02:21 Director notes (approved; five fixes)
1. Night and predawn darker and bluer: LOOK direct and sat lowered, GAIN table, RIM table (cool moon
   rim on crest and clump tops at night, faint mauve rose rim on the right before dawn), houses dimmed,
   valley clumps follow the hour. Night grass now about 0.10 to 0.20.
2. Dusk valley: haze cut, field pattern kept in the distance, 380 tree clumps with long shadows, mist
   in streaks and along the river, river brighter with glints and dark banks. River is in all five.
3. `launch` builds. `render_back(..., fire=(X, Y, radius, amount))`: warm pool on the grass (elliptic,
   masked by the hill) and on canopy undersides and the trunk side (cached `treefire` map).
4. `flatten=(X_centre, radius, amount)`: render_back redraws about 2000 hill blades lying outward
   around the centre (fall off exp(-(d/radius)^2), shudder on ones); grass_band and render_front lean
   their blades away from X_centre. Sway confirmed (mean abs diff f0 to f12: tree 0.013, front 0.052).
   Timing at 2 threads, quiet machine: render_back 0.40 s (0.65 with fire and flatten), band 0.04 s,
   front 0.20 s.
5. Camera tilt and pan: every screen position goes through cam.xf or cam.pt (layers, window lights,
   river reflections, front grass, band, fire pool, flatten blades). Tested with tilt=60, pan=-40.

## 02:39 State
Stills in build/review: hill-dusk.png, hill-dusk-crop.png, hill-night, hill-predawn, hill-launch,
hill-morning (each with -crop), hill-launch-fire.png (fire and flatten on), hill-sway.jpg.

## 05:43 to 05:56 Two cuts (director handed over s9a_home and s5b_fuse)
- `tr2/shots/s9a_home.py`: descent under the paper umbrella (pendulum, rocket lags by 0.13 s),
  touchdown squash, hop, settle, strings slip free, umbrella folds and drops right, dust puffs,
  torn grass, flatten, nozzle wisps. `pose(t)` is a pure function of time, so negative fl works.
  Exports `draw_parachute_rest(pic, cam, t, f)`; the fall ends exactly on the rest pose (REST dict).
  One frame cost about 7.4 s alone at 2 threads (over the 6 s guide).
- `tr2/shots/s5b_fuse.py`: macro at zoom 3.5. Spark steps hold 2 or 3 frames (`_step`), fuse pale
  ahead and charred behind with embers, ragged star head redrawn every frame, three overlapping
  `fx.sparks_layer` sprays, smoke thread, warm tint on blades near the spark, two blurred foreground
  planes (p 1.7 and 2.6), rack focus 1.55 to 2.2 s (whole back picture blurred until then).
  Not timed per frame. Not checked in motion (sheets only).

## 06:25 to 06:33 Director round on s9a
- Umbrella rescaled: LR 133 (canopy about 250 set px wide, 1.3 x the fin span), L_STR 150, HANDLE 62.
  Rest pose: open 0.08, hub at x 1886, 5 px above the ground line, `_rest_tufts` lays grass over
  its lower half (also used at the end of s9a).
- Landing: squash peaks at 1.45 for two frames, hop 27 px, flatten radius 300 at amount 1.0 then
  springing back by 0.68 s, 11 dust puffs, 20 floating seeds, camera thump, bigger umbrella wobble.
- Opening: tilt starts at 380 (was 560) so the rocket sits high in frame during the dissolve;
  swing amplitude raised to about 0.2 rad at the start.
- Checked: sheet-s9a_home.jpg, f1530.png (s9b with the rest umbrella), fl -16 to 11, fl 66 to 79.

## Known weak points, next steps
- Dusk river is pale gold and narrow through the village; could be wider and more orange.
- A straight band of mist sits on the horizon line; break it with far tree lines or low hills.
- Foliage tone steps are still a little posterised; sway moves whole sub layers by 2 to 4 px.
- Flatten is a Gaussian press from the centre, not a travelling ring: animate radius and amount.
- Houses are soft above 1.5x zoom (valley layer is 1.5x set resolution).
- `tod` must be a palette name; blended times of day are not cached. `sunrise` is not built.
- Not looked at: `defocus` (code path only), grass_band close up under feet.
- Coming cuts: s5b_fuse (ground level track along the fuse) and s9a_home (morning landing).
