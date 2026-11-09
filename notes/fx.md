# fx (builder D): rocket and effects log

Start: Tue Nov 10 01:54 IST 2026. Checkpoint 1 due 02:40 IST.

## Files I own
- `tr2/rocket.py`: the rocket cel prop (side, low34, below views), `State`, `draw`, `draw_glass`.
- `tr2/fx.py`: `SmokeSim` (2D fluid plus lobe tracers), `lobe_smoke`, `smoke_layers`, `sparks_layer`, `bloom`, `glow_at`, `ground_light`, `light_shafts`.
- `tr2/fx_look.py`: `sheet` mode renders `build/review/rocket-sheet.png`.
- `tr2/fx_launch_look.py`: `launch-look.png` (default) and `evolve` (contact sheet `launch-evolve.jpg`). Holds the placeholder hill and the emitter schedule `emit_at(t, k, noz)`.

## Log
- 01:55 read COMMON, brief D, PROMPT, core, palette.
- 02:00 rocket.py first pass. The rocket is a stack of bodies of revolution (nozzle bucket, tin skirt, paper body, saucer, teacup, kettle, knob) projected with one function `P(h, r, angle)`: rings arch more the higher they are above the eye (e0 + e1*h/500), plus a perspective taper k and a yaw per view. Attachments (big sail fin on a bamboo spar at -90 deg, two paper fins, cup handle at 100 deg, spout antenna with pennant, wind up key, three lashed bamboo legs) sit at body angles and sort by depth. Shadow tone is a surface patch on the side away from the sun, rim light a thin patch on the sun side, fire light a hard warm tone on undersides (cup bowl, saucer, skirt, nozzle, legs, fin hems). No clip masks and no alpha fills, so a full rocket draws in about 0.7 s.
- 02:03 looked at the sheet: design reads (teacup, saucer, kettle). Added the saucer, crooked paper body, bigger stars, height dependent ring arch.
- 02:05 fx.py first pass: fluid solver (numba), density thresholded into tones, sparks, bloom. First launch still: smoke was two huge marbled walls. Bad.
- 02:08 lower buoyancy, ground ring puffs (two fronts racing outward along the ground), smoke low and wide. Tones still marbled.
- 02:09 director note: smoke must be one low mass of cauliflower lobes, tones following the lobes, orange to gold band on the inner faces, warm pool on the grass, flame with a clear shape.
- 02:10 rewrote the smoke look: the sim now carries LOBE TRACERS (spawned at the jet splash and at the ring fronts, advected by the fluid, heavy ones lag and stay near the fire). `lobe_smoke` draws each as a dome in painter's order (lower lobes in front) through a noise warp, and shades the dome normal: off white top (sky), grey violet body, deep violet crescent underneath, orange then gold on the faces toward the fire (with distance falloff). Crisp edges, 4 tones plus the fire pair. This reads as cel effects animation.
- 02:12 flame: analytic jet tongue from the nozzle joined to the simulated flame field F, three flat tones (white core, yellow, orange). Layer order: back smoke, rocket, flame, front smoke (low lobes swallow the legs), jet, sparks, glow, bloom.
- 02:14 contact sheet of 8 times in the first 1.25 s (`launch-evolve.jpg`).

## Director notes received
- Rocket design approved: keep it exactly.
- Engine: `core.Camera` has `tilt` and `pan` (px, camera rotation, no parallax). Use tilt for the long move in shot 6 plus a small cy rise.
- Fireflies are no longer mine (`tr2/fireflies.py` by the director). Birds in `tr2/birds.py`.
- `tr2/shots/common.py`: `hilltop(cam, tod, f, t, mid=...)`, `draw_rocket`, `draw_panda`, `draw_map_prop`, `contact_shadow`.
- I own shots s5b_fuse, s6_lift, s9a_home. Read `briefs/SHOTS.md` after the checkpoint.

## Known weak points and next steps
- Lobe smoke is drawn in screen space from tracers; the lobe noise warp boils on twos but is not attached to the flow. Check in motion.
- Detached puffs are single round beads: make them small clusters or wisps.
- Flame jet is a blob: needs licking tongues and a flicker cycle on ones.
- Smoke render cost about 1.6 s per frame at 1080p (the old density tone path still runs under the lobes: remove it, keep only F for the flame).
- Not built yet: full launch sim cache (column under the climbing rocket, hanging trail), fuse spark head, match flare, star trail and lens flare for shot 8, parachute, star map, match, cockpit back and front layers, soot and legs fold checks, hatch.
