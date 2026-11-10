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

- 02:16 flame jet got a bell foot and a licking width; strays spawn as clusters of three; the old density tone path moved to `_density_smoke` (fallback only). Frame composite on the look still: 2.7 s (smoke and flame about 1.4 s, sparks cel 0.4 s, light passes the rest), on 2 threads.
- 02:17 tested rocket states in dusk, night (window glow), morning (soot, squash) and sunrise (below view, legs folded, rattle, rot): all draw, design holds. One rocket at s=1 costs 0.18 s.
- 02:18 checkpoint 1 stills saved: `build/review/launch-look.png`, `rocket-sheet.png`, `launch-evolve.jpg`. Reported to the director.

## Phase 2 (from 02:18)
- 02:19 scope changed by the director: B takes s5b_fuse and s9a_home, C the star of shot 8. Mine: cockpit layers first (A is blocked), then s6_lift. Keep `fx.sparks_layer` general for B. Next checkpoint 03:45: s6 contact sheet (`python3 render.py --sheet s6_lift 12`) and a half size clip (`--shot s6_lift --scale 0.5 --jobs 2`).
- 02:24 cockpit layers in rocket.py: `draw_cockpit_back`, `cockpit_mask`, `draw_cockpit_front`, `draw_cockpit_inside` (see the comment block "cockpit close ups" in rocket.py). Stills: `build/review/cockpit-look.png`, `cockpit-warm.png`, `cockpit-inside.png` via `python3 tr2/fx_look.py cockpit`. Told the director (SendMessage) so A can start.
- 02:26 `State.hatch_open` now draws: the kettle (with knob and spout) tips back 70 degrees on a hinge at the rim away from the handle (`_Proj.pa`, `_Proj.pv` swing a part about a pivot). Nothing else of the cel covers the space above the rim. Extra: `rocket.draw_hatch_back(...)` paints the dark mouth of the cup and its far rim on a cel that goes BEHIND the panda (optional). `rocket.mix_views(a, b, t)` blends two views (pass the dict as `view`).
- 02:31 `fx.sparks_layer` fixed for animation: birth times are fixed (new `dur`), and `xf=(s, tx, ty)` runs the sparks in SET px and draws them through the camera.
- 02:33 `tr2/fx_smoke.py`: smoke version 2 for motion. `Lobes` (spawn, step, alive, prune), `radius`, `render` (soft silhouette, wispy edge bites, haze, warp anchored to the set, fixed stack order: no popping, no boil), `flame` (ground splash field plus a jet tongue with a bell foot or a free tail).
- 02:36 `tr2/shots/s6_lift.py`: choreography (`height`, `motion`, `camera`), `build_sim` (cache `build/cache/fx/s6/v3_fNNN.npz`: lobes, flame field F, t), `render`, `smoketest`. Camera: zoom 1.75 close and low, tilt follows with a lag, zoom eases to 1.5. Ignition: white frame at fl 7, two blown out frames, shake 36 px decaying.
- 02:38 first full sheet of s6 worked end to end. Child lobes ("kids") on every big lobe give the cauliflower look; the crest grass band goes behind the smoke once it burns.
- 02:40 to 05:42: cut off by a usage limit. Director's finishing list on return: soft painted near clouds, a continuous trail column, softer ground smoke silhouette plus haze, one white flash frame, `render_back(fire=..., rocket_shadow=...)` and the travelling flatten ring.
- 05:44 done, sim cache is now `v7_fNNN.npz` (older v3 to v6 files in `build/cache/fx/s6/` are stale and can be deleted):
  - Trail: lobes are born 0.12 s "early" (already swollen), hang where they are made (150 px/s throw, low follow), sizes 22 to 92 px times the rocket scale, grow 2.4, three kids each: a continuous column that tapers to the rocket and widens below.
  - Near clouds: the lobe bank is rendered at half size, blurred (sigma 5 at half size), given a rose band along every lower edge and a brushed texture, then scaled up. The rocket's flame glows through as it enters.
  - Ignition: fl 7 is the white frame, fl 8 and 9 are over exposed with strong bloom (changed from pure white, not looked at since), shake 36 px decaying.
  - Ground smoke: soft=4.5 px edge, haze 0.36. `rocket_shadow` fades from 1 to 0 between ignition and lift off.
  - Looked at 12 consecutive frames (fl 14 to 25) as one tiled still: lobes grow and drift smoothly, no popping seen. The clip itself was not watched.
- Outputs: `build/review/sheet-s6_lift.jpg`, `build/seg/s6_lift.mp4` (half size clip).

## Still weak in s6 (not done)
- No torn leaves at ignition or as it passes the tree; the canopy only sways harder (`wind`).
- The orange lit kids make a busy row of small orange petals along the front smoke line.
- The front smoke is the same layer drawn again below a wandering line, not a separate nearer layer.
- My soft clouds and builder C's painted sky clouds are two different hands; mine are plainer.
- The view blend low34 to below changes the rocket's proportions while it climbs; watch it in motion.
- Frame cost about 4.5 to 5 s on one core.

## Manager notes for s6 (relayed 02:22)
1. Smoke must BILLOW and be tested in motion first; soft outer silhouette (2 to 4 px), wispy translucent edges, faint outer haze; tone boundaries inside stay crisp; orange inside, cool sky blue on top.
2. Ignition is a low angle hero shot, closer: rocket about 800 px tall, big soft foreground grass.
3. The ignition hits like a wall: one white flash frame, two frames of blown out bloom, heavy shake that decays, then the roll of smoke.
Director smoke notes: vary lobe size (few big, many small), taper the ends into small puffs and wisps, a second thinner faster ground layer racing ahead.

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
