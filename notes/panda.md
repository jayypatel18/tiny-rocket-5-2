# panda (builder A) log

Start: Tue Nov 10 01:55 IST 2026. Checkpoint 1 at 02:40 IST.

## Files
- `tr2/panda.py`: the rig and the drawing (Pose, draw, blend, bbox, POSES).
- `tr2/panda_sheet.py`: review stills. `python3 /abs/tr2/panda_sheet.py sheet|closeup|all`.
- Stills: `build/review/panda-sheet.png`, `build/review/panda-closeup.png`. JPEG previews in `build/cache/panda/`.

## How it is built
- Head and torso are ellipsoid volumes (`_Ball`): features are patches in (lon, lat) on the surface, rotated by yaw, pitch, roll, projected; points that go behind are pulled to the silhouette so patches wrap. Nose, mouth, eyes, muzzle sit on small boards glued to the surface (`ball.frame`) with a minimum width so they never become slivers.
- Each part (far ears, tail, legs, far arm, torso, head, near arms) is a `_Part`: its own supersampled PIL image. Fills first, then ONE shadow pass (form minus form moved toward the light, plus designed cast shapes via `part.shadow`), a thin rim on the sun side, then all ink lines, then composite on the caller's Cel.
- Colour: `_Env.lit` = local * key, clamped, 14 percent desaturated. White uses a softened key so the mask stays warm white. Fur local colour is my own (0.83, 0.50, 0.27), dark limbs (0.37, 0.215, 0.185), both in `LOCAL` in panda.py, not the palette ones. Shadow = lit * (amb/key) + a small violet add.
- Line boil: `drawing` seeds one smooth warp field used by every path of that drawing.

## Log
- 01:55 read briefs, core, palette, looked at the 5.1 close up.
- 02:07 first version of panda.py and the sheet script rendered first try. Saw: neon fur, far eye a sliver in three quarter view, giant highlights, ear glitches, sausage arms, goggles strap bug (band wrapped over the top of the head: lon range must stay within about 104 degrees).
- 02:10 round 2: eyes moved in, ears rebuilt, goggles fixed, muzzle protrudes in profile.
- 02:10 director notes (colour, white mask, body volume, designed shadows, eyes, cheek ruff, sleeper and blanket, close up framing, paw close up after the checkpoint).
- 02:15 round 3: new lit colour, warm white, tear marks from inner eye corner to mouth corner, eyes on boards (both match in three quarter view, two tone iris, lash ticks, lower lid), cheek ruff tufts, pear torso with chest ruff and tufted belly, elbows, mitten paws with 4 toe bumps, bigger feet with toes, head and arm cast shadows on the torso, quilt (`blanket`), close up at s 7.2 with taut string.
- 02:17 round 4: softer ear hollows, fewer bolder cheek tufts, smaller chest ruff, white mask lower on the side of the head.

## To do after checkpoint 1
- `draw_paw_closeup(cel, x, y, s, tod, drawing, paint, press)` for shot 9c.
- Remaining poses: step back walk keys, read map, wake and stretch, climb (3), cockpit bust, goggles down move, swallow, nod, strike match, paint with paw, climb out sooty, tail curl.
- Follow through helpers (tail and ear spring chain from motion history), smear.
- Remove the dead old eye loop in `_draw_head` (the `for sd, op in ():` block).
- Speed: sheet with 16 figures renders in about 4 s, the close up in about 1.6 s (ss 3, full frame cel).
