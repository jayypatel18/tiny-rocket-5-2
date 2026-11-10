# Tiny Rocket (Fable 5.2)

A 74 second anime short where **every frame and every sound is code written by Fable 5.2**. No image models, no video models, no stock footage, no samples, no font files, no downloaded assets. Python draws each of the 1,776 frames and synthesises each sample of the soundtrack.

**Watch it:** https://jayypatel18.github.io/tiny-rocket-5-2/

All summer, a small red panda has been building a rocket out of paper and scrap. Tonight it finds out whether a homemade dream can reach the sky.

## What it is

This is the second version of TINY ROCKET. The first was made by Fable 5.1 and lives in [jayypatel18/code-films](https://github.com/jayypatel18/code-films). This one tells the same story from the same prompt, with a harder brief on top, and was written from an empty folder by Fable 5.2 without reading the 5.1 code. The full prompt, including the extra "Fable 5.2 brief", is in [PROMPT.md](PROMPT.md).

## How it was made

One Fable 5.2 session directed and wrote the engine. It started builder agents, all Fable 5.2, and gave each a written brief (all in [briefs/](briefs/)): the character, the painted hilltop, the sky and cloudscapes, the rocket and effects, the sound. The director reviewed stills and sent notes in words, the way a studio works. A second session, the manager Jay assigned for the night, reviewed each checkpoint and sent notes too.

| Part | Where | How |
|---|---|---|
| Engine | `tr2/core.py`, `tr2/palette.py` | Float buffers, value noise, splines, a supersampled cel rasteriser with clip masks, hard shadows and pressure ink lines, a multiplane camera (parallax, dolly, tilt), six times of day in one palette |
| The red panda | `tr2/panda.py`, `tr2/acting.py` | A 2.5D rig: head and body are volumes, face features slide over them as the head turns. Every drawing is rebuilt from a pose; nothing is a rotated sprite. Key poses are timed from the cue sheet and blended on arcs, on ones for fast action and twos elsewhere |
| Painted backgrounds | `tr2/paint.py`, `tr2/hillset.py` | A gouache stroke painter (numba) over composed references: mountains, valley, village, hill, camphor tree, grass, as separate depth layers for five times of day |
| Sky and cloudscapes | `tr2/sky.py`, `tr2/cloudset.py`, `tr2/vista.py` | Painted clouds with light, a turning star field, the climb through the clouds, the sea of clouds at the climax |
| Rocket and effects | `tr2/rocket.py`, `tr2/fx.py` | The rocket is a body of revolution drawn as a cel. Smoke and fire are a 2D fluid simulation rendered in flat cel tones; sparks are particles with motion blur; bloom, glow and light shafts |
| Small things | `tr2/title.py`, `tr2/starmap.py`, `tr2/birds.py`, `tr2/fireflies.py` | The title is brushed with a per pixel brush model (no fonts), the hanko seal, the paw drawn star map, the flock, fireflies and pollen |
| The cuts | `tr2/shots/`, `tr2/timeline.py`, `tr2/film.py` | 16 cuts, one module each; one timeline drives both picture and sound |
| Sound | `sound/` | An original score and all foley, synthesised with numpy and scipy, placed from the same cue sheet |

## How long it took

Measured from the machine clock, logged in [NOTES.md](NOTES.md):

- Started 01:45:17 IST, final files built 11:00:09 IST on 10 November 2026.
- Two usage limit pauses are excluded: 02:39:50 to 05:41:28 (3 h 01 min 38 s) and 08:53:10 to 10:41:13 (1 h 48 min 03 s).
- **Working time: 4 hours 25 minutes**, from an empty folder to the finished film with sound. That includes about 70 minutes of rendering.
- The first complete cut with sound existed after 1 hour 41 minutes of working time; the rest was two rounds of notes and polish.

The film is 1,776 frames at 1920x1080, 24 fps, in 16 cuts. The soundtrack measures -14.0 LUFS integrated with a true peak of -1.3 dBTP and a loudness range of 15 LU.

The production log, with real clock times for every step, is [NOTES.md](NOTES.md).

## Rebuild it

Needs Python 3.12 with numpy, scipy, numba and Pillow, plus ffmpeg.

```
python3 -m tr2.timeline          # writes cues.json
python3 sound/score.py           # the soundtrack -> build/audio/tiny-rocket-5-2.wav
python3 render.py                # every frame, streamed into ffmpeg -> build/seg/full.mp4
python3 package.py all           # master, X file, web file, stills, loudness numbers
```

`python3 render.py --shot s6_lift` renders one cut, `--still 900` one frame. Rendering is deterministic: the same frame number always gives the same picture.

## Credit

The film format and the style of the original prompt come from [@chetaslua](https://x.com/chetaslua), who made the first films like this and published his prompt: [film one](https://x.com/chetaslua/status/2117169796290916732) and [THE LAST PAGE](https://x.com/chetaslua/status/2117224839010459987). The story is the one from the Fable 5.1 version in [jayypatel18/code-films](https://github.com/jayypatel18/code-films).
