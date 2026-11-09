# TINY ROCKET (Fable 5.2): production log

All times are IST, read from `date` on the machine that made the film. Nothing here is estimated.

## Who made it
Every line of picture and sound code was written by Fable 5.2 (model ID `claude-fable-5-2`): one director session (`fable film`) and builder agents it started, all on the same model. The director confirmed its model ID from its own system prompt before any work began. Nobody read the Fable 5.1 code; the only things taken from the 5.1 film were its prompt text and frames of the finished film, to know what to beat.

## Log
- **01:45:17** Task received. Start of the clock. Read the brief, the original PROMPT.md and its four briefs. Extracted a contact sheet and three stills from the 5.1 film into `build/ref51/` (frames only, no code).
- **01:50** Wrote the engine core from scratch: `tr2/core.py` (buffers, value noise, timing helpers, splines, the supersampled cel rasteriser with clip masks, crescent shadows and pressure ink strokes, compositing, blur, a numba bilinear warp, the multiplane camera) and `tr2/palette.py` (six times of day, cast colours, cel lighting). Tested: a lit, inked shape over a dusk sky renders in about 0.3 s.
- **01:54** Wrote `PROMPT.md` (original story text plus the "Fable 5.2 brief"), the common brief and four builder briefs in `briefs/`. Timeline set: 74.0 s, 16 cuts (`tr2/timeline.py`, `cues.json`).
- **01:55** Started four builders in parallel, all Fable 5.2: A panda (character rig), B hill (gouache painter and the hilltop set), C sky (sky, clouds, the two cloudscapes), D fx (the rocket, fluid smoke and fire, sparks, light). Checkpoint 1 for each is 02:40.
