# TINY ROCKET (Fable 5.2): production log

All times are IST, read from `date` on the machine that made the film. Nothing here is estimated.

## Who made it
Every line of picture and sound code was written by Fable 5.2 (model ID `claude-fable-5-2`): one director session (`fable film`) and builder agents it started, all on the same model. The director confirmed its model ID from its own system prompt before any work began. Nobody read the Fable 5.1 code; the only things taken from the 5.1 film were its prompt text and frames of the finished film, to know what to beat.

## Log
- **01:45:17** Task received. Start of the clock. Read the brief, the original PROMPT.md and its four briefs. Extracted a contact sheet and three stills from the 5.1 film into `build/ref51/` (frames only, no code).
- **01:50** Wrote the engine core from scratch: `tr2/core.py` (buffers, value noise, timing helpers, splines, the supersampled cel rasteriser with clip masks, crescent shadows and pressure ink strokes, compositing, blur, a numba bilinear warp, the multiplane camera) and `tr2/palette.py` (six times of day, cast colours, cel lighting). Tested: a lit, inked shape over a dusk sky renders in about 0.3 s.
- **01:54** Wrote `PROMPT.md` (original story text plus the "Fable 5.2 brief"), the common brief and four builder briefs in `briefs/`. Timeline set: 74.0 s, 16 cuts (`tr2/timeline.py`, `cues.json`).
- **01:55** Started four builders in parallel, all Fable 5.2: A panda (character rig), B hill (gouache painter and the hilltop set), C sky (sky, clouds, the two cloudscapes), D fx (the rocket, fluid smoke and fire, sparks, light). Checkpoint 1 for each is 02:40.
- **01:58 to 02:07** Director, while the builders work: the brushed title (`tr2/title.py`: a per pixel brush model with bristle streaks and a write on time map; no fonts), the hanko seal, the finishing pass (`tr2/finish.py`), the film and render framework (`tr2/film.py`, `render.py`: workers render frames in parallel and pipe raw RGB into ffmpeg), the bird flock (`tr2/birds.py`), the star map prop (`tr2/starmap.py`), the packaging script and the Pages shell.
- **02:07 to 02:10** First early notes to the builders after peeking at their stills: smoke read as blue marbled blobs (asked for low rolling cauliflower lobes lit from inside), dusk clouds had ruler flat bottoms, the vista read as purple rock instead of a sea of clouds, the panda was a neon orange sticker with a pasted belly.
- **02:10** First composite of a real frame: sky, hill set, rocket and star map together in the wide dusk shot (`build/review/f0240.jpg`). Wrote the storyboard for all 16 cuts (`briefs/SHOTS.md`) and drafts of the wide cuts s2, s3, s4a, s9b, s9c, s9d.
- **02:14** Added a true tilt and pan to the multiplane camera (a rotation shifts every layer equally, a crane move gives parallax) after the first night test showed the far valley hanging in frame during the tilt to the stars. The dusk to night dissolve now happens on the sky, with both cuts sharing one camera path.
