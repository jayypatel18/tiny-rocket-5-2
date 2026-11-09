# C: the sky, the clouds and the two cloudscapes (shots 7 and 8)

Read `ROOT/briefs/COMMON.md` first (ROOT = `/Users/jaypatel/Downloads/code-films-5-2`), then `ROOT/PROMPT.md`, `ROOT/tr2/core.py`, `ROOT/tr2/palette.py`. Your name for notes and cache: `sky`.

You own `ROOT/tr2/sky.py`, `ROOT/tr2/cloudset.py` (shot 7), `ROOT/tr2/vista.py` (shot 8), and `ROOT/notes/sky.md`. Agent B is writing a gouache stroke painter at `ROOT/tr2/paint.py` right now; when it exists (check after 02:20), use it for your painted layers. Until then write your own small brush code inside your modules so you are never blocked.

## `sky.py`: the living sky behind every hilltop shot
```python
from tr2 import sky
pic = sky.render(tod, cam, f, t, stars=None, clouds=1.0, star_turn=0.0, shooting=None, sun=True, fade_star=None)
#   Returns a float RGB 1920x1080 picture of the sky for this frame, seen through core.Camera `cam`
#   (sky parallax 0.02 for pans; but it must TILT correctly: when cam.cy rises by thousands of px in shot 6
#   the sky scrolls so the horizon glow sinks and the zenith colour arrives; define a tall sky, about 3 screens).
#   tod: a palette name or a blended dict (palette.blend_tod). t: seconds, drives cloud drift and twinkle.
```
- A painted gradient from the palette sky stops with brush variation (not a computer gradient), a sun or moon glow in the palette sun direction, the low rose band at predawn.
- Clouds as painted masses with real light: a lit edge toward the sun, a body tone, a violet shadow underside, wisps, at 3 depths that drift at different speeds (slow, visible over a 7 s shot) and evolve slightly. Each cloud different. Golden hour clouds at dusk, thin silver at night, rose lit at predawn, fresh white cumulus in the morning.
- Stars: thousands, varied in size, colour and brightness, a faint milky way band, gentle twinkle, `star_turn` (radians) rotates the field around a pole high on the right (shot 3: the sky visibly turns). First stars at dusk (`stars=0.15`), dense at night (1.0), fading at predawn.
- `shooting=(t0, x0, y0, x1, y1)`: a shooting star with a thin luminous tail (shot 3). `fade_star=(x, y, amount)`: the one star still fading in the morning sky (shot 9).
- Cost: under 0.8 s a frame. Cache the painted cloud sprites and the star catalogue.

## `cloudset.py`: shot 7a, through the clouds (sunrise)
A vertical journey. Tall painted cloud layers (at least 5 depths, parallax from 0.1 to 2.0) that stream downward past the camera as the rocket climbs: dark blue grey undersides below, then rose, then blazing gold tops as we break out above, with the sun low on the right, god rays through gaps, and a near cloud layer that whips past the lens. `cloudset.render(cam, f, t, climb)` returns the back picture, `cloudset.render_front(...)` an RGBA layer of near clouds and wisps that pass in front of the rocket. `climb` (0..1) is the journey; the director animates it. The last frames must be open sky above a floor of cloud, ready to cut to shot 8.

## `vista.py`: shot 8, the climax, the widest most awe inspiring image in the film
Seen from very far and very high at sunrise: the curve of the world along the horizon with a thin bright atmosphere line, the sun just risen at lower right pouring gold, a SEA of clouds stretching to the horizon in perspective (rows of cumulus tops getting smaller and hazier, lit gold on the sun side, violet blue in shadow, long shadows across the cloud floor), mountain ranges rising through the cloud sea as islands with snow catching pink light, rivers and a lake catching the light through gaps, deep blue above grading to stars at the top. At least 5 parallax layers so a slow push in and drift has depth. `vista.render(cam, f, t)` returns the picture WITHOUT the shooting star (agent D draws the star and trail; leave the upper two thirds of the sky open and calm for its arc from upper left to right of centre). The 5.1 version is `ROOT/build/ref51/still-climax.jpg`: flat brown lumps and cotton balls. Yours must look like a painted theatrical background: scale, perspective, light.

## Checkpoint 1: 02:40 IST (hard)
`ROOT/build/review/sky-dusk.png` (the dusk sky at the default camera, with clouds and first stars), `sky-night.png` (night, stars at 1.0, milky way, one shooting star), and a first pass `vista-look.png` (the shot 8 view, even if rough). All 1920x1080. End your turn with the report at 02:40 even if rough. After the director's notes: all times of day, the tilt for shot 6, cloudset, vista to final.
