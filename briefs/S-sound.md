# S: the score and the sound design

Read `ROOT/briefs/COMMON.md` first (ROOT = `/Users/jaypatel/Downloads/code-films-5-2`; the hard rules apply to sound too: no samples, no impulse response files, no borrowed melodies, nothing downloaded, numpy and scipy and numba only), then `ROOT/PROMPT.md` (the SOUND section and the "Fable 5.2 brief" sound line), then `ROOT/cues.json` and `ROOT/tr2/timeline.py`. Your name for notes and cache: `sound`.

You own `ROOT/sound/` (your code: entry script `ROOT/sound/score.py`, modules beside it, a short `README.md` that says what every layer is and how it is synthesised) and `ROOT/notes/sound.md`. You make every sound of the film in code. The Fable 5.1 film had a thin score; this one needs a real one. It is one of the four things people will compare (animation, picture, motion, sound).

## Timing
Everything is placed from `ROOT/cues.json` (74.0 s, 24 fps). READ THE CUES AT RUN TIME, never hard code a time: the director will shift cues by a few frames as the animation is finalised, and you must be able to re render in one command: `python3 ROOT/sound/score.py` writes
- `ROOT/build/audio/tiny-rocket-5-2.wav`: 48 kHz, stereo, 24 bit PCM, exactly 74.000 s (3,552,000 samples).
Every effect lands within one frame (42 ms) of its cue. A full render should take under about 4 minutes; cache instrument notes if needed.

## The score
- One original motif of 5 to 9 notes, simple enough to hum after one hearing, with a clear rising shape (a small thing reaching up). Write it down in notes/sound.md as note names and rhythm. Check it is not a known tune.
- Built up voice by voice across the film, so the climax feels earned:
  - Shot 1 (0 to 6): no music. Evening crickets, soft breeze in grass, the string creak and the tug. Maybe one held high string harmonic entering at the end.
  - Shot 2 (`motif_first`): the motif on a lone felt piano (soft hammers, a little key noise, room), unhurried, with space. A second phrase answers.
  - Shot 3 (night): the motif fragment on a music box or celesta over a very soft string pad; a glassy shimmer for the shooting star.
  - Shot 4 (predawn): a solo woodwind (a breathy flute or clarinet) takes a variation over pizzicato or a low drone; a tiny rise at the nod.
  - Shot 5: ticking, held breath tension (a soft clock like pulse made from the fuse and a muted string ostinato rising by half steps), the fuse fizz travelling across the stereo field in 5b, then at `stillness` TRUE silence: everything cuts dead, digital zero or near it, no reverb tail (gate the tails). The audience must feel the hole. At `one_eye_opens` one tiny sound (an eyelid, a cricket).
  - Shot 6 (`ignition`): the roar (see foley), and the orchestra enters under it: low strings and timpani like hits at `liftoff`, a rising string run as the camera tilts up, brass like swells.
  - Shot 7: the full orchestral swell: strings in octaves, horns, harp or piano arpeggios, choir pad ("ah") entering, a cymbal like swell into `break_into_open_sky`, where the texture opens (high strings, choir, air).
  - Shot 8 (`star_ignites`): the motif returns at FULL scale: piano in octaves plus strings and choir carrying the melody, bass and timpani underneath, the harmony finally resolving. This is the emotional peak and the loudest music. Let it ring and thin out as the star crosses.
  - Shot 9: quiet. Morning birds, breeze, the parachute creak, the soft touchdown. The motif once more, very simply, on the felt piano, slower. At `brush_stroke` exactly ONE soft piano note (the last note of the motif, the one it has been reaching for). The title: a brush on paper, the hanko a soft wooden thump, then the final chord dies before `fade_to_black`.
- Instruments are synthesised, and must sound like instruments, not beeps: the felt piano (inharmonic partials with per partial decay, hammer noise, sympathetic resonance, soft low pass, slight detune of unison strings), strings (many detuned bowed voices per section with vibrato that starts late, bow noise, slow attacks, body formants), woodwind (breath noise plus harmonics, pitch scoop), a choir pad (formant filtered voices, several vowels), music box (struck bars, bright fast decay), timpani and low drum (pitched membrane modes with a noise attack), a cymbal like swell (dense inharmonic noise bands). Humanise timing and velocity a little.
- A reverb you build yourself (a feedback delay network or a good Schroeder or Moorer design with early reflections, damping and stereo decorrelation): a warm hall for the score, a smaller outdoor space for foley.

## Foley and ambience (every action in cues.json gets a sound)
Crickets (many, each its own chirp rate and position), evening breeze through grass, night insects and a far owl like hoot, dawn birds (each call different), the knot (creak of fibre under tension, the tug snap), paws in grass (soft, each step different), ear flicks and blinks as tiny intimate sounds, tail swish, the paper map rustle, the yawn (no voice: a breath), paws on tin and paper, the goggles snap (elastic and a small thock), the swallow, the match strike (scratch, flare, burn), the fuse (a fizzing hiss with crackle that pans left to right across 5b and rises in pitch as it nears), the ignition (a crack, then a roar with sub bass, crackle, rattling tin and fluttering paper; it must feel huge after the silence), the shudder, the lift off whoosh rising away, birds bursting from the tree (many wing flaps), wind (rushing, with fins rattling), the break into open sky (the wind drops to a high thin air), the star (a shimmering rising tone), the parachute (paper umbrella creak and flutter), the touchdown (soft thud, legs creak), the climb out, the brush stroke on paper, the hanko, the tail curl.
Stereo with movement: place every source, pan the moving ones (fuse, rocket, birds, shooting star), keep the bass centred.

## Mix and master
Integrated loudness about -14 LUFS (measure with ffmpeg `ebur128` and print it), true peak at most -1 dBTP, loudness range around 14 to 18 LU but quiet scenes still clearly audible on a phone speaker (no content only below 150 Hz, gentle upward compression of the quiet scenes), nothing harsh above 8 kHz, no clipping, no DC. The silence at `stillness` is the only true silence. You cannot listen, so verify by measurement: per shot RMS and peak table, spectrogram images saved to `ROOT/build/review/sound-*.png` and looked at with the Read tool, onset times of every effect checked against the cues.

## Checkpoints
1. Temp score, within 75 minutes of your start: the whole 74 s with the motif, the main sections and the main effects in place, even if instruments are rough, at `ROOT/build/audio/tiny-rocket-5-2.wav`. End your turn with a report: path, loudness numbers, the motif, what is rough.
2. Final (after the director's notes): everything polished, measured, documented in `ROOT/sound/README.md`.
