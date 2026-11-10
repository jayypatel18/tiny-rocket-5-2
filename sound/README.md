# The sound of TINY ROCKET (Fable 5.2)

Every sound in the film is computed here from sines and random numbers. There are no samples, no impulse response files, no borrowed melodies and nothing downloaded. Libraries: numpy, scipy, numba.

## Render

    python3 sound/score.py            # about 1 minute; writes the wav, prints every measurement, saves spectrograms
    python3 sound/score.py --quick    # the same without the spectrogram images

Output: `build/audio/tiny-rocket-5-2.wav`, 48 kHz, stereo, 24 bit PCM, exactly the film length (74.000 s, 3,552,000 samples).

All times are read from `cues.json` at run time. Nothing is hard coded: move a cue, run the command again, and the music re-times itself (the phrase lengths are derived from the distances between cues, for example the unit of the first motif is a quarter of the gap from `motif_first` to `admire`, so the top note lands on `admire`).

## Files

| File | What it holds |
|---|---|
| `score.py` | Entry point. Builds foley and music, mixes, masters, writes the wav, prints the tables, checks the silence and every cue. |
| `synth.py` | Every instrument, the noise tools, the numba kernels and the reverb. |
| `music.py` | The score, section by section. |
| `foley.py` | The ambience beds and every spot effect. |
| `mixer.py` | Stems and buses, reverb sends, the master chain, BS.1770 loudness, true peak limiter, wav writer, tables. |

## The motif

Key of D major. Seven notes, three reaches from the same low note, each one higher, the last one arriving:

    A4  B4 | A4  D5 | A4  E5  F#5         degrees 5 6, 5 1, 5 2 3
    s   L  | s   L  | s   s   LL          s = half a unit, L = one unit, LL = two units or more

Where it appears:
1. Shot 2, felt piano alone. The F# lands on B minor and then G major 7, so it does not resolve. An answering phrase (F# E, D B, A D E) ends on the dominant.
2. Shot 3, music box: the first two reaches, then, after the shooting star, the third.
3. Shot 4, flute: the same shape a fourth higher (D E, D G, D A B) over a drone and pizzicato; a small rise A to D at the nod.
4. Shot 6, horns and violas over a bass that climbs F#, G, A, B while the rocket climbs.
5. Shot 7, fragments: a single horn plays the second reach at `wonder`, the horns start the third at `vista`. The harmony goes B flat major 7, C, C over D, with the high A held through all three as a common tone.
6. Shot 8, `star_ignites`: the whole motif, tutti, and for the first time the F# has D major under it. Then A B A an octave up (the first reach again, at the top) and it thins out.
7. Shot 9, felt piano, slower, six notes only. The seventh note, the F#, waits for `brush_stroke` and is played alone. It is the last piano note of the film. The last chord (G with the ninth, then D) is strings only and is gone before `fade_to_black`.

## Instruments (all in `synth.py`)

- **Felt piano** `piano()`: up to 48 stiff string partials (frequency n f0 sqrt(1 + B n^2), B rising with pitch), each with its own decay time; every partial is doubled into a fast "prompt" sound and a slow "aftersound" a cent or so apart, which gives the two stage decay and the slow beating of unison strings; hammer position comb in the spectrum; a velocity dependent low pass standing in for the felt; a soft noise thump for the hammer, a small key click, a damper noise at release. `music.sympathetic()` adds the undamped strings of the D major chord ringing along (a bank of narrow resonators fed by the piano bus).
- **Strings, horn, choir** `ens()` and the kernel `k_ensemble`: each section is 3 to 7 voices; each voice is a harmonic series whose harmonic gains are looked up, every sample, in a fixed body or formant table at the instantaneous frequency. Vibrato therefore moves the harmonics through the resonances, as on a real body or vocal tract. Voices differ in tuning, vibrato rate, depth, delay (vibrato starts late and fades in), slow pitch drift and stereo position. Strings: violin body table (air and wood resonances, bridge hill) scaled for viola, cello and bass; bow noise; tremolo by per voice bow strokes; darker at low dynamics. Horn: a spectrum that opens with loudness. Choir: three vowel tables (women "ah", women "oh", men "ah"), a steeper source spectrum, breath noise.
- **Flute** `flute()`: a monophonic line with legato, a small pitch scoop and a breath chiff on tongued notes, weak upper harmonics that grow with loudness, late vibrato in pitch and level, breath noise that follows the pitch (noise multiplied by the fundamental and its octave) plus air.
- **Music box** `musicbox()`: a clamped steel tooth (partials near 1, 6.27, 12.4 times the fundamental, the high ones dying fast), a tiny pluck tick.
- **Harp and pizzicato** `pluck()`: plucked string partials with pluck position comb, decay faster for higher partials, a finger noise.
- **Timpani and low drum** `timpani()`, `bassdrum()`: membrane modes (kettle: 1, 1.5, 2, 2.44, 2.9; open drum: 1, 1.59, 2.14, 2.30, 2.65), pitch that settles after the stroke, mallet noise. Rolls are many strokes with uneven spacing.
- **Cymbal swell** `cymbal_swell()`: six noise bands with independent slow shimmer plus about 70 inharmonic partials; the high bands arrive later in the swell.
- **Reverb** `Reverb` and `k_fdn`: early reflections (8 taps a side, some crossing), 4 allpass diffusers per channel, an 8 line feedback delay network mixed by a Hadamard matrix with a damping filter in each line, two decorrelated outputs. Two instances: a hall (measured RT60 about 2.1 s) for the score, a small outdoor space (0.44 s) for foley.

## Foley and ambience (all in `foley.py`)

Beds: a field of 16 crickets (each its own pitch, pulse count, chirp rate, rests and place), breeze through grass, night insects with a far owl, a thin predawn air with the first birds, wind in the clouds, thin high air above them, morning birds (six families of call, every call drawn anew) and breeze.

Spot effects, one for every action in `cues.json`: the knot creak (stick and slip impulses through fibre resonances) and the tug, ear flicks, tail swishes, blinks, breaths (band noise with a moving centre, no voice), paws in grass, village lights (tiny far glints), the map rustle, the shooting star shimmer, blanket, yawn, paws on tin, the drop into the cup, the hatch, the goggles (elastic stretch, snap, thock), the swallow, the match (scratch, flare, burn), the fuse (hiss and crackle, panned left to right across shot 5b and rising in pitch as it nears, muffled inside the cup, dying at `fuse_end`), the eyelid, the ignition (a short fizz, the crack, a sub boom, a roar of sub, body, tearing mids and crackle, rattling tin, fluttering paper), the lift off whoosh, leaves, 16 birds with their own wing beats and paths, wind and fin rattle, wisps past the lens, the mist burst at the break into open sky, buffeting in the cup, the star (bloom, shimmer that climbs through D major, falling glitter, all panned along the arc), the parachute (bamboo creaks and paper flutter), touchdown, hatch, soot shake, the brush strokes, the hanko.

## Mix and master (`mixer.py`)

- 16 buses, each with a level and sends to the hall and the outdoor reverb. Bass instruments sit in the centre and everything below 125 Hz is made mono.
- **The silence.** The mix is built as two halves split at `stillness[0]`. Anything that starts before the cut is truncated at the cut, and each half gets its own reverb pass, so no tail can cross. The master then gates the beat to digital zero. Only the eyelid at `one_eye_opens` passes (dry, 100 ms).
- Each music section is rendered into its own stems and scaled as a whole to a target level, so the balance between sections is set by numbers, not by luck.
- Master: rumble filter, bass to the centre, a slight air roll off, gentle upward compression of the quiet scenes (look ahead, so it is already down before anything loud), loudness normalisation to -14 LUFS (own BS.1770 meter, checked against ffmpeg `ebur128`), a two stage limiter (a slow gain ride for sustained overs, then a 4x oversampled look ahead peak limiter) with a ceiling of -1.3 dBTP.

## What `score.py` prints

File facts, integrated loudness and true peak (own meter and ffmpeg), loudness range, stems per shot, the final master per shot (RMS, peak, LUFS, share of energy below 150 Hz and above 8 kHz, stereo side share), how much the quiet scenes were lifted and how hard the limiter worked per shot, proof of digital zero in the stillness, the level step at ignition and at the climax, and a table of every effect against its cue with the error in milliseconds.

Spectrograms: `build/review/sound-spectrogram-*.png`.
