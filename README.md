# prism-share

A measurement instrument for screen-to-camera optical data links. It generates
*cimbar-class* codes, a grid of cells in which each cell encodes bits by
**glyph shape** (16 glyphs = 4 bits) and **colour** (up to 16 colours = up to
4 bits), and it measures how end-to-end goodput varies with **colour depth**
and **cell size**.

**Hypothesis under test.** Once camera demosaicing and auto white balance have
degraded colour, colour does not pay for itself: monochrome codes with smaller
cells deliver more goodput than colour codes with larger cells.

This is a research tool, not a product. It is built for correctness,
reproducibility and easy parameterisation. Runtime speed is not a goal.

> **Status:** build steps 1–4 are complete: the codec (1–2), synthetic
> degradations with a simulated pilot study (3, see [docs/pilot.md](docs/pilot.md)),
> and the transmitter with its reference frame (4). Ingest (5b) implements
> [docs/capture-format.md](docs/capture-format.md) and its six rules; the
> captured-data sweep, three-outcome yield and provenance-stamped figures (6)
> are built. Of step 5a, the lens-independent geometry is built; the fiducial
> front end (which image regions are markers under a real lens) is an interface
> that raises `NotImplementedError` until real captures exist, so **no real
> capture can be processed yet**. The dashboard (7) is not started.
>
> **Frame format 2 (2026-09-12)** adds the index band, so frames drawn before it
> are not interchangeable with frames drawn after: see §3.1.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # or: pip install -r requirements.lock.txt -e .
pytest                              # ~2 min
python -m prism_share.sim.pilot     # simulated pilot study, ~4 min on 10 cores -> docs/pilot.md
```

```python
from prism_share.codec.params import CodecParams
from prism_share.codec.encoder import encode, save_frames
from prism_share.codec.decoder import decode_payload

p = CodecParams(colour_depth=1, cell_px=5)
frames = encode(b"hello thesis", p)          # list of (1024, 1024, 3) uint8 RGB
save_frames(frames, "data/demo")              # deterministic PNGs
assert decode_payload(frames, p) == b"hello thesis"
```

`requirements.lock.txt` pins the exact versions the tests were last run with
(Python 3.12.6, macOS arm64).

---

## Repository layout

```
prism_share/
  codec/
    params.py     CodecParams + every fixed design constant (the only place numbers live)
    prng.py       SHA-256 counter-mode keystream: the only source of randomness
    glyphs.py     deterministic glyph generation + committed cache (glyph_cache/*.json)
    palette.py    colour palettes for depths 1, 2, 4, 8, 16
    layout.py     frame geometry: fiducials, static region, cell grid, reference masks
    framing.py    one frame's bytes <-> cell symbols (header, CRC, RS, interleave, whitening)
    ecc.py        Reed-Solomon encode/decode with erasures; the RS recoverability bound
    fountain.py   rateless erasure code over frames
    encoder.py    payload -> frame images; deterministic PNG writer
    decoder.py    rectified frame -> symbols + confidences; frames -> payload
  colourspace.py  sRGB transfer, linear luminance, exact 8-bit Y'CbCr and 4:2:0
  sim/
    degrade.py    synthetic degradations: blur, noise, perspective, white balance, chroma
    pilot.py      simulated sweep of every configuration x degradation -> docs/pilot.md
  analysis/
    metrics.py    shape SER, colour SER, byte errors, goodput formula
    ecc_sim.py    any RS(n, k) from a recorded byte error pattern
    plots.py      thesis figures (vector PDF + SVG)
    detect.py     fiducial refinement, homography, rectification at scale k, flat field, overlay
    sweep.py      ingested runs -> per-frame and per-run tables (Parquet), three-outcome yield
  ingest/
    manifest.py   the capture contract: manifest.json + frames.jsonl, schema version 1
    rundef.py     laptop-side run definitions (experiments/runs/<run_id>.yaml)
    ingest.py     the six rules; loading Y/U/V/RGB planes; merge with detection
    pull.py       adb pull wrapper
  transmit/
    display.py    fullscreen player, calibration patches, screenshot verification
    reference.py  the reference frame the capture app locks onto
  export/
    bench.py      decode-benchmark bundle (frames + self-describing ground truth) for the Android spike
tests/
experiments/      pilot.yaml; runs/<run_id>.yaml laptop-side run definitions
docs/             capture-format.md (the phone contract), pilot report, reference-frame table
data/             captures, generated frames, logs (gitignored)
```

`prng.py`, `layout.py`, `framing.py`, `colourspace.py` and `sim/` were not in
the original plan. They hold code that several modules share (encoder, decoder,
simulator, and later `detect.py` and ingest), so it is not duplicated.

---

## 1. `CodecParams`: the independent variables

Every function that generates or reads a frame takes a `CodecParams`. It is a
frozen dataclass: hashable, comparable, and serialisable with `to_dict()` /
`from_dict()` / `to_json()`.

| Field | Default | Swept values | Meaning |
|---|---|---|---|
| `colour_depth` | 4 | 1, 2, 4, 8, 16 | Number of ink colours. 1 = monochrome (white ink). Carries log2(depth) bits per cell. |
| `cell_px` | 8 | 4, 5, 6, 8, 10 | Side of a glyph bitmap in screen pixels. One bitmap pixel = one screen pixel. |
| `cell_gap_px` | 1 | — | Background pixels between cells, and between the grid and anything white. |
| `glyph_count` | 16 | — | Distinct glyph shapes (power of two). Carries log2(count) bits per cell. |
| `frame_px` | 1024 | — | Side of the square frame in screen pixels. |
| `ecc_data` | 125 | via `ecc_sim` | Reed-Solomon message bytes per codeword, *k*. |
| `ecc_total` | 155 | via `ecc_sim` | Reed-Solomon codeword bytes, *n* (≤ 255). Corrects ⌊(n−k)/2⌋ byte errors. |
| `seed` | 0 | — | Seeds every payload-dependent pseudo-random stream (whitening, fountain). |

Derived properties: `glyph_bits`, `colour_bits`, `bits_per_cell`, `pitch_px`
(= cell_px + cell_gap_px), `glyph_weight` (= ⌊cell_px²/2⌋), `ecc_parity`,
`ecc_correctable`, `label` (a filesystem-safe ID) and `fingerprint()` (an
unsigned 32-bit value from SHA-256 of the canonical JSON, embedded in every
frame so a frame cannot be decoded with the wrong parameters by accident).

## 2. Fixed design constants

These are held constant across the whole experiment. All of them are in
`params.py`, and `tests/test_no_magic_numbers.py` fails if the literal `4`,
`8` or `16` appears anywhere else in the package.

| Constant | Value | Rationale |
|---|---|---|
| `BORDER_PX` | 16 | White quiet zone around the frame. |
| `FIDUCIAL_DICTIONARY` | `DICT_4X4_50` | OpenCV ArUco. Robust detection, sub-pixel corners, and ids give orientation. |
| `FIDUCIAL_IDS` | 0, 1, 2, 3 | Clockwise from top-left. Distinct ids make rotation and mirroring unambiguous. |
| `FIDUCIAL_MODULE_PX` | 12 | Marker = 6×6 modules = 72 px. |
| `FIDUCIAL_MARGIN_PX` | 12 | White between marker and data (≥ 1 module, as ArUco requires). |
| `KEEPOUT_PX` | 100 | Corner square with no data: border + marker + margin. |
| `FRAME_FORMAT_VERSION` | 2 | Pixel layout of a frame. Enters the fingerprint, so formats cannot be mixed. |
| `INDEX_BAND_BITS`, `_REPEATS` | 8, 3 | Frame index, repeated for a majority vote (§3.1). |
| `INDEX_BAND_BLOCK_PX` | 32 | Block side: far larger than any data cell, so the band reads at every configuration. |
| `INDEX_BAND_REFERENCE` | 0 | Index reserved for the reference frame. |
| `BACKGROUND_RGB` | (0, 0, 0) | Data region is dark, like cimbar's dark mode. |
| `MONO_INK_RGB` | (255, 255, 255) | Ink for colour_depth = 1. |
| `PALETTE_LEVELS` | 0, 128, 255 | Per-channel code values palette candidates may use. |
| `GLYPH_POOL_SEED` | 0x5052534D | Glyph set is a property of (glyph_count, cell_px), **not** of `seed`. |
| `GLYPH_POOL_SIZE` | 3000 | Candidate patterns per cell size. |
| `GLYPH_SMOOTH_KERNEL`, `_PASSES` | (1,2,1), 1 | Spatial low-pass of candidate noise (see §4). |
| `GLYPH_SELECT_RESTARTS` | 16 | Restarts of the max-min selection heuristic. |
| `HEADER_FORMAT` | `>IIII` | 16-byte frame header. |
| `CRC_FORMAT` | `>I` | CRC-32 over header + block. |
| `FOUNTAIN_REPAIR_FRACTION`, `_MIN_REPAIR` | 0.25, 4 | Default number of repair frames. |
| `ASSUMED_FPS` | 30 | Frame rate in the goodput formula. Never measured. |
| `RS_SELECTION_PERIOD` | 2 | Even frame positions choose the RS code, odd positions score it (§10.1). |
| `YUV_DEFAULT_MATRIX` | `bt601` | Y′CbCr matrix of simulated YUV_420_888 (BT.709 selectable). |
| `YUV_LIMITED_*`, `YUV_C_OFFSET` | 16, 219, 224, 128 | 8-bit limited-range Y′CbCr. |
| `YUV_420_FACTOR` | 2 | Chroma subsampling per axis in YUV_420_888. |
| `DECODER_LUMA_MATRIX` | `bt601` | Luma used by the `luma` shape channel. |
| `COLOUR_CORE_FRACTION` | 0.25 | Share of ink pixels averaged by the `saturated` colour estimator. |
| `HEADLINE_SHAPE_CHANNEL`, `_COLOUR_ESTIMATOR` | `luma`, `saturated` | Pre-registered headline decoder (§8.1); the rest are sensitivity analysis. |
| `HEADLINE_DECODER_REGISTERED` | 2026-09-14 | Date of that pre-registration, before any real capture. |
| `NEAR_TIE_MARGIN_PCT` | 5.0 | A winner ahead of the other class by less than this is flagged as a near-tie. |
| `REFERENCE_TARGET_LINEAR_MEAN` | 0.308 | Reference frame's whole-frame mean linear luminance (see §14). |
| `REFERENCE_DITHER_SIZE`, `_BLOCK_PX` | 16, 2 | Bayer matrix side; screen pixels per dither element. |
| `DISPLAY_SURROUND_RGB` | (0, 0, 0) | Screen outside the frame, for every configuration. |
| `DISPLAY_DEFAULT_INTERVAL_MS` | 200 | Free-running time per frame. |
| `CALIBRATION_GREY_LEVELS` | 17 | Grey ramp of `display calibrate`. |

## 3. Frame anatomy

A frame is a `frame_px × frame_px` RGB image (origin top-left, x right, y down):

1. **White border**, `BORDER_PX` wide, all round.
2. **Four keep-out squares** (`KEEPOUT_PX`) at the corners. Each is white and
   holds an ArUco marker (black on white) at offset `BORDER_PX` from both edges.
3. **The index band** (§3.1), between the two top keep-out squares.
4. **Data region**: everything else. Black background with the cell grid.

**Static region invariance (hard requirement 1).** The border band and the
keep-out squares make up the *static region*. It depends on `frame_px` only:
it is the same for every colour depth, cell size, gap, ECC rate and seed, and it
contains only pure black (0,0,0) and pure white (255,255,255). If detection
difficulty changed with the parameter under test, the experiment would be
confounded, and this invariance rules that out. It is enforced three ways:

* `test_static_region_identical_across_all_configurations`: every one of the 25
  sweep configurations, plus seed, ECC and gap variants, is compared
  pixel-for-pixel against a single reference;
* `test_static_region_is_pinned`: a SHA-256 of the static canvas is pinned;
* the marker bit patterns are hard-coded in `params.py` (so rendering never
  depends on the OpenCV version) and checked against `cv2.aruco`.

What the fiducials *cannot* hold constant is the image content next to the
keep-out margin: blur can carry data-region light into the margin. The 12 px
margin is chosen to be several blur widths.

### 3.1 The index band: every frame says which frame it is

A reserved horizontal band of large black-and-white blocks carries an 8-bit
**frame index**. Index 0 is the reference frame; code frames are 1 upward.

| Property | Value |
|---|---|
| Blocks | 24 = 8 bits × 3 repeats, majority vote on read |
| Block size | 32 × 32 px (every data cell is 4–10 px) |
| Position | x 128–896, y 24–56; centred between the top keep-out squares |
| Area | 768 × 32 px = 24 576 px², 2.34 % of the frame |
| Colour | pure black and white only, at every colour depth |

Its **geometry is constant for every configuration**, exactly like the
fiducials; only the index changes what it says. It is read after rectification,
from the same normalised levels the decoder uses, and it is unaffected by
colour depth or cell size because its blocks are an order of magnitude larger
than any cell.

**Why the frame carries its own identity.** The alternative, working out which
displayed frame a capture shows by seeing which one it decodes closest to,
makes identification depend on decode quality, which is the thing being
measured. A badly degraded capture is exactly the one that fails to identify
and gets excluded, so exclusion correlates with the outcome: configurations
that degrade more lose more frames, and their surviving frames are their
easiest ones. That compresses the measured difference between configurations
by a configuration-dependent amount. The band removes the question: it is
readable in every case where detection succeeded at all, and when detection
fails the frame is already counted as "detection failed" and needs no ground
truth. There is deliberately **no content-matching fallback**: a fallback would
fire precisely on degraded frames, which is where the bias does its damage.

The band costs a fixed area of cells:

| cell_px | cells before | cells now | lost |
|---|---|---|---|
| 4 | 38 048 | 36 970 | 1 078 (2.83 %) |
| 5 | 26 441 | 25 667 | 774 (2.93 %) |
| 6 | 19 305 | 18 639 | 666 (3.45 %) |
| 8 | 11 700 | 11 270 | 430 (3.68 %) |
| 10 | 7 844 | 7 564 | 280 (3.57 %) |

The index wraps after 255, because a real payload can need thousands of
frames. A measurement run displays exactly one frame (§13), so the wrap never
matters there.

**The band is apparatus, not codec.** A deployed system would carry its frame
index in the fountain header, not in a 768 × 32 px strip. Because the band's
cell cost varies with cell_px (2.83 % at 4 px, 3.57 % at 10 px), it would
enter the between-configuration comparison as a term that is not physics. Every
goodput from captured data is therefore reported twice (§17): as measured, and
with the band's cells credited back.

**Known failure mode: a horizontal artefact across y 24–56.** All three copies
of the index sit in the same rows. A horizontal artefact over those rows, such
as glare across the top of the panel or a rolling-shutter dark band, takes out
all three at once, so the majority vote cannot save the read. This is accepted
deliberately. Within a run the geometry is locked, so such an artefact repeats
on every capture: it kills the whole run loudly (`band_agreement` drops and
captures stop matching the run's frame) rather than biasing it quietly.
**Diagnosis:** an artefact tied to the rig, not the code. **Fix:** change the
phone's angle or the lighting, and repeat the run.

**Frame format version.** `FRAME_FORMAT_VERSION` (now 2) enters
`CodecParams.fingerprint()`, which every frame carries in its header. A capture
of a format-1 frame (no index band) therefore fails the fingerprint check
instead of being decoded with today's geometry: **old and new captures cannot
be silently mixed.** Any run captured before 2026-09-12 must be recaptured, and
its frames regenerated with `display frames`.

**Cell grid.** The pitch is `cell_px + cell_gap_px`. The number of cells per
side is the largest count that fits inside the border with at least
`cell_gap_px` of background on every side, and the grid is centred. Cells whose
gap-padded box would touch a keep-out square are dropped. The remaining cells
are numbered row-major, and that number is the **cell index** used throughout.

## 4. Glyphs

A glyph is a `cell_px × cell_px` binary bitmap drawn at native resolution. It is
never resampled, so it is never interpolated.

**Candidate pool.** Integer noise from the keystream (seed `GLYPH_POOL_SEED`) is
smoothed by a separable integer binomial kernel and then thresholded at its rank,
so that exactly `glyph_weight` = ⌊cell_px²/2⌋ pixels are ink. Patterns with an
*isolated pixel* (one whose in-grid 4-neighbours all have the opposite value)
are rejected. Duplicates are dropped, and the first 3000 survivors form the pool.

* *Constant weight* has two consequences. Every glyph has the same number of ink
  pixels, so colour can be read from any glyph, and every glyph has the same
  mean, so correlation matching is unbiased between glyphs.
* *No isolated pixels + smoothing* keeps features at least 2 px across. A
  single-pixel feature is below what the camera's optical PSF resolves, and a
  pure Hamming criterion would favour exactly such high-frequency patterns.

**Selection.** The 16 glyphs are chosen to **maximise the minimum pairwise
Hamming distance**. The method is greedy farthest-point construction from 16
deterministic starts, each followed by best-improvement swap search. Ties go to
fewer pairs at the minimum, then larger total distance. The chosen set is sorted
by its row-major bit string, and that order fixes glyph index → bitmap.

**Cache.** Sets are stored in `prism_share/codec/glyph_cache/` and committed.
`test_committed_cache_matches_regeneration` regenerates each set and requires
it to equal the cache, so the glyphs used in the thesis can never change
silently.

| cell_px | Grid | Data cells | min Hamming *d* (upper bound†) | Pairs at *d* |
|---|---|---|---|---|
| 4 | 198×198 | 38 048 | 8 of 16 px (8) | 109 |
| 5 | 165×165 | 26 441 | 10 of 25 px (12) | 2 |
| 6 | 141×141 | 19 305 | 16 of 36 px (18) | 14 |
| 8 | 110×110 | 11 700 | 30 of 64 px (34) | 18 |
| 10 | 90×90 | 7 844 | 48 of 100 px (52) | 32 |

† Plotkin average-distance bound *d* ≤ *nM* / (2(*M*−1)) with *M* = 16, rounded
down to even (distances between constant-weight words are even). This bound
ignores the smoothness constraint, so it may not be attainable. cell_px = 4
reaches it.

## 5. Palettes

Ink colours are drawn on the black background.

* **Depth 1** is white.
* **Depth ≥ 2** comes from an *exhaustive* search over every subset of the
  candidate colours: RGB triples over {0, 128, 255} with at least one channel at
  255 (19 candidates). Requiring a full-scale channel gives every palette colour
  the same HSV value, so every colour has the same shape contrast against
  black, measured the way the `max` shape channel measures it (per-pixel channel
  maximum). The pre-registered headline reads shape from luma instead, where the
  contrast is not equal. §8.1 states that cost.
* **Objective** (lexicographic, exact integer arithmetic): (1) maximise the
  minimum pairwise Euclidean distance in 8-bit code values; (2) maximise the
  minimum Rec. 709 luma (sensor SNR); (3) minimise the number of pairs at the
  minimum; (4) maximise the total distance; (5) take the first subset.
* Each depth is optimised **independently** (the palettes are not nested). Each
  colour condition gets the best palette the rule can produce, which is the
  fair way to test a hypothesis that predicts colour will lose.

| Depth | Colours (RGB) | Min distance |
|---|---|---|
| 1 | (255,255,255) | — |
| 2 | (0,255,0) (255,0,255) | 441.7 |
| 4 | (0,0,255) (0,255,0) (255,0,0) (255,255,255) | 360.6 |
| 8 | (0,255,0) (0,255,255) (128,128,255) (128,255,128) (255,0,255) (255,128,128) (255,255,0) (255,255,255) | 179.6 |
| 16 | (0,128,255) (0,255,0) (0,255,128) (0,255,255) (128,128,255) (128,255,0) (128,255,128) (128,255,255) (255,0,128) (255,0,255) (255,128,0) (255,128,128) (255,128,255) (255,255,0) (255,255,128) (255,255,255) | 127.0 |

The order of the list is the colour index. These are the exact values written to
the framebuffer, and the transmitter will log them again at display time.

## 6. Framing: bytes ↔ cells

For one frame with RS(*n*, *k*) = (`ecc_total`, `ecc_data`):

1. **Frame data** = `header (16 B) ‖ block ‖ CRC-32 (4 B)`, where
   n_codewords = ⌊capacity_bytes / *n*⌋ and data_bytes = n_codewords · *k*.
   The header holds `block_id, n_source_blocks, payload_len, params_fingerprint`
   (four big-endian uint32).
2. **RS encoding**: data is cut into consecutive *k*-byte messages. Message *j*
   becomes systematic codeword *j*.
3. **Interleaving**: symbol *s* of codeword *j* goes to stream byte
   *s*·n_codewords + *j*. Adjacent cells therefore feed different codewords, and
   every codeword samples the whole frame evenly, which spreads spatially
   clustered errors (glare, corner defocus) across codewords.
4. **Padding**: zeros up to capacity_bytes.
5. **Whitening**: XOR with the keystream (`seed`, domain "whitening"). This
   makes the glyph and colour distribution uniform whatever the payload, so frame
   statistics do not depend on content. It also means parity and payload
   symbols are statistically alike, which `ecc_sim` relies on.
6. **Bits → cells**: the stream is unpacked MSB-first, followed by < 8 keystream
   pad bits. Cell *i* takes bits [*i·b*, (*i*+1)·*b*), with *b* = bits_per_cell:
   first `glyph_bits` (the glyph index), then `colour_bits` (the colour index).

Steps 3–6 do not depend on *k*, and depend on *n* only through the interleaving
depth. A recorded per-cell error pattern can therefore be re-grouped into
codewords of any RS(*n*′, *k*′) (hard requirement 5).

**Payload bytes per frame** (`FrameCapacity.block_bytes`) exclude the header,
CRC, RS parity and padding. At the default RS(155, 125):

| cell_px | depth 1 | depth 2 | depth 4 | depth 8 | depth 16 |
|---|---|---|---|---|---|
| 4 | 14 855 B | 18 605 B | 22 230 B | 25 980 B | 29 730 B |
| 5 | 10 230 B | 12 855 B | 15 480 B | 17 980 B | 20 605 B |
| 6 | 7 480 B | 9 355 B | 11 230 B | 13 105 B | 14 980 B |
| 8 | 4 480 B | 5 605 B | 6 730 B | 7 855 B | 8 980 B |
| 10 | 2 980 B | 3 730 B | 4 480 B | 5 230 B | 5 980 B |

This is the *ceiling* (frame yield = 1). For example, monochrome at 4 px already
has 2.2× the raw capacity of the cimbar-like 4-colour 8 px code. Whether that
ceiling survives the camera is what the experiment measures.

## 7. Fountain code

The payload is split into *K* source blocks of `block_bytes`, one block per
frame. Frames 0…*K*−1 carry the source blocks (systematic). Frame *i* ≥ *K*
carries the XOR of a pseudo-random subset of blocks. Its coefficient vector
comes from the keystream (`seed`, index *i*), and an all-zero draw is replaced
by a unit vector. Decoding is Gaussian elimination over GF(2), which succeeds as
soon as the received coefficient vectors reach rank *K*.

This is a **dense random linear fountain**, used instead of an LT/peeling code.
For the tens to hundreds of blocks a thesis payload needs, LT codes need large
reception overhead. For a dense code, the probability that *K* + *m* received
frames are still rank-deficient falls roughly as 2^−*m*. That is close to the
ideal erasure code the goodput formula assumes. Its O(*K*²) decoding cost does
not matter offline.

## 8. Decoder

`decoder.read_symbols(rectified, params)` is pure: it takes an array, returns
arrays, and does no I/O and keeps no state. The input is a `frame_px`-square
frame on a 0–255 scale, either RGB `(H, W, 3)` or a single plane `(H, W)` (for
example the Y plane of a YUV capture, which is valid for monochrome only).

1. **Level normalisation** (default on). Per channel, map the median of the
   *white reference* (the middle of the border band) to 1 and the median of the
   *black reference* (the centre of the markers' black border modules) to 0,
   then clip. Both references lie in the static region, so the correction is
   computed identically for every configuration. Scaling each channel to the
   display's own white is also a von Kries white balance.
2. **Shape.** Cell intensity is the per-pixel channel maximum. Each cell is
   Pearson-correlated with every glyph and the argmax wins.
   *Glyph confidence* = best − second-best correlation ∈ [0, 2].
3. **Colour**, independent of the shape decision. The `glyph_weight` brightest
   pixels are taken as ink, and their mean RGB is scaled so its maximum channel
   is 1. The nearest palette colour (palette scaled the same way, Euclidean)
   wins. *Colour confidence* = second-nearest − nearest distance. For
   colour_depth 1 no colour decision is made: the colour is 0 and the confidence
   is +∞.

Because colour never uses the decoded glyph, the decoder cannot turn a shape
error into a colour error or the reverse. Shape SER and colour SER (step 6)
therefore measure separate physical effects (hard requirement 6).

**Decoder variants** (`DecoderOptions`). Every analysis decodes with all four:

* shape channel `max` (as above) or `luma`, which reads shape, and ranks ink
  pixels, from Y′ (BT.601) of the normalised RGB;
* colour estimator `mean` (as above) or `saturated`, which averages only the
  most saturated `COLOUR_CORE_FRACTION` (¼) of the ink pixels.

`DecoderOptions()` with no arguments is still `max`/`mean`, the plain reference
decoder described above. No analysis uses that default implicitly: the pilot
and the sweep both iterate over `decoder.ALL_DECODERS`, headline first.

### 8.1 Pre-registered headline decoder: `luma`/`saturated`

**Every headline number, table and figure uses `luma`/`saturated`. The other
three variants are reported alongside it, everywhere, as sensitivity analysis.**
The choice is fixed in `params.py` (`HEADLINE_SHAPE_CHANNEL`,
`HEADLINE_COLOUR_ESTIMATOR`, `HEADLINE_DECODER_REGISTERED = 2026-09-14`). It is
exported as `decoder.HEADLINE_DECODER`, and no experiment file can override it:
`sim.pilot.load_config` rejects a `headline_decoder` key.

**Provenance, stated plainly.** This variant was first picked *after* the
4-frame pilot, because it had the smallest total goodput loss in the decoder
ablation. That is a post-hoc choice. It is pre-registered now, on 2026-09-14,
before any real capture exists. The pilot numbers are **not** the
justification. The justification is the argument below, which would hold
whatever the pilot had said, and the headline will not be changed after real
captures are seen.

The guiding principle is the same as for the palettes (§5). The hypothesis
predicts that colour loses. A fair test must therefore give colour its best
chance, and it must not handicap colour through a decoder that ignores how the
camera actually delivers pixels. The decoder choice cannot move monochrome: a
white-on-black cell has equal channels, so its luma and its channel maximum
are the same signal, and monochrome makes no colour decision. Only colour
goodput depends on this choice.

**Why `luma` for shape.**

1. **It reads shape where the camera keeps shape.** The capture format is
   `YUV_420_888`, and JPEG is also Y′CbCr with subsampled chroma. Y is the only
   full-resolution plane. U and V have one sample per 2×2 pixels, so any
   spatial detail finer than that exists only in Y. Every RGB triple, the
   phone's or ours, is Y plus upsampled chroma. Its channel maximum therefore
   mixes chroma interpolation into glyph edges, and that happens on every
   capture, not only under a bad condition. A glyph is spatial detail at 4–10
   px, so shape should come from the plane that carries spatial detail.
2. **It keeps the error decomposition and the mechanism experiment clean.**
   Y is the same, up to rounding, in the phone's RGB and in our
   `yuv_nearest` / `yuv_bilinear` conversions. So under `luma`, shape SER is
   essentially identical across pixel sources, and any difference between
   sources lands in colour SER. The mechanism experiment (native U/V planes vs
   the phone's upsampled RGB) then compares colour alone, which is exactly
   what it is for. Under `max`, the shape reading would change with the chroma
   upsampler, and the comparison would mix shape and colour effects (hard
   requirement 6).
3. **Its cost is known and named.** A low-luma ink (pure blue, Y′ = 0.114) has
   about one ninth of the channel-maximum contrast in luma. It is therefore the
   first to lose shape under noise and blur. The palette rule's criterion (2)
   (maximise the minimum luma) exists because of this, but it is only a
   tie-break after colour distance, so the 4-colour palette still contains
   pure blue. This cost falls on colour, against the hypothesis's favoured
   outcome. It is accepted because the alternative, reading shape through
   chroma interpolation, costs colour on every capture and at every noise
   level.

**Why `saturated` for colour.**

1. **4:2:0 dilutes chroma at every glyph edge, on every capture.** An ink pixel
   whose 2×2 chroma block overlaps the black gap or background has its colour
   pulled toward neutral. Ink pixels are spread across a 4–10 px cell, so a
   large share of them are edge pixels. The interior pixels, whose chroma
   block lies wholly in ink, are the most saturated ones. Averaging only those
   estimates the colour that was displayed. Averaging all of them estimates
   that colour mixed with black's chroma.
2. **It does not depend on the shape decision.** It uses the same ink ranking
   as `mean`, so colour SER still cannot inherit shape errors.
3. **Its cost is also known and named.** Selecting the most saturated pixels
   under heavy noise selects noise excursions, and averaging a quarter of the
   ink uses fewer pixels. At low noise that cost is small. At high noise it can
   exceed the dilution it corrects. Real sensor noise on a bright, static,
   close-range screen is expected to be low, but that is an expectation, not a
   measurement. If captures turn out noisy, the `mean` rows of the sensitivity
   analysis show it. The headline is not switched.

**Would another default be better?** `max`/`saturated` is the strongest
alternative, because the palettes' full-value constraint gives every colour
equal channel-maximum contrast. It is rejected on reason 2 for `luma`: it makes
shape SER depend on the chroma path, and so confounds the one experiment
designed to isolate the chroma path. The `mean` estimators are rejected on
reason 1 for `saturated`: they ignore a dilution that 4:2:0 guarantees.

`read_index_band(rectified, params)` reads the frame's own index from the band
(§3.1) using the same normalised levels, and reports the majority-vote
agreement and the smallest decision margin. It is what identifies a capture;
nothing in the analysis matches captures against candidate frames by content.

`decode_frame` adds RS error decoding, CRC and fingerprint checks.
`framing.decode_frame_symbols(..., erased_cells=mask)` treats low-confidence
cells as RS erasures. `decode_payload` runs the fountain decoder over any
collection of frames, in any order.

## 9. Determinism (hard requirement 2)

* **All randomness comes from `prng.keystream`**: SHA-256 in counter mode over
  `(seed, domain, index, counter)`. Its output is fixed by the SHA-256
  specification. numpy's RNGs are not used, because numpy guarantees stable bit
  streams but not stable output from the methods built on them.
* **Glyph and palette selection** uses integer arithmetic with explicit
  tie-breaking, and glyph sets are also committed to the repository.
* **PNGs** are written by a minimal encoder (`encoder.png_bytes`) that uses
  *stored* (uncompressed) deflate blocks and no metadata chunks. Compressed
  output differs between zlib builds (zlib vs zlib-ng), whereas stored blocks,
  CRC-32 and Adler-32 are fully specified. The file is therefore a pure function
  of the pixels, at a cost of about 3 MB per 1024 px frame. The PNGs carry no
  gAMA, sRGB, iCCP or cHRM chunk, so nothing downstream is invited to
  colour-manage them.
* **Tests**: `test_golden_png_hash` pins SHA-256 values of encoded PNGs.
  Running the suite on another machine is the cross-machine check.
  `test_fresh_interpreter_with_different_hash_seed_matches` rules out any
  dependence on `PYTHONHASHSEED` or in-process state.

## 10. Metrics, ECC simulation and goodput

`analysis/metrics.py` compares decoded symbols with ground truth and reports
**shape SER** (glyph wrong) and **colour SER** (colour wrong) as separate
numbers, plus the stream **byte error** pattern an RS decoder would see.
`analysis/ecc_sim.py` regroups that byte pattern into codewords of any
RS(*n*, *k*), exactly as `framing.py` interleaves them. Per frame, only the
maximum per-codeword error count matters, so it is stored, and yield follows for
every *k* at once (hard requirement 5).

```
goodput = payload_bytes_per_frame × frame_yield × ASSUMED_FPS
```

* `payload_bytes_per_frame` = `FrameCapacity.block_bytes` for the RS(*n*, *k*)
  being evaluated (`ecc_sim.payload_bytes_per_frame`).
* `frame_yield` = the fraction of displayed frames whose every codeword is
  recoverable, i.e. 2·errors + erasures ≤ *n* − *k* in every codeword.
* `ASSUMED_FPS` is a documented constant (30, `params.py`). The capture frame
  rate never enters the result: goodput is derived, never timed.
* The formula assumes an **ideal erasure code across frames**. The fountain
  code's reception overhead (a few frames per payload, see §7) is not charged.

### 10.1 The RS code is chosen out of sample

Hard requirement 5 lets every configuration use its goodput-maximising
RS(*n*, *k*). Choosing that code on the same frames its goodput is then scored
on is an **in-sample optimum**. It picks the code that happens to fit those
frames' worst codewords, so it overstates goodput, and it does so most for
configurations near their yield cliff. More frames dilute the bias; they do not
remove it. On real captures, where 200 frames per condition is a lab day, it
would be worse than in the pilot.

So every reported goodput is **out of sample** (`ecc_sim.out_of_sample`):

* **Split by position, fixed before analysis.** A frame is in the *selection*
  half iff its position is even (`RS_SELECTION_PERIOD` = 2), otherwise in the
  *evaluation* half. Position means the frame index in the pilot, and the
  capture `index` from `frames.jsonl` for real captures. The rule looks at
  position only, never at outcomes, so every configuration in a condition gets
  the same split and comparisons stay paired.
* **Interleaved, not first half / second half.** Slow drift over a run (panel
  warm-up, room light) then lands in both halves equally, instead of
  separating the frames the code was chosen on from the frames it is scored on.
* **Chosen on selection, scored on evaluation.** RS(*n*, *k*) is chosen on the
  selection half. Yield, the three-outcome counts and goodput are counted on
  the evaluation half only. The fixed RS(155,125) columns are counted on the
  evaluation half too, so every yield in a table uses the same frames. SERs are
  not selected on anything and use all frames.
* **No held-out frames, no number.** If either half is empty, `best_*` goodput
  is NaN, never an in-sample value.

In-sample selection remains available only to measure the bias. The pilot keeps
`in_sample_goodput_mbps` and `evaluation_oracle_goodput_mbps` as labelled
diagnostic columns and reports the gap. `sweep.py` refuses to produce in-sample
goodput unless it is run with `--in-sample`. That flag prints a warning, raises
`InSampleWarning` from `SweepConfig`, writes `rs_selection = in_sample` on every
row, and stamps "IN-SAMPLE RS SELECTION (optimistic, not a result)" on every
figure.

## 11. Synthetic degradations (build step 3)

`sim/degrade.py`. Every degradation is a pure function
`fn(frame, severity, params, *, index=0, **options) -> frame`. Frames are float
RGB on a 0–255 scale, and each function has exactly one severity parameter with
a physical unit. Random draws come from `params.seed`, the degradation's name and
`index` (the frame number), so a step's randomness does not depend on where it
sits in a chain. `apply_chain` composes steps in any order; end with `quantize`
to model an 8-bit capture.

| Degradation | Severity (unit) | Model |
|---|---|---|
| `blur` | Gaussian σ (screen px) | Optical defocus/PSF. |
| `noise` | σ (8-bit code values) | Additive white Gaussian, independent per pixel and channel, clipped. |
| `perspective` | max corner displacement (fraction of frame side) | Warp to a random inward quadrilateral at unchanged resolution, then back through the exact inverse homography (bilinear both ways). Resampling loss only; no detection error. |
| `white_balance` | R/B imbalance (stops) | Gains 2^(s/2) on R and 2^(−s/2) on B in *linear* light (sRGB EOTF), clipped at full scale. |
| `chroma` | chroma sample pitch (screen px) | **YUV_420_888 round trip**: R′G′B′ → 8-bit Y′CbCr (BT.601 limited range by default; BT.709 and full range selectable) → Cb, Cr box-filtered to 1/pitch resolution and rounded to 8 bits → upsampled by `nearest` or `bilinear` → R′G′B′, clipped. Pitch 2 = 4:2:0 with the camera sampling the screen 1:1; a camera with *m* sensor pixels per screen pixel has an effective pitch of 2/*m*. |

Chroma siting is centred (the 2×2 box mean), and bilinear upsampling
interpolates between those centres. Android leaves YUV_420_888 siting to the
device. The Y′CbCr functions live in `prism_share/colourspace.py`, so ingesting
real YUV captures (step 5) will use the same implementation.

Gaussian variates are Box–Muller on numpy's PCG64 *bit stream*, which numpy
guarantees stable across versions. numpy's `Generator.normal` is avoided.

## 12. Pilot study

`python -m prism_share.sim.pilot` runs every configuration against every
degradation ladder in `experiments/pilot.yaml`, one degradation at a time, and
writes [docs/pilot.md](docs/pilot.md) with tables, figures (`docs/pilot/*.pdf|svg`)
and `docs/pilot/pilot_summary.csv`. The results are **simulated predictions,
not measurements**. In short: at 200 frames per condition, with the
pre-registered decoder (§8.1) and the RS code chosen out of sample (§10.1), a
colour depth above 1 maximises goodput in 42 of 46 conditions (32 of the
original 34, unchanged by out-of-sample scoring). Every winner is reported with
its runner-up and margin, and the band-credited column gives the same winner in
all 46. Chroma subsampling is the only degradation that selectively harms
colour. On the chroma-pitch ladder (1 to 4 in steps as fine as 0.25), the best
depth falls from 16 to 4, then 2, and monochrome wins from pitch 3.75. The
crossing lies between 3.5 and 3.75 with `luma/saturated`, but at about 2.3–3.0
with the `mean` colour estimators, the largest decoder sensitivity in the pilot.
The optimal cell size is set almost entirely by blur. The report lists what
the model leaves out: Bayer demosaicing, combined degradations, realistic noise
and compression.

## 13. Transmitter (build step 4)

`transmit/display.py`:

```bash
python -m prism_share.transmit.display frames --colour-depth 4 --cell-px 8 --payload-bytes 200000 --n-frames 5 --out data/runs/c4_px8
python -m prism_share.ingest.rundef new --run-id d07_lux200_f0 --frames data/runs/c4_px8 --block 0 --condition distance_m=0.7
python -m prism_share.transmit.display measure experiments/runs/d07_lux200_f0.yaml   # THE measurement path
python -m prism_share.transmit.display play data/runs/c4_px8 --reference            # demonstrator only, never a measurement
python -m prism_share.transmit.display calibrate                                   # solid patches (add --auto to cycle)
python -m prism_share.transmit.display verify shot.png --frame data/runs/c4_px8/frame_00000.png
```

**Measurement protocol: one run, one static frame.** A measurement run holds
ONE frame on screen for the whole capture and never advances, so a capture
cannot be torn between two frames: tearing is impossible rather than accounted
for. Frame yield is the fraction of captures that are recoverable, and the
variation between captures comes from the channel (noise, rolling-shutter
phase, micro-motion), not from the stimulus changing. Content variation is
covered by running several distinct frames per configuration, one per run
(`--block 0`, `--block 1`, …). `display measure <run definition>` shows, in
order, advancing only on → :

1. the **reference frame**: lock AE, AF and AWB on it;
2. the **brightest** code configuration: check clipping;
3. the **darkest** code configuration: check clipping (the reference passing a
   clip check proves neither extreme passes; they sit about ±0.5 stop from it);
4. the **run's frame**, held while the phone captures. Nothing follows it, and
   space never starts playback.

`display play` remains as a free-running **demonstrator**. It warns on start,
logs `measurement: false`, and nothing it shows is a valid capture for the
sweep.

* **Window:** a borderless fullscreen OpenCV HighGUI window. The buffer handed
  to it is always exactly the window size, so HighGUI has nothing to rescale.
* **Scaling:** integer nearest-neighbour only (`np.repeat`), at the largest
  scale that fits (or `--scale`). A frame larger than the screen is refused,
  never shrunk. Outside the frame the screen is `DISPLAY_SURROUND_RGB` (black)
  in every configuration.
* **Colour:** PNG values reach the window unmodified. No gamma, ICC or colour
  management is applied, and any colour chunk found in an input PNG is
  reported, not applied.
* **Log** (JSON Lines, `data/display_logs/`): the session (screen size, scale,
  placement, surround, interval); each item's exact distinct RGB triples with
  pixel counts, plus SHA-256 hashes of the frame and of the composed screen
  buffer; every show event with its monotonic time. The thesis can quote the
  RGB values from here.
* **Timing:** the free-running interval (`--interval-ms`, default 200) follows a
  fixed schedule, so draw time does not accumulate. HighGUI gives no vsync
  guarantee, so actual times are logged. Timing never enters a result.
* **Keys:** space play/pause (ignored in `measure`), → ↓ next, ← ↑ previous, `r` reference, `q`/Esc quit.
* **`frames`** writes a run's PNGs plus `frames.json` (params, fingerprint,
  payload source and per-frame hashes). By default the payload is the
  deterministic keystream, so ground truth can be regenerated.
* **`calibrate`** shows full-screen solid patches: black, white, the six
  primaries and secondaries, every palette colour of every depth, and a
  17-level grey ramp.

**What the software cannot guarantee**: the OS and the panel. On macOS the
window server may colour-manage windows (ColorSync) and, on Retina displays,
scale from points to pixels with interpolation. The panel may apply its own
picture processing. Before trusting a setup:

1. set the display to its native resolution at 100 % scaling, and disable
   Night Shift, True Tone and auto-brightness (or the equivalents on other
   platforms);
2. show a frame (`measure`, or `play --hold`), take a lossless full-screen screenshot, and run
   `verify`. It locates the frame at an integer scale and requires **every
   pixel to match exactly**. It also reports whether a mismatch looks like
   colour management (small differences everywhere) or interpolated scaling
   (localised differences);
3. use `calibrate` with a colorimeter or the capture phone to characterise what
   the panel actually emits.

## 14. Reference frame

`transmit/reference.py`. The capture app shows this frame first and locks AE,
AF and AWB on it:

* **Geometry:** the codes' static region verbatim (same `frame_px`, border,
  keep-out squares and fiducials), so it is geometrically identical to them.
* **Content:** pure black and white only. The data region is an ordered
  (Bayer 16×16) dither of 2×2 px blocks, giving fine texture for
  contrast-detect AF and no feature finer than a glyph's.
* **Luminance:** blocks are turned white, in dither order, until the
  *whole-frame mean linear luminance* (sRGB EOTF, Rec. 709 weights) reaches
  `REFERENCE_TARGET_LINEAR_MEAN`. Linear, because auto-exposure meters light,
  not code values: flat sRGB 128 is 0.216, while a 50/50 black/white pattern is 0.5.
* **One reference for all runs, by design.** The code configurations span 0.217
  (2 colours, 4 px) to 0.438 (mono, 10 px) linear, about one stop, so no single
  frame can match them all. The target is pinned at their geometric midpoint,
  0.308, which limits the worst mismatch to ±0.51 stops and gives **every
  configuration identical camera settings**. A per-configuration reference
  (`--match DEPTH,CELL_PX`) is available, but using it would tie exposure to the
  parameter under test.
* `python -m prism_share.transmit.reference --report docs/reference_frame.md`
  prints and records every configuration's mean linear luminance and its
  mismatch in stops ([docs/reference_frame.md](docs/reference_frame.md)). Its
  PNG hash is pinned by `tests/test_reference.py`.

## 15. Detection and rectification (build step 5a, geometry half)

`analysis/detect.py`: `detect(image, params, *, kernel, flat_field, front_end, k)`
takes a captured frame (Y plane or RGB) and returns a `Detection`: the
rectified array plus a flat `DetectionRecord`. Failure is returned, never
raised.

| Stage | Status |
|---|---|
| **Front end**: decide which regions are the four markers; return coarse outer corners (±2 px) | **Interface only** (`FrontEnd`, `FrontEndResult`); `locate_fiducials` raises `NotImplementedError`. To be designed on real captures. |
| Sub-pixel refinement | Built. Edge profiles across each marker's outer square; gradient centroid around the 50 % crossing; TLS line per edge; corners = line intersections. |
| Marker centres → homography | Built. Diagonal intersections; H from the four centre correspondences. |
| Reprojection error | Built. Measured on the 16 marker corners, which the fit does not use (a four-point fit always has zero residual). |
| Rectification | Built. Integer scale k ≥ the largest local magnification (never downsamples); kernel `nearest`, `bilinear` (default) or `lanczos`, recorded. The decoder infers k from the array shape and averages each k×k block. |
| Flat field | Built, off by default. Local mean of the captured reference / local mean of the ideal dither, normalised. |
| Record, overlay | Built. See `DetectionRecord`; `write_overlay` draws the frame outline, fiducial quad, refined marker outlines, cell grid and cell centres. |

**One detector.** The detection constants (`REFINE_*`, `RECTIFY_*`,
`FLAT_FIELD_*`) are single values in `params.py`, and refinement reads only the
static region. `tests/test_detect.py` checks that refined corners are
bit-identical across all 25 configurations for the same capture geometry.

**Validation (synthetic, pure geometry).** A generated frame is imaged through a
known homography (2× supersampled, area-averaged), and the front end is a stub
returning the true corners ±2 px. Over magnifications of 0.8–3, oblique and
rotated views, blur up to 1.5 px and noise up to 4 code values, the recovered
homography matches to ≤ 0.06 source px across the whole code (the test asserts
< 0.1 px), and symbols decode with zero errors where the pilot says they
should.

**Not yet validated:** anything lens-dependent. The model is a pure homography,
so lens distortion inside the frame is not corrected. The 16-corner reprojection
error exposes it only at the markers; look at the cell grid in the middle of the
overlay on the first real capture.

## 16. Ingest (build step 5b)

Implements [docs/capture-format.md](docs/capture-format.md), schema version 1,
field for field. `ingest/manifest.py` is the only module that knows the
contract, and it adds no fields.

```bash
python -m prism_share.ingest.pull d07_a30_lux200_oled          # adb pull, then validate
python -m prism_share.ingest.ingest data/d07_a30_lux200_oled   # validate and report (--detect needs the front end)
```

| Rule (from the contract) | Enforcement |
|---|---|
| 1. unknown `schema_version` | run rejected |
| 2. no laptop-side run definition | run rejected; the message lists the known run ids, because `run_id` is typed by hand on the phone |
| 3. tainted frames | dropped; count per `taint_reasons` value in the report and `ingest_report.json` |
| 4. `frames.tainted / frames.written > 0.01` | run refused: "must be repeated" (exactly 1 % passes) |
| 5. `written != requested` | warning |
| 6. resolution, YUV matrix and range from the manifest | read, never assumed; every plane is checked against `capture_resolution`; values the contract does not define reject the run |

Ingest also rejects files that break the contract: missing or mistyped
fields, taint reasons outside the closed set, `tainted` inconsistent with
`taint_reasons`, `frames.jsonl` counts that contradict the manifest, and
and a `pixel_format` that is neither documented path. Files are always found
through `file_stem`, never rebuilt from `index`.

Where the contract is silent, ingest refuses rather than guessing:

* **`yuv_matrix` / `yuv_range` values are not enumerated.** Only `BT601`,
  `BT709`, `limited` and `full` are accepted.
* **`capture_resolution` order is not stated.** It is read as [width, height]
  (the Camera2 `Size` order) and verified against every PNG.
* **Nothing marks reference-frame captures or says which code frame was on
  screen.** Neither needs a field: every frame carries its own index band
  (§3.1), and index 0 is the reference.

`color.pixel_format` selects the path: `YUV_420_888` (Y, U, V and RGB PNGs) or
`JPEG` (one `frame_NNNN.jpg` per frame, a comparison baseline that is never a
headline measurement). Any other value rejects the run.

**Laptop-side run definitions** (`ingest/rundef.py`). The contract makes
`run_id` the join key with a laptop-side run definition, but does not define
that file. It is defined here, and written by
`python -m prism_share.ingest.rundef new`:

```yaml
# experiments/runs/<run_id>.yaml
run_id: d07_a30_lux200_oled      # must equal manifest.json's run_id
frames: data/runs/c4_px8          # folder written by `display frames` (frames.json)
displayed: [3]                    # exactly ONE block_id: the static frame held for the whole capture
condition: {distance_m: 0.7, illuminance_lux: 200, display: oled}
clip_check:                       # shown by `display measure` right after locking
  brightest: {colour_depth: 1, cell_px: 10, frames: data/clip_check/c1_px10, frame_mean_linear: 0.43751, stops_vs_reference: 0.506}
  darkest:   {colour_depth: 4, cell_px: 4,  frames: data/clip_check/c4_px4,  frame_mean_linear: 0.21741, stops_vs_reference: -0.502}
```

One run holds one codec configuration and **one static frame**. Anything else
in `displayed` (a list of several, `all`, nothing) rejects the run. The
configuration and payload come from `frames.json`, so there is a single source
of truth. On load, the frame's ground truth is regenerated and must reproduce
the hash recorded at display time, which catches any change to the codec
between display and analysis. A capture whose band names any other frame is
counted as detected but not recoverable; it is never matched to ground truth,
and never excluded.

`clip_check` is required. `rundef new` fills it from the same linear-luminance
table as [docs/reference_frame.md](docs/reference_frame.md), picking the
brightest and darkest code configurations (currently mono at 10 px, +0.51
stops, and 4 colours at 4 px, −0.50 stops; 2 colours at 4 px is within
frame-to-frame spread of the darkest). It also writes one deterministic frame of
each to `data/clip_check/`, shared by every run. A run id that already exists is
never overwritten.

## 17. Captured-data sweep and figures (build step 6)

`analysis/sweep.py`:

```bash
python -m prism_share.analysis.sweep data/<run_id> ... --out data/sweep --figures docs/figures
```

For each run, in two passes:

1. **Detect** every capture on its Y plane (the JPEG image on that path), then
   read its **index band**. Index 0 means a capture of the reference frame:
   those are excluded from yield and, with `--flat-field`, the first one
   supplies the run's flat field.
2. **Decode** every code capture from each pixel source with each decoder
   variant, against the ground truth of the frame its band names. The sources
   are: `y` (monochrome only), the phone's `rgb`, and our own `yuv_nearest` /
   `yuv_bilinear` conversion of the native planes using the manifest's matrix
   and range; on the JPEG path there is one source, `jpeg`. Detection geometry
   is shared, so the sources differ only in their pixels.

**The mechanism experiment** is this comparison: the same captures decoded from
the native quarter-resolution U and V planes (upsampled by a method we choose
and record) against the phone's own upsampled RGB. It replaces the RAW
demosaic comparison, because the target device does not expose RAW; there is
no RAW/DNG path anywhere in the project.

**Three-outcome yield.** Every code frame is exactly one of: detection failed,
detected but not recoverable (including a band index no displayed frame
carries), or recovered.
This holds for the configuration's own RS(n, k) (`fixed_*` columns) and for the
goodput-maximising RS (`best_*` columns). Both are counted on the run's
**evaluation half** (odd capture indices). The `best_*` code is chosen on the
selection half (even indices), and `n_selection_frames` / `n_evaluation_frames`
record the split (§10.1). `--in-sample` chooses and scores on all frames, with a
warning, for measuring the bias only. Frames below
`--min-px-per-cell` are excluded and counted (`n_below_resolution`).

**Two goodput columns.** Every goodput is reported twice:

| Column | Meaning |
|---|---|
| `*_goodput_mbps` | as measured: payload of the frame as drawn, with the index band |
| `*_goodput_band_credited_mbps` | the band's cells credited back: same RS code, same measured yield, payload recomputed for the grid without the band |

`band_cells_lost` and `band_cell_cost_pct` state the cost per configuration
(1 078 cells, 2.83 % at 4 px; 280 cells, 3.57 % at 10 px), and
`*_payload_bytes` / `*_payload_bytes_band_credited` give both payloads.
`winners.csv` lists the winning configuration per device, condition, pixel
source and decoder under each column, with `same_winner`; the sweep warns if
the band's cost changes any winner. If the headline holds in both columns, the
band is demonstrably not driving it. Crediting assumes the displaced cells would
have had the same error statistics as the rest of the frame.

**Tables.** `frames.parquet` has one row per capture × pixel source × decoder.
`summary.parquet` / `.csv` has one row per run × pixel source × decoder; the run
fixes configuration, condition and device. `winners.csv` is described above.

**Figures** (`plots.captured_figures`): goodput surface, three-outcome bars,
error decomposition and per-device comparison. The goodput surface and the
per-device comparison show both goodput columns side by side on one scale, and
the credited panel states the band's cost in every cell. Every figure shows its
run ids, device model, pixel source, decoder, RS policy, kernel, flat-field
state and YUV matrix/range (with "read" or "assumed"), plus the number of frames
behind **each** point.
The figures of the pre-registered decoder go under `headline/` and every other
decoder's under `sensitivity/`, and the footer labels each decoder
"(pre-registered headline)" or "(sensitivity)".

---

## Assumptions and limitations (so far)

1. The display shows 8-bit code values unmodified. No gamma or ICC handling is
   applied anywhere in this package; that the OS does not add any is checked
   per setup with `display verify`, and what the panel emits is characterised
   with `display calibrate`.
2. Palette distances are Euclidean in **code values**. This is a proxy for what
   an ISP-processed camera image records; it is not a perceptual or linear-light
   metric.
3. The level 128 in `PALETTE_LEVELS` is a code value. On a gamma-2.2 panel it
   emits about 22 % of full-scale light.
4. The glyph distance metric is pixel Hamming distance. The smoothness
   constraint on the pool is what accounts for optical blur; distances are not
   computed after a modelled PSF.
5. The decoder assumes rectification is accurate to well under one pixel. Cell
   positions are fixed to the nominal grid, and nothing is re-aligned per cell.
6. Bit-to-cell mapping is plain binary. There is no Gray coding of glyph or
   colour indices, which would not help anyway because RS works on bytes and
   any wrong symbol already costs at least one byte.
7. Shape and colour bits share codewords. `ecc_sim` can still evaluate
   alternative mappings from the same captures, because the per-cell decisions
   are recorded.
8. Linear-luminance calculations (reference frame, exposure mismatch) assume an
   sRGB panel. They predict, not measure, the light emitted.
9. Synthetic degradations are applied one at a time in the pilot, and there is
   no Bayer mosaic/demosaic degradation yet (see docs/pilot.md).
10. Auto-exposure is assumed to meter the mean linear luminance of the frame;
    real AE algorithms may weight the centre or use highlights.
11. Which frame was on screen is read from its index band, never inferred from
    content. Measurement runs hold one static frame, so no capture can be torn
    between frames. A capture whose band names another frame (a protocol slip)
    counts as not recoverable.
12. A capture whose detection failed carries no readable index, so if the phone
    captured reference frames and one of them failed detection, it is counted
    as a failed code frame.
13. The mechanism experiment compares native U/V planes against upsampled RGB.
    The target device exposes no RAW, so there is no demosaic comparison and no
    RAW/DNG path.
14. The band-credited goodput column assumes the cells the band displaces would
    have had the same error statistics as the rest of the frame. The band sits
    near the top edge, where vignetting and lens falloff are worse than average,
    so if anything the credit is slightly generous.

## Decode-benchmark bundle (Android spike)

```
python -m prism_share.export.bench --out data/bench/ --frames 50 \
    --package io.github.ahmedov.prismshare.testspike
```

`--package` is the Android application ID of the app that reads the bundle. It is
required and has no default, because the generated mkdir, push and chmod commands
target that app's storage. An ID that does not parse is refused before anything
is written.

This bundle is for a Kotlin decoder in a separate project that cannot see this
source. The benchmark measures decode compute, so the bundle carries everything
as data and the consumer ports no layout logic.

* **`manifest.json`** lists every configuration folder (`c1_px4`, `c16_px4`)
  and every file in each, so the consumer enumerates instead of guessing.
* **Each folder** holds the frames (1024×1024 lossless PNG, no degradation,
  byte for byte what `display frames` writes), `params.json` (flat
  `CodecParams`) and `ground_truth.json` (per frame: index band value, and
  `glyph_ids` / `colour_ids` parallel to `cells`).
* **`codec.json`** holds `glyphs` at rendered pixel size, `palette` in
  encoder index order, `background`, and `cells` (x, y, width, height of every
  data cell, in decode order). It also holds the other two things the decoder
  reads, as rectangles: the `index_band` blocks, each with the bit it carries,
  and the white/black `level_references` used to normalise levels.

`tests/test_bench_export.py` acts as the consumer. Starting from
`manifest.json`, it redraws every frame from the JSON alone and requires every
cell, band block and reference pixel to match the PNG. `data/bench/README.md`
gives the frame format version, payload bytes per frame, and the `adb push`
commands, ending with `adb shell chmod -R o+rX` on the pushed folder. The chmod
is required: `adb push` creates directories owned by `shell` in group
`ext_data_rw`, which the app process is not in, so the app cannot read the
pushed files until permissions are widened.

## Build plan

1. ✅ params, glyphs, palette + tests
2. ✅ encoder, decoder, ECC, fountain + round-trip at every colour depth × cell size
3. ✅ Synthetic degradation: blur, noise, perspective warp, white-balance shift, chroma subsampling; pilot study
4. ✅ `transmit/display.py` + reference frame
5. ◐ `analysis/detect.py` geometry ✅, fiducial front end (needs real captures) ❌; `ingest/` ✅
6. ✅ metrics (shape SER, colour SER, three-outcome yield), `ecc_sim`, sweep → Parquet, provenance-stamped plots
7. dashboard (optional)
