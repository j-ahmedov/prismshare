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
> and the transmitter with its reference frame (4). Detection, ingest, the
> capture sweep and the dashboard (5–7) are not built yet. Parts of step 6
> (`analysis/metrics.py`, `analysis/ecc_sim.py`, `analysis/plots.py`) exist
> because the pilot needed them.

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
  transmit/
    display.py    fullscreen player, calibration patches, screenshot verification
    reference.py  the reference frame the capture app locks onto
tests/
experiments/      YAML run definitions (pilot.yaml)
docs/             pilot report and figures, reference-frame luminance table
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
| `YUV_DEFAULT_MATRIX` | `bt601` | Y′CbCr matrix of simulated YUV_420_888 (BT.709 selectable). |
| `YUV_LIMITED_*`, `YUV_C_OFFSET` | 16, 219, 224, 128 | 8-bit limited-range Y′CbCr. |
| `YUV_420_FACTOR` | 2 | Chroma subsampling per axis in YUV_420_888. |
| `DECODER_LUMA_MATRIX` | `bt601` | Luma used by the `luma` shape channel. |
| `COLOUR_CORE_FRACTION` | 0.25 | Share of ink pixels averaged by the `saturated` colour estimator. |
| `REFERENCE_TARGET_LINEAR_MEAN` | 0.312 | Reference frame's whole-frame mean linear luminance (see §14). |
| `REFERENCE_DITHER_SIZE`, `_BLOCK_PX` | 16, 2 | Bayer matrix side; screen pixels per dither element. |
| `DISPLAY_SURROUND_RGB` | (0, 0, 0) | Screen outside the frame, for every configuration. |
| `DISPLAY_DEFAULT_INTERVAL_MS` | 200 | Free-running time per frame. |
| `CALIBRATION_GREY_LEVELS` | 17 | Grey ramp of `display calibrate`. |

## 3. Frame anatomy

A frame is a `frame_px × frame_px` RGB image (origin top-left, x right, y down):

1. **White border**, `BORDER_PX` wide, all round.
2. **Four keep-out squares** (`KEEPOUT_PX`) at the corners. Each is white and
   holds an ArUco marker (black on white) at offset `BORDER_PX` from both edges.
3. **Data region**: everything else. Black background with the cell grid.

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
  black, measured the way the decoder measures it (per-pixel channel maximum).
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
| 4 | 15 230 B | 19 105 B | 22 980 B | 26 730 B | 30 605 B |
| 5 | 10 605 B | 13 230 B | 15 855 B | 18 605 B | 21 230 B |
| 6 | 7 730 B | 9 605 B | 11 605 B | 13 480 B | 15 480 B |
| 8 | 4 605 B | 5 855 B | 6 980 B | 8 230 B | 9 355 B |
| 10 | 3 105 B | 3 855 B | 4 605 B | 5 480 B | 6 230 B |

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

**Decoder variants** (`DecoderOptions`). The pilot study compares four:

* shape channel `max` (default, as above) or `luma`, which reads shape, and
  ranks ink pixels, from Y′ (BT.601) of the normalised RGB;
* colour estimator `mean` (default, as above) or `saturated`, which averages
  only the most saturated `COLOUR_CORE_FRACTION` (¼) of the ink pixels.

No variant is best everywhere. `luma` is immune to chroma subsampling but weak
for dark-luma inks (pure blue) under noise and blur. `saturated` cuts colour
SER 1.8–6.7× under 4:2:0, but is far worse under strong noise. The codec
default is unchanged (`max`/`mean`). The pilot reports `luma`/`saturated` as its
headline because it has the smallest total loss; see
[docs/pilot.md](docs/pilot.md#decoder-ablation).

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
not measurements**. In short: with single degradations, a colour depth above 1
maximises goodput in 33 of 34 conditions. Chroma subsampling is the only
degradation that selectively harms colour, pushing the best depth down from 16
to 2–4, and the optimal cell size is set by blur alone. The report lists what
the model leaves out: Bayer demosaicing, combined degradations, realistic noise
and compression.

## 13. Transmitter (build step 4)

`transmit/display.py`:

```bash
python -m prism_share.transmit.display frames --colour-depth 4 --cell-px 8 --payload-bytes 200000 --out data/runs/c4_px8
python -m prism_share.transmit.display play data/runs/c4_px8 --reference     # free-running, opens with the reference frame
python -m prism_share.transmit.display play data/runs/c4_px8 --hold          # single-frame hold, arrow keys step
python -m prism_share.transmit.display calibrate                            # solid patches (add --auto to cycle)
python -m prism_share.transmit.display verify shot.png --frame data/runs/c4_px8/frame_00000.png
```

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
* **Keys:** space play/pause, → ↓ next, ← ↑ previous, `r` reference, `q`/Esc quit.
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
2. play a frame with `--hold`, take a lossless full-screen screenshot, and run
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
* **One reference for all runs, by design.** The code configurations span 0.218
  (2 colours, 4 px) to 0.447 (mono, 10 px) linear, about one stop, so no single
  frame can match them all. The target is pinned at their geometric midpoint,
  0.312, which limits the worst mismatch to ±0.53 stops and gives **every
  configuration identical camera settings**. A per-configuration reference
  (`--match DEPTH,CELL_PX`) is available, but using it would tie exposure to the
  parameter under test.
* `python -m prism_share.transmit.reference --report docs/reference_frame.md`
  prints and records every configuration's mean linear luminance and its
  mismatch in stops ([docs/reference_frame.md](docs/reference_frame.md)). Its
  PNG hash is pinned by `tests/test_reference.py`.

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

## Build plan

1. ✅ params, glyphs, palette + tests
2. ✅ encoder, decoder, ECC, fountain + round-trip at every colour depth × cell size
3. ✅ Synthetic degradation: blur, noise, perspective warp, white-balance shift, chroma subsampling; pilot study
4. ✅ `transmit/display.py` + reference frame
5. `analysis/detect.py` + `ingest/`
6. metrics (shape SER, colour SER, yield), `ecc_sim`, sweep → Parquet, plots
7. dashboard (optional)
