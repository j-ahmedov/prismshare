"""Codec parameters: the single place where every tunable number lives.

Two kinds of numbers are defined here:

* ``CodecParams`` fields are the independent variables of the experiment. Every
  function that generates or reads a frame takes a ``CodecParams``.
* Module-level constants are fixed design choices that are *not* swept. They are
  documented in README.md ("Fixed design constants") so the thesis can state them.

No other module may hard-code a codec number (tests/test_no_magic_numbers.py
enforces that the literals 4 and 8 appear nowhere else in the package).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

# --------------------------------------------------------------------------- #
# Sweep domains
# --------------------------------------------------------------------------- #

#: Symbol colour depths with a defined palette. 1 means monochrome.
ALLOWED_COLOUR_DEPTHS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)
#: Cell sizes (screen pixels per glyph side) swept in the thesis.
SWEEP_CELL_PX: Final[tuple[int, ...]] = (4, 5, 6, 8, 10)
#: Smallest cell for which a 16-glyph set with a useful distance exists.
MIN_CELL_PX: Final[int] = 4

#: Bumped whenever the pixel layout of a frame changes. It enters
#: CodecParams.fingerprint(), so a frame drawn by an older format fails the
#: fingerprint check instead of being decoded with the wrong geometry.
#: 1 = fiducials + cell grid. 2 = adds the index band (2026-09-12).
FRAME_FORMAT_VERSION: Final[int] = 2

BITS_PER_BYTE: Final[int] = 8
#: Reed-Solomon over GF(2^8): a codeword is at most 255 symbols.
RS_MAX_CODEWORD: Final[int] = 255

# --------------------------------------------------------------------------- #
# Frame geometry that never varies with CodecParams
# --------------------------------------------------------------------------- #
# Layout from the frame edge inwards, at each corner:
#   BORDER_PX white | fiducial FIDUCIAL_PX | FIDUCIAL_MARGIN_PX white | data
# Everything inside the KEEPOUT_PX corner squares plus the border band is the
# "static region": pixel-identical for every CodecParams with the same frame_px.

#: White quiet zone around the whole code.
BORDER_PX: Final[int] = 16
#: OpenCV ArUco dictionary the fiducials are drawn from (used by detect.py).
FIDUCIAL_DICTIONARY: Final[str] = "DICT_4X4_50"
#: Interior bits per marker side; must match FIDUCIAL_DICTIONARY.
FIDUCIAL_BITS: Final[int] = 4
#: Black border around the marker interior, in modules.
FIDUCIAL_BORDER_BITS: Final[int] = 1
#: Screen pixels per marker module.
FIDUCIAL_MODULE_PX: Final[int] = 12
#: White margin between the marker and the data region.
FIDUCIAL_MARGIN_PX: Final[int] = 12
#: Marker ids placed at the corners, clockwise from top-left: TL, TR, BR, BL.
#: Distinct ids make the orientation of a captured frame unambiguous.
FIDUCIAL_IDS: Final[tuple[int, ...]] = (0, 1, 2, 3)
#: Interior bit patterns of FIDUCIAL_IDS in FIDUCIAL_DICTIONARY, row-major,
#: 1 = white. Hard-coded so rendering never depends on the OpenCV version;
#: tests/test_layout.py checks them against cv2.aruco.
FIDUCIAL_PATTERNS: Final[tuple[tuple[str, ...], ...]] = (
    ("1011", "0101", "0011", "0010"),
    ("0000", "1111", "1001", "1010"),
    ("0011", "0011", "0010", "1101"),
    ("1001", "1001", "0100", "0110"),
)
FIDUCIAL_MODULES: Final[int] = FIDUCIAL_BITS + 2 * FIDUCIAL_BORDER_BITS
FIDUCIAL_PX: Final[int] = FIDUCIAL_MODULES * FIDUCIAL_MODULE_PX
#: Side of the square at each corner that holds no data cells.
KEEPOUT_PX: Final[int] = BORDER_PX + FIDUCIAL_PX + FIDUCIAL_MARGIN_PX

# --------------------------------------------------------------------------- #
# Index band: the frame's own identity, readable at every configuration
# --------------------------------------------------------------------------- #
# A reserved horizontal band of large black/white blocks carrying an 8-bit
# frame index, centred between the two top keep-out squares. Its geometry is
# constant for every configuration, exactly like the fiducials, and it never
# uses colour. Index 0 is the reference frame; code frames are 1 upward.

#: Bits of frame index.
INDEX_BAND_BITS: Final[int] = 8
#: Copies of the index across the band; the reader takes a majority vote.
INDEX_BAND_REPEATS: Final[int] = 3
#: Side of one block. An order of magnitude larger than any data cell, so the
#: band is readable wherever the code is, at every cell_px.
INDEX_BAND_BLOCK_PX: Final[int] = 32
#: Background margin around the band, keeping it clear of the white border and
#: of the data cells.
INDEX_BAND_MARGIN_PX: Final[int] = 8
#: Index reserved for the reference frame.
INDEX_BAND_REFERENCE: Final[int] = 0

#: The decoder's white level is sampled from the border band, and its black
#: level from the markers' black border modules, each inset by this many pixels
#: from every black/white edge so blur cannot mix the two.
WHITE_REF_INSET_PX: Final[int] = BORDER_PX // 4
BLACK_REF_INSET_PX: Final[int] = FIDUCIAL_MODULE_PX // 4

# --------------------------------------------------------------------------- #
# Colours (8-bit RGB code values, written to the display unmodified)
# --------------------------------------------------------------------------- #

WHITE_RGB: Final[tuple[int, int, int]] = (255, 255, 255)
BLACK_RGB: Final[tuple[int, int, int]] = (0, 0, 0)
#: Background of the data region (between and inside glyphs).
BACKGROUND_RGB: Final[tuple[int, int, int]] = BLACK_RGB
#: Ink colour of the colour_depth == 1 (monochrome) palette.
MONO_INK_RGB: Final[tuple[int, int, int]] = WHITE_RGB
#: Per-channel code values palette candidates are drawn from.
PALETTE_LEVELS: Final[tuple[int, ...]] = (0, 128, 255)
#: Rec. 709 luma weights x 10000, integer so palette selection is exact.
LUMA_WEIGHTS_709: Final[tuple[int, int, int]] = (2126, 7152, 722)

# --------------------------------------------------------------------------- #
# Glyph generation (see glyphs.py)
# --------------------------------------------------------------------------- #

#: Fixed seed of the glyph candidate pool. Deliberately NOT CodecParams.seed:
#: the glyph set is a property of (glyph_count, cell_px) and must not change
#: between runs that use different payload seeds.
GLYPH_POOL_SEED: Final[int] = 0x5052534D  # "PRSM"
#: Number of distinct candidate patterns to collect per cell_px.
GLYPH_POOL_SIZE: Final[int] = 3000
#: Give up collecting candidates after this many draws per wanted candidate.
GLYPH_POOL_MAX_DRAWS_PER_CANDIDATE: Final[int] = 50
#: Integer smoothing kernel applied to the candidate noise field (binomial).
GLYPH_SMOOTH_KERNEL: Final[tuple[int, ...]] = (1, 2, 1)
#: Number of separable smoothing passes.
GLYPH_SMOOTH_PASSES: Final[int] = 1
#: Deterministic restarts of the greedy max-min selection.
GLYPH_SELECT_RESTARTS: Final[int] = 16
#: Bumped whenever the generation algorithm changes; stored in the cache file.
GLYPH_ALGORITHM_VERSION: Final[int] = 1

# --------------------------------------------------------------------------- #
# Framing (see framing.py)
# --------------------------------------------------------------------------- #

#: Frame header, big-endian: block_id, n_source_blocks, payload_len,
#: params_fingerprint. Four unsigned 32-bit integers.
HEADER_FORMAT: Final[str] = ">IIII"
#: CRC-32 (zlib polynomial) over header + block, appended to the frame data.
CRC_FORMAT: Final[str] = ">I"

# --------------------------------------------------------------------------- #
# Fountain code (see fountain.py)
# --------------------------------------------------------------------------- #

#: Repair frames generated by default, as a fraction of source blocks...
FOUNTAIN_REPAIR_FRACTION: Final[float] = 0.25
#: ...but never fewer than this many.
FOUNTAIN_MIN_REPAIR: Final[int] = 4

# --------------------------------------------------------------------------- #
# Goodput (see analysis/metrics.py)
# --------------------------------------------------------------------------- #

#: Frame rate used to turn payload per frame into goodput. A documented
#: constant: capture timing never enters a result.
ASSUMED_FPS: Final[float] = 30.0

#: RS codes are chosen out of sample. A frame at position p (pilot: frame
#: index; capture: the capture's ``index`` in frames.jsonl, assigned before any
#: analysis) is in the *selection* half iff p % RS_SELECTION_PERIOD == 0 and in
#: the *evaluation* half otherwise. The goodput-maximising RS(n, k) is chosen on
#: the selection half and goodput is scored only on the evaluation half. The
#: halves interleave rather than split first/second so that slow drift over a
#: run (panel warm-up, room light) lands in both equally instead of
#: separating them. The rule depends on position only, so every configuration
#: in a condition gets the same split and comparisons stay paired.
RS_SELECTION_PERIOD: Final[int] = 2

#: (colour_depth, cell_px) of the decode-benchmark bundle for the Android spike
#: (prism_share/export/bench.py): the two ends of the colour axis at the
#: smallest cell, i.e. the most cells and the most work per frame.
BENCH_CONFIGURATIONS: Final[tuple[tuple[int, int], ...]] = ((1, 4), (16, 4))

#: Version of the benchmark bundle's ground_truth.json layout. Bump on any change
#: a reader could notice.
BENCH_SCHEMA_VERSION: Final[int] = 1

# --------------------------------------------------------------------------- #
# Colour spaces (see prism_share/colourspace.py)
# --------------------------------------------------------------------------- #

#: Luma coefficients (Kr, Kb) of the Y'CbCr matrices; Kg = 1 - Kr - Kb.
YUV_MATRICES: Final[dict[str, tuple[float, float]]] = {
    "bt601": (0.299, 0.114),
    "bt709": (0.2126, 0.0722),
}
#: Default Y'CbCr matrix for simulated YUV_420_888 captures.
YUV_DEFAULT_MATRIX: Final[str] = "bt601"
#: 8-bit limited ("video") range: Y' in [16, 235], Cb/Cr in [16, 240].
YUV_LIMITED_Y_OFFSET: Final[int] = 16
YUV_LIMITED_Y_SPAN: Final[int] = 219
YUV_LIMITED_C_SPAN: Final[int] = 224
#: Chroma zero level (both ranges) and full-range span.
YUV_C_OFFSET: Final[int] = 128
YUV_FULL_SPAN: Final[int] = 255
#: Chroma subsampling factor of YUV_420_888 (per axis).
YUV_420_FACTOR: Final[int] = 2
#: Linear-light luminance weights of sRGB / Rec. 709 primaries.
LINEAR_LUMINANCE_WEIGHTS: Final[tuple[float, float, float]] = (0.2126, 0.7152, 0.0722)

# --------------------------------------------------------------------------- #
# Decoder options (see decoder.py)
# --------------------------------------------------------------------------- #

#: Luma matrix used when the decoder reads shape from luma.
DECODER_LUMA_MATRIX: Final[str] = "bt601"
#: 'saturated' colour estimator: average this fraction of a cell's ink pixels,
#: the most saturated ones (the least diluted by neighbouring black).
COLOUR_CORE_FRACTION: Final[float] = 0.25
#: PRE-REGISTERED headline decoder (2026-09-14, before any real capture exists).
#: Every headline number, table and figure uses it; the other three variants are
#: reported as sensitivity analysis, never promoted. Reasons (README, section 8.1):
#: shape from luma because Y is the only full-resolution plane of YUV_420_888;
#: colour from the most saturated ink pixels because 4:2:0 dilutes chroma at every
#: glyph edge on every capture. Changing these values invalidates the pre-registration.
HEADLINE_SHAPE_CHANNEL: Final[str] = "luma"
HEADLINE_COLOUR_ESTIMATOR: Final[str] = "saturated"
HEADLINE_DECODER_REGISTERED: Final[str] = "2026-09-14"
#: A winner whose goodput exceeds the comparison by less than this (percent) is a
#: near-tie and is flagged wherever winners are reported.
NEAR_TIE_MARGIN_PCT: Final[float] = 5.0

# --------------------------------------------------------------------------- #
# Reference frame (see transmit/reference.py)
# --------------------------------------------------------------------------- #

#: Whole-frame mean linear luminance the reference frame is built to. Pinned so
#: the reference is identical for every run. Value: geometric midpoint of the
#: darkest (0.2174, 4 colours / 4 px; 2 colours / 4 px is within frame-to-frame spread)
#: and brightest (0.4375, mono / 10 px)
#: sweep configurations, which minimises the worst-case exposure mismatch
#: (about +/-0.50 stops). See transmit/reference.py and docs/reference_frame.md.
REFERENCE_TARGET_LINEAR_MEAN: Final[float] = 0.308
#: Side of the ordered-dither (Bayer) threshold matrix; must be a power of two.
REFERENCE_DITHER_SIZE: Final[int] = 16
#: Screen pixels per dither element: 2 keeps every feature >= 2 px, like the glyphs.
REFERENCE_DITHER_BLOCK_PX: Final[int] = 2

# --------------------------------------------------------------------------- #
# Detection and rectification (see analysis/detect.py)
# ONE parameter set for every configuration: nothing here may depend on
# colour_depth or cell_px (the fiducials are identical so detection is too).
# --------------------------------------------------------------------------- #

#: Resampling kernels for rectification, name -> OpenCV interpolation flag name.
RECTIFY_KERNELS: Final[dict[str, str]] = {
    "nearest": "INTER_NEAREST",
    "bilinear": "INTER_LINEAR",
    "lanczos": "INTER_LANCZOS4",
}
RECTIFY_DEFAULT_KERNEL: Final[str] = "bilinear"
#: Intensity profiles taken across each marker edge during sub-pixel refinement.
REFINE_EDGE_SAMPLES: Final[int] = 24
#: Part of each edge that is sampled, as fractions of its length (corners are
#: rounded by blur, so the ends are avoided and corners come from line fits).
REFINE_EDGE_SPAN: Final[tuple[float, float]] = (0.15, 0.85)
#: Half-length of each profile in marker modules: must stay inside the black
#: border ring (1 module) and the white margin (1 module).
REFINE_PROFILE_HALF_MODULES: Final[float] = 0.75
#: Sampling step along a profile, in source pixels.
REFINE_PROFILE_STEP_PX: Final[float] = 0.25
#: Refinement passes (each re-centres profiles on the previous fit).
REFINE_ITERATIONS: Final[int] = 3
#: Minimum white-minus-black step (8-bit code values) for an edge to count as found.
REFINE_MIN_EDGE_CONTRAST: Final[float] = 20.0
#: Side, in frame pixels, of the window over which the flat-field gain is estimated.
FLAT_FIELD_WINDOW_FRAME_PX: Final[int] = 64

# --------------------------------------------------------------------------- #
# Captured-data analysis (see analysis/sweep.py)
# --------------------------------------------------------------------------- #

#: Pixel sources a captured YUV_420_888 frame can be decoded from:
#: 'y' = the Y plane alone (monochrome only), 'rgb' = the phone's own RGB PNG,
#: 'yuv_nearest' / 'yuv_bilinear' = our conversion of the native Y, U, V planes
#: with the manifest's matrix and range, chroma upsampled by that method.
PIXEL_SOURCES: Final[tuple[str, ...]] = ("y", "rgb", "yuv_nearest", "yuv_bilinear")
#: The JPEG comparison path carries one image per frame, so one pixel source.
JPEG_PIXEL_SOURCE: Final[str] = "jpeg"
#: Fraction of each index-band block (centred) averaged when reading a bit,
#: keeping block edges - the only part blur can reach - out of the measurement.
INDEX_BAND_SAMPLE_FRACTION: Final[float] = 0.5
#: Stand-in maximum codeword error count for a frame that was not detected:
#: larger than any codeword, so no RS(n, k) ever recovers it.
UNDETECTED_CW_ERRORS: Final[int] = 10**6

# --------------------------------------------------------------------------- #
# Display (see transmit/display.py)
# --------------------------------------------------------------------------- #

#: Colour of the screen outside the frame. Constant for every configuration.
DISPLAY_SURROUND_RGB: Final[tuple[int, int, int]] = BLACK_RGB
#: Default interval between frames in free-running mode.
DISPLAY_DEFAULT_INTERVAL_MS: Final[int] = 200
#: Grey ramp of the calibration sequence: this many evenly spaced levels 0..255.
CALIBRATION_GREY_LEVELS: Final[int] = 17


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class CodecParams:
    """Complete description of one code configuration.

    Two CodecParams that compare equal produce byte-identical frames for the
    same payload. Defaults are the reference configuration used in the README.
    """

    colour_depth: int = 4
    """Number of symbol colours: 1, 2, 4, 8 or 16. 1 = monochrome."""
    cell_px: int = 8
    """Glyph side in screen pixels; each glyph is a cell_px x cell_px bitmap."""
    cell_gap_px: int = 1
    """Background pixels between adjacent cells (and around the grid)."""
    glyph_count: int = 16
    """Number of distinct glyph shapes; must be a power of two."""
    frame_px: int = 1024
    """Side of the square frame in screen pixels."""
    ecc_data: int = 125
    """Reed-Solomon data symbols (bytes) per codeword, k."""
    ecc_total: int = 155
    """Reed-Solomon total symbols (bytes) per codeword, n."""
    seed: int = 0
    """Seed of every payload-dependent pseudo-random stream."""

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field.name} must be int, got {value!r}")
        if self.colour_depth not in ALLOWED_COLOUR_DEPTHS:
            raise ValueError(
                f"colour_depth must be one of {ALLOWED_COLOUR_DEPTHS}, got {self.colour_depth}"
            )
        if self.cell_px < MIN_CELL_PX:
            raise ValueError(f"cell_px must be >= {MIN_CELL_PX}, got {self.cell_px}")
        if self.cell_gap_px < 0:
            raise ValueError(f"cell_gap_px must be >= 0, got {self.cell_gap_px}")
        if self.glyph_count < 2 or not _is_power_of_two(self.glyph_count):
            raise ValueError(f"glyph_count must be a power of two >= 2, got {self.glyph_count}")
        if self.frame_px <= 2 * KEEPOUT_PX:
            raise ValueError(f"frame_px must exceed {2 * KEEPOUT_PX}, got {self.frame_px}")
        if not 0 < self.ecc_data < self.ecc_total <= RS_MAX_CODEWORD:
            raise ValueError(
                "need 0 < ecc_data < ecc_total <= "
                f"{RS_MAX_CODEWORD}, got k={self.ecc_data}, n={self.ecc_total}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    # ----------------------------------------------------------------- derived

    @property
    def glyph_bits(self) -> int:
        """Bits carried by the glyph shape of one cell."""
        return self.glyph_count.bit_length() - 1

    @property
    def colour_bits(self) -> int:
        """Bits carried by the colour of one cell (0 for monochrome)."""
        return self.colour_depth.bit_length() - 1

    @property
    def bits_per_cell(self) -> int:
        return self.glyph_bits + self.colour_bits

    @property
    def pitch_px(self) -> int:
        """Distance between the top-left corners of adjacent cells."""
        return self.cell_px + self.cell_gap_px

    @property
    def glyph_weight(self) -> int:
        """Ink pixels in every glyph (glyphs are constant-weight)."""
        return (self.cell_px * self.cell_px) // 2

    @property
    def ecc_parity(self) -> int:
        """Parity symbols per codeword, n - k."""
        return self.ecc_total - self.ecc_data

    @property
    def ecc_correctable(self) -> int:
        """Symbol errors per codeword RS(n, k) is guaranteed to correct."""
        return self.ecc_parity // 2

    @property
    def label(self) -> str:
        """Short human-readable identifier, stable and filesystem-safe."""
        return (
            f"c{self.colour_depth}_px{self.cell_px}_gap{self.cell_gap_px}"
            f"_g{self.glyph_count}_f{self.frame_px}"
            f"_rs{self.ecc_total}-{self.ecc_data}_s{self.seed}"
        )

    # ----------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodecParams:
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown CodecParams fields: {sorted(unknown)}")
        return cls(**{k: int(v) for k, v in data.items()})

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, no whitespace. Basis of fingerprint()."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> int:
        """Unsigned 32-bit hash of the canonical JSON and the frame format version.

        The format version is included so a capture of a frame drawn by an
        earlier layout fails the header check rather than being decoded with
        today's geometry.
        """
        digest = hashlib.sha256(f"v{FRAME_FORMAT_VERSION}|{self.to_json()}".encode("ascii")).digest()
        return int.from_bytes(digest[:4], "big")

    def replace(self, **changes: int) -> CodecParams:
        return dataclasses.replace(self, **changes)
