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

# --------------------------------------------------------------------------- #
# Reference frame (see transmit/reference.py)
# --------------------------------------------------------------------------- #

#: Whole-frame mean linear luminance the reference frame is built to. Pinned so
#: the reference is identical for every run. Value: geometric midpoint of the
#: darkest (0.2178, 2 colours / 4 px) and brightest (0.4470, mono / 10 px)
#: sweep configurations, which minimises the worst-case exposure mismatch
#: (about +/-0.52 stops). See transmit/reference.py and docs/reference_frame.md.
REFERENCE_TARGET_LINEAR_MEAN: Final[float] = 0.312
#: Side of the ordered-dither (Bayer) threshold matrix; must be a power of two.
REFERENCE_DITHER_SIZE: Final[int] = 16
#: Screen pixels per dither element: 2 keeps every feature >= 2 px, like the glyphs.
REFERENCE_DITHER_BLOCK_PX: Final[int] = 2

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
        """Unsigned 32-bit hash of the canonical JSON, embedded in every frame."""
        digest = hashlib.sha256(self.to_json().encode("ascii")).digest()
        return int.from_bytes(digest[:4], "big")

    def replace(self, **changes: int) -> CodecParams:
        return dataclasses.replace(self, **changes)
