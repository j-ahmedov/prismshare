"""Locate, register and rectify a captured frame (build step 5a).

Two halves, deliberately separated:

1. **Front end - NOT IMPLEMENTED.** ``locate_fiducials`` decides which image
   regions are the four corner markers and returns each marker's four outer
   corners to within a couple of source pixels. How to do that robustly
   depends on the real lens, focus, glare and moire, so it will be designed on
   real captures; until then it raises ``NotImplementedError``. Its contract is
   ``FrontEnd`` / ``FrontEndResult`` below.
2. **Geometry - implemented, lens-independent.** Given coarse corners:
   sub-pixel refinement of each marker's outer square, marker centres,
   homography from the four centre correspondences, reprojection error,
   rectification at an integer scale k with an explicit resampling kernel,
   optional flat-field correction, cell centres, the detection record and a
   debug overlay.

Conventions
-----------
* Points are (x, y) in OpenCV's pixel-centre convention: the centre of pixel
  (col, row) is at (col, row). ``layout.fiducial_corners`` uses pixel-edge
  coordinates, so ideal points here are those minus 0.5.
* The homography ``H`` maps **frame coordinates** (pixel centres of the
  frame_px x frame_px code) to **source coordinates** (the captured image).
* One detector, one parameter set, for every configuration: nothing below
  reads ``colour_depth`` or ``cell_px`` except to report source pixels per
  cell. The refinement only ever samples the static region around the markers,
  which is pixel-identical across configurations (tests/test_detect.py checks
  that refined corners are bit-identical across all 25 configurations).

Sub-pixel refinement
--------------------
Each marker's outer boundary is a straight black/white step (black border ring
inside, white margin outside, each one module wide). For every edge,
``REFINE_EDGE_SAMPLES`` intensity profiles are taken across it (bilinear
sampling, over the middle ``REFINE_EDGE_SPAN`` of the edge, +/-
``REFINE_PROFILE_HALF_MODULES`` modules). On each profile the 50 % crossing
between the edge's own black and white levels locates the edge roughly; the
edge position is then the centroid of the intensity gradient in a window
centred there. The centroid is exact for an area-sampled sharp step (where the
50 % crossing alone is biased by up to ~0.1 px, depending on where the edge
falls within a pixel) and unbiased under any symmetric blur. A total-least-squares line is
fitted per edge, and the corners are the intersections of adjacent lines,
repeated ``REFINE_ITERATIONS`` times. Corners come from line intersections, so
corner rounding by blur does not bias them. Measured on synthetic captures:
<= 0.06 px homography error for magnifications 0.8-3, blur <= 1.5 px, noise
<= 4 code values (tests/test_detect.py asserts < 0.1 px).

Homography and reprojection error
---------------------------------
H is estimated from exactly four correspondences - the marker centres (the
intersection of each refined quad's diagonals, which is the projective image
of the square's centre) - as specified. Four points determine a homography
exactly, so the fit residual is identically zero and says nothing. The
reported ``reprojection_error_px`` is therefore measured on the 16 refined
marker corners, which are not used in the fit: RMS distance between them and
the ideal corners mapped through H. It exposes corner-localisation error and
any departure from a pure projective model (e.g. lens distortion) at the
markers.

Rectification
-------------
The output is (k*frame_px, k*frame_px[, C]) float32, where k is the smallest
integer not below the largest local magnification of H over the frame
(largest singular value of its Jacobian, on a grid), so no region of the
source is ever downsampled. Resampling uses the named kernel
(``RECTIFY_KERNELS``, default bilinear) via ``cv2.remap``; the kernel is
recorded. ``decoder.read_symbols`` infers k from the array shape.

Flat field
----------
The reference frame is a black/white *dither*, not a uniform field, so it
cannot be divided out pixel by pixel (black blocks would divide by ~0).
``FlatField.from_reference`` divides the local mean of the captured reference
by the local mean of the ideal pattern, both over ``FLAT_FIELD_WINDOW_FRAME_PX``
windows (whole dither periods), and normalises the result to median 1. That is
"the normalised reference" at the resolution a dither can provide, and local
means are insensitive to the lens's position-dependent blur. It assumes black is
dark relative to white: spatially varying glare leaks into the gain. Valid only
because exposure is locked for the whole run. Off by default, and the record
states whether it was applied.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from prism_share.codec.layout import fiducial_corners, grid_layout
from prism_share.codec.params import (
    FIDUCIAL_IDS,
    FIDUCIAL_MODULES,
    FLAT_FIELD_WINDOW_FRAME_PX,
    RECTIFY_DEFAULT_KERNEL,
    RECTIFY_KERNELS,
    REFINE_EDGE_SAMPLES,
    REFINE_EDGE_SPAN,
    REFINE_ITERATIONS,
    REFINE_MIN_EDGE_CONTRAST,
    REFINE_PROFILE_HALF_MODULES,
    REFINE_PROFILE_STEP_PX,
    CodecParams,
)

FloatArray = npt.NDArray[np.float64]

# --------------------------------------------------------------------------- #
# Front end: interface only
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FrontEndResult:
    """What a fiducial front end must return for one captured image.

    ``corners`` maps each marker id in ``FIDUCIAL_IDS`` to a (4, 2) array of
    that marker's OUTER corners in source pixels (pixel-centre convention),
    ordered as drawn: top-left, top-right, bottom-right, bottom-left of the
    marker in its own (unrotated) orientation - the order OpenCV ArUco uses.
    Accuracy of about +/-2 source pixels is enough; the geometry stage refines
    them. On failure, ``corners`` holds whatever was found and
    ``failure_reason`` says why; detection then fails without raising.
    """

    corners: dict[int, FloatArray]
    failure_reason: str | None = None


#: A front end takes the captured image (Y plane (H, W) or RGB (H, W, 3), 0-255)
#: and the CodecParams, and returns a FrontEndResult. It must use one fixed set
#: of parameters for every configuration.
FrontEnd = Callable[[npt.NDArray[Any], CodecParams], FrontEndResult]


def locate_fiducials(image: npt.NDArray[Any], params: CodecParams) -> FrontEndResult:
    """The real-lens fiducial front end. Deliberately not implemented yet.

    It must be designed and validated on real captures (focus, glare, moire,
    rolling-shutter banding), not on synthetic blur. See ``FrontEndResult``
    for the contract it has to meet.
    """
    raise NotImplementedError(
        "fiducial front end not implemented: it will be designed on real captures. "
        "Pass front_end= to detect() to supply coarse marker corners."
    )


# --------------------------------------------------------------------------- #
# Detection record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetectionRecord:
    """Everything downstream needs about one frame's detection. Flat and serialisable."""

    detected: bool
    failure_reason: str | None
    reprojection_error_px: float | None
    """RMS over the 16 refined marker corners (not used in the fit), source px."""
    reprojection_error_max_px: float | None
    fiducial_centres_px: tuple[tuple[float, float], ...] | None
    """Marker centres in source px, in FIDUCIAL_IDS slot order (TL, TR, BR, BL)."""
    quad_area_px: float | None
    """Area of the quadrilateral through the four fiducial centres, source px^2."""
    source_px_per_cell: float | None
    """Mean linear source pixels per cell pitch: sqrt(quad area / frame-space area) * pitch."""
    source_px_per_cell_min: float | None
    """Worst cell: smallest local magnification over all cell centres * pitch."""
    rectify_scale_k: int | None
    resampling_kernel: str
    flat_field_applied: bool
    homography: tuple[tuple[float, float, float], ...] | None
    """H, frame (pixel-centre) -> source (pixel-centre), row-major."""

    def to_row(self) -> dict[str, Any]:
        """Flatten for a table: centres and homography become scalar columns."""
        row = asdict(self)
        centres = row.pop("fiducial_centres_px")
        hom = row.pop("homography")
        for slot, marker_id in enumerate(FIDUCIAL_IDS):
            row[f"fiducial{marker_id}_x"] = None if centres is None else centres[slot][0]
            row[f"fiducial{marker_id}_y"] = None if centres is None else centres[slot][1]
        for i in range(3):
            for j in range(3):
                row[f"h{i}{j}"] = None if hom is None else hom[i][j]
        return row


@dataclass(frozen=True)
class Detection:
    record: DetectionRecord
    rectified: npt.NDArray[np.float32] | None = None
    refined_corners: FloatArray | None = field(default=None, repr=False)
    """(4 markers, 4 corners, 2) in source px, slot order."""


def _failed(reason: str, kernel: str) -> Detection:
    return Detection(
        DetectionRecord(
            detected=False, failure_reason=reason, reprojection_error_px=None, reprojection_error_max_px=None,
            fiducial_centres_px=None, quad_area_px=None, source_px_per_cell=None, source_px_per_cell_min=None,
            rectify_scale_k=None, resampling_kernel=kernel, flat_field_applied=False, homography=None,
        )
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


class RefinementError(ValueError):
    """An edge could not be localised; becomes a detection failure, never a crash."""


def ideal_marker_corners(frame_px: int) -> FloatArray:
    """(4 markers, 4 corners, 2) outer corners in frame pixel-centre coordinates."""
    return fiducial_corners(frame_px) - 0.5


def ideal_marker_centres(frame_px: int) -> FloatArray:
    return ideal_marker_corners(frame_px).mean(axis=1)


def luminance(image: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Single-plane intensity for geometry: the plane itself, or the channel mean of RGB."""
    arr = np.asarray(image, dtype=np.float32)
    return arr if arr.ndim == 2 else arr.mean(axis=2)


def _sample_bilinear(plane: npt.NDArray[np.float32], points: FloatArray) -> FloatArray:
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2).astype(np.float32)
    values = cv2.remap(plane, flat[:, :1], flat[:, 1:], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return values.reshape(shape).astype(np.float64)


def _intersect(p1: FloatArray, d1: FloatArray, p2: FloatArray, d2: FloatArray) -> FloatArray:
    """Intersection of lines p1 + s d1 and p2 + t d2."""
    matrix = np.array([d1, -d2]).T
    s, _ = np.linalg.solve(matrix, p2 - p1)
    return p1 + s * d1


def refine_marker(plane: npt.NDArray[np.float32], coarse: FloatArray) -> tuple[FloatArray, float]:
    """Sub-pixel outer corners of one marker from coarse corners. Returns (corners (4, 2), min edge contrast)."""
    corners = np.asarray(coarse, dtype=np.float64).copy()
    n = len(corners)
    min_contrast = math.inf
    for _ in range(REFINE_ITERATIONS):
        side = np.mean([np.linalg.norm(corners[(i + 1) % n] - corners[i]) for i in range(n)])
        half = REFINE_PROFILE_HALF_MODULES * side / FIDUCIAL_MODULES
        offsets = np.arange(-half, half + REFINE_PROFILE_STEP_PX / 2, REFINE_PROFILE_STEP_PX)
        lines = []
        for i in range(n):
            a, b = corners[i], corners[(i + 1) % n]
            direction = (b - a) / np.linalg.norm(b - a)
            outward = np.array([direction[1], -direction[0]])  # corners run clockwise in image coordinates
            ts = np.linspace(*REFINE_EDGE_SPAN, REFINE_EDGE_SAMPLES)
            base = a + ts[:, None] * (b - a)
            profiles = _sample_bilinear(plane, base[:, None, :] + offsets[None, :, None] * outward)  # dark -> bright
            tail = max(2, len(offsets) // 5)
            black, white = float(np.median(profiles[:, :tail])), float(np.median(profiles[:, -tail:]))
            contrast = white - black
            min_contrast = min(min_contrast, contrast)
            if contrast < REFINE_MIN_EDGE_CONTRAST:
                raise RefinementError(f"edge {i} contrast {contrast:.1f} < {REFINE_MIN_EDGE_CONTRAST:g}")
            level = (black + white) / 2
            window = half / 2  # gradient-centroid half-window around the 50 % crossing
            mids = (offsets[:-1] + offsets[1:]) / 2
            points = []
            for s, prof in enumerate(profiles):
                above = prof >= level
                crossings = np.flatnonzero(~above[:-1] & above[1:])
                if not len(crossings):
                    continue
                j = crossings[np.argmin(np.abs(offsets[crossings]))]  # the crossing nearest the current edge
                frac = (level - prof[j]) / (prof[j + 1] - prof[j])
                crossing = offsets[j] + frac * (offsets[j + 1] - offsets[j])
                # Edge = centroid of the gradient in a window centred on the crossing. Exact for an
                # area-sampled sharp step (where the 50 % crossing is biased by pixel phase) and
                # unbiased under symmetric blur.
                grad = np.diff(prof)
                sel = np.abs(mids - crossing) <= window
                weight = grad[sel].sum()
                if weight <= 0:
                    continue
                t = float((mids[sel] * grad[sel]).sum() / weight)
                points.append(base[s] + t * outward)
            if len(points) < REFINE_EDGE_SAMPLES // 2:
                raise RefinementError(f"edge {i}: only {len(points)} of {REFINE_EDGE_SAMPLES} profiles crossed the edge")
            pts = np.array(points)
            centroid = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - centroid)
            lines.append((centroid, vt[0]))
        corners = np.array([_intersect(*lines[(i - 1) % n], *lines[i]) for i in range(n)])
    return corners, min_contrast


def quad_centre(corners: FloatArray) -> FloatArray:
    """Intersection of the diagonals: the projective image of a square's centre."""
    return _intersect(corners[0], corners[2] - corners[0], corners[1], corners[3] - corners[1])


def homography_from_centres(source_centres: FloatArray, frame_px: int) -> FloatArray:
    """H (frame -> source) from the four marker-centre correspondences."""
    src = ideal_marker_centres(frame_px).astype(np.float32)
    dst = np.asarray(source_centres, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def project(homography: FloatArray, points: FloatArray) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    hom = np.c_[flat, np.ones(len(flat))] @ homography.T
    return (hom[:, :2] / hom[:, 2:]).reshape(pts.shape)


def local_magnification(homography: FloatArray, points: FloatArray) -> FloatArray:
    """Singular values (n, 2) of H's Jacobian at frame points: source px per frame px, per direction."""
    h = homography
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    w = h[2, 0] * flat[:, 0] + h[2, 1] * flat[:, 1] + h[2, 2]
    mapped = project(h, flat)
    jac = np.empty((len(flat), 2, 2))
    for r in range(2):
        for c in range(2):
            jac[:, r, c] = (h[r, c] - mapped[:, r] * h[2, c]) / w
    return np.linalg.svd(jac, compute_uv=False)


def choose_scale(homography: FloatArray, frame_px: int) -> int:
    """Smallest integer k with k >= the largest local magnification anywhere on the frame."""
    grid = np.linspace(-0.5, frame_px - 0.5, 33)
    gx, gy = np.meshgrid(grid, grid)
    smax = float(local_magnification(homography, np.c_[gx.ravel(), gy.ravel()]).max())
    return max(1, math.ceil(smax - 1e-9))


def rectify(image: npt.NDArray[Any], homography: FloatArray, frame_px: int, k: int, kernel: str) -> npt.NDArray[np.float32]:
    """Resample the source into a (k*frame_px)^2 frame-aligned image with the named kernel."""
    if kernel not in RECTIFY_KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {sorted(RECTIFY_KERNELS)}")
    size = k * frame_px
    # Rectified pixel u (centre convention) is frame coordinate x = (u - (k - 1) / 2) / k.
    to_frame = np.array([[1 / k, 0, -(k - 1) / (2 * k)], [0, 1 / k, -(k - 1) / (2 * k)], [0, 0, 1]])
    m = homography @ to_frame
    u = np.arange(size, dtype=np.float64)
    uu, vv = np.meshgrid(u, u)
    w = m[2, 0] * uu + m[2, 1] * vv + m[2, 2]
    map_x = ((m[0, 0] * uu + m[0, 1] * vv + m[0, 2]) / w).astype(np.float32)
    map_y = ((m[1, 0] * uu + m[1, 1] * vv + m[1, 2]) / w).astype(np.float32)
    src = np.asarray(image, dtype=np.float32)
    flag = getattr(cv2, RECTIFY_KERNELS[kernel])
    return cv2.remap(src, map_x, map_y, interpolation=flag, borderMode=cv2.BORDER_REPLICATE)


def rectify_with(image: npt.NDArray[Any], record: DetectionRecord, params: CodecParams,
                 flat_field: FlatField | None = None) -> npt.NDArray[np.float32]:
    """Rectify another pixel source of the same capture with an existing detection record.

    Detection runs once per capture (on the Y plane); every pixel source is then
    resampled with the same homography, scale and kernel, so sources differ only
    in their pixel values, never in their geometry.
    """
    if not record.detected or record.homography is None or record.rectify_scale_k is None:
        raise ValueError("cannot rectify with a failed detection")
    out = rectify(image, np.array(record.homography), params.frame_px, record.rectify_scale_k, record.resampling_kernel)
    return flat_field.apply(out) if flat_field is not None else out


def cell_centres(params: CodecParams) -> FloatArray:
    """(n_cells, 2) cell centres in frame pixel-centre coordinates, in cell-index order."""
    lay = grid_layout(params)
    offset = (params.cell_px - 1) / 2
    return np.c_[lay.cell_x + offset, lay.cell_y + offset].astype(np.float64)


def cell_centres_source(homography: FloatArray, params: CodecParams) -> FloatArray:
    """(n_cells, 2) cell centres in the captured image."""
    return project(homography, cell_centres(params))


def _shoelace(poly: FloatArray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


# --------------------------------------------------------------------------- #
# Flat field
# --------------------------------------------------------------------------- #


def _box_reduce(rectified: npt.NDArray[Any], frame_px: int) -> FloatArray:
    """Average each k x k block of a rectified image down to frame resolution."""
    arr = np.asarray(rectified, dtype=np.float64)
    k = arr.shape[0] // frame_px
    shape = (frame_px, k, frame_px, k) + arr.shape[2:]
    return arr.reshape(shape).mean(axis=(1, 3))


@dataclass(frozen=True)
class FlatField:
    """Multiplicative gain map at frame resolution, median 1, from a captured reference frame."""

    gain: FloatArray  # (frame_px, frame_px) or (frame_px, frame_px, C)

    @classmethod
    def from_reference(cls, rectified_reference: npt.NDArray[Any], params: CodecParams) -> FlatField:
        """Gain = local mean of the captured reference / local mean of the ideal reference, normalised.

        Local means over FLAT_FIELD_WINDOW_FRAME_PX windows (a whole number of
        dither periods) are insensitive to the lens's position-dependent blur,
        unlike any estimate built on the dither's fine detail. Assumes the
        display's black is dark relative to its white; spatially varying glare
        adds to the estimate (see README, flat field).
        """
        from prism_share.transmit.reference import reference_frame

        captured = _box_reduce(rectified_reference, params.frame_px)
        ideal = (np.asarray(reference_frame(params.frame_px))[:, :, 0] > 0).astype(np.float64)
        size = (FLAT_FIELD_WINDOW_FRAME_PX, FLAT_FIELD_WINDOW_FRAME_PX)
        ideal_mean = cv2.blur(ideal, size, borderType=cv2.BORDER_REFLECT)
        if captured.ndim == 3:
            ideal_mean = ideal_mean[:, :, None]
        captured_mean = cv2.blur(captured, size, borderType=cv2.BORDER_REFLECT).reshape(captured.shape)
        gain = captured_mean / np.maximum(ideal_mean, 1e-6)
        median = np.median(gain.reshape(-1, *gain.shape[2:]), axis=0)
        return cls(gain=gain / median)

    def apply(self, rectified: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        arr = np.asarray(rectified, dtype=np.float32)
        k = arr.shape[0] // self.gain.shape[0]
        g = np.repeat(np.repeat(self.gain, k, axis=0), k, axis=1)
        if arr.ndim == 3 and g.ndim == 2:
            g = g[:, :, None]
        return (arr / np.maximum(g, 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def detect(
    image: npt.NDArray[Any],
    params: CodecParams,
    *,
    kernel: str = RECTIFY_DEFAULT_KERNEL,
    flat_field: FlatField | None = None,
    front_end: FrontEnd = locate_fiducials,
    k: int | None = None,
) -> Detection:
    """Detect, register and rectify one captured frame. Failure is returned, never raised.

    ``NotImplementedError`` from the default front end does propagate: it is
    not a detection outcome, it means the front end has not been built yet.
    """
    if kernel not in RECTIFY_KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {sorted(RECTIFY_KERNELS)}")
    arr = np.asarray(image)
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] != 3):
        return _failed(f"unsupported image shape {arr.shape}", kernel)
    frame_px = params.frame_px

    found = front_end(arr, params)
    missing = [m for m in FIDUCIAL_IDS if m not in found.corners]
    if found.failure_reason or missing:
        reason = found.failure_reason or ""
        return _failed(f"front end: {reason}{'; ' if reason and missing else ''}{'missing markers ' + str(missing) if missing else ''}", kernel)

    plane = luminance(arr)
    try:
        refined = np.array([refine_marker(plane, np.asarray(found.corners[m], dtype=np.float64))[0] for m in FIDUCIAL_IDS])
        centres = np.array([quad_centre(c) for c in refined])
        homography = homography_from_centres(centres, frame_px)
    except (RefinementError, np.linalg.LinAlgError, cv2.error) as exc:
        return _failed(f"refinement: {exc}", kernel)
    if not np.all(np.isfinite(homography)):
        return _failed("degenerate homography", kernel)
    if np.linalg.det(homography[:2, :2]) <= 0:
        return _failed("homography reverses orientation (mirrored or marker order wrong)", kernel)

    predicted = project(homography, ideal_marker_corners(frame_px))
    distances = np.linalg.norm(predicted - refined, axis=-1)
    frame_quad = _shoelace(ideal_marker_centres(frame_px))
    source_quad = _shoelace(centres)
    min_scale = float(local_magnification(homography, cell_centres(params)).min())
    scale_k = choose_scale(homography, frame_px) if k is None else k
    rectified = rectify(arr, homography, frame_px, scale_k, kernel)
    if flat_field is not None:
        rectified = flat_field.apply(rectified)

    record = DetectionRecord(
        detected=True,
        failure_reason=None,
        reprojection_error_px=float(np.sqrt((distances**2).mean())),
        reprojection_error_max_px=float(distances.max()),
        fiducial_centres_px=tuple((float(x), float(y)) for x, y in centres),
        quad_area_px=source_quad,
        source_px_per_cell=math.sqrt(source_quad / frame_quad) * params.pitch_px,
        source_px_per_cell_min=min_scale * params.pitch_px,
        rectify_scale_k=scale_k,
        resampling_kernel=kernel,
        flat_field_applied=flat_field is not None,
        homography=tuple(tuple(float(v) for v in row) for row in homography),  # type: ignore[misc]
    )
    return Detection(record, rectified, refined)


# --------------------------------------------------------------------------- #
# Debug overlay
# --------------------------------------------------------------------------- #

_SUBPIXEL_BITS = 3  # OpenCV drawing shift: coordinates in 1/8 pixel


def overlay_image(image: npt.NDArray[Any], detection: Detection, params: CodecParams) -> npt.NDArray[np.uint8]:
    """The captured frame (BGR, 8-bit) with the detected quad, cell grid and cell centres drawn on it."""
    arr = np.clip(np.asarray(image, dtype=np.float64), 0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR) if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    rec = detection.record
    scale = 1 << _SUBPIXEL_BITS

    def pt(p: FloatArray) -> tuple[int, int]:
        return int(round(p[0] * scale)), int(round(p[1] * scale))

    if not rec.detected or rec.homography is None:
        cv2.putText(canvas, f"NOT DETECTED: {rec.failure_reason}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return canvas
    h = np.array(rec.homography)
    lay = grid_layout(params)
    # Cell grid: every row and column boundary of the full grid (lines stay lines under a homography).
    edges = lay.origin_px - 0.5 + np.arange(lay.cells_per_side + 1) * params.pitch_px - params.cell_gap_px / 2
    lo, hi = edges[0], edges[-1]
    for e in edges:
        for a, b in (((e, lo), (e, hi)), ((lo, e), (hi, e))):
            pa, pb = project(h, np.array([a, b], dtype=np.float64))
            cv2.line(canvas, pt(pa), pt(pb), (255, 200, 0), 1, cv2.LINE_AA, _SUBPIXEL_BITS)
    for c in cell_centres_source(h, params):
        cv2.circle(canvas, pt(c), 0, (0, 255, 255), 1, cv2.LINE_8, _SUBPIXEL_BITS)
    # Frame outline and fiducial quad.
    f = params.frame_px
    outline = project(h, np.array([[-0.5, -0.5], [f - 0.5, -0.5], [f - 0.5, f - 0.5], [-0.5, f - 0.5]]))
    cv2.polylines(canvas, [np.array([pt(p) for p in outline])], True, (0, 255, 0), 2, cv2.LINE_AA, _SUBPIXEL_BITS)
    centres = np.array(rec.fiducial_centres_px)
    cv2.polylines(canvas, [np.array([pt(p) for p in centres])], True, (255, 0, 255), 1, cv2.LINE_AA, _SUBPIXEL_BITS)
    if detection.refined_corners is not None:
        for quad in detection.refined_corners:
            cv2.polylines(canvas, [np.array([pt(p) for p in quad])], True, (0, 0, 255), 1, cv2.LINE_AA, _SUBPIXEL_BITS)
    for c in centres:
        cv2.drawMarker(canvas, pt(c), (0, 0, 255), cv2.MARKER_CROSS, 12 * scale, 1, cv2.LINE_AA)
    label = (
        f"reproj {rec.reprojection_error_px:.3f}px  px/cell {rec.source_px_per_cell:.2f} (min {rec.source_px_per_cell_min:.2f})"
        f"  k={rec.rectify_scale_k}  {rec.resampling_kernel}  flat={rec.flat_field_applied}"
    )
    cv2.putText(canvas, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return canvas


def write_overlay(path: str | Path, image: npt.NDArray[Any], detection: Detection, params: CodecParams) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay_image(image, detection, params))

