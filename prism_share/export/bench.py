"""Export a decode-benchmark bundle for a decoder in another project (the Android spike).

    python -m prism_share.export.bench --out data/bench/ --frames 50 \
        --package io.github.ahmedov.prismshare.testspike

``--package`` is the Android application ID of the app that reads the bundle.
It is required and has no default: the README's mkdir, push and chmod commands
target that app's storage, and a wrong default would produce commands that look
right and fail late on the phone.

The consumer is a Kotlin decoder that cannot see this source. The bundle
therefore carries every piece of geometry and codebook as **data**, so nothing
has to be ported: the benchmark measures decode compute, and ported layout
logic would be both wasted effort and a source of bugs.

Layout of the bundle::

    <out>/manifest.json          frame format version, frame size, and every configuration
                                 folder with its files listed (enumerate, never guess)
    <out>/README.md              frame format version, payload bytes per frame, adb push + chmod commands
    <out>/c1_px4/                one folder per configuration in params.BENCH_CONFIGURATIONS,
    <out>/c16_px4/               named c<colour_depth>_px<cell_px>
        frame_0000.png ...       frame_px x frame_px 8-bit RGB, lossless, no degradation
        params.json              CodecParams as a flat JSON object
        codec.json               everything needed to decode, as data
        ground_truth.json        per frame: index band value, glyph_ids, colour_ids

``codec.json`` holds:

* ``glyphs``: the glyph bitmaps at their rendered pixel size (cell_px rows of
  cell_px 0/1 values; 1 = ink), in glyph-id order;
* ``palette``: the ink colours as [R, G, B], in colour-id order;
* ``background``: the RGB of every non-ink pixel of a cell;
* ``cells``: one {x, y, width, height} per data cell, in decode order. That is
  the order of ``glyph_ids`` / ``colour_ids``, and cells covered by fiducials
  or the index band are simply absent;
* ``index_band``: every block as {x, y, width, height, bit}, where ``bit`` is
  the bit of the frame index the block carries (0 = least significant). White
  means 1 and black means 0. Each bit appears ``repeats`` times; the Python
  decoder takes a majority vote;
* ``level_references``: the white and black reference regions, as disjoint
  rectangles, that the Python decoder takes per-channel medians of to
  normalise levels before decoding.

The frames are byte for byte what ``display frames --payload-bytes`` writes for
the same payload, and every frame is a systematic source frame.
``tests/test_bench_export.py`` redraws every frame from codec.json and
ground_truth.json alone and checks it against the PNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from prism_share.codec.encoder import INDEX_BAND_CODE_INDICES, EncodedFrame, encode_frames, png_bytes
from prism_share.codec.framing import CRC_BYTES, HEADER_BYTES, frame_capacity
from prism_share.codec.glyphs import glyphs_for
from prism_share.codec.layout import black_reference_mask, grid_layout, index_band, white_reference_mask
from prism_share.codec.palette import palette_for
from prism_share.codec.params import (
    BACKGROUND_RGB,
    BENCH_CONFIGURATIONS,
    BENCH_SCHEMA_VERSION,
    FRAME_FORMAT_VERSION,
    INDEX_BAND_BITS,
    INDEX_BAND_REPEATS,
    CodecParams,
)
from prism_share.codec.prng import keystream
from prism_share.transmit.display import RUN_PAYLOAD_DOMAIN

#: Android application ID: at least two dot-separated segments, each starting with a letter and
#: containing only ASCII letters, digits and underscores.
_APPLICATION_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+")
DEFAULT_OUT = Path("data/bench")
MANIFEST, README = "manifest.json", "README.md"
PARAMS_FILE, CODEC_FILE, GROUND_TRUTH_FILE = "params.json", "codec.json", "ground_truth.json"


def validate_package(package: str) -> str:
    """Return ``package`` if it parses as an Android application ID, else raise ValueError."""
    if not _APPLICATION_ID.fullmatch(package):
        raise ValueError(f"{package!r} is not an Android application ID (e.g. io.github.ahmedov.prismshare.testspike): "
                         "it needs two or more dot-separated segments, each starting with a letter and containing "
                         "only letters, digits and underscores")
    return package


def phone_bench_dir(package: str) -> str:
    """Where the bundle goes on the phone: the consuming app's app-specific external storage."""
    return f"/sdcard/Android/data/{validate_package(package)}/files/bench"


def frame_filename(i: int) -> str:
    return f"frame_{i:04d}.png"


def folder_name(params: CodecParams) -> str:
    return f"c{params.colour_depth}_px{params.cell_px}"


def bench_params(seed: int = 0) -> list[CodecParams]:
    return [CodecParams(colour_depth=d, cell_px=c, seed=seed) for d, c in BENCH_CONFIGURATIONS]


def bench_frames(params: CodecParams, n_frames: int) -> tuple[bytes, list[EncodedFrame]]:
    """Payload and frames as ``display frames --payload-bytes`` would make them, one source block per frame."""
    if not 1 <= n_frames <= INDEX_BAND_CODE_INDICES:
        raise ValueError(f"--frames must be in [1, {INDEX_BAND_CODE_INDICES}] so every frame has a distinct index band value")
    payload = keystream(params.seed, RUN_PAYLOAD_DOMAIN, n_frames * frame_capacity(params).block_bytes)
    return payload, encode_frames(payload, params, n_frames=n_frames)


# --------------------------------------------------------------------------- #
# Geometry as data
# --------------------------------------------------------------------------- #


def rect(x: int, y: int, width: int, height: int) -> dict[str, int]:
    return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}


def mask_rectangles(mask: npt.NDArray[np.bool_]) -> list[dict[str, int]]:
    """Disjoint rectangles whose union is exactly ``mask``.

    Each row is split into runs, and a run extends the rectangle above it when
    that rectangle spans exactly the same columns. Rectangles are listed in
    order of their top-left corner (y, then x).
    """
    open_rects: dict[tuple[int, int], dict[str, int]] = {}
    done: list[dict[str, int]] = []
    height, _ = mask.shape
    for y in range(height + 1):
        runs: set[tuple[int, int]] = set()
        if y < height:
            row = np.concatenate(([False], mask[y], [False])).astype(np.int8)
            edges = np.flatnonzero(np.diff(row))
            runs = {(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)}
        for key in [k for k in open_rects if k not in runs]:
            done.append(open_rects.pop(key))
        for a, b in runs:
            if (a, b) in open_rects:
                open_rects[(a, b)]["height"] += 1
            else:
                open_rects[(a, b)] = rect(a, y, b - a, 1)
    return sorted(done, key=lambda r: (r["y"], r["x"]))


def codec_document(params: CodecParams) -> dict[str, Any]:
    """codec.json: glyphs, palette, background, cells, index band blocks, level references."""
    layout = grid_layout(params)
    glyphs = glyphs_for(params)
    if glyphs.shape[1:] != (params.cell_px, params.cell_px):
        raise AssertionError("glyph bitmaps are not at rendered cell size")
    band = index_band(params.frame_px)
    blocks = []
    for position in range(band.blocks):
        x, y = band.block_origin(position)
        blocks.append({**rect(x, y, band.block_px, band.block_px), "bit": INDEX_BAND_BITS - 1 - position % INDEX_BAND_BITS})
    return {
        "frame_format_version": FRAME_FORMAT_VERSION,
        "frame_width": params.frame_px,
        "frame_height": params.frame_px,
        "coordinates": "x = column from the left, y = row from the top, 0-based, in frame pixels; a rectangle covers "
                       "x <= px < x + width and y <= py < y + height",
        "glyph_count": params.glyph_count,
        "glyph_ink_pixels": params.glyph_weight,
        "glyphs": [g.astype(int).tolist() for g in glyphs],
        "glyphs_format": "glyphs[glyph_id][row][column], row 0 at the top of the cell; 1 = ink, 0 = background",
        "colour_depth": params.colour_depth,
        "palette": palette_for(params).tolist(),
        "palette_format": "palette[colour_id] = [R, G, B], 8-bit code values",
        "background": list(BACKGROUND_RGB),
        "n_cells": layout.n_cells,
        "cells": [rect(x, y, params.cell_px, params.cell_px) for x, y in zip(layout.cell_x, layout.cell_y, strict=True)],
        "cells_format": "cells[i] is the cell that glyph_ids[i] and colour_ids[i] describe, in decode order; pixel "
                        "(x + column, y + row) of cell i is palette[colour_ids[i]] where glyphs[glyph_ids[i]][row][column] "
                        "is 1, and background elsewhere",
        "index_band": {
            "bits": INDEX_BAND_BITS,
            "repeats": INDEX_BAND_REPEATS,
            "blocks": blocks,
            "format": "each block is solid white [255, 255, 255] if its bit of the frame index is 1 and solid black "
                      "[0, 0, 0] if 0; bit 0 is the least significant; index 0 is reserved for the reference frame",
        },
        "level_references": {
            "white": mask_rectangles(white_reference_mask(params.frame_px)),
            "black": mask_rectangles(black_reference_mask(params.frame_px)),
            "format": "pixels that are always white / always black in every frame; the Python decoder maps the "
                      "per-channel median over each set to 1.0 / 0.0 before decoding (a no-op on these ideal frames)",
        },
    }


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _dump(doc: dict[str, Any], compact_keys: tuple[str, ...]) -> str:
    """Indented JSON, except that the listed keys' values (long arrays) each go on one line."""
    placeholders: dict[str, str] = {}

    def swap(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k in compact_keys:
                    token = f"@@{len(placeholders)}@@"
                    placeholders[token] = json.dumps(v, separators=(",", ":"))
                    out[k] = token
                else:
                    out[k] = swap(v)
            return out
        if isinstance(node, list):
            return [swap(v) for v in node]
        return node

    text = json.dumps(swap(doc), indent=1)
    for token, value in placeholders.items():
        text = text.replace(json.dumps(token), value, 1)
    return text + "\n"


def ground_truth_document(frames: list[EncodedFrame]) -> dict[str, Any]:
    return {
        "format": "frames[k] describes frame_<k, 4 digits>.png; glyph_ids and colour_ids are parallel arrays in the "
                  "order of codec.json cells",
        "frames": [
            {"file": frame_filename(k), "index_band": f.frame_index,
             "png_sha256": hashlib.sha256(png_bytes(f.image)).hexdigest(),
             "glyph_ids": f.glyphs.tolist(), "colour_ids": f.colours.tolist()}
            for k, f in enumerate(frames)
        ],
    }


def configuration_entry(params: CodecParams, n_frames: int) -> dict[str, Any]:
    cap = frame_capacity(params)
    return {
        "folder": folder_name(params),
        "colour_depth": params.colour_depth,
        "cell_px": params.cell_px,
        "params_fingerprint": params.fingerprint(),
        "n_cells": cap.n_cells,
        "rs_code": [params.ecc_total, params.ecc_data],
        "payload_bytes_per_frame": cap.block_bytes,
        "params": PARAMS_FILE,
        "codec": CODEC_FILE,
        "ground_truth": GROUND_TRUTH_FILE,
        "frames": [frame_filename(i) for i in range(n_frames)],
    }


def manifest_document(params_list: list[CodecParams], n_frames: int) -> dict[str, Any]:
    first = params_list[0]
    return {
        "bench_schema_version": BENCH_SCHEMA_VERSION,
        "frame_format_version": FRAME_FORMAT_VERSION,
        "frame_width": first.frame_px,
        "frame_height": first.frame_px,
        "frame_pixel_format": "PNG, 8-bit RGB, no alpha, no colour-management chunks",
        "frames_per_configuration": n_frames,
        "configurations": [configuration_entry(p, n_frames) for p in params_list],
    }


def push_commands(out: Path, params_list: list[CodecParams], package: str) -> list[str]:
    """mkdir, push, then widen permissions: without the chmod the app cannot read what was pushed (CHMOD_REASON)."""
    local = out.as_posix().rstrip("/")
    phone = phone_bench_dir(package)
    sources = " ".join(f"{local}/{name}" for name in [MANIFEST, README, *(folder_name(p) for p in params_list)])
    return [
        f"adb shell mkdir -p {phone}",
        f"adb push {sources} {phone}/",
        f"adb shell chmod -R o+rX {phone}",
    ]


#: Why push_commands ends with a chmod. Written into every generated README.
CHMOD_REASON = (
    "**Why the `chmod`.** `adb push` creates directories owned by `shell` in group `ext_data_rw`, mode `drwxrws---`. "
    "The app process runs as its own uid and is not in `ext_data_rw`, so it cannot enter those directories. Every "
    "pushed file is unreadable to the app until permissions are widened, even inside the app's own storage, and from "
    "the app it looks like \"file not found\". `chmod -R o+rX` adds read, plus traverse on directories, for "
    "\"others\". Found 2026-09-16 on SM-A346E (Android 16) with the TestSpike app. The same applies to anything "
    "pushed into an app's `Android/data` folder, including reference frames pushed to the capture app for its "
    "display mode."
)


def readme(out: Path, params_list: list[CodecParams], n_frames: int, package: str) -> str:
    first = params_list[0]
    rows = [
        f"| `{folder_name(p)}` | {p.colour_depth} | {p.cell_px} | {frame_capacity(p).n_cells} | "
        f"RS({p.ecc_total},{p.ecc_data}) | **{frame_capacity(p).block_bytes}** |"
        for p in params_list
    ]
    return "\n".join([
        "# Decode benchmark bundle",
        "",
        f"Generated by `python -m prism_share.export.bench --out {out.as_posix().rstrip('/')}/ --frames {n_frames} "
        f"--package {package}`. "
        "Regenerate; do not edit.",
        "",
        f"* **Frame format version: {FRAME_FORMAT_VERSION}.** Also in `manifest.json` and every `codec.json`; "
        "refuse a version you do not know.",
        f"* **Frames per configuration:** {n_frames}, {first.frame_px}×{first.frame_px} lossless 8-bit RGB PNG, "
        "no degradation. Each is a systematic frame carrying its own payload block.",
        "* **Start from `manifest.json`.** It lists every configuration folder and every file in it.",
        "",
        "## Configurations",
        "",
        f"Payload bytes per frame = RS message bytes − {HEADER_BYTES}-byte header − {CRC_BYTES}-byte CRC.",
        "",
        "| folder | colour depth | cell px | cells | RS code | payload bytes per frame |",
        "|---|---:|---:|---:|---|---:|",
        *rows,
        "",
        "## Files in each folder",
        "",
        "* `frame_0000.png` …: the rendered frames.",
        "* `params.json`: the full CodecParams, flat.",
        "* `codec.json`: everything needed to decode, as data: `glyphs` (bitmaps at rendered pixel size), "
        "`palette`, `background`, `cells` (x, y, width, height of every data cell, in decode order), `index_band` "
        "(every block with the bit it carries) and `level_references` (white and black reference rectangles).",
        "* `ground_truth.json`: per frame, `index_band`, and `glyph_ids` / `colour_ids` in the order of `cells`.",
        "",
        "## Push to the phone",
        "",
        "From the repository root:",
        "",
        "```",
        *push_commands(out, params_list, package),
        "```",
        "",
        f"The bundle lands in `{phone_bench_dir(package)}/`, with `manifest.json` at its top: the app-specific "
        f"storage of `{package}`, the app that reads it. To target another app, regenerate with its `--package`; "
        "do not edit these lines.",
        "",
        CHMOD_REASON,
        "",
    ])


def export(out: Path, n_frames: int, package: str, seed: int = 0) -> list[Path]:
    """Write the bundle; returns the configuration folders. Refuses to write over an existing bundle."""
    validate_package(package)  # before anything is written
    params_list = bench_params(seed)
    folders = [out / folder_name(p) for p in params_list]
    clashes = [p for p in [out / MANIFEST, out / README, *folders] if p.exists() and (p.is_file() or any(p.iterdir()))]
    if clashes:
        raise FileExistsError(f"{', '.join(map(str, clashes))} already exist; remove them first (stale frames would "
                              "silently join the bundle)")
    for params, folder in zip(params_list, folders, strict=True):
        _, frames = bench_frames(params, n_frames)
        folder.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            (folder / frame_filename(i)).write_bytes(png_bytes(f.image))
        (folder / PARAMS_FILE).write_text(json.dumps(params.to_dict(), indent=1, sort_keys=True) + "\n")
        (folder / CODEC_FILE).write_text(_dump(codec_document(params), ("glyphs", "palette", "background", "cells", "blocks", "white", "black")))
        (folder / GROUND_TRUTH_FILE).write_text(_dump(ground_truth_document(frames), ("glyph_ids", "colour_ids")))
    (out / MANIFEST).write_text(_dump(manifest_document(params_list, n_frames), ("frames", "rs_code")))
    (out / README).write_text(readme(out, params_list, n_frames, package))
    return folders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--frames", type=int, required=True, help=f"frames per configuration, 1..{INDEX_BAND_CODE_INDICES}")
    parser.add_argument("--package", required=True,
                        help="Android application ID of the app that reads the bundle (no default: it decides where "
                             "the README's adb commands put the files)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        folders = export(Path(args.out), args.frames, args.package, args.seed)
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for folder in folders:
        print(f"wrote {folder}")
    print(f"wrote {Path(args.out) / MANIFEST} and {Path(args.out) / README}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
