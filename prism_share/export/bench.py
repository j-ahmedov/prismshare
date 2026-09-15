"""Export a decode-benchmark bundle for a second decoder implementation (the Android spike).

    python -m prism_share.export.bench --out data/bench/ --frames 50

For each configuration in ``params.BENCH_CONFIGURATIONS`` it writes one folder,
named by ``CodecParams.label``, containing:

* ``frame_0000.png`` ... : the rendered frames, frame_px x frame_px 8-bit RGB,
  lossless and byte-deterministic (``encoder.write_png``), with no degradation.
  They are exactly what ``display frames`` writes and ``display`` shows (the
  screen only scales them by an integer factor, nearest neighbour): the
  payload is the same keystream, and every frame is a systematic source frame.
* ``params.json`` : ``CodecParams.to_dict()``, a flat JSON object.
* ``ground_truth.json`` : for each frame, its index band value and the symbol
  stream as two parallel arrays ``glyph_ids`` / ``colour_ids``, in the order
  ``decoder.read_symbols`` walks the cells. Every convention needed to use them
  is written into the file itself: cell order and positions, glyph bitmaps,
  palette and index band encoding. ``tests/test_bench_export.py`` rebuilds the
  frames from that JSON alone, so a description that disagrees with the code
  fails the test.

It also writes ``README.md`` at the bundle root, with the frame format version,
payload bytes per frame and the adb command to put the bundle on the phone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from prism_share.codec.encoder import INDEX_BAND_CODE_INDICES, EncodedFrame, encode_frames, png_bytes
from prism_share.codec.framing import CRC_BYTES, HEADER_BYTES, FrameCapacity, frame_capacity
from prism_share.codec.glyphs import glyphs_for
from prism_share.codec.layout import grid_layout, index_band, keepout_origins
from prism_share.codec.palette import palette_for
from prism_share.codec.params import (
    BACKGROUND_RGB,
    BENCH_CONFIGURATIONS,
    BENCH_SCHEMA_VERSION,
    BLACK_RGB,
    BORDER_PX,
    FRAME_FORMAT_VERSION,
    INDEX_BAND_BITS,
    INDEX_BAND_REFERENCE,
    INDEX_BAND_REPEATS,
    KEEPOUT_PX,
    WHITE_RGB,
    CodecParams,
)
from prism_share.codec.prng import keystream
from prism_share.ingest.manifest import PHONE_CAPTURES_DIR
from prism_share.transmit.display import RUN_PAYLOAD_DOMAIN

#: Where the bundle goes on the phone: next to the capture app's captures folder, in its app-specific storage.
PHONE_BENCH_DIR = str(Path(PHONE_CAPTURES_DIR).parent / "bench")
DEFAULT_OUT = Path("data/bench")
KEEPOUT_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


def frame_filename(i: int) -> str:
    return f"frame_{i:04d}.png"


def bench_params(seed: int = 0) -> list[CodecParams]:
    return [CodecParams(colour_depth=d, cell_px=c, seed=seed) for d, c in BENCH_CONFIGURATIONS]


def bench_frames(params: CodecParams, n_frames: int) -> tuple[bytes, list[EncodedFrame]]:
    """The payload and frames, exactly as ``display frames --payload-bytes`` would make them.

    The payload is sized to ``n_frames`` source blocks, so every frame is a
    systematic frame carrying its own block and no repair frames are needed.
    """
    if not 1 <= n_frames <= INDEX_BAND_CODE_INDICES:
        raise ValueError(f"--frames must be in [1, {INDEX_BAND_CODE_INDICES}] so every frame has a distinct index band value")
    payload = keystream(params.seed, RUN_PAYLOAD_DOMAIN, n_frames * frame_capacity(params).block_bytes)
    return payload, encode_frames(payload, params, n_frames=n_frames)


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _dump(doc: dict[str, Any], arrays: dict[str, Any]) -> str:
    """Indented JSON for the descriptive fields, each long array on one line.

    ``doc`` holds placeholder strings "@@name@@" where ``arrays[name]`` belongs.
    """
    text = json.dumps(doc, indent=1, ensure_ascii=False)
    for name, value in arrays.items():
        placeholder = json.dumps(f"@@{name}@@")
        if text.count(placeholder) != 1:
            raise AssertionError(f"placeholder {name} not found exactly once")
        text = text.replace(placeholder, _compact(value))
    return text + "\n"


def ground_truth(params: CodecParams, payload: bytes, frames: list[EncodedFrame]) -> str:
    """The ground_truth.json text for one configuration."""
    layout = grid_layout(params)
    band = index_band(params.frame_px)
    reserved = [
        {"name": f"keepout_{name}", "x": x, "y": y, "width": KEEPOUT_PX, "height": KEEPOUT_PX}
        for name, (x, y) in zip(KEEPOUT_NAMES, keepout_origins(params.frame_px), strict=True)
    ] + [{"name": "index_band", "x": band.x, "y": band.y, "width": band.width, "height": band.height}]
    glyphs = glyphs_for(params)
    arrays: dict[str, Any] = {
        "cell_x": layout.cell_x.tolist(),
        "cell_y": layout.cell_y.tolist(),
        "glyphs": [g.astype(int).tolist() for g in glyphs],
        "palette_rgb": palette_for(params).tolist(),
    }
    frame_docs = []
    for i, f in enumerate(frames):
        arrays[f"glyph_ids_{i}"] = f.glyphs.tolist()
        arrays[f"colour_ids_{i}"] = f.colours.tolist()
        frame_docs.append({
            "file": frame_filename(i),
            "index_band": f.frame_index,
            "png_sha256": hashlib.sha256(png_bytes(f.image)).hexdigest(),
            "header": {"block_id": f.header.block_id, "n_source_blocks": f.header.n_source_blocks,
                       "payload_len": f.header.payload_len, "params_fingerprint": f.header.params_fingerprint},
            "glyph_ids": f"@@glyph_ids_{i}@@",
            "colour_ids": f"@@colour_ids_{i}@@",
        })
    doc = {
        "bench_schema_version": BENCH_SCHEMA_VERSION,
        "frame_format_version": FRAME_FORMAT_VERSION,
        "params_label": params.label,
        "params_fingerprint": params.fingerprint(),
        "image": {
            "files": "frame_0000.png, frame_0001.png, ... (zero-padded to 4 digits; frame i is frames[i] below)",
            "width": params.frame_px,
            "height": params.frame_px,
            "pixel_format": "8-bit RGB, no alpha, no colour-management chunks: each value is the code value written to the framebuffer",
            "coordinates": "x = column from the left, y = row from the top, both 0-based, in pixels of the PNG",
            "degradation": "none: these are the ideal frames, already rectified, at 1 PNG pixel per screen pixel",
        },
        "symbols": {
            "n_cells": layout.n_cells,
            "glyph_ids": "glyph_ids[i] is the glyph of cell i: an index into codebook.glyphs, 0..glyph_count-1",
            "colour_ids": "colour_ids[i] is the ink colour of cell i: an index into codebook.palette_rgb, "
                          "0..colour_depth-1; always 0 when colour_depth is 1",
            "rendering": "for 0 <= r, c < cell_px, pixel (x = cell_x[i] + c, y = cell_y[i] + r) is "
                         "codebook.palette_rgb[colour_ids[i]] if codebook.glyphs[glyph_ids[i]][r][c] == 1, "
                         "else codebook.background_rgb",
        },
        "cell_order": {
            "rule": "Cell i is the i-th cell kept in a row-major scan of a square grid: rows from top to bottom, "
                    "and within a row columns from left to right. A grid cell is skipped, and gets no index, "
                    "if its box grown by cell_gap_px on every side overlaps any rectangle in reserved_rectangles "
                    "(the four fiducial keep-out squares and the index band). This is the order "
                    "decoder.read_symbols reads cells in, and the order the symbol stream is written in.",
            "grid": {
                "cells_per_side": layout.cells_per_side,
                "origin_px": layout.origin_px,
                "pitch_px": params.pitch_px,
                "cell_px": params.cell_px,
                "cell_gap_px": params.cell_gap_px,
                "formula": "grid cell (row, col), 0 <= row, col < cells_per_side, has its top-left pixel at "
                           "x = origin_px + col * pitch_px, y = origin_px + row * pitch_px, and covers "
                           "cell_px x cell_px pixels",
            },
            "skip_test": "with gap = cell_gap_px, a cell at (x, y) overlaps rectangle (rx, ry, width, height) iff "
                         "x - gap < rx + width AND x + cell_px + gap > rx AND y - gap < ry + height AND "
                         "y + cell_px + gap > ry (all strict inequalities)",
            "reserved_rectangles": reserved,
            "cell_x": "@@cell_x@@",
            "cell_y": "@@cell_y@@",
            "cell_xy_note": "cell_x[i], cell_y[i] = top-left pixel of cell i; these are the result of the rule "
                            "above, given so that the rule can be checked rather than trusted",
        },
        "codebook": {
            "glyphs": "@@glyphs@@",
            "glyphs_note": "glyphs[g][r][c]: r = row from the top of the cell, c = column from the left, 1 = ink, "
                           "0 = background",
            "palette_rgb": "@@palette_rgb@@",
            "palette_note": "palette_rgb[k] = [R, G, B] code values of ink colour k",
            "background_rgb": list(BACKGROUND_RGB),
        },
        "index_band": {
            "value": "frames[k].index_band; 0 is reserved for the reference frame (not in this bundle); frame k "
                     "of this bundle carries block_id k and index_band k + 1",
            "x": band.x,
            "y": band.y,
            "block_px": band.block_px,
            "blocks": band.blocks,
            "bits": INDEX_BAND_BITS,
            "repeats": INDEX_BAND_REPEATS,
            "encoding": "blocks[p] for p = 0 .. blocks-1, left to right, block p covering x in "
                        "[x + p*block_px, x + (p+1)*block_px) and y in [y, y + block_px); block p is white "
                        f"{list(WHITE_RGB)} for bit 1 and black {list(BLACK_RGB)} for bit 0; the bit of block p is bit "
                        f"(bits - 1 - (p mod bits)) of the value, i.e. the value MSB-first, written repeats times "
                        "in a row",
            "reference_value": INDEX_BAND_REFERENCE,
        },
        "static_region": {
            "note": "every pixel that is neither inside a cell box nor inside an index band block is identical in "
                    f"every frame of every configuration in this bundle: the {BORDER_PX} px white border, the four "
                    "ArUco markers (DICT_4X4_50, ids 0-3 clockwise from top-left) in the keep-out squares, and background",
        },
        "payload": {
            "length": len(payload),
            "source": f"prism_share.codec.prng.keystream(seed={params.seed}, domain={RUN_PAYLOAD_DOMAIN!r}, "
                      f"n_bytes={len(payload)})",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "frames": frame_docs,
    }
    return _dump(doc, arrays)


def capacity_row(params: CodecParams) -> str:
    cap: FrameCapacity = frame_capacity(params)
    return (f"| `{params.label}` | {params.colour_depth} | {params.cell_px} | {cap.n_cells} | {cap.capacity_bytes} | "
            f"RS({params.ecc_total},{params.ecc_data}) x {cap.n_codewords} | **{cap.block_bytes}** |")


def readme(out: Path, params_list: list[CodecParams], n_frames: int) -> str:
    phone_dir = PHONE_BENCH_DIR
    return "\n".join([
        "# Decode benchmark bundle",
        "",
        f"Generated by `python -m prism_share.export.bench --out {out} --frames {n_frames}`. Do not edit: regenerate.",
        "",
        f"* **Frame format version:** {FRAME_FORMAT_VERSION} (`params.FRAME_FORMAT_VERSION`, part of every "
        "params fingerprint). A reader must refuse a bundle whose `frame_format_version` it does not know.",
        f"* **Bundle schema version:** {BENCH_SCHEMA_VERSION} (`ground_truth.json` → `bench_schema_version`).",
        f"* **Frames per configuration:** {n_frames}, each a systematic fountain frame with its own source "
        f"block; index band values 1 to {n_frames}.",
        "* **Degradation:** none. The PNGs are the ideal frames, 1 PNG pixel = 1 screen pixel, exactly as "
        "`python -m prism_share.transmit.display` shows them (the screen only adds integer nearest-neighbour scaling).",
        "",
        "## Configurations",
        "",
        f"*Payload bytes per frame* is the fountain block each frame carries: RS message bytes minus the {HEADER_BYTES}-byte "
        f"header and {CRC_BYTES}-byte CRC, at the configuration's own RS code.",
        "",
        "| folder | colour depth | cell px | cells | capacity bytes | RS code × codewords | payload bytes per frame |",
        "|---|---:|---:|---:|---:|---|---:|",
        *[capacity_row(p) for p in params_list],
        "",
        "## Files in each folder",
        "",
        f"* `frame_0000.png` … : {params_list[0].frame_px}×{params_list[0].frame_px} 8-bit RGB PNG, stored (uncompressed) "
        "deflate, no colour chunks; `ground_truth.json` → `frames[k].png_sha256` is the SHA-256 of the file's bytes.",
        "* `params.json` : the full `CodecParams` as a flat JSON object.",
        "* `ground_truth.json` : per frame, `index_band` and the parallel arrays `glyph_ids` / `colour_ids`. "
        "The file also spells out the cell order (`cell_order.rule`, with every cell's position), the glyph "
        "bitmaps and palette (`codebook`) and the index band encoding (`index_band`). Read those fields; do "
        "not assume a convention that is not written there.",
        "",
        "## Put it on the phone",
        "",
        "From the repository root, with one device connected:",
        "",
        "```",
        f"adb shell mkdir -p {phone_dir}",
        "adb push " + " ".join(f"{(out / name).as_posix()}" for name in ["README.md", *[p.label for p in params_list]]) + f" {phone_dir}/",
        "```",
        "",
        f"This puts each configuration folder at `{phone_dir}/<folder>/`, in the capture app's app-specific "
        "storage, where that app can read it without a storage permission. A different app must use its "
        "own package name in place of `io.github.ahmedov.prismshare.capture`.",
        "",
    ])


def export(out: Path, n_frames: int, seed: int = 0) -> list[Path]:
    """Write the bundle; returns the configuration folders. Refuses to write into a non-empty folder."""
    params_list = bench_params(seed)
    folders = [out / p.label for p in params_list]
    for folder in folders:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"{folder} is not empty; remove it first (a stale frame would silently join the bundle)")
    for params, folder in zip(params_list, folders, strict=True):
        payload, frames = bench_frames(params, n_frames)
        folder.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            (folder / frame_filename(i)).write_bytes(png_bytes(f.image))
        (folder / "params.json").write_text(json.dumps(params.to_dict(), indent=1, sort_keys=True) + "\n")
        (folder / "ground_truth.json").write_text(ground_truth(params, payload, frames))
    (out / "README.md").write_text(readme(out, params_list, n_frames))
    return folders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--frames", type=int, required=True, help=f"frames per configuration, 1..{INDEX_BAND_CODE_INDICES}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        folders = export(Path(args.out), args.frames, args.seed)
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for folder in folders:
        print(f"wrote {folder}")
    print(f"wrote {Path(args.out) / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
