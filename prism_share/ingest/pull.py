"""Fetch a run folder from the phone with adb.

    python -m prism_share.ingest.pull <run_id> [--dest data]

Runs exactly the command docs/capture-format.md gives:

    adb pull /sdcard/Android/data/io.github.ahmedov.prismshare.capture/files/captures/<run_id> ./data/

and refuses to overwrite a run folder that already exists locally (pulled data
is raw evidence; a second pull into the same folder would mix two copies).
After pulling, it runs ingest validation so a bad run is caught on the spot.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from prism_share.ingest.manifest import PHONE_CAPTURES_DIR

_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class PullError(RuntimeError):
    pass


def phone_path(run_id: str) -> str:
    if not _RUN_ID.match(run_id) or run_id in (".", ".."):
        raise PullError(f"run_id {run_id!r} is not a safe folder name")
    return f"{PHONE_CAPTURES_DIR}/{run_id}"


def pull_command(run_id: str, dest: Path, adb: str = "adb") -> list[str]:
    return [adb, "pull", phone_path(run_id), f"{dest}/"]


def pull(run_id: str, dest: Path = Path("data"), adb: str | None = None) -> Path:
    """Pull one run; returns the local run folder."""
    exe = adb or shutil.which("adb")
    if exe is None:
        raise PullError("adb not found. Install Android platform-tools (macOS: brew install android-platform-tools).")
    local = dest / run_id
    if local.exists():
        raise PullError(f"{local} already exists; move it away first (pulled runs are never overwritten)")
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(pull_command(run_id, dest, exe), capture_output=True, text=True)
    if result.returncode != 0:
        raise PullError(f"adb pull failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    if not (local / "manifest.json").exists():
        raise PullError(f"adb reported success but {local}/manifest.json is missing")
    return local


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--dest", default="data")
    parser.add_argument("--no-validate", action="store_true", help="skip ingest validation after pulling")
    args = parser.parse_args(argv)
    try:
        local = pull(args.run_id, Path(args.dest))
    except PullError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"pulled {local}")
    if args.no_validate:
        return 0
    from prism_share.ingest.ingest import main as ingest_main

    return ingest_main([str(local)])


if __name__ == "__main__":
    sys.exit(main())
