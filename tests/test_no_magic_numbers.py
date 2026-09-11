"""No hard-coded 4, 8 or 16 anywhere in the package outside params.py.

Those are the values of glyph bits, glyph count, colour depths, cell sizes and
bits per byte that the experiment varies or depends on; a stray literal would
silently pin one of them. Strings (docstrings, struct formats) are not checked.
"""

from __future__ import annotations

import ast
from pathlib import Path

import prism_share

FORBIDDEN = {4, 8, 16}
PACKAGE_DIR = Path(prism_share.__file__).parent
EXEMPT = {PACKAGE_DIR / "codec" / "params.py"}


def _offences(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
            and node.value in FORBIDDEN
        ):
            found.append(f"{path.name}:{node.lineno}: literal {node.value!r}")
    return found


def test_no_forbidden_literals_outside_params() -> None:
    sources = [p for p in PACKAGE_DIR.rglob("*.py") if p not in EXEMPT]
    assert len(sources) > 5, "package sources not found"
    offences = [o for p in sources for o in _offences(p)]
    assert not offences, "hard-coded 4/8/16 outside params.py:\n" + "\n".join(offences)


def test_checker_catches_a_literal(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text('"""Docstring with 8."""\nx = 8\ny = "eight 8"\nz = 4.0\nw = True\nv = 16\n')
    assert _offences(probe) == ["probe.py:2: literal 8", "probe.py:4: literal 4.0", "probe.py:6: literal 16"]
