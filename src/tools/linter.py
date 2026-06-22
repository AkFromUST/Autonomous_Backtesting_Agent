from __future__ import annotations

import py_compile
import subprocess
import tempfile
import os

from src.state import LintError

# Style-only codes that don't prevent execution — never block the pipeline
_STYLE_ONLY_CODES = {"F541", "E501", "W291", "W293", "W292", "E302", "E303", "B007", "E711", "E712"}


def lint(code: str) -> list[LintError]:
    """Run ruff + py_compile on code string. Returns list of errors (empty = clean)."""
    errors: list[LintError] = []

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # Syntax check first — fast, catches the worst issues
        try:
            py_compile.compile(tmp_path, doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(LintError(line=None, code="SyntaxError", message=str(e)))
            return errors  # no point running ruff on broken syntax

        # Ruff for style/logic issues
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", tmp_path],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            import json
            for item in json.loads(result.stdout):
                code = item.get("code", "")
                if code in _STYLE_ONLY_CODES:
                    continue
                errors.append(LintError(
                    line=item.get("location", {}).get("row"),
                    code=code,
                    message=item.get("message", ""),
                ))
    finally:
        os.unlink(tmp_path)

    return errors
