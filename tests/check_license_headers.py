#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Check release attribution files and source license headers."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".c",
    ".cmake",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".in",
    ".md",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {".clang-format", ".gitignore", "CMakeLists.txt"}
DERIVED_PREFIXES = ("include/common/", "src/common/")


def repository_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [Path(line) for line in result.stdout.splitlines() if line]
    return [path for path in paths if (ROOT / path).is_file()]


def main():
    errors = []
    for required in ("LICENSE", "NOTICE"):
        if not (ROOT / required).is_file():
            errors.append(f"missing required attribution file: {required}")

    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for phrase in (
        "Copyright 2026 Hongpei Li",
        "Copyright 2025-2026 Daniel Cederberg",
        "https://github.com/dance858/PSLP",
    ):
        if phrase not in notice:
            errors.append(f"NOTICE is missing: {phrase}")

    for path in repository_files():
        if path.name in {"LICENSE", "NOTICE"}:
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
            continue
        text = (ROOT / path).read_text(encoding="utf-8")
        header = "\n".join(text.splitlines()[:20])
        if "SPDX-License-Identifier: Apache-2.0" not in header:
            errors.append(f"missing Apache-2.0 SPDX header: {path}")
        if "Copyright 2026 Hongpei Li" not in header:
            errors.append(f"missing PreFOS copyright header: {path}")
        path_string = path.as_posix()
        if path_string.startswith(DERIVED_PREFIXES):
            if "Copyright 2025-2026 Daniel Cederberg" not in header:
                errors.append(f"missing upstream copyright header: {path}")
            if "Modified for PreFOS in 2026." not in header:
                errors.append(f"missing modification notice: {path}")

    if errors:
        print("License audit failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("License audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
