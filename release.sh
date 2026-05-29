#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./release.sh [--repository NAME] [--version X.Y.Z] [--dry-run]

By default, the script bumps the patch segment of the current version:
  0.2.0 -> 0.2.1

Options:
  --repository NAME  Twine repository alias to upload to (default: pypi)
  --version X.Y.Z    Set an explicit version instead of auto-bumping the patch
  --dry-run          Print the next version and planned upload target only
  -h, --help         Show this help text
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi
REPOSITORY="pypi"
TARGET_VERSION=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository)
      [[ $# -ge 2 ]] || {
        echo "Missing value for --repository" >&2
        exit 1
      }
      REPOSITORY="$2"
      shift 2
      ;;
    --version)
      [[ $# -ge 2 ]] || {
        echo "Missing value for --version" >&2
        exit 1
      }
      TARGET_VERSION="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}

for required_file in pyproject.toml src/anomx/__init__.py tests/test_package.py; do
  [[ -f "$required_file" ]] || {
    echo "Required file not found: $required_file" >&2
    exit 1
  }
done

CURRENT_VERSION="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import re

content = Path("pyproject.toml").read_text(encoding="utf-8")
match = re.search(r'^version = "([^"]+)"$', content, re.MULTILINE)
if not match:
    raise SystemExit("Could not find project version in pyproject.toml")
print(match.group(1))
PY
)"

if [[ -n "$TARGET_VERSION" ]]; then
  NEXT_VERSION="$TARGET_VERSION"
else
  IFS=. read -r major minor patch <<<"$CURRENT_VERSION"
  [[ -n "${major:-}" && -n "${minor:-}" && -n "${patch:-}" ]] || {
    echo "Expected a semantic version like X.Y.Z, found: $CURRENT_VERSION" >&2
    exit 1
  }
  NEXT_VERSION="${major}.${minor}.$((patch + 1))"
fi

[[ "$NEXT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "Version must match X.Y.Z, found: $NEXT_VERSION" >&2
  exit 1
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Current version: $CURRENT_VERSION"
  echo "Next version:    $NEXT_VERSION"
  echo "Repository:      $REPOSITORY"
  exit 0
fi

"$PYTHON_BIN" - "$CURRENT_VERSION" "$NEXT_VERSION" <<'PY'
from pathlib import Path
import re
import sys

current_version, next_version = sys.argv[1:3]

patterns = {
    "pyproject.toml": [
        (
            rf'^(version = "){re.escape(current_version)}(")$',
            rf"\g<1>{next_version}\g<2>",
        ),
    ],
    "src/anomx/__init__.py": [
        (
            rf'^(__version__ = "){re.escape(current_version)}(")$',
            rf"\g<1>{next_version}\g<2>",
        ),
    ],
    "tests/test_package.py": [
        (
            rf'^(    assert anomx.__version__ == "){re.escape(current_version)}(")$',
            rf"\g<1>{next_version}\g<2>",
        ),
        (
            rf'^(    assert "anomx ){re.escape(current_version)}(" in result\.stdout)$',
            rf"\g<1>{next_version}\g<2>",
        ),
    ],
}

for file_path, replacements in patterns.items():
    path = Path(file_path)
    original_text = path.read_text(encoding="utf-8")
    updated_text = original_text

    for pattern, replacement in replacements:
        updated_text, count = re.subn(pattern, replacement, updated_text, flags=re.MULTILINE)
        if count != 1:
            raise SystemExit(
                f"Failed to update version in {file_path}. "
                f"Expected exactly one match for pattern: {pattern}"
            )

    path.write_text(updated_text, encoding="utf-8")
PY

echo "Updated version: $CURRENT_VERSION -> $NEXT_VERSION"

rm -rf dist build
find . -maxdepth 1 -type d -name "*.egg-info" -exec rm -rf {} +

"$PYTHON_BIN" -m build
"$PYTHON_BIN" -m twine check dist/*
"$PYTHON_BIN" -m twine upload --repository "$REPOSITORY" dist/*
