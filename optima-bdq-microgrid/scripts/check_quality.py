"""Run the repeatable local quality and dependency-security checks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT_ROOT = PROJECT_ROOT.parent
CHECKS = (
    ("Dependency audit", [sys.executable, "-m", "pip_audit", "-r", "requirements-dev.txt"]),
    ("Test suite", [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"]),
    (
        "Python compilation",
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "app",
            "src",
            "scripts",
            "tests",
        ],
    ),
    ("Git whitespace check", ["git", "diff", "--check"]),
    ("Staged Git whitespace check", ["git", "diff", "--cached", "--check"]),
)


def main() -> int:
    failed = []
    for label, command in CHECKS:
        cwd = GIT_ROOT if label.endswith("whitespace check") else PROJECT_ROOT
        print(f"\n== {label} ==", flush=True)
        result = subprocess.run(command, cwd=cwd, check=False)
        if result.returncode:
            failed.append((label, result.returncode))

    if failed:
        print("\nFailed checks:", file=sys.stderr)
        for label, returncode in failed:
            print(f"- {label} (exit {returncode})", file=sys.stderr)
        return 1

    print("\nAll quality checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
