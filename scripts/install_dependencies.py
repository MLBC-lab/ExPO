from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _detect_venv_python(venv_dir: Path) -> Path:
    """Return the python executable inside a virtual environment."""
    if os.name == "nt":
        cand = venv_dir / "Scripts" / "python.exe"
    else:
        cand = venv_dir / "bin" / "python"
    if not cand.exists():
        raise FileNotFoundError(f"Python executable not found in venv: {cand}")
    return cand


def _run(cmd: List[str]) -> None:
    print(f"[install] Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Install ExPO Python dependencies.")
    ap.add_argument(
        "--venv",
        type=str,
        default=".venv",
        help="Path to virtual environment directory (default: .venv).",
    )
    ap.add_argument(
        "--create-venv",
        action="store_true",
        help="Create the virtual environment before installing packages.",
    )
    ap.add_argument(
        "--upgrade-pip",
        action="store_true",
        help="Upgrade pip in the chosen environment before installing packages.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    venv_dir = PROJECT_ROOT / args.venv
    python_exe = Path(sys.executable)

    if args.create_venv:
        print(f"[install] Creating virtual environment at: {venv_dir}")
        _run([str(python_exe), "-m", "venv", str(venv_dir)])
        python_exe = _detect_venv_python(venv_dir)
    else:
        # If the venv directory exists but we are not explicitly creating it,
        # prefer using its Python interpreter for installation.
        if venv_dir.exists():
            try:
                python_exe = _detect_venv_python(venv_dir)
                print(f"[install] Using existing virtual environment: {venv_dir}")
            except FileNotFoundError:
                print(f"[install] Warning: {venv_dir} exists but no Python executable found; using current interpreter.")

    pip_cmd = [str(python_exe), "-m", "pip"]

    if args.upgrade_pip:
        print("[install] Upgrading pip...")
        _run(pip_cmd + ["install", "--upgrade", "pip"])

    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        raise FileNotFoundError(f"requirements.txt not found at: {req_file}")

    print(f"[install] Installing dependencies from {req_file} using {python_exe}")
    _run(pip_cmd + ["install", "-r", str(req_file)])

    print("[install] Dependency installation completed successfully.")


if __name__ == "__main__":
    main()
