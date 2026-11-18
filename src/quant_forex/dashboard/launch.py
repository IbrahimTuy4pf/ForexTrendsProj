"""Launch the Streamlit research dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def app_path() -> Path:
    return Path(__file__).resolve().parent / "app.py"


def launch_dashboard(extra_args: list[str] | None = None) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'Streamlit is not installed. From the project root: python -m pip install -e ".[dashboard]"'
        ) from exc
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path())]
    if extra_args:
        cmd.extend(extra_args)
    return int(subprocess.call(cmd))
