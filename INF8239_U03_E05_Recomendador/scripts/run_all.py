"""Flujo completo: descarga -> auditoría -> LAB08 -> experimento principal (LAB09).

Uso:  uv run python scripts/run_all.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    for script in ("download_data.py", "audit_data.py", "lab08_content.py", "lab09_hybrid.py"):
        print(f"\n===== {script}")
        sys.argv = [script]
        try:
            runpy.run_path(str(HERE / script), run_name="__main__")
        except SystemExit as exit_:
            if exit_.code not in (0, None):
                return int(exit_.code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
