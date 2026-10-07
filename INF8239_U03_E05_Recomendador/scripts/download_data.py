"""Descarga MovieLens ml-latest-small desde GroupLens y registra hashes.

Los datos NO se versionan ni se redistribuyen (data/raw está en .gitignore): cada persona
los descarga desde la fuente oficial aceptando sus términos de uso.

Uso:  uv run python scripts/download_data.py
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from inf8239_u03.config import EXPECTED_COUNTS, MOVIELENS_DIR, MOVIELENS_URL, REPORTS, ROOT
from inf8239_u03.data import download_zip, sha256_file

MANIFEST = REPORTS / "data_manifest.json"


def main() -> int:
    if (MOVIELENS_DIR / "ratings.csv").exists():
        print("MovieLens ya está en", MOVIELENS_DIR.relative_to(ROOT), "(no se descarga de nuevo)")
        zip_digest = json.loads(MANIFEST.read_text())["zip_sha256"] if MANIFEST.exists() else None
    else:
        _, zip_digest = download_zip(MOVIELENS_URL, MOVIELENS_DIR.parent)
        print("Descargado de", MOVIELENS_URL)
    ratings = pd.read_csv(MOVIELENS_DIR / "ratings.csv")
    movies = pd.read_csv(MOVIELENS_DIR / "movies.csv")
    counts = {"ratings": len(ratings), "movies": len(movies), "users": int(ratings["userId"].nunique())}
    matches = counts == EXPECTED_COUNTS
    manifest = {
        "source": MOVIELENS_URL,
        "downloaded_or_checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "zip_sha256": zip_digest,
        "files_sha256": {name: sha256_file(MOVIELENS_DIR / name)
                         for name in ("ratings.csv", "movies.csv", "tags.csv", "links.csv", "README.txt")
                         if (MOVIELENS_DIR / name).exists()},
        "counts": counts,
        "matches_documented_version_2018_09_26": matches,
    }
    REPORTS.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Conteos:", counts, "| coincide con la versión documentada:", matches)
    print("SHA-256 ZIP:", zip_digest)
    if not matches:
        print("AVISO: la versión descargada difiere de la documentada; los resultados pueden variar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
