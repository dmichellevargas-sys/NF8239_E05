"""Descarga segura, carga, validación y auditoría de MovieLens."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def download_zip(url: str, destination: Path) -> tuple[Path, str]:
    """Descarga el ZIP oficial y lo extrae verificando que ninguna ruta salga del destino."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    digest = sha256_bytes(response.content)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Ruta insegura en ZIP: {member.filename}")
        archive.extractall(destination)
    return destination, digest


def load_movielens(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(directory)
    if not (directory / "ratings.csv").exists():
        raise FileNotFoundError(
            f"No existe {directory / 'ratings.csv'}. Ejecute: uv run python scripts/download_data.py"
        )
    ratings = pd.read_csv(directory / "ratings.csv")
    movies = pd.read_csv(directory / "movies.csv")
    validate_movielens(ratings, movies)
    return ratings, movies


def validate_movielens(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    required_ratings = {"userId", "movieId", "rating", "timestamp"}
    required_movies = {"movieId", "title", "genres"}
    if not required_ratings <= set(ratings.columns):
        raise ValueError("ratings.csv no tiene las columnas requeridas")
    if not required_movies <= set(movies.columns):
        raise ValueError("movies.csv no tiene las columnas requeridas")
    if not ratings["movieId"].isin(movies["movieId"]).all():
        raise ValueError("Existen ratings sin película correspondiente")
    if not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Existen ratings fuera del rango 0.5–5.0")
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Existen pares usuario-película duplicados")
    if not movies["movieId"].is_unique:
        raise ValueError("movieId repetido en movies.csv")


def movie_year(titles: pd.Series) -> pd.Series:
    """Año entre paréntesis al final del título (NaN si no existe)."""
    return pd.to_numeric(titles.str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")


def audit(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    users, items = ratings["userId"].nunique(), ratings["movieId"].nunique()
    per_user = ratings.groupby("userId").size()
    per_item = ratings.groupby("movieId").size()
    dates = pd.to_datetime(ratings["timestamp"], unit="s")
    sorted_counts = np.sort(per_item.to_numpy())[::-1]
    top10 = sorted_counts[: max(1, int(len(sorted_counts) * 0.10))].sum() / sorted_counts.sum()
    return {
        "ratings": len(ratings), "movies": len(movies), "users": int(users),
        "rated_movies": int(items),
        "density": float(len(ratings) / (users * items)),
        "first_rating": str(dates.min()), "last_rating": str(dates.max()),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_distribution": {str(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()},
        "share_ratings_ge_4": float((ratings["rating"] >= 4).mean()),
        "ratings_per_user": {"min": int(per_user.min()), "median": float(per_user.median()),
                             "max": int(per_user.max())},
        "ratings_per_movie": {"median": float(per_item.median()),
                              "share_movies_with_1_rating": float((per_item == 1).mean()),
                              "share_movies_with_lt5_ratings": float((per_item < 5).mean())},
        "share_ratings_top10pct_movies": float(top10),
        "movies_without_ratings": int((~movies["movieId"].isin(ratings["movieId"])).sum()),
        "movies_without_genres": int(movies["genres"].eq("(no genres listed)").sum()),
        "movies_without_year": int(movie_year(movies["title"]).isna().sum()),
        "duplicate_user_movie_pairs": int(ratings.duplicated(["userId", "movieId"]).sum()),
        "null_values": int(ratings.isna().sum().sum() + movies.isna().sum().sum()),
    }
