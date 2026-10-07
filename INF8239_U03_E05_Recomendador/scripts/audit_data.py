"""Auditoría de MovieLens: tamaño, densidad, distribución, cola larga y evolución temporal.

Uso:  uv run python scripts/audit_data.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inf8239_u03.config import MOVIELENS_DIR, REPORTS, TEST_QUANTILE
from inf8239_u03.data import audit, load_movielens


def run() -> dict:
    REPORTS.mkdir(exist_ok=True)
    ratings, movies = load_movielens(MOVIELENS_DIR)
    summary = audit(ratings, movies)
    (REPORTS / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dates = pd.to_datetime(ratings["timestamp"], unit="s")
    cutoff = pd.to_datetime(int(ratings["timestamp"].quantile(TEST_QUANTILE)), unit="s")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ratings["rating"].value_counts().sort_index().plot.bar(ax=axes[0], color="#1f77b4")
    axes[0].set(title="Distribución de valoraciones", xlabel="Valoración", ylabel="Cantidad")
    per_year = dates.dt.year.value_counts().sort_index()
    axes[1].bar(per_year.index, per_year.values, color="#94a3b8")
    axes[1].axvline(cutoff.year + cutoff.dayofyear / 366, color="#c0392b", ls="--",
                    label=f"Corte temporal ({cutoff.date()})")
    axes[1].set(title="Valoraciones por año", xlabel="Año", ylabel="Cantidad")
    axes[1].legend(fontsize=8)
    counts = np.sort(ratings.groupby("movieId").size().to_numpy())[::-1]
    axes[2].plot(np.arange(1, len(counts) + 1), counts, color="#0f7c7b")
    axes[2].set(xscale="log", yscale="log", title="Cola larga: valoraciones por película",
                xlabel="Posición de la película (log)", ylabel="Valoraciones (log)")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_audit.png", dpi=170)
    plt.close(fig)
    return summary


def main() -> None:
    s = run()
    print(f"Ratings: {s['ratings']} | Películas: {s['movies']} | Usuarios: {s['users']}")
    print(f"Densidad: {s['density']:.4f} | Rango: {s['first_rating']} a {s['last_rating']}")
    print("Por usuario:", s["ratings_per_user"], "| Por película:", s["ratings_per_movie"])
    print(f"El 10 % de películas más valoradas concentra {s['share_ratings_top10pct_movies']:.1%} "
          "de las valoraciones")
    print("Sin géneros:", s["movies_without_genres"], "| Sin año:", s["movies_without_year"],
          "| Duplicados:", s["duplicate_user_movie_pairs"], "| Nulos:", s["null_values"])


if __name__ == "__main__":
    main()
