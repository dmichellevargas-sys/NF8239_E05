"""Métricas de ranking, cobertura, novedad y diversidad."""
from __future__ import annotations

import numpy as np
import pandas as pd


def hit_rate_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    """Proporción de filas (usuario, película) de `truth` que aparecen en el top-k del usuario."""
    hits = []
    for row in truth.itertuples():
        if row.userId in recommendations:
            hits.append(int(row.movieId in recommendations[row.userId][:k]))
    return sum(hits) / len(hits) if hits else 0.0


def catalog_coverage(recommendations: dict[int, list[int]], catalog_size: int) -> float:
    recommended = {item for values in recommendations.values() for item in values}
    return len(recommended) / catalog_size if catalog_size else 0.0


def precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    """NDCG binario: ganancia 1 si la película es relevante, descontada por la posición."""
    gains = [1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(sum(gains) / ideal) if ideal else 0.0


def user_hit(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    return float(bool(set(recommended[:k]) & relevant))


def novelty(recommended: list[int], popularity: dict[int, float]) -> float:
    """Autoinformación media: -log2 de la proporción de usuarios que valoraron el ítem."""
    values = [-np.log2(max(popularity.get(item, 0.0), 1e-6)) for item in recommended]
    return float(np.mean(values)) if values else 0.0


def intra_list_diversity(recommended: list[int], item_vectors: dict[int, np.ndarray]) -> float:
    """1 - similitud coseno media entre pares de la lista (vectores de géneros normalizados)."""
    vectors = [item_vectors[i] for i in recommended if i in item_vectors]
    if len(vectors) < 2:
        return 0.0
    matrix = np.vstack(vectors)
    sims = matrix @ matrix.T
    n = len(vectors)
    return float(1 - (sims.sum() - np.trace(sims)) / (n * (n - 1)))
