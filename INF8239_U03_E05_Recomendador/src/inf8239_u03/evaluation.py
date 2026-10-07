"""Protocolo de evaluación offline: tareas por usuario, recomendación top-k y métricas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import intra_list_diversity, ndcg_at_k, novelty, precision_at_k, recall_at_k, user_hit


@dataclass
class UserTask:
    user_id: int
    segment: str                 # "calido", "frio_k5" o "frio_k0"
    history: pd.DataFrame        # valoraciones que el modelo puede usar
    relevant: set[int]           # películas valoradas >= umbral en el periodo evaluado
    history_size: int
    target: pd.DataFrame | None = None  # valoraciones evaluadas (para RMSE)


def build_tasks(train: pd.DataFrame, test: pd.DataFrame, catalog: np.ndarray,
                onboarding_k: int = 5, threshold: float = 4.0) -> list[UserTask]:
    """Usuarios cálidos: historial = entrenamiento. Usuarios fríos (sin entrenamiento):
    frio_k5 usa sus primeras k valoraciones posteriores al corte como historial;
    frio_k0 no usa ninguna. Solo cuentan como relevantes películas del catálogo no vistas."""
    catalog_set = {int(m) for m in catalog}
    train_users = set(train["userId"])
    tasks = []
    for user_id, events in test.sort_values(["userId", "timestamp"], kind="stable").groupby("userId"):
        if user_id in train_users:
            history = train.loc[train["userId"].eq(user_id), ["movieId", "rating"]]
            target = events
            segments = [("calido", history, target)]
        else:
            first, rest = events.iloc[:onboarding_k], events.iloc[onboarding_k:]
            segments = [("frio_k0", events.iloc[:0][["movieId", "rating"]], events),
                        ("frio_k5", first[["movieId", "rating"]], rest)]
        for segment, history, target in segments:
            seen = set(history["movieId"].astype(int))
            relevant = {int(m) for m in target.loc[target["rating"] >= threshold, "movieId"]
                        if int(m) in catalog_set and int(m) not in seen}
            if relevant:
                tasks.append(UserTask(int(user_id), segment, history.reset_index(drop=True),
                                      relevant, len(history), target.reset_index(drop=True)))
    return tasks


def recommend(model, task: UserTask, k: int = 10) -> list[int]:
    scores = model.score(task.history, task.user_id if task.segment == "calido" else None)
    order = np.argsort(-scores, kind="stable")
    seen = set(task.history["movieId"].astype(int))
    out = []
    for j in order:
        item = int(model.catalog[j])
        if item not in seen:
            out.append(item)
            if len(out) == k:
                break
    return out


def evaluate(model, tasks: list[UserTask], popularity: dict[int, float],
             item_vectors: dict[int, np.ndarray], k: int = 10) -> tuple[pd.DataFrame, dict]:
    rows, lists = [], {}
    for task in tasks:
        recs = recommend(model, task, k)
        lists[(task.user_id, task.segment)] = recs
        rows.append({
            "userId": task.user_id, "segment": task.segment, "history_size": task.history_size,
            "n_relevant": len(task.relevant),
            "precision": precision_at_k(recs, task.relevant, k),
            "recall": recall_at_k(recs, task.relevant, k),
            "ndcg": ndcg_at_k(recs, task.relevant, k),
            "hit": user_hit(recs, task.relevant, k),
            "novelty": novelty(recs, popularity),
            "diversity": intra_list_diversity(recs, item_vectors),
            "mean_popularity_rank": float(np.mean([popularity_rank(popularity).get(i, len(popularity))
                                                   for i in recs])),
        })
    return pd.DataFrame(rows), lists


_RANK_CACHE: dict[int, dict[int, int]] = {}


def popularity_rank(popularity: dict[int, float]) -> dict[int, int]:
    key = id(popularity)
    if key not in _RANK_CACHE:
        ordered = sorted(popularity, key=lambda m: -popularity[m])
        _RANK_CACHE[key] = {m: r + 1 for r, m in enumerate(ordered)}
    return _RANK_CACHE[key]


def coverage(lists: dict, catalog_size: int) -> float:
    return len({i for recs in lists.values() for i in recs}) / catalog_size


def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(per_user: pd.DataFrame, lists: dict, catalog_size: int, model_name: str) -> pd.DataFrame:
    rows = []
    for segment, group in [("todos", per_user), *per_user.groupby("segment")]:
        keys = [k for k in lists if segment == "todos" or k[1] == segment]
        low, high = bootstrap_mean_ci(group["ndcg"])
        rows.append({
            "model": model_name, "segment": segment, "users": len(group),
            "precision@10": group["precision"].mean(), "recall@10": group["recall"].mean(),
            "ndcg@10": group["ndcg"].mean(), "ndcg_ci95_low": low, "ndcg_ci95_high": high,
            "ndcg_se": group["ndcg"].std(ddof=1) / np.sqrt(len(group)) if len(group) > 1 else np.nan,
            "hit@10": group["hit"].mean(),
            "coverage": coverage({k: lists[k] for k in keys}, catalog_size),
            "novelty": group["novelty"].mean(), "diversity": group["diversity"].mean(),
            "mean_popularity_rank": group["mean_popularity_rank"].mean(),
        })
    return pd.DataFrame(rows)


def pareto_flags(df: pd.DataFrame, gain: str, other: str) -> list[bool]:
    """True si ninguna otra fila es al menos igual en ambas métricas y mejor en una."""
    flags = []
    for _, row in df.iterrows():
        dominated = ((df[gain] >= row[gain]) & (df[other] >= row[other])
                     & ((df[gain] > row[gain]) | (df[other] > row[other]))).any()
        flags.append(not bool(dominated))
    return flags


def one_se_choice(df: pd.DataFrame, gain: str = "ndcg@10", se: str = "ndcg_se",
                  secondary: str = "coverage") -> str:
    """Regla de un error estándar: entre las configuraciones cuyo NDCG no es inferior al mejor
    menos un error estándar, elige la de mayor cobertura."""
    best = df.loc[df[gain].idxmax()]
    eligible = df[df[gain] >= best[gain] - best[se]]
    return str(eligible.sort_values([secondary, gain], ascending=False).iloc[0]["model"])
