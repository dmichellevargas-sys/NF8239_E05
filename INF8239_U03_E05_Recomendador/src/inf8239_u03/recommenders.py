"""Recomendadores: popularidad, contenido (géneros), factorización matricial e híbrido.

Todos los modelos personalizados exponen `score(history, user_id)`, que devuelve un puntaje
para cada película del catálogo (`model.catalog`), de modo que se evalúan con el mismo código.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import movie_year


# --------------------------------------------------------------------------- popularidad
def weighted_popularity(ratings: pd.DataFrame, movies: pd.DataFrame, quantile: float = 0.80) -> pd.DataFrame:
    """Media bayesiana (estilo IMDb): mezcla la media de la película con la media global
    según su número de valoraciones; solo conserva películas con conteo >= cuantil."""
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    global_mean = float(ratings["rating"].mean())
    minimum = float(stats["count"].quantile(quantile))
    stats["weighted_score"] = (
        stats["count"] / (stats["count"] + minimum) * stats["mean"]
        + minimum / (stats["count"] + minimum) * global_mean
    )
    return stats.query("count >= @minimum").join(movies.set_index("movieId")).sort_values(
        "weighted_score", ascending=False)


class PopularityRecommender:
    """Línea base no personalizada: películas con más valoraciones en entrenamiento.
    Es también el fallback para usuarios sin historial."""

    name = "popularidad"

    def fit(self, ratings: pd.DataFrame, catalog: np.ndarray) -> PopularityRecommender:
        self.catalog = np.asarray(catalog)
        counts = ratings.groupby("movieId").size()
        means = ratings.groupby("movieId")["rating"].mean()
        # desempate por media para que el orden sea determinista
        self._scores = (counts.reindex(self.catalog).fillna(0).to_numpy()
                        + means.reindex(self.catalog).fillna(0).to_numpy() / 10)
        return self

    def score(self, history: pd.DataFrame | None = None, user_id: int | None = None) -> np.ndarray:
        return self._scores.copy()


class RandomRecommender:
    """Piso de referencia: orden aleatorio reproducible por usuario."""

    name = "aleatorio"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, ratings: pd.DataFrame, catalog: np.ndarray) -> RandomRecommender:
        self.catalog = np.asarray(catalog)
        return self

    def score(self, history: pd.DataFrame | None = None, user_id: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed + int(user_id or 0))
        return rng.random(len(self.catalog))


# --------------------------------------------------------------------------- contenido
def content_text(movies: pd.DataFrame) -> pd.Series:
    """Géneros como tokens + década de estreno (p. ej. 'decade1990')."""
    genres = movies["genres"].str.replace("(no genres listed)", "", regex=False)
    genres = genres.str.replace("-", "", regex=False).str.replace("|", " ", regex=False)
    decade = (movie_year(movies["title"]) // 10 * 10).astype("Int64").astype(str)
    decade = ("decade" + decade).where(~decade.eq("<NA>"), "")
    return (genres + " " + decade).str.strip()


class ContentRecommender:
    """Ítem a ítem (LAB08) y perfil de usuario: TF-IDF de géneros y década, similitud coseno."""

    name = "contenido"

    def fit(self, movies: pd.DataFrame, catalog: np.ndarray | None = None) -> ContentRecommender:
        self.movies = movies.reset_index(drop=True).copy()
        self.movies["genres_text"] = content_text(self.movies)
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=True)
        self.matrix = self.vectorizer.fit_transform(self.movies["genres_text"])
        self.row = {int(m): i for i, m in enumerate(self.movies["movieId"])}
        self.catalog = np.asarray(catalog if catalog is not None else self.movies["movieId"])
        self.catalog_matrix = self.matrix[[self.row[int(m)] for m in self.catalog]]
        return self

    def recommend(self, title: str, k: int = 10) -> pd.DataFrame:
        matches = self.movies.index[self.movies["title"].eq(title)]
        if len(matches) == 0:
            raise KeyError(f"Título no encontrado: {title}")
        index = int(matches[0])
        scores = cosine_similarity(self.matrix[index], self.matrix).ravel()
        order = [value for value in scores.argsort(kind="stable")[::-1] if value != index][:k]
        result = self.movies.loc[order, ["movieId", "title", "genres"]].copy()
        result["content_score"] = scores[order]
        return result.reset_index(drop=True)

    def profile(self, history: pd.DataFrame):
        rows = [self.row[int(m)] for m in history["movieId"] if int(m) in self.row]
        if not rows:
            return None
        ratings = history.loc[history["movieId"].astype(int).isin(self.row), "rating"].to_numpy()
        weights = np.clip(ratings - 2.5, 0.1, None)  # las películas que gustaron pesan más
        return self.matrix[rows].multiply(weights[:, None]).sum(axis=0) / weights.sum()

    def score(self, history: pd.DataFrame, user_id: int | None = None) -> np.ndarray:
        profile = self.profile(history)
        if profile is None:
            return np.zeros(len(self.catalog))
        return cosine_similarity(np.asarray(profile), self.catalog_matrix).ravel()


# --------------------------------------------------------------------------- factorización
class MatrixFactorization:
    """Factorización con sesgos (Koren et al., 2009) entrenada con SGD:
    r̂(u, i) = μ + b_u + b_i + p_u · q_i."""

    name = "factorizacion"

    def __init__(self, factors: int = 20, learning_rate: float = 0.01,
                 regularization: float = 0.05, seed: int = 42, fold_in_lambda: float = 1.0):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed
        self.fold_in_lambda = fold_in_lambda

    def fit(self, ratings: pd.DataFrame, epochs: int = 12, catalog: np.ndarray | None = None
            ) -> MatrixFactorization:
        self.users = sorted(ratings["userId"].unique())
        self.items = sorted(ratings["movieId"].unique())
        self.user_index = {value: index for index, value in enumerate(self.users)}
        self.item_index = {value: index for index, value in enumerate(self.items)}
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (len(self.users), self.factors))
        self.item_factors = rng.normal(0, 0.1, (len(self.items), self.factors))
        self.user_bias = np.zeros(len(self.users))
        self.item_bias = np.zeros(len(self.items))
        self.global_mean = float(ratings["rating"].mean())
        u = ratings["userId"].map(self.user_index).to_numpy()
        i = ratings["movieId"].map(self.item_index).to_numpy()
        r = ratings["rating"].to_numpy(dtype=float)
        lr, reg = self.learning_rate, self.regularization
        P, Q, bu, bi = self.user_factors, self.item_factors, self.user_bias, self.item_bias
        for _ in range(epochs):
            for idx in rng.permutation(len(r)):
                user, item = u[idx], i[idx]
                error = r[idx] - (self.global_mean + bu[user] + bi[item] + P[user] @ Q[item])
                bu[user] += lr * (error - reg * bu[user])
                bi[item] += lr * (error - reg * bi[item])
                previous_user = P[user].copy()
                P[user] += lr * (error * Q[item] - reg * P[user])
                Q[item] += lr * (error * previous_user - reg * Q[item])
        self.catalog = np.asarray(catalog if catalog is not None else self.items)
        self._catalog_rows = np.array([self.item_index.get(m, -1) for m in self.catalog])
        return self

    def predict_indices(self, user_index: int, item_index: int) -> float:
        return float(self.global_mean + self.user_bias[user_index] + self.item_bias[item_index]
                     + self.user_factors[user_index] @ self.item_factors[item_index])

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_index or movie_id not in self.item_index:
            return self.global_mean
        return self.predict_indices(self.user_index[user_id], self.item_index[movie_id])

    def fold_in(self, history: pd.DataFrame) -> tuple[float, np.ndarray]:
        """Estima sesgo y factores de un usuario nuevo con los factores de ítems fijos
        (regresión ridge sobre su historial)."""
        known = history[history["movieId"].isin(self.item_index)]
        if known.empty:
            return 0.0, np.zeros(self.factors)
        rows = known["movieId"].map(self.item_index).to_numpy()
        x = np.hstack([np.ones((len(rows), 1)), self.item_factors[rows]])
        y = known["rating"].to_numpy() - self.global_mean - self.item_bias[rows]
        w = np.linalg.solve(x.T @ x + self.fold_in_lambda * np.eye(x.shape[1]), x.T @ y)
        return float(w[0]), w[1:]

    def user_vector(self, history: pd.DataFrame | None, user_id: int | None):
        if user_id is not None and user_id in self.user_index:
            j = self.user_index[user_id]
            return self.user_bias[j], self.user_factors[j]
        return self.fold_in(history if history is not None else pd.DataFrame(columns=["movieId", "rating"]))

    def score(self, history: pd.DataFrame | None = None, user_id: int | None = None) -> np.ndarray:
        bias, vector = self.user_vector(history, user_id)
        rows = self._catalog_rows
        scores = np.full(len(rows), self.global_mean + bias)
        valid = rows >= 0
        scores[valid] += self.item_bias[rows[valid]] + self.item_factors[rows[valid]] @ vector
        return scores

    def top_n(self, user_id: int, seen: set[int], k: int = 10) -> pd.DataFrame:
        if user_id not in self.user_index:
            return pd.DataFrame(columns=["movieId", "collaborative_score"])
        scores = self.score(user_id=user_id)
        order = [j for j in np.argsort(-scores, kind="stable") if self.catalog[j] not in seen][:k]
        return pd.DataFrame({"movieId": self.catalog[order], "collaborative_score": scores[order]})


# --------------------------------------------------------------------------- híbrido
def minmax(values: np.ndarray) -> np.ndarray:
    low, high = float(np.min(values)), float(np.max(values))
    return (values - low) / (high - low) if high - low > 1e-12 else np.zeros_like(values, dtype=float)


class HybridRecommender:
    """Combinación lineal de tres señales normalizadas a [0, 1]:
    w_pop · popularidad (log del conteo) + w_mf · factorización + w_content · contenido.
    Los pesos suman 1; con w_pop = 0 y w_mf = alpha se obtiene el híbrido del LAB09."""

    def __init__(self, mf: MatrixFactorization, content: ContentRecommender,
                 popularity: PopularityRecommender, w_mf: float = 0.5, w_content: float = 0.5,
                 w_pop: float = 0.0):
        total = w_mf + w_content + w_pop
        if total <= 0 or min(w_mf, w_content, w_pop) < 0:
            raise ValueError("Los pesos deben ser no negativos y sumar más que cero")
        self.mf, self.content, self.popularity = mf, content, popularity
        self.w_mf, self.w_content, self.w_pop = w_mf / total, w_content / total, w_pop / total
        self.catalog = mf.catalog
        self._pop = minmax(np.log1p(popularity.score()))
        self.name = f"hibrido_p{self.w_pop:.2f}_m{self.w_mf:.2f}_c{self.w_content:.2f}"

    def score(self, history: pd.DataFrame, user_id: int | None = None) -> np.ndarray:
        total = self.w_pop * self._pop
        if self.w_mf > 0:
            total = total + self.w_mf * minmax(self.mf.score(history, user_id))
        if self.w_content > 0:
            total = total + self.w_content * minmax(self.content.score(history, user_id))
        return total


class FallbackRecommender:
    """Sistema final: híbrido si el usuario tiene historial; popularidad si no tiene."""

    def __init__(self, primary, fallback: PopularityRecommender, name: str = "sistema"):
        self.primary, self.fallback, self.name = primary, fallback, name
        self.catalog = primary.catalog

    def score(self, history: pd.DataFrame, user_id: int | None = None) -> np.ndarray:
        if history is None or history.empty:
            return self.fallback.score()
        return self.primary.score(history, user_id)


# --------------------------------------------------------------------------- particiones
def temporal_leave_one_out(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición del laboratorio: última valoración de cada usuario como prueba.
    Ojo: deja en entrenamiento valoraciones de otros usuarios posteriores a la prueba."""
    ordered = ratings.sort_values(["userId", "timestamp"], kind="stable")
    test = ordered.groupby("userId", sort=False).tail(1)
    train = ordered.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def temporal_cutoff_split(ratings: pd.DataFrame, quantile: float = 0.80
                          ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Corte temporal global: todo lo anterior al cuantil es entrenamiento; lo posterior, prueba.
    Ningún modelo ve información posterior al corte."""
    cutoff = int(ratings["timestamp"].quantile(quantile))
    train = ratings[ratings["timestamp"] < cutoff].reset_index(drop=True)
    test = ratings[ratings["timestamp"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff
