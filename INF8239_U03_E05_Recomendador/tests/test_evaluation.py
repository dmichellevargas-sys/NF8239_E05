import numpy as np
import pandas as pd
import pytest

from inf8239_u03.evaluation import build_tasks, one_se_choice, pareto_flags, recommend
from inf8239_u03.recommenders import (
    ContentRecommender,
    FallbackRecommender,
    HybridRecommender,
    MatrixFactorization,
    PopularityRecommender,
)


@pytest.fixture
def split():
    train = pd.DataFrame({"userId": [1, 1, 2, 2], "movieId": [1, 2, 1, 3],
                          "rating": [5.0, 4.0, 4.0, 5.0], "timestamp": [1, 2, 3, 4]})
    test = pd.DataFrame({  # usuario 1 cálido; usuario 7 frío con 7 valoraciones
        "userId": [1, 1] + [7] * 7, "movieId": [3, 4, 1, 2, 3, 4, 5, 2, 1],
        "rating": [5.0, 4.5, 5.0, 4.0, 4.0, 5.0, 4.5, 3.0, 2.0],
        "timestamp": [10, 11, 20, 21, 22, 23, 24, 25, 26]})
    return train, test


def test_build_tasks_creates_warm_and_cold_segments(split):
    train, test = split
    tasks = build_tasks(train, test, np.array([1, 2, 3, 4, 5]), onboarding_k=5)
    segments = {(t.user_id, t.segment) for t in tasks}
    assert {(1, "calido"), (7, "frio_k0")} <= segments
    warm = next(t for t in tasks if t.segment == "calido")
    assert warm.relevant == {3, 4} and warm.history_size == 2


def test_relevant_items_exclude_history_and_items_outside_catalog(split):
    train, test = split
    tasks = build_tasks(train, test, np.array([1, 2, 3]), onboarding_k=5)
    warm = next(t for t in tasks if t.segment == "calido")
    assert warm.relevant == {3}  # 4 no está en el catálogo
    # frío k=5: sus primeras 5 valoraciones son historial; lo que queda no es relevante (< 4)
    assert not any(t.segment == "frio_k5" for t in tasks)


def test_recommend_never_returns_seen_items(sample_ratings, sample_movies):
    catalog = np.array([1, 2, 3, 4, 5])
    pop = PopularityRecommender().fit(sample_ratings, catalog)
    train, test = sample_ratings.iloc[:6], sample_ratings.iloc[6:]
    task = build_tasks(train, test.assign(userId=1), catalog, threshold=0.5)[0]
    recs = recommend(pop, task, k=5)
    assert not set(recs) & set(task.history["movieId"])


def test_hybrid_normalizes_weights_and_rejects_negative(sample_ratings, sample_movies):
    catalog = np.array([1, 2, 3, 4, 5])
    pop = PopularityRecommender().fit(sample_ratings, catalog)
    mf = MatrixFactorization(factors=2).fit(sample_ratings, epochs=1, catalog=catalog)
    content = ContentRecommender().fit(sample_movies, catalog)
    hybrid = HybridRecommender(mf, content, pop, w_mf=2, w_content=1, w_pop=1)
    assert (hybrid.w_mf, hybrid.w_content, hybrid.w_pop) == (0.5, 0.25, 0.25)
    with pytest.raises(ValueError):
        HybridRecommender(mf, content, pop, w_mf=-1, w_content=1)


def test_fallback_uses_popularity_without_history(sample_ratings, sample_movies):
    catalog = np.array([1, 2, 3, 4, 5])
    pop = PopularityRecommender().fit(sample_ratings, catalog)
    content = ContentRecommender().fit(sample_movies, catalog)
    system = FallbackRecommender(content, pop)
    empty = pd.DataFrame(columns=["movieId", "rating"])
    np.testing.assert_array_equal(system.score(empty), pop.score())


def test_pareto_and_one_se_rule():
    df = pd.DataFrame({"model": ["a", "b", "c", "d"], "ndcg@10": [0.30, 0.29, 0.10, 0.05],
                       "ndcg_se": [0.02, 0.02, 0.01, 0.01], "coverage": [0.01, 0.03, 0.02, 0.10]})
    assert pareto_flags(df, "ndcg@10", "coverage") == [True, True, False, True]
    assert one_se_choice(df) == "b"
