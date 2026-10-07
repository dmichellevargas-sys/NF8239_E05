import numpy as np
import pandas as pd
import pytest

from inf8239_u03.recommenders import MatrixFactorization, temporal_cutoff_split, temporal_leave_one_out


def test_temporal_split_keeps_last_event(sample_ratings):
    train, test = temporal_leave_one_out(sample_ratings)
    assert len(test) == sample_ratings["userId"].nunique()
    assert all(test.groupby("userId")["timestamp"].max() >= train.groupby("userId")["timestamp"].max())


def test_matrix_factorization_predicts_finite_value(sample_ratings):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=2)
    assert np.isfinite(model.predict(1, 1))


def test_cutoff_split_has_no_future_in_train(sample_ratings):
    train, test, cutoff = temporal_cutoff_split(sample_ratings, 0.5)
    assert train["timestamp"].max() < cutoff <= test["timestamp"].min()
    assert len(train) + len(test) == len(sample_ratings)


def test_item_bias_learns_well_and_badly_rated_items():
    rng = np.random.default_rng(0)
    rows = [(u, 1, 5.0) for u in range(30)] + [(u, 2, 1.0) for u in range(30)]
    rows += [(u, 3, float(rng.integers(2, 5))) for u in range(30)]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    model = MatrixFactorization(factors=2, seed=1).fit(ratings, epochs=30)
    assert model.item_bias[model.item_index[1]] > 0 > model.item_bias[model.item_index[2]]


def test_fold_in_and_score_cover_whole_catalog(sample_ratings):
    model = MatrixFactorization(factors=3, seed=42).fit(sample_ratings, epochs=5,
                                                        catalog=np.array([1, 2, 3, 4, 5, 99]))
    new_user = pd.DataFrame({"movieId": [1, 2], "rating": [5.0, 4.5]})
    scores = model.score(new_user, user_id=None)
    assert scores.shape == (6,) and np.isfinite(scores).all()
    bias, _ = model.fold_in(new_user)
    assert scores[-1] == pytest.approx(model.global_mean + bias)  # ítem sin factores: media + sesgo
    bias, vector = model.fold_in(pd.DataFrame(columns=["movieId", "rating"]))
    assert bias == 0.0 and not vector.any()


def test_unknown_user_or_item_predicts_global_mean(sample_ratings):
    model = MatrixFactorization(factors=2, seed=42).fit(sample_ratings, epochs=1)
    assert model.predict(999, 1) == model.global_mean
