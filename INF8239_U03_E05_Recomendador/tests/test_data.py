import pandas as pd
import pytest

from inf8239_u03.data import audit, movie_year, validate_movielens


def test_valid_contract(sample_ratings, sample_movies):
    validate_movielens(sample_ratings, sample_movies)


def test_rejects_orphan_rating(sample_ratings, sample_movies):
    changed = sample_ratings.copy()
    changed.loc[0, "movieId"] = 999
    with pytest.raises(ValueError, match="sin película"):
        validate_movielens(changed, sample_movies)


def test_rejects_duplicate_user_movie_pair(sample_ratings, sample_movies):
    changed = pd.concat([sample_ratings, sample_ratings.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicados"):
        validate_movielens(changed, sample_movies)


def test_rejects_out_of_range_rating(sample_ratings, sample_movies):
    changed = sample_ratings.copy()
    changed.loc[0, "rating"] = 6
    with pytest.raises(ValueError, match="fuera del rango"):
        validate_movielens(changed, sample_movies)


def test_movie_year_extracts_trailing_year():
    years = movie_year(pd.Series(["Toy Story (1995)", "Sin año", "Blade Runner 2049 (2017) "]))
    assert years.iloc[0] == 1995 and pd.isna(years.iloc[1]) and years.iloc[2] == 2017


def test_audit_reports_density_and_counts(sample_ratings, sample_movies):
    summary = audit(sample_ratings, sample_movies)
    assert summary["ratings"] == 9 and summary["users"] == 3
    assert summary["density"] == pytest.approx(9 / (3 * 5))
    assert summary["duplicate_user_movie_pairs"] == 0
