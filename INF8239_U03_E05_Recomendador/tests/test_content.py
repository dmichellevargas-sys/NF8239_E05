import numpy as np
import pandas as pd

from inf8239_u03.recommenders import ContentRecommender, content_text, weighted_popularity


def test_content_excludes_query_and_returns_unique_items(sample_movies):
    result = ContentRecommender().fit(sample_movies).recommend("Alpha", 3)
    assert "Alpha" not in set(result["title"])
    assert result["movieId"].is_unique


def test_popularity_contains_weighted_score(sample_ratings, sample_movies):
    result = weighted_popularity(sample_ratings, sample_movies, quantile=0.0)
    assert "weighted_score" in result.columns


def test_weighted_score_shrinks_toward_global_mean():
    ratings = pd.DataFrame({"userId": [1, 2, 3, 4], "movieId": [1, 1, 1, 2],
                            "rating": [4.0, 4.0, 4.0, 5.0], "timestamp": [1, 2, 3, 4]})
    movies = pd.DataFrame({"movieId": [1, 2], "title": ["A", "B"], "genres": ["Drama", "Drama"]})
    result = weighted_popularity(ratings, movies, quantile=0.0)
    assert result.loc[2, "weighted_score"] < 5.0


def test_content_text_adds_decade_and_drops_missing_genres():
    movies = pd.DataFrame({"title": ["Film (1994)", "Other"],
                           "genres": ["Sci-Fi|Drama", "(no genres listed)"]})
    text = content_text(movies)
    assert text.iloc[0] == "SciFi Drama decade1990"
    assert text.iloc[1] == ""


def test_profile_prefers_items_with_liked_genres(sample_movies):
    model = ContentRecommender().fit(sample_movies)
    history = pd.DataFrame({"movieId": [3], "rating": [5.0]})  # Gamma: Drama
    scores = dict(zip(model.catalog, model.score(history)))
    assert scores[4] > scores[1]  # Delta (Drama|Romance) antes que Alpha (Animation|Comedy)


def test_empty_history_gives_zero_scores(sample_movies):
    model = ContentRecommender().fit(sample_movies)
    empty = pd.DataFrame(columns=["movieId", "rating"])
    assert np.all(model.score(empty) == 0)
