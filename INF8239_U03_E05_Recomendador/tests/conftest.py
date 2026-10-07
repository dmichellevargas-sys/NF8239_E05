import pandas as pd
import pytest


@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "genres": ["Animation|Comedy", "Animation|Adventure", "Drama", "Drama|Romance", "Comedy"],
    })


@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "movieId": [1, 2, 3, 1, 3, 4, 2, 4, 5],
        "rating": [5.0, 4.0, 2.0, 4.0, 3.5, 5.0, 4.5, 3.0, 4.0],
        "timestamp": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    })
