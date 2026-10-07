import pandas as pd

from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import ContentRecommender, weighted_popularity

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 10)
ratings, movies = load_movielens(MOVIELENS_DIR)
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
popular = weighted_popularity(ratings, movies).head(10)
popular[["title", "count", "mean", "weighted_score"]].to_csv(reports / "popular_top10.csv")
title = "Toy Story (1995)"
recommendations = ContentRecommender().fit(movies).recommend(title, 10)
recommendations.to_csv(reports / "content_recommendations.csv", index=False)
print("Popularidad:\n", popular[["title", "count", "mean", "weighted_score"]])
print(f"\nSimilares a {title}:\n", recommendations)
