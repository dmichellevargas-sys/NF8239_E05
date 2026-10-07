"""Demostración local del recomendador (requiere haber descargado MovieLens)."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from inf8239_u03.config import MOVIELENS_DIR, RANDOM_STATE, REPORTS, TEST_QUANTILE
from inf8239_u03.data import load_movielens
from inf8239_u03.recommenders import (
    ContentRecommender,
    FallbackRecommender,
    HybridRecommender,
    MatrixFactorization,
    PopularityRecommender,
    temporal_cutoff_split,
)

st.title("INF-8239 · Recomendador académico de películas")
st.caption("Demostración con MovieLens (ml-latest-small). Las listas son estimaciones offline, "
           "no preferencias garantizadas.")
try:
    ratings, movies = load_movielens(MOVIELENS_DIR)
except FileNotFoundError:
    st.error("Descargue MovieLens con: uv run python scripts/download_data.py")
    st.stop()


@st.cache_resource
def build_system():
    results = json.loads((REPORTS / "results.json").read_text()) if (REPORTS / "results.json").exists() else {}
    weights = results.get("selected_hybrid", {"w_pop": 0.25, "w_mf": 0.5, "w_content": 0.25})
    train, _, _ = temporal_cutoff_split(ratings, TEST_QUANTILE)
    catalog = np.array(sorted(train["movieId"].unique()))
    pop = PopularityRecommender().fit(train, catalog)
    content = ContentRecommender().fit(movies, catalog)
    mf = MatrixFactorization(factors=results.get("factors", 20), seed=RANDOM_STATE,
                             fold_in_lambda=results.get("fold_in_lambda", 10.0)).fit(
        train, results.get("epochs", 20), catalog)
    hybrid = HybridRecommender(mf, content, pop, weights["w_mf"], weights["w_content"], weights["w_pop"])
    return FallbackRecommender(hybrid, pop), content, catalog, weights


system, content, catalog, weights = build_system()
titles = movies.set_index("movieId")["title"]
tab1, tab2 = st.tabs(["Usuario nuevo", "Películas similares"])
with tab1:
    st.write("Valore hasta 5 películas (encuesta de bienvenida). Sin valoraciones se usa popularidad.")
    choices = st.multiselect("Películas", options=list(catalog), format_func=lambda m: titles[m],
                             max_selections=5)
    history = pd.DataFrame({"movieId": choices,
                            "rating": [st.slider(titles[m], 0.5, 5.0, 4.0, 0.5, key=int(m)) for m in choices]})
    if st.button("Recomendar", key="user"):
        scores = system.score(history)
        order = [int(catalog[j]) for j in np.argsort(-scores, kind="stable") if catalog[j] not in set(choices)][:10]
        st.dataframe(movies.set_index("movieId").loc[order, ["title", "genres"]], use_container_width=True)
        st.info(f"Pesos del híbrido: popularidad {weights['w_pop']:.2f}, factorización {weights['w_mf']:.2f}, "
                f"contenido {weights['w_content']:.2f}. Riesgo: sesgo hacia películas populares.")
with tab2:
    title = st.selectbox("Película de referencia", movies["title"].sort_values())
    if st.button("Buscar similares", key="item"):
        st.dataframe(content.recommend(title, 10), use_container_width=True)
        st.info("Método: TF-IDF de géneros y década, similitud coseno. Riesgo: sobre-especialización.")
