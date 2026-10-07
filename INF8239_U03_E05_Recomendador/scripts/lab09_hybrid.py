"""Experimento principal del Ejercicio 05: popularidad, contenido, factorización e híbrido.

Protocolo (declarado antes de mirar la prueba):
  1. Corte temporal global en el cuantil 0.80 de las fechas: nada posterior al corte se usa
     para entrenar, calcular popularidad ni definir el catálogo.
  2. Usuarios de prueba: cálidos (tienen historial previo) y fríos (aparecen después del corte).
     Fríos k=0: sin historial (solo cabe el fallback). Fríos k=5: sus 5 primeras valoraciones
     posteriores al corte hacen de encuesta de bienvenida y se evalúa el resto.
  3. Relevante = valoración >= 4 sobre películas del catálogo no vistas. Top-10.
  4. Ajuste (factores, regularización del fold-in y pesos del híbrido) con un segundo corte temporal
     DENTRO de entrenamiento (validación). La prueba se usa una sola vez.
  5. Decisión: frontera de Pareto NDCG@10 vs. cobertura en validación; regla de un error
     estándar: entre las configuraciones a menos de 1 EE del mejor NDCG, la de mayor cobertura.

Uso:  uv run python scripts/lab09_hybrid.py [--epochs 20] [--factors 10 20 40]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import normalize

from inf8239_u03.config import (
    MOVIELENS_DIR,
    ONBOARDING_K,
    RANDOM_STATE,
    RELEVANT_THRESHOLD,
    REPORTS,
    TEST_QUANTILE,
    TOP_K,
    VALID_QUANTILE,
)
from inf8239_u03.data import load_movielens
from inf8239_u03.evaluation import (
    build_tasks,
    evaluate,
    one_se_choice,
    pareto_flags,
    popularity_rank,
    summarize,
)
from inf8239_u03.recommenders import (
    ContentRecommender,
    FallbackRecommender,
    HybridRecommender,
    MatrixFactorization,
    PopularityRecommender,
    RandomRecommender,
    temporal_cutoff_split,
    temporal_leave_one_out,
)

WEIGHT_STEP = 0.25
FOLD_IN_LAMBDAS = [1.0, 5.0, 10.0, 25.0, 50.0]
LABELS = {"aleatorio": "Aleatorio", "popularidad": "Popularidad", "contenido": "Contenido",
          "factorizacion": "Factorización", "hibrido_lab09": "Híbrido LAB09 (0.75 FM + 0.25 cont.)",
          "hibrido_seleccionado": "Híbrido seleccionado", "sistema": "Sistema final (con fallback)"}


class Context:
    """Todo lo que se calcula a partir de un conjunto de entrenamiento (sin mirar el futuro)."""

    def __init__(self, train: pd.DataFrame, movies: pd.DataFrame):
        self.train = train
        self.catalog = np.array(sorted(train["movieId"].unique()))
        self.popularity = (train.groupby("movieId")["userId"].nunique()
                           / train["userId"].nunique()).to_dict()
        self.pop_model = PopularityRecommender().fit(train, self.catalog)
        self.content = ContentRecommender().fit(movies, self.catalog)
        unit = normalize(self.content.matrix).toarray()
        self.item_vectors = {int(m): unit[self.content.row[int(m)]] for m in self.catalog}

    def run(self, model, tasks, name):
        per_user, lists = evaluate(model, tasks, self.popularity, self.item_vectors, TOP_K)
        per_user.insert(0, "model", name)
        return per_user, lists, summarize(per_user, lists, len(self.catalog), name)


def weight_grid(step: float = WEIGHT_STEP):
    values = np.round(np.arange(0, 1 + 1e-9, step), 2)
    return [(p, m, round(1 - p - m, 2)) for p in values for m in values if p + m <= 1 + 1e-9
            and p != 1.0]  # popularidad pura ya es una línea base


def split_report(ratings, train, test, cutoff):
    train_users, test_users = set(train["userId"]), set(test["userId"])
    new_items = set(test["movieId"]) - set(train["movieId"])
    loo_train, loo_test = temporal_leave_one_out(ratings)
    later = loo_train["timestamp"].to_numpy()
    leak = float(np.mean([np.any(later > t) for t in loo_test["timestamp"].to_numpy()]))
    share_later = float(np.mean(later > np.median(loo_test["timestamp"])))
    return {
        "cutoff_quantile": TEST_QUANTILE, "cutoff_timestamp": cutoff,
        "cutoff_date": str(pd.to_datetime(cutoff, unit="s").date()),
        "train_ratings": len(train), "test_ratings": len(test),
        "train_users": len(train_users), "test_users": len(test_users),
        "warm_test_users": len(test_users & train_users),
        "cold_test_users": len(test_users - train_users),
        "catalog_size": int(train["movieId"].nunique()),
        "test_ratings_on_items_new_after_cutoff": float(test["movieId"].isin(new_items).mean()),
        "leave_one_out_test_events_with_later_training_data": leak,
        "leave_one_out_train_ratings_after_median_test_time": share_later,
    }


def tune(ratings_train, movies, factors_grid, epochs):
    """Validación: segundo corte temporal dentro del entrenamiento."""
    inner_train, inner_valid, _ = temporal_cutoff_split(ratings_train, VALID_QUANTILE)
    ctx = Context(inner_train, movies)
    tasks = [t for t in build_tasks(inner_train, inner_valid, ctx.catalog, ONBOARDING_K,
                                    RELEVANT_THRESHOLD) if t.segment != "frio_k0"]
    mf_rows, models = [], {}
    for f in factors_grid:
        start = perf_counter()
        mf = MatrixFactorization(factors=f, seed=RANDOM_STATE).fit(inner_train, epochs, ctx.catalog)
        seconds = perf_counter() - start
        _, _, summ = ctx.run(mf, tasks, f"factorizacion_f{f}")
        models[f] = mf
        mf_rows.append({"factors": f, "fit_seconds": seconds,
                        **summ.iloc[0][["ndcg@10", "ndcg_se", "coverage"]].to_dict()})
    mf_table = pd.DataFrame(mf_rows)
    mf_table.to_csv(REPORTS / "validation_factors.csv", index=False)
    best = mf_table.loc[mf_table["ndcg@10"].idxmax()]
    eligible = mf_table[mf_table["ndcg@10"] >= best["ndcg@10"] - best["ndcg_se"]]
    factors = int(eligible.sort_values("fit_seconds").iloc[0]["factors"])  # el más barato

    # Regularización del fold-in (usuarios nuevos): menor RMSE en fríos k=5 de validación
    all_tasks = build_tasks(inner_train, inner_valid, ctx.catalog, ONBOARDING_K, RELEVANT_THRESHOLD)
    lam_rows = []
    for lam in FOLD_IN_LAMBDAS:
        models[factors].fold_in_lambda = lam
        r = rmse_table(ctx, models[factors], all_tasks)
        lam_rows.append({"fold_in_lambda": lam,
                         "rmse_frio_k5": float(r.query("segment == 'frio_k5' and model == 'factorización'")["rmse"].iloc[0])})
    lam_table = pd.DataFrame(lam_rows)
    lam_table.to_csv(REPORTS / "validation_fold_in.csv", index=False)
    fold_in_lambda = float(lam_table.loc[lam_table["rmse_frio_k5"].idxmin(), "fold_in_lambda"])
    models[factors].fold_in_lambda = fold_in_lambda

    rows = [ctx.run(ctx.pop_model, tasks, "popularidad")[2].iloc[0]]
    for wp, wm, wc in weight_grid():
        model = HybridRecommender(models[factors], ctx.content, ctx.pop_model, wm, wc, wp)
        rows.append(ctx.run(model, tasks, model.name)[2].iloc[0])
    grid = pd.DataFrame(rows).reset_index(drop=True)
    grid["is_pareto"] = pareto_flags(grid, "ndcg@10", "coverage")
    choice = one_se_choice(grid)
    grid["selected"] = grid["model"] == choice
    grid.to_csv(REPORTS / "validation_grid.csv", index=False)
    return factors, fold_in_lambda, choice, grid, len(tasks)


def weights_from_name(name: str) -> tuple[float, float, float]:
    parts = dict(p[0] and (p[0], float(p[1:])) for p in name.split("_")[1:])
    return parts["p"], parts["m"], parts["c"]


def plot_pareto(grid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in grid.iterrows():
        color = "#0f7c7b" if r["selected"] else ("#1f77b4" if r["is_pareto"] else "#cbd5e1")
        ax.errorbar(r["coverage"], r["ndcg@10"], yerr=r["ndcg_se"], fmt="o", color=color,
                    ms=7 if r["selected"] else 5, capsize=2)
        if r["is_pareto"] or r["model"] == "popularidad":
            label = "popularidad" if r["model"] == "popularidad" else r["model"].replace("hibrido_", "")
            ax.annotate(label, (r["coverage"], r["ndcg@10"]), fontsize=7, xytext=(4, 3),
                        textcoords="offset points")
    front = grid[grid["is_pareto"]].sort_values("coverage")
    ax.plot(front["coverage"], front["ndcg@10"], "--", color="#1f77b4", lw=1)
    ax.set(xlabel="Cobertura del catálogo", ylabel="NDCG@10 (validación) ± 1 EE",
           title="Híbridos en validación: azul = Pareto, verde = seleccionado")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_pareto_validation.png", dpi=170)
    plt.close(fig)


def rmse_table(ctx: Context, mf: MatrixFactorization, tasks_all) -> pd.DataFrame:
    """Error de predicción de la valoración en usuarios cálidos y fríos k=5."""
    train = ctx.train
    item_mean = train.groupby("movieId")["rating"].mean()
    rows = []
    for segment in ("calido", "frio_k5"):
        y, p_global, p_item, p_mf = [], [], [], []
        for task in [t for t in tasks_all if t.segment == segment]:
            target = task.target
            known = target[target["movieId"].isin(mf.item_index)]
            bias, vector = mf.user_vector(task.history, task.user_id if segment == "calido" else None)
            for row in known.itertuples():
                j = mf.item_index[row.movieId]
                y.append(row.rating)
                p_global.append(mf.global_mean)
                p_item.append(item_mean[row.movieId])
                p_mf.append(float(np.clip(mf.global_mean + bias + mf.item_bias[j]
                                          + mf.item_factors[j] @ vector, 0.5, 5.0)))
        for name, pred in (("media global", p_global), ("media de la película", p_item),
                           ("factorización", p_mf)):
            rows.append({"segment": segment, "model": name, "ratings": len(y),
                         "rmse": root_mean_squared_error(y, pred)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--factors", type=int, nargs="+", default=[10, 20, 40])
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    ratings, movies = load_movielens(MOVIELENS_DIR)

    train, test, cutoff = temporal_cutoff_split(ratings, TEST_QUANTILE)
    split = split_report(ratings, train, test, cutoff)
    (REPORTS / "split_summary.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print("Corte:", split["cutoff_date"], "| cálidos:", split["warm_test_users"],
          "| fríos:", split["cold_test_users"])

    factors, fold_in_lambda, choice, grid, n_valid = tune(train, movies, args.factors, args.epochs)
    plot_pareto(grid)
    w_pop, w_mf, w_content = weights_from_name(choice) if choice != "popularidad" else (1, 0, 0)
    print(f"Validación ({n_valid} tareas): factores = {factors}; lambda fold-in = {fold_in_lambda}; "
          f"elegido = {choice}")

    # ---------------- reentrenamiento con todo el entrenamiento y evaluación única en prueba
    ctx = Context(train, movies)
    tasks = build_tasks(train, test, ctx.catalog, ONBOARDING_K, RELEVANT_THRESHOLD)
    start = perf_counter()
    mf = MatrixFactorization(factors=factors, seed=RANDOM_STATE, fold_in_lambda=fold_in_lambda).fit(
        train, args.epochs, ctx.catalog)
    mf_seconds = perf_counter() - start
    selected = HybridRecommender(mf, ctx.content, ctx.pop_model, w_mf, w_content, w_pop)
    models = {
        "aleatorio": RandomRecommender(RANDOM_STATE).fit(train, ctx.catalog),
        "popularidad": ctx.pop_model, "contenido": ctx.content, "factorizacion": mf,
        "hibrido_lab09": HybridRecommender(mf, ctx.content, ctx.pop_model, 0.75, 0.25, 0.0),
        "hibrido_seleccionado": selected,
        "sistema": FallbackRecommender(selected, ctx.pop_model),
    }
    summaries, per_users, lists_by_model, latency = [], [], {}, {}
    for name, model in models.items():
        use = tasks if name in ("aleatorio", "popularidad", "sistema") else \
            [t for t in tasks if t.segment != "frio_k0"]
        start = perf_counter()
        per_user, lists, summ = ctx.run(model, use, name)
        latency[name] = (perf_counter() - start) * 1000 / max(len(use), 1)
        summaries.append(summ)
        per_users.append(per_user)
        lists_by_model[name] = lists
    table = pd.concat(summaries, ignore_index=True)
    table["ms_per_user"] = table["model"].map(latency)
    table.to_csv(REPORTS / "test_metrics.csv", index=False)
    per_user_all = pd.concat(per_users, ignore_index=True)
    per_user_all.to_csv(REPORTS / "test_per_user.csv", index=False)

    # ---------------- figuras y análisis de errores
    plot_segments(table)
    plot_history_buckets(per_user_all)
    plot_popularity_bias(ctx, tasks, lists_by_model)
    rmse = rmse_table(ctx, mf, tasks)
    rmse.to_csv(REPORTS / "rmse.csv", index=False)
    examples(ctx, movies, tasks, lists_by_model)
    fallback = (pd.DataFrame({"movieId": ctx.catalog, "score": ctx.pop_model.score()})
                .nlargest(10, "score").merge(movies, on="movieId"))
    fallback[["movieId", "title", "genres"]].to_csv(REPORTS / "cold_start_fallback.csv", index=False)

    results = {
        "split": split, "factors": factors, "fold_in_lambda": fold_in_lambda, "epochs": args.epochs,
        "selected_hybrid": {"w_pop": w_pop, "w_mf": w_mf, "w_content": w_content, "name": choice},
        "selection_rule": "Pareto NDCG@10 vs cobertura en validación; regla de 1 EE -> mayor cobertura",
        "mf_train_seconds": mf_seconds,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor() or platform.machine(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__},
    }
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    cols = ["model", "segment", "users", "ndcg@10", "ndcg_ci95_low", "ndcg_ci95_high",
            "precision@10", "recall@10", "hit@10", "coverage", "novelty", "diversity"]
    print(table[cols].round(3).to_string(index=False))
    print(rmse.round(4).to_string(index=False))


def plot_segments(table: pd.DataFrame) -> None:
    order = ["aleatorio", "popularidad", "contenido", "factorizacion", "hibrido_lab09",
             "hibrido_seleccionado", "sistema"]
    segs = [("calido", "Usuarios cálidos"), ("frio_k5", "Fríos con 5 valoraciones"),
            ("frio_k0", "Fríos sin historial")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (seg, title) in zip(axes, segs):
        sub = table[table["segment"] == seg].set_index("model").reindex(order).dropna(subset=["ndcg@10"])
        err = np.vstack([sub["ndcg@10"] - sub["ndcg_ci95_low"], sub["ndcg_ci95_high"] - sub["ndcg@10"]])
        colors = ["#0f7c7b" if m in ("sistema", "hibrido_seleccionado") else "#94a3b8" for m in sub.index]
        ax.barh([LABELS[m] for m in sub.index], sub["ndcg@10"], xerr=err, color=colors, capsize=3)
        ax.set(title=f"{title} (n = {int(sub['users'].iloc[0])})", xlabel="NDCG@10 (IC 95 %)")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_test_segments.png", dpi=170)
    plt.close(fig)


def plot_history_buckets(per_user: pd.DataFrame) -> None:
    data = per_user[per_user["model"].isin(["popularidad", "sistema"])].copy()
    data["bucket"] = pd.cut(data["history_size"], [-1, 0, 5, 50, 200, 10_000],
                            labels=["0", "1-5", "6-50", "51-200", ">200"])
    table = data.groupby(["bucket", "model"], observed=True)["ndcg"].mean().unstack()
    counts = data[data["model"] == "sistema"].groupby("bucket", observed=True).size()
    table.to_csv(REPORTS / "ndcg_by_history_size.csv")
    table = table.rename(columns={"popularidad": "Popularidad", "sistema": "Sistema final"})
    ax = table.plot.bar(figsize=(8, 4), color=["#94a3b8", "#0f7c7b"], rot=0)
    ax.legend(title="")
    for i, n in enumerate(counts):
        ax.annotate(f"n={n}", (i, 0.01), ha="center", fontsize=8)
    ax.set(xlabel="Valoraciones en el historial del usuario", ylabel="NDCG@10 medio",
           title="Desempeño según el tamaño del historial (prueba)")
    plt.tight_layout()
    plt.savefig(REPORTS / "fig_history_size.png", dpi=170)
    plt.close()


def plot_popularity_bias(ctx: Context, tasks, lists_by_model) -> None:
    """Posición de popularidad (1 = la más valorada) de lo recomendado frente a lo relevante."""
    ranks = popularity_rank(ctx.popularity)
    data = {"Relevantes (verdad)": [ranks.get(m, len(ranks)) for t in tasks
                                   if t.segment != "frio_k0" for m in t.relevant]}
    for name in ("popularidad", "factorizacion", "contenido", "hibrido_seleccionado"):
        data[LABELS[name]] = [ranks.get(m, len(ranks)) for (u, seg), recs in lists_by_model[name].items()
                              if seg != "frio_k0" for m in recs]
    summary = pd.DataFrame({k: pd.Series(v).describe(percentiles=[0.5]) for k, v in data.items()}).T
    summary["share_top100"] = [float(np.mean(np.asarray(v) <= 100)) for v in data.values()]
    summary.to_csv(REPORTS / "popularity_bias.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(list(data.values()), vert=False, showfliers=False)
    ax.set_yticks(range(1, len(data) + 1), list(data.keys()))
    ax.set(xscale="log", xlabel="Posición de popularidad (1 = película más valorada; escala log)",
           title="Sesgo de popularidad: qué tan populares son las películas recomendadas")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_popularity_bias.png", dpi=170)
    plt.close(fig)


def examples(ctx, movies, tasks, lists_by_model) -> None:
    """Tres usuarios cálidos ilustrativos: historial favorito, aciertos y recomendaciones."""
    title = movies.set_index("movieId")["title"]
    rows = []
    warm = [t for t in tasks if t.segment == "calido"][:3]
    for task in warm:
        liked = task.history.sort_values("rating", ascending=False).head(3)["movieId"]
        for name in ("popularidad", "hibrido_seleccionado"):
            recs = lists_by_model[name][(task.user_id, task.segment)]
            rows.append({"userId": task.user_id, "model": name,
                         "historial_favorito": " | ".join(title[m] for m in liked),
                         "recomendaciones": " | ".join(title[m] for m in recs),
                         "aciertos": " | ".join(title[m] for m in recs if m in task.relevant)})
    pd.DataFrame(rows).to_csv(REPORTS / "example_recommendations.csv", index=False)


if __name__ == "__main__":
    main()
