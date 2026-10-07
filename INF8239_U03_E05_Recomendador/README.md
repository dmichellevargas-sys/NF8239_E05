# INF-8239 · Ejercicio 05 · Motor recomendador reproducible

Recomendador top-10 de películas con **MovieLens ml-latest-small**: popularidad, contenido,
factorización matricial e híbrido, evaluados con un **corte temporal global**, usuarios cálidos y
fríos, cobertura, novedad y diversidad, y una decisión sobre la **frontera de Pareto**.
Proyecto individual de la Unidad 03 (LAB08 + LAB09). Proyecto base del curso: Edwin Ramón José Nolasco.

> MovieLens **no se incluye** en el repositorio (términos de GroupLens). El script lo descarga
> desde la fuente oficial. Lea `data/raw/ml-latest-small/README.txt` después de descargarlo.

## Requisitos
[uv](https://docs.astral.sh/uv/) ≥ 0.5, que instala Python 3.12, e internet la primera vez (≈ 1 MB).
CPU común; la ejecución completa tarda unos minutos.

## Ejecución (uv, recomendada)
```bash
uv python install 3.12
uv sync
uv run pytest -q                       # 28 pruebas
uv run python scripts/run_all.py       # descarga -> auditoría -> LAB08 -> experimento principal
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/E05_recomendador.ipynb
uv run streamlit run app/streamlit_app.py   # demostración opcional
```
Pasos individuales: `scripts/download_data.py`, `audit_data.py`, `lab08_content.py`,
`lab09_hybrid.py [--epochs 20] [--factors 10 20 40]`. Variables opcionales en `.env.example`.

## Ejecución alternativa (pip)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
$env:PYTHONPATH="src;."             # macOS/Linux: export PYTHONPATH=src:.
python -m pytest -q
python scripts/run_all.py
```

## Estructura
```
app/streamlit_app.py      demostración: usuario nuevo y películas similares
data/raw/                 MovieLens descargado (no versionado)
docs/                     DATASET_CARD, SYSTEM_CARD y declaración de uso de IA
notebooks/                notebook ejecutado
reports/                  métricas (CSV/JSON) y figuras
scripts/                  descarga, auditoría, LAB08, LAB09 (experimento principal), run_all
src/inf8239_u03/          config, data, metrics, recommenders, evaluation
tests/                    pruebas de datos, métricas, modelos y protocolo
```

## Protocolo
| Elemento | Decisión | Razón |
|---|---|---|
| Partición | Corte temporal global en el cuantil 0.80 de las fechas (22-03-2016) | El *leave-one-out* por usuario del laboratorio deja en entrenamiento valoraciones de otros usuarios posteriores a la prueba |
| Segmentos | Cálidos (con historial), fríos k=0 (sin historial), fríos k=5 (5 valoraciones de bienvenida) | Medir el arranque en frío de forma explícita |
| Relevancia | Valoración ≥ 4 sobre películas del catálogo no vistas | Recomendación top-10 de lo que gustará |
| Catálogo | Películas valoradas antes del corte | No usar información futura |
| Ajuste | Segundo corte temporal dentro del entrenamiento | La prueba se usa una sola vez |
| Decisión | Frontera de Pareto NDCG@10–cobertura y regla de un error estándar | Ante empate estadístico, preferir la lista más diversa en catálogo |

## Resultados en prueba (NDCG@10 [IC 95 %])
| Modelo | Cálidos (n = 28) | Fríos k=5 (n = 88) | Fríos k=0 (n = 88) | Cobertura* |
|---|---|---|---|---|
| Aleatorio | 0.010 [0.000, 0.021] | 0.019 [0.008, 0.031] | 0.019 [0.008, 0.031] | 0.036 |
| Popularidad | 0.086 [0.021, 0.164] | 0.269 [0.215, 0.320] | 0.379 [0.325, 0.432] | 0.008 |
| Contenido | 0.019 [0.003, 0.041] | 0.045 [0.027, 0.063] | — | 0.075 |
| Factorización | 0.037 [0.006, 0.081] | 0.038 [0.023, 0.056] | — | 0.010 |
| Híbrido LAB09 (0.75 FM + 0.25 contenido) | 0.039 [0.017, 0.067] | 0.206 [0.158, 0.252] | — | 0.031 |
| **Híbrido seleccionado (0.25 pop. + 0.50 FM + 0.25 cont.)** | **0.090 [0.043, 0.151]** | **0.332 [0.275, 0.387]** | fallback | **0.023** |
| **Sistema final** (híbrido + fallback de popularidad) | 0.090 | 0.332 | 0.379 | 0.023 |

\*Cobertura = proporción del catálogo de 7 867 películas que aparece en alguna lista.
RMSE en usuarios cálidos: factorización 0.922, media de la película 1.059, media global 1.122.
Detalle en `reports/test_metrics.csv`, `rmse.csv` y el notebook.

## Git
```bash
git init
git add pyproject.toml uv.lock requirements.txt .gitignore .python-version .env.example
git commit -m "chore: configure reproducible recommender environment"
git add src/inf8239_u03/config.py src/inf8239_u03/data.py scripts/download_data.py scripts/audit_data.py docs/DATASET_CARD.md tests/conftest.py tests/test_data.py
git commit -m "feat: official MovieLens download, data contract and audit"
git add src scripts tests
git commit -m "feat: temporal protocol, baselines, MF with fold-in and hybrid"
git add reports notebooks app
git commit -m "docs: executed notebook, reports and demo app"
git add README.md docs/SYSTEM_CARD.md docs/AI_USE_DECLARATION.md
git commit -m "docs: README, system card and AI use statement"
git tag u03-ejercicio05
```

## Referencias
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), Artículo 19. https://doi.org/10.1145/2827872
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2.ª ed.). Springer.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer, 42*(8), 30–37. https://doi.org/10.1109/MC.2009.263
