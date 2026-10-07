# System Card · Recomendador de películas (Ejercicio 05)

## Usuarios y propósito
Demostración académica de un recomendador top-10 de películas para usuarios de MovieLens: usuarios con historial y usuarios nuevos. El objetivo es comparar enfoques y documentar sus límites, no desplegar un servicio. Usuarios previstos: estudiantes y docentes de INF-8239.

## Catálogo y candidatos
- **Catálogo:** películas con al menos una valoración anterior al corte temporal (22-03-2016): 7 867 títulos según `reports/split_summary.json`. Las películas posteriores al corte no se recomiendan (15 % de las valoraciones de prueba).
- **Candidatos:** todo el catálogo menos lo que el usuario ya valoró. No hay filtro previo por popularidad.

## Señales utilizadas
| Señal | Origen | Uso |
|---|---|---|
| Valoraciones explícitas (0.5–5) | ratings.csv anterior al corte | Factorización, popularidad, perfil de contenido |
| Número de valoraciones por película | ratings.csv anterior al corte | Popularidad (log del conteo) |
| Géneros y década de estreno | movies.csv | Contenido (TF-IDF y coseno) |
| Encuesta de bienvenida (5 valoraciones) | Usuarios nuevos | Fold-in de la factorización y perfil de contenido |

No se usan datos demográficos, etiquetas libres ni texto.

## Modelos y fallback
- **Popularidad:** películas más valoradas antes del corte.
- **Contenido:** perfil del usuario = promedio TF-IDF de géneros y década de lo que valoró, ponderado por max(valoración − 2.5, 0.1).
- **Factorización:** r̂ = μ + b_u + b_i + p_u·q_i, SGD (20 épocas, tasa 0.01, regularización 0.05). Factores y regularización del fold-in elegidos en validación. Usuarios nuevos: fold-in por regresión ridge con los factores de las películas fijos.
- **Híbrido:** suma ponderada de las tres señales normalizadas a [0, 1]. Pesos elegidos en validación.
- **Fallback:** si el usuario no tiene ninguna valoración, se usa popularidad (lista en `reports/cold_start_fallback.csv`).

## Métricas offline
Protocolo: corte temporal global (cuantil 0.80), validación con un segundo corte dentro del entrenamiento, relevante = valoración ≥ 4, top-10, intervalos bootstrap por usuario. Resultados completos en `reports/test_metrics.csv` y en el informe.

## Cold start, cobertura y diversidad
- **Usuario sin historial:** popularidad.
- **Usuario con 5 valoraciones:** híbrido con fold-in.
- **Película nueva:** no entra al catálogo hasta acumular valoraciones. El contenido podría puntuarla por géneros, pero no se evaluó.
- **Cobertura:** baja (≈ 2 % del catálogo en prueba), aunque unas tres veces la de la popularidad.
- **Diversidad intralista:** intermedia; la novedad es baja porque el sistema favorece películas populares.

## Riesgos y monitoreo
- **Sesgo de popularidad:** refuerza lo ya popular y reduce la exposición de la cola larga. Monitorear cobertura, novedad y proporción de recomendaciones en el top-100 de popularidad.
- **Burbuja de filtro:** el contenido solo usa géneros y tiende a repetir el mismo tipo de película.
- **Incertidumbre:** con 28 usuarios cálidos en prueba, los intervalos de NDCG son muy amplios; no deben presentarse como mejoras seguras.
- **Deriva temporal:** el catálogo y los gustos cambian; reentrenar y reevaluar con nuevos cortes temporales.
- **Privacidad:** los historiales de visualización son sensibles; no combinar con datos que permitan reidentificar personas.
- **Comunicación:** presentar las listas como sugerencias, no como verdades ni juicios sobre la persona.
