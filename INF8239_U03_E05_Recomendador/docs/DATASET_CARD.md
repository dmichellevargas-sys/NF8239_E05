# Dataset Card · MovieLens ml-latest-small

## Fuente, fecha, versión y hash
- **Fuente oficial:** GroupLens Research, https://files.grouplens.org/datasets/movielens/ml-latest-small.zip (página: https://grouplens.org/datasets/movielens/latest/).
- **Versión:** "ml-latest-small", generada el 26 de septiembre de 2018 según el `README.txt` incluido en el ZIP. GroupLens actualiza los conjuntos *latest* sin numerarlos, por eso `scripts/download_data.py` comprueba los conteos documentados (100 836 valoraciones, 9 742 películas, 610 usuarios).
- **Hashes:** `reports/data_manifest.json` registra el SHA-256 del ZIP y de cada CSV descargado, con la fecha de verificación.

## Licencia y cita
Uso según los términos del `README.txt` de GroupLens (revíselos en el archivo descargado): uso para investigación y docencia; **no se permite redistribuir** los datos sin permiso separado; no se permite uso comercial sin permiso; no se puede afirmar respaldo de la Universidad de Minnesota o de GroupLens; se debe citar:

Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), Artículo 19. https://doi.org/10.1145/2827872

En consecuencia, el repositorio **no incluye** los CSV (`data/raw/` está en `.gitignore`); solo contiene código, métricas agregadas y listas de títulos derivadas.

## Archivos y diccionario

| Archivo | Columna | Tipo | Descripción |
|---|---|---|---|
| ratings.csv | userId | entero | Usuario anonimizado (sin datos demográficos) |
| | movieId | entero | Película (coincide con movies.csv) |
| | rating | decimal | 0.5 a 5.0 en pasos de 0.5 |
| | timestamp | entero | Segundos desde 1970-01-01 UTC |
| movies.csv | movieId | entero | Identificador |
| | title | texto | Título con año entre paréntesis |
| | genres | texto | Géneros separados por `|`; "(no genres listed)" si no tiene |
| tags.csv, links.csv | — | — | Descargados pero no usados en este proyecto |

## Procedimiento de descarga
`uv run python scripts/download_data.py` descarga el ZIP oficial, rechaza rutas inseguras al extraer, guarda los CSV en `data/raw/ml-latest-small/` y escribe el manifiesto. Si los conteos no coinciden con la versión documentada, avisa.

## Población observada y exclusiones
610 usuarios seleccionados al azar por GroupLens entre quienes valoraron al menos 20 películas, de marzo de 1996 a septiembre de 2018. No hay datos demográficos. Solo aparecen películas con al menos una valoración o etiqueta. En el experimento:
- Entrenamiento: valoraciones anteriores al corte temporal (22-03-2016, cuantil 0.80 de las fechas).
- Catálogo: películas con al menos una valoración antes del corte; el 15 % de las valoraciones de prueba recae en películas posteriores al corte y no puede recomendarse.

## Calidad, sesgos y usos prohibidos
- Sin nulos ni pares usuario-película duplicados; todas las valoraciones referencian una película existente.
- Densidad ≈ 1.7 %; cola larga: el 10 % de películas más valoradas concentra el 60 % de las valoraciones y el 35 % tiene una sola valoración.
- 34 películas sin géneros y 13 sin año.
- **Sesgos:** usuarios de una sola plataforma, mayoritariamente angloparlantes y cinéfilos; los nuevos usuarios valoran películas populares durante el registro; la actividad es muy desigual (de 20 a 2 698 valoraciones por usuario).
- **Usos prohibidos:** redistribuir los datos, intentar reidentificar usuarios, uso comercial sin permiso.
