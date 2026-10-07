# Declaración de uso de inteligencia artificial

## Herramientas utilizadas
- **Claude (Anthropic)**, interfaz web claude.ai, dentro de un proyecto de la asignatura. Octubre de 2026.

## Propósito de cada uso
- Revisar el proyecto base de LAB08 y LAB09 y detectar debilidades frente a la rúbrica.
- Generar código: corte temporal global, protocolo de usuarios cálidos y fríos, factorización con sesgos y fold-in, híbrido de tres señales, métricas de ranking, cobertura, novedad y diversidad, selección por frontera de Pareto, figuras y pruebas.
- Redactar README, Dataset Card, System Card, notebook y el informe.

## Prompts o decisiones relevantes
1. Se compartieron los archivos del proyecto base con el enunciado y la rúbrica del Ejercicio 05; luego: *"listo"* para iniciar.
2. Decisiones tomadas durante el desarrollo y revisadas por la estudiante:
   - Reemplazar el *leave-one-out* por usuario por un corte temporal global, porque el primero deja en entrenamiento valoraciones posteriores a la prueba.
   - Evaluar por separado usuarios cálidos, fríos sin historial y fríos con 5 valoraciones de bienvenida.
   - Añadir la popularidad como tercera señal del híbrido, porque en validación ningún modelo personalizado la superaba solo.
   - Elegir pesos y factores con un segundo corte temporal dentro del entrenamiento y la regla de un error estándar sobre la frontera NDCG@10–cobertura.

## Elementos verificados por el equipo
> Marque solo lo que haya comprobado personalmente.

- [ ] Descargué MovieLens desde GroupLens, leí su `README.txt` (licencia) y los conteos coinciden con la versión documentada.
- [ ] Ejecuté `uv run pytest -q` y `uv run python scripts/run_all.py` y obtuve las métricas del README.
- [ ] Revisé que ningún modelo usa información posterior al corte (popularidad, catálogo y entrenamiento).
- [ ] Revisé las listas de `reports/example_recommendations.csv` y la interpretación de errores.
- [ ] Verifiqué las referencias bibliográficas.
- [ ] Puedo explicar cada archivo de `src/` y `scripts/`.

## Cambios realizados sobre resultados generados
| Problema detectado | Corrección |
|---|---|
| El entorno de la IA no podía acceder a files.grouplens.org; verificó el código con una copia local de ml-latest-small con los mismos conteos documentados, que no se incluyó en el repositorio. | El código descarga siempre desde GroupLens; la estudiante debe ejecutar el flujo y confirmar los resultados. |
| La regla de un error estándar elegía el híbrido de solo contenido porque la tabla de validación tenía índices repetidos. | Se reinició el índice de la tabla y se añadió una prueba unitaria de la regla. |
| Con fold-in poco regularizado, el RMSE de usuarios nuevos era peor que la media de la película. | Se ajustó la regularización del fold-in en validación (λ = 10). |
| `top_n` evaluaba película por película en Python (muy lento). | Se vectorizó el cálculo de puntajes sobre todo el catálogo. |
| La figura de sesgo de popularidad con histogramas logarítmicos era engañosa. | Se reemplazó por diagramas de caja y una tabla con la proporción de recomendaciones en el top-100. |

## Responsabilidad asumida
La estudiante responde por el código, los resultados, las referencias y las conclusiones, y declara poder explicar todo lo entregado.
