"""Configuración central del proyecto (variables de entorno con valores por defecto)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOVIELENS_URL = os.getenv(
    "MOVIELENS_URL", "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)
MOVIELENS_DIR = ROOT / os.getenv("MOVIELENS_DIR", "data/raw/ml-latest-small")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
REPORTS = ROOT / "reports"

# Valores publicados por GroupLens para ml-latest-small (README del 26-09-2018).
# Se usan para verificar que la descarga corresponde a la versión documentada.
EXPECTED_COUNTS = {"ratings": 100_836, "movies": 9_742, "users": 610}

# Protocolo de evaluación (declarado antes de entrenar)
TEST_QUANTILE = 0.80        # corte temporal global: el 20 % más reciente de las valoraciones es prueba
VALID_QUANTILE = 0.80       # dentro de entrenamiento, el 20 % más reciente se usa para ajustar
ONBOARDING_K = 5            # usuarios fríos: primeras 5 valoraciones como "encuesta de bienvenida"
RELEVANT_THRESHOLD = 4.0    # una película es relevante si el usuario la valoró con 4 o más
TOP_K = 10
