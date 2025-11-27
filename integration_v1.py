# integration_v1.py
#
# Cálculo del Índice de Mérito para la Retención (IMR) para una vaca nueva
# a partir de su CSV de ordeños, usando:
# - Pipeline de Random Forest (comportamiento)
# - Pipeline de Isolation Forest (sanidad)
# - Mérito productivo precalculado
#
# Uso desde consola (desde la raíz del repo):
#   python3 integration_v1.py --csv data/input/1554.csv --cow-id 1554
#

import os
import sys
import argparse
import re
import unicodedata

import numpy as np
import pandas as pd

from data.etl_vaca_single import build_merged_from_single  # asegúrate de que data/ tenga __init__.py

# --- asegurar raíz del proyecto en sys.path ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# helpers de almacenamiento (S3 + logs)
from util.storage import load_model, load_csv

# ==========================
# 0) Rutas de modelos y datos
# ==========================

BEHAVIOR_MODEL_PATH = "trained_models/randomForest/comportamiento_rf_best_pipeline_v2.1.joblib"
HEALTH_MODEL_PATH   = "trained_models/isolationForest/iso_sanidad_pipeline_v2.1.joblib"
MERITO_CSV_PATH     = "data/merito_productivo/merito_productivo_vacas.csv"


# ==========================
# 1) Helpers para cargar modelos
# ==========================

def load_behavior_model(path: str):
    """
    Carga el pipeline de Random Forest de comportamiento guardado.
    Debe ser un Pipeline sklearn con:
        imputer -> scaler -> RandomForestClassifier
    """
    return load_model(
        path,
        resource_type="model",
        purpose="integration_behavior_model",
        script_name="integration_v1.py",
    )


def load_health_model(path: str):
    """
    Carga el pipeline de IsolationForest de sanidad guardado.
    Debe ser un Pipeline sklearn con:
        imputer -> scaler -> IsolationForest
    """
    return load_model(
        path,
        resource_type="model",
        purpose="integration_health_model",
        script_name="integration_v1.py",
    )


# ==========================
# 2) Helpers de ETL desde merged (una vaca)
# ==========================

def normalize_column(name: str) -> str:
    """
    MISMA función que en etl_sesiones.py:
    - quita BOM y acentos
    - pasa a minúsculas
    - reemplaza espacios, barras, etc. por '_'
    """
    name = name.replace("\ufeff", "")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^0-9a-z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def parse_duration_mm_ss(value):
    """
    MISMA lógica que en etl_sesiones.py.
    Convierte 'mm:ss' a segundos (float). Si está vacío o mal formado, regresa NaN.
    """
    if isinstance(value, str) and ":" in value:
        try:
            m, s = value.split(":")
            return int(m) * 60 + int(s)
        except Exception:
            return np.nan
    return np.nan


def build_behavior_features_from_merged(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Toma el DataFrame 'merged' de UNA vaca (estructura como registros_sesiones_merged),
    normaliza nombres de columnas igual que etl_sesiones.py y construye EXACTAMENTE
    las mismas columnas de features que se usaron en sessions_behavior.csv.
    """
    # 1) Normalizar nombres de columnas
    df = df_merged.rename(columns={c: normalize_column(c) for c in df_merged.columns})

    # 2) Asegurar dur_seconds (igual que en etl_sesiones.py)
    if "dur_seconds" not in df.columns:
        if "main_duracion_mm_ss" in df.columns:
            df["dur_seconds"] = df["main_duracion_mm_ss"].apply(parse_duration_mm_ss)
        else:
            df["dur_seconds"] = np.nan

    # 3) Misma lista de columnas que en etl_sesiones.py (behavior_feature_cols)
    behavior_feature_cols = [
        "dur_seconds",
        "main_produccion_kg",
        "estado_numero_de_ordeno",
        "media_de_los_flujos_kg_min_di",
        "media_de_los_flujos_kg_min_dd",
        "media_de_los_flujos_kg_min_ti",
        "media_de_los_flujos_kg_min_td",
        "flujos_maximos_kg_min_di",
        "flujos_maximos_kg_min_dd",
        "flujos_maximos_kg_min_ti",
        "flujos_maximos_kg_min_td",
        "producciones_kg_di",
        "producciones_kg_dd",
        "producciones_kg_ti",
        "producciones_kg_td",
    ]

    missing = [c for c in behavior_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de comportamiento en df_merged normalizado: {missing}")

    X_beh = df[behavior_feature_cols].copy()

    # Asegurar numérico + imputar medianas
    for c in X_beh.columns:
        X_beh[c] = pd.to_numeric(X_beh[c], errors="coerce")
        med = X_beh[c].median()
        X_beh[c] = X_beh[c].fillna(med)

    return X_beh


def build_health_features_from_merged(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Toma el DataFrame 'merged' de UNA vaca, normaliza columnas igual que etl_sesiones.py
    y construye EXACTAMENTE las mismas columnas que se usaron en sessions_health.csv.
    """
    df = df_merged.rename(columns={c: normalize_column(c) for c in df_merged.columns})

    health_feature_cols = [
        "sangre_ppm_di",
        "sangre_ppm_dd",
        "sangre_ppm_ti",
        "sangre_ppm_td",
        "conductividad_ms_cm_di",
        "conductividad_ms_cm_dd",
        "conductividad_ms_cm_ti",
        "conductividad_ms_cm_td",
        "media_de_los_flujos_kg_min_di",
        "media_de_los_flujos_kg_min_dd",
        "media_de_los_flujos_kg_min_ti",
        "media_de_los_flujos_kg_min_td",
        "flujos_maximos_kg_min_di",
        "flujos_maximos_kg_min_dd",
        "flujos_maximos_kg_min_ti",
        "flujos_maximos_kg_min_td",
        "producciones_kg_di",
        "producciones_kg_dd",
        "producciones_kg_ti",
        "producciones_kg_td",
    ]

    missing = [c for c in health_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de sanidad en df_merged normalizado: {missing}")

    X_health = df[health_feature_cols].copy()

    for c in X_health.columns:
        X_health[c] = pd.to_numeric(X_health[c], errors="coerce")
        med = X_health[c].median()
        X_health[c] = X_health[c].fillna(med)

    return X_health


# ==========================
# 3) Cargar mérito productivo de esa vaca
# ==========================

def load_merito_for_cow(cow_id: int, merito_csv_path: str) -> float:
    """
    Carga el mérito productivo precomputado para una vaca desde
    merito_productivo_vacas.csv, donde tienes columnas:
    id, merito_productivo, n_ordenos, ...
    """
    df_mer = load_csv(
        merito_csv_path,
        resource_type="data",
        purpose="integration_merito_productivo",
        script_name="integration_v1.py",
    )
    row = df_mer[df_mer["id"] == cow_id]

    if row.empty:
        raise ValueError(f"No se encontró mérito productivo para vaca id={cow_id}")

    return float(row["merito_productivo"].iloc[0])


# ==========================
# 4) Parámetros globales del índice (Z() y umbrales)
# ==========================

# Pesos del índice, según la documentación:
# IMR_i = wG·Z(MeritoProductivo_i) − wC·Z(RiesgoComport_i) − wS·Z(RiesgoSanidad_i)
W_G = 0.5   # peso del mérito productivo
W_C = 0.2   # peso del riesgo de comportamiento
W_S = 0.3   # peso del riesgo sanitario

# Estos valores deben obtenerse del dataset global:
# medias y desviaciones estándar para Z(x) = (x - mu) / sigma
MU_G, SIG_G = 0.0, 1.0   # TODO: media y sigma reales de MéritoProductivo
MU_C, SIG_C = 0.5, 0.1   # TODO: media y sigma reales de RiesgoComportamiento
MU_S, SIG_S = 0.5, 0.1   # TODO: media y sigma reales de RiesgoSanidad

# Umbrales p40 y p75 de la distribución global del IMR
P40_IMR = -0.2   # TODO: percentil 40 real
P75_IMR =  0.5   # TODO: percentil 75 real


def z_score(x: float, mu: float, sigma: float) -> float:
    """Z(x) = (x - mu) / sigma, con protección si sigma = 0."""
    if sigma <= 0:
        return 0.0
    return (x - mu) / sigma


def clasificar_imr(imr: float) -> str:
    """
    Clasifica el IMR según los umbrales globales:
      - IMR ≥ p75: Retener / Reproducir
      - p40 ≤ IMR < p75: Supervisar / Manejo dirigido
      - IMR < p40: Descartar
    """
    if imr >= P75_IMR:
        return "Retener / Reproducir"
    elif imr >= P40_IMR:
        return "Supervisar / Manejo dirigido"
    else:
        return "Descartar"


# ==========================
# 5) Función principal de integración
# ==========================

def compute_imr_for_cow(cow_csv_path: str, cow_id: int | None = None):
    # 1) ETL: De CSV crudo de la vaca -> merged-like (misma estructura que registros_sesiones_merged)
    df_merged = build_merged_from_single(cow_csv_path, cow_id=cow_id)
    cow_id_final = int(df_merged["id"].iloc[0])

    # 2) Construir features para cada modelo (ya normalizados)
    X_beh = build_behavior_features_from_merged(df_merged)
    X_health = build_health_features_from_merged(df_merged)

    # 3) Cargar pipelines sklearn (Random Forest + IsolationForest)
    beh_pipeline = load_behavior_model(BEHAVIOR_MODEL_PATH)
    iso_pipeline = load_health_model(HEALTH_MODEL_PATH)

    # 4) Predicciones de comportamiento
    prob_inquieta_sesion = beh_pipeline.predict_proba(X_beh)[:, 1]
    riesgo_comportamiento = float(prob_inquieta_sesion.mean())

    # 5) Predicciones de sanidad
    # Soporta ambos formatos de pipeline:
    # - v2.0: imputer -> scaler -> iso
    # - v2.1: preprocessor -> iso
    if "preprocessor" in iso_pipeline.named_steps:
        # v2.1
        X_trans = iso_pipeline.named_steps["preprocessor"].transform(X_health)
        iso_model = iso_pipeline.named_steps["iso"]
    else:
        # v2.0
        X_imp = iso_pipeline.named_steps["imputer"].transform(X_health)
        X_trans = iso_pipeline.named_steps["scaler"].transform(X_imp)
        iso_model = iso_pipeline.named_steps["iso"]

    anomaly_score_sesion = -iso_model.score_samples(X_trans)
    riesgo_sanidad = float(anomaly_score_sesion.mean())

    # 6) Cargar mérito productivo de esa vaca
    merito_prod = load_merito_for_cow(cow_id_final, MERITO_CSV_PATH)

    # 7) Calcular Z() para cada componente
    Z_G = z_score(merito_prod, MU_G, SIG_G)
    Z_C = z_score(riesgo_comportamiento, MU_C, SIG_C)
    Z_S = z_score(riesgo_sanidad, MU_S, SIG_S)

    # 8) IMR según la fórmula de la documentación:
    imr = W_G * Z_G - W_C * Z_C - W_S * Z_S

    # 9) Clasificación según p40/p75
    decision = clasificar_imr(imr)

    return {
        "id_vaca": cow_id_final,
        "merito_productivo": merito_prod,
        "riesgo_comportamiento": riesgo_comportamiento,
        "riesgo_sanidad": riesgo_sanidad,
        "Z_merito": Z_G,
        "Z_riesgo_comport": Z_C,
        "Z_riesgo_san": Z_S,
        "IMR": imr,
        "decision": decision,
    }

# ==========================
# 6) CLI para meter el CSV de una vaca
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Calcular IMR para una vaca a partir de su CSV de ordeños."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Ruta al CSV crudo de la vaca (ej. data/input/1554.csv)",
    )
    parser.add_argument(
        "--cow-id",
        type=int,
        default=None,
        help="ID de la vaca (opcional, si no se infiere del nombre del archivo).",
    )

    args = parser.parse_args()

    resultado = compute_imr_for_cow(args.csv, cow_id=args.cow_id)

    print("\n=== Resultado índice para vaca ===")
    for k, v in resultado.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()