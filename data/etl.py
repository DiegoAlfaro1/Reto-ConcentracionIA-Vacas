# etl_sesiones.py
# Uso desde consola:
# python3 data/etl.py

import unicodedata
import re

import numpy as np
import pandas as pd


CSV_INPUT = "datos/registros_sesiones_merged.csv"
CSV_BEHAVIOR = "sessions_behavior.csv"
CSV_HEALTH = "datos/sessions_health.csv"


def normalize_column(name: str) -> str:
    """
    Normaliza nombres de columnas:
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
    Convierte 'mm:ss' a segundos (float).
    Si está vacío o mal formado, regresa NaN.
    """
    if isinstance(value, str) and ":" in value:
        try:
            m, s = value.split(":")
            return int(m) * 60 + int(s)
        except Exception:
            return np.nan
    return np.nan


def main():
    print(f"Leyendo CSV de sesiones: {CSV_INPUT}")
    df = pd.read_csv(CSV_INPUT)

    # Normalizar nombres de columnas
    df = df.rename(columns={c: normalize_column(c) for c in df.columns})
    print("Columnas normalizadas:")
    print(df.columns.tolist())

    # Crear duración en segundos
    df["dur_seconds"] = df["main_duracion_mm_ss"].apply(parse_duration_mm_ss)

    # Flags de comportamiento
    patada_flag = df["estado_patada"].notna().astype(int)
    incompleto_flag = df["estado_incompleto"].notna().astype(int)
    pezones_flag = df["estado_pezones_no_encontrados"].notna().astype(int)

    # Umbral de duración (percentil 95)
    dur_threshold = df["dur_seconds"].quantile(0.95)
    dur_larga_flag = (df["dur_seconds"] > dur_threshold).astype(int)

    df["label_inquieta"] = (
        patada_flag
        + incompleto_flag
        + pezones_flag
        + dur_larga_flag
    ) > 0
    df["label_inquieta"] = df["label_inquieta"].astype(int)

    print("\nDistribución de label_inquieta (0 = tranquila, 1 = inquieta):")
    print(df["label_inquieta"].value_counts())

    # ------------------------------
    # Dataset de COMPORTAMIENTO (RF)
    # ------------------------------
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

    missing_behavior = [c for c in behavior_feature_cols if c not in df.columns]
    if missing_behavior:
        raise ValueError(f"Faltan columnas de comportamiento: {missing_behavior}")

    df_behavior = df[behavior_feature_cols + ["label_inquieta"]].copy()
    print(f"\nGuardando dataset de comportamiento en {CSV_BEHAVIOR}")
    print(df_behavior.head())
    df_behavior.to_csv(CSV_BEHAVIOR, index=False)

    # --------------------------
    # Dataset de SANIDAD (ISO)
    # --------------------------
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

    missing_health = [c for c in health_feature_cols if c not in df.columns]
    if missing_health:
        raise ValueError(f"Faltan columnas de sanidad: {missing_health}")

    df_health = df[health_feature_cols].copy()
    print(f"\nGuardando dataset de sanidad en {CSV_HEALTH}")
    print(df_health.head())
    df_health.to_csv(CSV_HEALTH, index=False)

    print("\nETL terminado correctamente.")


if __name__ == "__main__":
    main()
