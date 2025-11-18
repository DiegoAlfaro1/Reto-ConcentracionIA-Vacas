# merito_productivo.py

import os
import unicodedata
import re

import numpy as np
import pandas as pd

# Rutas de entrada/salida (ajusta si tu estructura es distinta)
CSV_INPUT = "../data/registros_sesiones_merged.csv"
CSV_SESIONES_OUT = "../data/merito_productivo/sessions_with_prod_ajustada.csv"
CSV_VACAS_OUT = "../data/merito_productivo/merito_productivo_vacas.csv"


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


def main():
    print(f"Leyendo sesiones desde: {CSV_INPUT}")
    df = pd.read_csv(CSV_INPUT)

    # Normalizar nombres de columnas
    df = df.rename(columns={c: normalize_column(c) for c in df.columns})
    print("Columnas disponibles después de normalizar:")
    print(df.columns.tolist())

    # --- 1) Identificadores importantes ---
    # En tu CSV, 'id' es el ID de la vaca (se repite para cada sesión)
    COW_ID_COL = "id"

    # Columna de producción observada de la sesión
    PROD_COL = "main_produccion_kg"

    # Columna de fecha/hora de inicio de la sesión
    DATETIME_COL = "main_hora_de_inicio"

    # Si en el futuro tienes robot, podrías usar algo como:
    # ROBOT_COL = "robot_id"
    # y agregarlo a group_cols
    ROBOT_COL = None  # por ahora no está presente en este archivo

    # --- 2) Parsear fecha/hora y extraer mes / hora ---
    print("\nParseando fecha/hora de inicio...")
    df[DATETIME_COL] = pd.to_datetime(
        df[DATETIME_COL],
        dayfirst=True,
        errors="coerce",
    )

    df["mes"] = df[DATETIME_COL].dt.month
    df["hora"] = df[DATETIME_COL].dt.hour

    # Eliminar filas sin producción o sin fecha/hora válida
    before = len(df)
    df = df.dropna(subset=[PROD_COL, "mes", "hora"])
    after = len(df)
    print(f"Filas válidas para mérito productivo: {after} (droppeadas {before - after})")

    # --- 3) Calcular Producción esperada por (robot x hora x mes) ---
    # Fórmula de referencia:
    # ProduccionAjustada_i = ProduccionObservada_i - E(Produccion | robot x hora x mes)
    # Aquí usamos (mes, hora) porque robot no está disponible explícitamente.

    if ROBOT_COL is not None and ROBOT_COL in df.columns:
        group_cols = [ROBOT_COL, "mes", "hora"]
        print(f"\nCalculando producción esperada condicional a {group_cols}...")
    else:
        group_cols = ["mes", "hora"]
        print(f"\nCalculando producción esperada condicional a {group_cols}...")

    df["prod_esperada"] = (
        df.groupby(group_cols)[PROD_COL]
        .transform("mean")
    )

    # Producción Ajustada por sesión
    df["produccion_ajustada"] = df[PROD_COL] - df["prod_esperada"]

    print("\nEjemplo de ProduccionObservada vs Esperada vs Ajustada:")
    print(
        df[
            [
                COW_ID_COL,
                DATETIME_COL,
                PROD_COL,
                "mes",
                "hora",
                "prod_esperada",
                "produccion_ajustada",
            ]
        ]
        .head()
    )

    # --- 4) Calcular Mérito Productivo por vaca ---
    # Fórmula:
    # MéritoProductivo_i = (1 / N_i) * sum_{j=1..N_i} ProduccionAjustada_{i,j}
    # donde:
    #   i = vaca
    #   j = sesión
    #   N_i = número de ordeños de esa vaca

    print("\nCalculando Mérito Productivo por vaca...")

    agg = (
        df.groupby(COW_ID_COL)
        .agg(
            merito_productivo=("produccion_ajustada", "mean"),   # 1/N * suma
            produccion_ajustada_total=("produccion_ajustada", "sum"),
            n_ordenos=("produccion_ajustada", "size"),
            produccion_media_observada=(PROD_COL, "mean"),
        )
        .reset_index()
    )

    print("\nEjemplo de tabla de Mérito Productivo por vaca:")
    print(agg.head())

    # --- 5) Guardar resultados ---
    out_dir = os.path.dirname(CSV_SESIONES_OUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(CSV_SESIONES_OUT, index=False)
    print(f"\nSesiones con ProduccionAjustada guardadas en: {CSV_SESIONES_OUT}")

    out_dir_v = os.path.dirname(CSV_VACAS_OUT)
    if out_dir_v:
        os.makedirs(out_dir_v, exist_ok=True)

    agg.to_csv(CSV_VACAS_OUT, index=False)
    print(f"Tabla de Mérito Productivo por vaca guardada en: {CSV_VACAS_OUT}")

    print("\nCálculo de Mérito Productivo terminado correctamente.")


if __name__ == "__main__":
    main()