# etl_vaca_single.py

import os
import sys
import pandas as pd

# --- asegurar raíz del proyecto en sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# helpers de almacenamiento (S3 + logs)
from util.storage import load_csv


REG_COLUMNS = [
    "id",
    "﻿Main | Hora de inicio",
    "﻿Main | Acción",
    "﻿Main | Duración (mm:ss)",
    "﻿Main | Producción (kg)",
    "Estado | Número de ordeño",
    "Estado | Patada",
    "Estado | Incompleto",
    "Estado | Pezones no encontrados",
    "Estado AMD | Ubre",
    "Estado AMD | Pezón",
    "Media de los flujos (kg/min) | DI",
    "Media de los flujos (kg/min) | DD",
    "Media de los flujos (kg/min) | TI",
    "Media de los flujos (kg/min) | TD",
    "Sangre (ppm) | DI",
    "Sangre (ppm) | DD",
    "Sangre (ppm) | TI",
    "Sangre (ppm) | TD",
    "Conductividad (mS / cm) | DI",
    "Conductividad (mS / cm) | DD",
    "Conductividad (mS / cm) | TI",
    "Conductividad (mS / cm) | TD",
    "Misc | EO/PO",
    "Misc | Destino Leche",
    "Flujos máximos (kg/min) | DI",
    "Flujos máximos (kg/min) | DD",
    "Flujos máximos (kg/min) | TI",
    "Flujos máximos (kg/min) | TD",
    "Producciones (kg) | DI",
    "Producciones (kg) | DD",
    "Producciones (kg) | TI",
    "Producciones (kg) | TD",
    "Misc | Razón de la desviación",
]


def build_merged_from_single(raw_path: str, cow_id: int | None = None) -> pd.DataFrame:
    """
    Toma un CSV tipo 6178.csv (una sola vaca, formato DeLaval crudo)
    y lo convierte a un DataFrame con las MISMAS columnas que
    registros_sesiones_merged.csv.
    """
    # Antes: df = pd.read_csv(raw_path)
    # Ahora: usar load_csv para soportar S3 + logs
    df = load_csv(
        raw_path,
        resource_type="data",
        purpose="etl_vaca_single_input",
        script_name="etl_vaca_single.py",
    )

    # Fila 0 = nombres "bonitos" en español (no los usamos directamente aquí)
    header = df.iloc[0]
    df_data = df.iloc[1:].reset_index(drop=True).copy()

    # Mapear columnas base
    new_cols = {}
    # Índices de columnas según 6178.csv
    new_cols[0] = "Main | Hora de inicio"
    new_cols[1] = "Main | Acción"
    new_cols[2] = "Main | Duración (mm:ss)"
    new_cols[3] = "Main | Producción (kg)"
    new_cols[4] = "Estado | Número de ordeño"
    new_cols[6] = "Estado | Patada"
    new_cols[7] = "Estado | Incompleto"
    new_cols[8] = "Estado | Pezones no encontrados"
    new_cols[9] = "Estado AMD | Ubre"
    new_cols[10] = "Estado AMD | Pezón"

    # Bloque: Media de los flujos (kg/min) + DI/DD/TI/TD
    prefix = df.columns[11]  # "Media de los flujos (kg/min)"
    for offset, quarter in enumerate(["DI", "DD", "TI", "TD"]):
        new_cols[11 + offset] = f"{prefix} | {quarter}"

    # Bloque: Sangre (ppm) + DI/DD/TI/TD
    prefix = df.columns[15]  # "Sangre (ppm)"
    for offset, quarter in enumerate(["DI", "DD", "TI", "TD"]):
        new_cols[15 + offset] = f"{prefix} | {quarter}"

    # Bloque: Conductividad (mS / cm) + DI/DD/TI/TD
    prefix = df.columns[19]  # "Conductividad (mS / cm)"
    for offset, quarter in enumerate(["DI", "DD", "TI", "TD"]):
        new_cols[19 + offset] = f"{prefix} | {quarter}"

    # Misc: EO/PO, Destino Leche, Razón de la desviación
    new_cols[23] = "Misc | EO/PO"
    new_cols[25] = "Misc | Destino Leche"
    new_cols[26] = "Misc | Razón de la desviación"

    # Bloque: Flujos máximos (kg/min) + DI/DD/TI/TD
    prefix = df.columns[27]  # "Flujos máximos (kg/min)"
    for offset, quarter in enumerate(["DI", "DD", "TI", "TD"]):
        new_cols[27 + offset] = f"{prefix} | {quarter}"

    # Bloque: Producciones (kg) + DI/DD/TI/TD
    prefix = df.columns[31]  # "Producciones (kg)"
    for offset, quarter in enumerate(["DI", "DD", "TI", "TD"]):
        new_cols[31 + offset] = f"{prefix} | {quarter}"

    # Renombrar columnas
    rename_map = {df.columns[idx]: name for idx, name in new_cols.items()}
    df_data = df_data.rename(columns=rename_map)

    # Añadir id de vaca
    if cow_id is None:
        base = os.path.basename(raw_path)
        cow_id = int(os.path.splitext(base)[0])  # 6178.csv -> 6178
    df_data.insert(0, "id", cow_id)

    # Ajustar nombres con BOM (para que coincidan exactamente con registros_sesiones_merged)
    bom_map = {
        "Main | Hora de inicio": "﻿Main | Hora de inicio",
        "Main | Acción": "﻿Main | Acción",
        "Main | Duración (mm:ss)": "﻿Main | Duración (mm:ss)",
        "Main | Producción (kg)": "﻿Main | Producción (kg)",
    }
    df_data = df_data.rename(columns=bom_map)

    # Dejar solo las columnas que existen en registros_sesiones_merged.csv
    keep_cols = [c for c in df_data.columns if c in REG_COLUMNS]
    df_data = df_data[keep_cols]

    # Asegurar orden de columnas igual a REG_COLUMNS
    df_data = df_data[[c for c in REG_COLUMNS if c in df_data.columns]]

    return df_data
