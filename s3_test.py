# test_s3_head.py
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # lee .env

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX_PROCESSED_DATA = os.getenv("S3_PREFIX_PROCESSED_DAT", "processed/").lstrip("/")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Construye el URI S3 completo
S3_URI = f"s3://{S3_BUCKET}/{S3_PREFIX_PROCESSED_DATA}registros_sesiones_merged.csv"

print(f"Intentando leer: {S3_URI}")

def _flatten_columns(columns):
    # Aplana MultiIndex si viene con múltiples niveles
    flat = []
    for c in columns:
        if isinstance(c, tuple):
            flat.append("_".join([str(x) for x in c]).strip())
        else:
            flat.append(str(c))
    return flat

def read_multiheader_csv(uri: str) -> pd.DataFrame:
    """
    Lee el CSV asumiendo que puede venir con multi-header (2 niveles).
    Si no tiene multi-header, cae en lectura normal.
    """
    try:
        # Intenta multi-header (dos filas de encabezado)
        df = pd.read_csv(uri, header=[0], encoding='utf-8')
        # Si realmente venía de un solo nivel, pandas no crea MultiIndex y no pasa nada
    except Exception:
        # Si falla, prueba lectura normal
        df = pd.read_csv(uri, encoding='utf-8')

    # Aplana columnas si son MultiIndex
    try:
        df.columns = _flatten_columns(df.columns.values)
    except Exception:
        pass
    return df

# Importante: s3fs usa las variables de entorno AWS_* automáticamente
# También podrías pasar storage_options={'client_kwargs': {'region_name': AWS_REGION}}
df = read_multiheader_csv(S3_URI)


print("Shape:", df.shape)
print("\nHEAD (5 filas):")
print(df.head(5))