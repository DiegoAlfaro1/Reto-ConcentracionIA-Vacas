import json
import logging
import re
import io
import boto3
import pandas as pd

# Configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3')

def read_loose(file_content):
    """Lee CSV desde contenido en memoria con autodetección de separador y codificación."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                io.StringIO(file_content.decode(enc)),
                sep=None,
                engine="python",
                header=None,
                dtype=str
            )
        except Exception:
            continue
    # Último intento: coma fija
    return pd.read_csv(
        io.StringIO(file_content.decode("utf-8")),
        header=None,
        dtype=str
    )

def sanitize(s):
    """Devuelve texto limpio; si viene NaN/None/float vacío, regresa ''."""
    try:
        if s is None:
            return ""
        if isinstance(s, float):
            if pd.isna(s):
                return ""
        s = str(s)
        s = re.sub(r"\s+", " ", s.strip())
        return s
    except Exception:
        return ""

def ffill_row(values):
    """Forward-fill en la fila de grupos, tolerante a vacíos/NaN."""
    out, last = [], ""
    for v in values:
        v = sanitize(v)
        if v != "":
            last = v
        out.append(last)
    return out

def make_unique(names):
    """Genera nombres de columna únicos."""
    seen, out = {}, []
    for n in names:
        base = n if n else "col"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
    return out

def read_with_grouped_header(file_content):
    """
    Procesa CSV con header agrupado desde contenido en memoria.
    Fila 0: GRUPO (Main, Estado, Conductividad …)  [con forward-fill]
    Fila 1: SUBCOLUMNA (Hora de inicio, Acción, DI, DD, TI, TD, …)
    Nombres: '<Grupo> | <Sub>' (o '<Sub>' si no hay grupo)
    Devuelve data desde fila 2 hacia abajo, sin transformar valores.
    """
    df_raw = read_loose(file_content)
    if len(df_raw) < 2:
        return pd.DataFrame()

    groups = ffill_row(list(df_raw.iloc[0].values))
    subs = [sanitize(x) for x in list(df_raw.iloc[1].values)]

    cols = []
    for g, s in zip(groups, subs):
        if g and s:
            cols.append(f"{g} | {s}")
        elif s:
            cols.append(s)
        elif g:
            cols.append(g)
        else:
            cols.append("")

    cols = make_unique(cols)
    body = df_raw.iloc[2:].copy()
    body.columns = cols
    body = body.dropna(axis=1, how="all")
    return body

def get_csv_files_from_s3(bucket, prefix="raw/"):
    """Obtiene lista de archivos CSV desde S3."""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            return []
        
        csv_files = [
            obj['Key'] for obj in response['Contents']
            if obj['Key'].lower().endswith('.csv') and obj['Key'] != prefix
        ]
        
        return sorted(csv_files)
    
    except Exception as e:
        logger.error(f"Error listing S3 objects: {str(e)}")
        raise

def process_csv_from_s3(bucket, key):
    """
    Descarga y procesa un archivo CSV desde S3.
    Retorna DataFrame procesado o None si hay error.
    """
    try:
        # Descargar archivo desde S3
        response = s3.get_object(Bucket=bucket, Key=key)
        file_content = response['Body'].read()
        
        # Procesar con header agrupado
        df = read_with_grouped_header(file_content)
        
        if df.empty:
            logger.warning(f"⚠️ {key}: sin datos válidos (¿archivo vacío o solo encabezados?)")
            return None
        
        # Extraer ID de vaca del nombre del archivo
        filename = key.split('/')[-1]
        cow_id = re.sub(r"\D", "", filename.replace('.csv', '')) or None
        df.insert(0, "id", cow_id)
        
        logger.info(f"✓ {key}: {len(df)} filas, {len(df.columns)} columnas")
        return df
    
    except Exception as e:
        logger.error(f"Error processing {key}: {str(e)}")
        return None

def merge_csv_files_from_s3(bucket, input_prefix="raw/"):
    """
    Función principal que:
    1. Lista archivos CSV en S3
    2. Procesa cada uno
    3. Merge todos los DataFrames
    4. Retorna el DataFrame merged
    
    Returns:
        tuple: (merged_df, metadata_dict)
    """
    try:
        # Obtener lista de archivos CSV
        csv_files = get_csv_files_from_s3(bucket, input_prefix)
        
        if not csv_files:
            raise ValueError(f"No se encontraron archivos CSV en s3://{bucket}/{input_prefix}")
        
        logger.info(f"Encontrados {len(csv_files)} archivos CSV para procesar")
        
        # Procesar cada archivo
        frames = []
        for key in csv_files:
            df = process_csv_from_s3(bucket, key)
            if df is not None:
                frames.append(df)
        
        if not frames:
            raise ValueError("No se generaron datos válidos de ningún archivo")
        
        # Merge todos los DataFrames
        merged = pd.concat(frames, ignore_index=True, sort=False)
        logger.info(f"Merge completado: {len(merged)} filas totales, {len(merged.columns)} columnas")
        
        metadata = {
            'total_rows': len(merged),
            'total_columns': len(merged.columns),
            'files_processed': len(frames),
            'sample_columns': list(merged.columns)[:12]
        }
        
        return merged, metadata
    
    except Exception as e:
        logger.error(f"Error en merge_csv_files_from_s3: {str(e)}", exc_info=True)
        raise