# util/storage.py
import os
import io
import joblib
import pandas as pd
import boto3

from config import (
    USE_S3,
    S3_BUCKET,
    S3_PREFIX_DATA,
    S3_PREFIX_RESULTS,
    S3_PREFIX_TRAINED_MODELS,
    S3_PREFIX_MODELS,
)
from .audit import log_event

_s3_client = boto3.client("s3") if USE_S3 and S3_BUCKET else None


def _s3_key_for_local_path(local_path: str) -> str:
    """
    Mapea un path local a la key de S3 usando los prefijos:
    - data/...           -> S3_PREFIX_DATA/...
    - results/...        -> S3_PREFIX_RESULTS/...
    - trained_models/... -> S3_PREFIX_TRAINED_MODELS/...
    - models/...         -> S3_PREFIX_MODELS/...
    """
    local = local_path.lstrip("./")

    if local.startswith("data/"):
        rel = local[len("data/") :]
        prefix = S3_PREFIX_DATA
    elif local.startswith("results/"):
        rel = local[len("results/") :]
        prefix = S3_PREFIX_RESULTS
    elif local.startswith("trained_models/"):
        rel = local[len("trained_models/") :]
        prefix = S3_PREFIX_TRAINED_MODELS
    elif local.startswith("models/"):
        rel = local[len("models/") :]
        prefix = S3_PREFIX_MODELS
    else:
        # Genérico: deja el path tal cual
        rel = local
        prefix = ""

    if prefix:
        key = f"{prefix.rstrip('/')}/{rel.lstrip('/')}"
    else:
        key = rel

    return key.lstrip("/")


# ---------- CSV / DataFrames ----------

def load_csv(
    local_path: str,
    resource_type: str = "data",
    purpose: str | None = None,
    script_name: str | None = None,
) -> pd.DataFrame:
    """
    Lee un CSV. Si USE_S3=true intenta primero desde S3.
    Siempre loguea la lectura.
    """
    s3_key = _s3_key_for_local_path(local_path)
    source = "local"

    if USE_S3 and _s3_client and S3_BUCKET:
        try:
            obj = _s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            source = "s3"
        except Exception as exc:
            print(f"[storage] No se pudo leer de S3 ({s3_key}), usando local. Error: {exc}")
            df = pd.read_csv(local_path)
    else:
        df = pd.read_csv(local_path)

    log_event(
        action="read",
        resource_type=resource_type,
        local_path=local_path if source == "local" else None,
        s3_key=s3_key if source == "s3" else None,
        extra={"purpose": purpose, "source": source, "shape": df.shape},
        script_name=script_name,
    )

    return df


def save_csv(
    df: pd.DataFrame,
    local_path: str,
    resource_type: str = "data",
    purpose: str | None = None,
    script_name: str | None = None,
):
    """
    Guarda un DataFrame:
    - Siempre local.
    - Si USE_S3=true, también en S3.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False)

    s3_key = _s3_key_for_local_path(local_path)
    if USE_S3 and _s3_client and S3_BUCKET:
        with open(local_path, "rb") as f:
            _s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=f.read())

    log_event(
        action="write",
        resource_type=resource_type,
        local_path=local_path,
        s3_key=s3_key if USE_S3 else None,
        extra={"purpose": purpose, "shape": df.shape},
        script_name=script_name,
    )


# ---------- Modelos ----------

def save_model(
    model,
    local_path: str,
    resource_type: str = "model",
    purpose: str | None = None,
    script_name: str | None = None,
):
    """
    Guarda un modelo (joblib) local + (opcional) S3.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    joblib.dump(model, local_path)

    s3_key = _s3_key_for_local_path(local_path)
    if USE_S3 and _s3_client and S3_BUCKET:
        with open(local_path, "rb") as f:
            _s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=f.read())

    log_event(
        action="write",
        resource_type=resource_type,
        local_path=local_path,
        s3_key=s3_key if USE_S3 else None,
        extra={"purpose": purpose},
        script_name=script_name,
    )


def load_model(
    local_path: str,
    resource_type: str = "model",
    purpose: str | None = None,
    script_name: str | None = None,
):
    """
    Carga un modelo desde S3 si USE_S3=true, si no, desde local.
    """
    s3_key = _s3_key_for_local_path(local_path)
    source = "local"

    if USE_S3 and _s3_client and S3_BUCKET:
        try:
            obj = _s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            bio = io.BytesIO(obj["Body"].read())
            model = joblib.load(bio)
            source = "s3"
        except Exception as exc:
            print(f"[storage] No se pudo leer modelo de S3 ({s3_key}), usando local. Error: {exc}")
            model = joblib.load(local_path)
    else:
        model = joblib.load(local_path)

    log_event(
        action="read",
        resource_type=resource_type,
        local_path=local_path if source == "local" else None,
        s3_key=s3_key if source == "s3" else None,
        extra={"purpose": purpose, "source": source},
        script_name=script_name,
    )

    return model