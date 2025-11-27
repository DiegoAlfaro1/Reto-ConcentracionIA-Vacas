# util/storage.py
import os
import io
from pathlib import Path

import joblib
import pandas as pd
import boto3

# Cargar .env desde la raíz del proyecto (solo para S3)
try:
    from dotenv import load_dotenv

    ROOT_DIR = Path(__file__).resolve().parent.parent  # sube de util/ a raíz
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    ROOT_DIR = Path(__file__).resolve().parent.parent

from .audit import log_event

# === Config desde variables de entorno ===
USE_S3 = os.getenv("USE_S3", "false").lower() in ("true", "1", "yes")
S3_BUCKET = os.getenv("S3_BUCKET", "")

S3_PREFIX_DATA = os.getenv("S3_PREFIX_DATA", "data/")
S3_PREFIX_RESULTS = os.getenv("S3_PREFIX_RESULTS", "results/")
S3_PREFIX_TRAINED_MODELS = os.getenv("S3_PREFIX_TRAINED_MODELS", "trained_models/")
S3_PREFIX_MODELS = os.getenv("S3_PREFIX_MODELS", "models/")

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
        rel = local[len("data/"):]
        prefix = S3_PREFIX_DATA
    elif local.startswith("results/"):
        rel = local[len("results/"):]
        prefix = S3_PREFIX_RESULTS
    elif local.startswith("trained_models/"):
        rel = local[len("trained_models/"):]
        prefix = S3_PREFIX_TRAINED_MODELS
    elif local.startswith("models/"):
        rel = local[len("models/"):]
        prefix = S3_PREFIX_MODELS
    else:
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
    Siempre loguea la lectura en logs/events.csv.
    """
    s3_key = _s3_key_for_local_path(local_path)
    source = "local"

    if USE_S3 and _s3_client and S3_BUCKET:
        try:
            obj = _s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            source = "s3"
        except Exception:
            df = pd.read_csv(local_path)
            source = "local"
    else:
        df = pd.read_csv(local_path)
        source = "local"

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
    dir_name = os.path.dirname(local_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df.to_csv(local_path, index=False)

    s3_key = _s3_key_for_local_path(local_path)

    if USE_S3 and _s3_client and S3_BUCKET:
        try:
            with open(local_path, "rb") as f:
                _s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=f.read())
        except Exception:
            # Si falla S3, no rompemos la ejecución: el CSV local ya quedó
            pass

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
    dir_name = os.path.dirname(local_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    joblib.dump(model, local_path)

    s3_key = _s3_key_for_local_path(local_path)
    if USE_S3 and _s3_client and S3_BUCKET:
        try:
            with open(local_path, "rb") as f:
                _s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=f.read())
        except Exception:
            pass

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
        except Exception:
            model = joblib.load(local_path)
            source = "local"
    else:
        model = joblib.load(local_path)
        source = "local"

    log_event(
        action="read",
        resource_type=resource_type,
        local_path=local_path if source == "local" else None,
        s3_key=s3_key if source == "s3" else None,
        extra={"purpose": purpose, "source": source},
        script_name=script_name,
    )

    return model
