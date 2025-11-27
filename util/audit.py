# util/audit.py
import os
import csv
import json
from datetime import datetime
from pathlib import Path

import boto3

# Cargar .env desde la raíz del proyecto (para S3)
try:
    from dotenv import load_dotenv

    ROOT_DIR = Path(__file__).resolve().parent.parent
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        ROOT_DIR = Path(__file__).resolve().parent.parent
except ImportError:
    ROOT_DIR = Path(__file__).resolve().parent.parent

# Config S3 para logs
USE_S3 = os.getenv("USE_S3", "false").lower() in ("true", "1", "yes")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX_LOGS = os.getenv("S3_PREFIX_LOGS", "logs/")

_s3_client = boto3.client("s3") if USE_S3 and S3_BUCKET else None

# Carpeta local de logs
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def _get_log_paths_for_today():
    """
    Devuelve (log_path_local, s3_key) para el día actual.
    Ejemplo local: logs/events_2025-11-26.csv
    Ejemplo S3:   logs/events_2025-11-26.csv (con prefijo S3_PREFIX_LOGS)
    """
    today_str = datetime.now().date().isoformat()  # YYYY-MM-DD
    filename = f"events_{today_str}.csv"
    local_path = LOGS_DIR / filename

    # Key en S3
    prefix = S3_PREFIX_LOGS.rstrip("/")
    s3_key = f"{prefix}/{filename}".lstrip("/")

    return local_path, s3_key


def _ensure_log_file_with_header(log_path: Path):
    """
    Crea el CSV con encabezado si no existe.
    """
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "date",         # YYYY-MM-DD
                "time",         # HH:MM:SS
                "action",       # read / write
                "resource_type",# data / model / results / etc
                "file",         # ruta local o key de S3 (lo que tengamos)
                "script",       # nombre del script
                "extra"         # JSON con info adicional (opcional)
            ])


def log_event(
    action: str,
    resource_type: str,
    local_path: str | None = None,
    s3_key: str | None = None,
    extra: dict | None = None,
    script_name: str | None = None,
):
    """
    Registra un evento en:
    - CSV local: logs/events_YYYY-MM-DD.csv
    - S3:        S3_PREFIX_LOGS/events_YYYY-MM-DD.csv  (si USE_S3=True)
    """
    now = datetime.now()
    date_str = now.date().isoformat()
    time_str = now.time().strftime("%H:%M:%S")

    # Archivo local del día
    log_path, s3_log_key = _get_log_paths_for_today()
    _ensure_log_file_with_header(log_path)

    # Prioridad para el campo "file": primero S3 si existe, si no local
    file_ref = s3_key or local_path or ""

    extra_str = json.dumps(extra, ensure_ascii=False) if extra else ""

    # 1) Escribir en CSV local
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            date_str,
            time_str,
            action,
            resource_type,
            file_ref,
            script_name or "",
            extra_str,
        ])

    # 2) Subir/actualizar el CSV en S3 (opcional)
    if USE_S3 and _s3_client and S3_BUCKET:
        try:
            with open(log_path, "rb") as f:
                _s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=s3_log_key,
                    Body=f.read(),
                )
        except Exception:
            pass