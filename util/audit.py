# util/audit.py
import json
import uuid
from datetime import datetime, timezone
import os

import boto3

from config import (
    USE_S3,
    S3_BUCKET,
    S3_PREFIX_LOGS,
    ENV,
    PROJECT_NAME,
)

_s3_client = boto3.client("s3") if USE_S3 and S3_BUCKET else None


def _build_log_key() -> str:
    """
    Genera la key de S3 para el log:
    [S3_PREFIX_LOGS]/YYYY/MM/DD/event_<uuid>.json
    """
    now = datetime.now(timezone.utc)
    date_prefix = now.strftime("%Y/%m/%d")
    event_id = str(uuid.uuid4())

    base = f"{S3_PREFIX_LOGS.rstrip('/')}/{date_prefix}/event_{event_id}.json"
    return base.lstrip("/")  # por si alguien pone "/" al inicio


def log_event(
    action: str,
    resource_type: str,
    local_path: str | None = None,
    s3_key: str | None = None,
    extra: dict | None = None,
    script_name: str | None = None,
):
    """
    Crea un evento de auditoría.
    action: "read", "write", "train", "predict", etc.
    resource_type: "data", "model", "result", "config", etc.
    """
    now = datetime.now(timezone.utc).isoformat()
    event = {
        "timestamp": now,
        "env": ENV,
        "project": PROJECT_NAME,
        "action": action,
        "resource_type": resource_type,
        "local_path": local_path,
        "s3_key": s3_key,
        "script": script_name or os.getenv("PYTHON_SCRIPT") or "unknown",
        "run_id": os.getenv("RUN_ID"),
        "user": os.getenv("USER") or os.getenv("USERNAME"),
    }

    if extra:
        event["extra"] = extra

    body = json.dumps(event, ensure_ascii=False).encode("utf-8")

    # Si S3 está habilitado, intentamos subir
    if USE_S3 and _s3_client and S3_BUCKET:
        key = _build_log_key()
        try:
            _s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=body)
            return
        except Exception as exc:
            # Si falla, cae a log local
            print(f"[audit] Error subiendo log a S3: {exc}")

    # Fallback: log local
    os.makedirs("logs", exist_ok=True)
    fname = f"logs/audit_fallback_{datetime.now().strftime('%Y%m%d')}.log"
    with open(fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
