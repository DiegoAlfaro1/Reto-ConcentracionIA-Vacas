# config.py
import os

from dotenv import load_dotenv

# Cargar .env (si usas python-dotenv)
load_dotenv()

USE_S3 = os.getenv("USE_S3", "false").lower() == "true"

S3_BUCKET = os.getenv("S3_BUCKET", "")

S3_PREFIX_DATA = os.getenv("S3_PREFIX_DATA", "data")
S3_PREFIX_RESULTS = os.getenv("S3_PREFIX_RESULTS", "results")
S3_PREFIX_TRAINED_MODELS = os.getenv("S3_PREFIX_TRAINED_MODELS", "trained_models")
S3_PREFIX_MODELS = os.getenv("S3_PREFIX_MODELS", "models")
S3_PREFIX_LOGS = os.getenv("S3_PREFIX_LOGS", "logs")

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Extras opcionales
ENV = os.getenv("ENV", "local")
PROJECT_NAME = os.getenv("PROJECT_NAME", "caetec-ia")
