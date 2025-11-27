# comportamiento_lr.py
# Regresión Logística para predicción de comportamiento (inquieta/tranquila)
# Modelo de línea base para comparación con modelos más complejos
# Uso desde consola (desde la raíz del repo):
#   python3 models/comportamiento_lr.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- asegurar raíz del proyecto en sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- helpers de almacenamiento (S3 + logs) ---
from util.storage import load_csv, save_csv, save_model

CSV_BEHAVIOR = "data/sessions_behavior.csv"
RESULTS_DIR = "results/logisticRegression/"
MODELS_DIR = "trained_models/logisticRegression/"

SCRIPT_NAME = "comportamiento_lr.py"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ==========================
    # 1) Cargar dataset (con logs)
    # ==========================
    print(f"Leyendo dataset de comportamiento: {CSV_BEHAVIOR}")
    df = load_csv(
        CSV_BEHAVIOR,
        resource_type="data",
        purpose="lr_comportamiento_train_baseline",
        script_name=SCRIPT_NAME,
    )

    # Separar features y target
    X = df.drop(columns=["label_inquieta"])
    y = df["label_inquieta"]

    print("Shape X:", X.shape)
    print("Distribución de y:")
    print(y.value_counts(), "\n")

    # ==========================
    # 2) Definir pipeline de Regresión Logística
    # ==========================
    lr_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # K-Fold (3 folds estratificado)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    # ==========================
    # 3) Cross-validation
    # ==========================
    print("Ejecutando 3-fold cross-validation...")
    cv_results = cross_validate(lr_pipeline, X, y, cv=cv, scoring=scoring)

    print("\n=== Resultados Regresión Logística (comportamiento) - 3 folds ===")
    metrics_data = {"metric": [], "mean": [], "std": []}
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        metrics_data["metric"].append(metric)
        metrics_data["mean"].append(scores.mean())
        metrics_data["std"].append(scores.std())
        print(f"{metric:9s}: mean={scores.mean():.3f}  std={scores.std():.3f}  folds={np.round(scores, 3)}")

    # Guardar métricas en CSV (con logs)
    df_metrics = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(RESULTS_DIR, "lr_cv_metrics.csv")
    save_csv(
        df_metrics,
        metrics_path,
        resource_type="results",
        purpose="lr_comportamiento_cv_metrics",
        script_name=SCRIPT_NAME,
    )
    print(f"\nMétricas guardadas en: {metrics_path}")

    # ==========================
    # 4) Matriz de confusión
    # ==========================
    y_pred_cv = cross_val_predict(lr_pipeline, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred_cv)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión - Regresión Logística (3-fold CV)")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "lr_cv_confusion_matrix.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {out_path}")

    # ==========================
    # 5) Entrenar modelo final y guardar (con logs)
    # ==========================
    print("\nEntrenando modelo final con todos los datos...")
    lr_pipeline.fit(X, y)

    model_path = os.path.join(MODELS_DIR, "comportamiento_lr_pipeline.joblib")
    save_model(
        lr_pipeline,
        model_path,
        resource_type="model",
        purpose="lr_comportamiento_baseline_final",
        script_name=SCRIPT_NAME,
    )
    print(f"Pipeline guardado en: {model_path}")


if __name__ == "__main__":
    main()
