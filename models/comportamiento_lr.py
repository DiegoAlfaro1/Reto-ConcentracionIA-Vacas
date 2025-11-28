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
    metric_names = []
    means = []
    stds = []
    per_fold_dict = {}

    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        metric_names.append(metric)
        means.append(scores.mean())
        stds.append(scores.std())
        per_fold_dict[metric] = scores

        print(
            f"{metric:9s}: "
            f"mean={scores.mean():.3f}  std={scores.std():.3f}  "
            f"folds={np.round(scores, 3)}"
        )

    # ==========================
    # 4) Tabla de métricas por fold (CSV, estilo RF)
    # ==========================
    n_folds = len(per_fold_dict[metric_names[0]])
    table_data = {"metric": metric_names}

    for fold_idx in range(n_folds):
        col_name = f"fold_{fold_idx + 1}"
        table_data[col_name] = [per_fold_dict[m][fold_idx] for m in metric_names]

    table_data["mean"] = means
    table_data["std"] = stds

    df_metrics = pd.DataFrame(table_data)
    metrics_path = os.path.join(RESULTS_DIR, "lr_cv_metrics_table.csv")

    save_csv(
        df_metrics,
        metrics_path,
        resource_type="results",
        purpose="lr_comportamiento_cv_metrics_table",
        script_name=SCRIPT_NAME,
    )
    print("\nTabla de métricas por fold guardada en:")
    print(metrics_path)
    print(df_metrics, "\n")

    # ==========================
    # 5) Visualización de métricas (HISTOGRAMA + BARPLOT MEAN±STD)
    # ==========================
    import seaborn as sns

    # --- Histograma de distribución de métricas por fold ---
    for metric in metric_names:
        scores = per_fold_dict[metric]
        plt.figure(figsize=(7,5))
        sns.histplot(scores, kde=True, bins=8)
        plt.title(f"Distribución CV - {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frecuencia")
        hist_path = os.path.join(RESULTS_DIR, f"lr_hist_{metric}.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"Histograma guardado en: {hist_path}")


    # ==========================
    # 6) Entrenar modelo final y guardar (con logs)
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