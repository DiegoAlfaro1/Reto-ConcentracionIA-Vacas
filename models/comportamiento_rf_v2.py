# comportamiento_rf_v2.py
# Uso desde consola:
# python3 models/comportamiento_rf_v2.py
# Random Forest

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate


CSV_BEHAVIOR = "../data/sessions_behavior.csv"
RESULTS_DIR = "../results"
MODELS_DIR = "../models"


def main():
    # Crear directorios de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Leyendo dataset de comportamiento: {CSV_BEHAVIOR}")
    df = pd.read_csv(CSV_BEHAVIOR)

    # Separar features y target
    X = df.drop(columns=["label_inquieta"])
    y = df["label_inquieta"]

    print("Shape X:", X.shape)
    print("Distribución de y:")
    print(y.value_counts(), "\n")

    # Pipeline: imputar -> escalar -> Random Forest
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
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

    print("Ejecutando 3-fold cross-validation...")
    cv_results = cross_validate(
        rf_pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    print("\n=== Resultados Random Forest (comportamiento) - 3 folds ===")
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

    # ---------------------------------------------------
    # Gráfica 1: barra con media y desviación estándar
    # ---------------------------------------------------
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest - 3-fold CV (métrica promedio ± std)")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_bar.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nGráfica de métricas promedio guardada en: {out_path}")

    # ---------------------------------------------------
    # Gráfica 2: métricas por fold (líneas)
    # ---------------------------------------------------
    folds = np.arange(1, 4)

    fig, ax = plt.subplots()
    for metric in metric_names:
        scores = per_fold_dict[metric]
        ax.plot(folds, scores, marker="o", label=metric)

    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Random Forest - 3-fold CV por métrica")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_per_fold.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Gráfica de métricas por fold guardada en: {out_path}")

    # ---------------------------------------------------
    # Entrenar modelo final con todos los datos
    # ---------------------------------------------------
    print("\nEntrenando modelo final con todos los datos...")
    rf_pipeline.fit(X, y)
    print("Modelo entrenado.")

    # Guardar el pipeline completo (imputer + scaler + RF)
    model_path = os.path.join(MODELS_DIR, "comportamiento_rf_pipeline.joblib")
    joblib.dump(rf_pipeline, model_path)
    print(f"Pipeline de Random Forest guardado en: {model_path}")


if __name__ == "__main__":
    main()
