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
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CSV_BEHAVIOR = "data/sessions_behavior.csv"
RESULTS_DIR = "results/"
MODELS_DIR = "models/trained_models/"


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
    # Tabla de métricas (por métrica y por fold)
    # ---------------------------------------------------
    n_folds = len(per_fold_dict[metric_names[0]])
    table_data = {"metric": metric_names}

    for fold_idx in range(n_folds):
        col_name = f"fold_{fold_idx + 1}"
        table_data[col_name] = [
            per_fold_dict[m][fold_idx] for m in metric_names
        ]

    table_data["mean"] = means
    table_data["std"] = stds

    df_metrics = pd.DataFrame(table_data)
    metrics_csv_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_table.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    print("\nTabla de métricas por fold guardada en:")
    print(metrics_csv_path)
    print(df_metrics, "\n")

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
    out_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_bar_v2.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Gráfica de métricas promedio guardada en: {out_path}")

    # ---------------------------------------------------
    # Gráfica 2: métricas por fold (líneas)
    # ---------------------------------------------------
    folds = np.arange(1, n_folds + 1)

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

    out_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_per_fold_v2.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Gráfica de métricas por fold guardada en: {out_path}")

    # ---------------------------------------------------
    # Matriz de confusión usando predicciones de CV
    # ---------------------------------------------------
    print("\nCalculando matriz de confusión con cross_val_predict...")
    y_pred_cv = cross_val_predict(
        rf_pipeline, X, y, cv=cv, n_jobs=-1
    )

    cm = confusion_matrix(y, y_pred_cv)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión - Random Forest (3-fold CV)")
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "rf_cv_confusion_matrix.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {out_path}")

    # ---------------------------------------------------
    # Entrenar modelo final con todos los datos
    # ---------------------------------------------------
    print("\nEntrenando modelo final con todos los datos...")
    rf_pipeline.fit(X, y)
    print("Modelo entrenado.")

    # Guardar el pipeline completo (imputer + scaler + RF)
    model_path = os.path.join(MODELS_DIR, "comportamiento_rf_pipeline_v2.joblib")
    joblib.dump(rf_pipeline, model_path)
    print(f"Pipeline de Random Forest guardado en: {model_path}")


if __name__ == "__main__":
    main()
