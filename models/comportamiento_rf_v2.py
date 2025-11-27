# models/comportamiento_rf_v2.py
# Uso desde consola (desde la raíz del repo):
#   python3 models/comportamiento_rf_v2.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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
RESULTS_DIR = "results/randomForest/"
MODELS_DIR = "trained_models/randomForest/"


def main():
    # Crear directorios de resultados y modelos (por si acaso)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ==========================
    # 1) Cargar dataset (con logs)
    # ==========================
    print(f"Leyendo dataset de comportamiento: {CSV_BEHAVIOR}")
    df = load_csv(
        CSV_BEHAVIOR,
        resource_type="data",
        purpose="rf_comportamiento_train_v2",
        script_name="comportamiento_rf_v2.py",
    )

    # Separar features y target
    X = df.drop(columns=["label_inquieta"])
    y = df["label_inquieta"]

    print("Shape X:", X.shape)
    print("Distribución de y:")
    print(y.value_counts(), "\n")

    # ==========================
    # 2) Definir pipeline de Random Forest
    # ==========================
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

    # ==========================
    # 3) Cross-validation
    # ==========================
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

    # ==========================
    # 4) Tabla de métricas por fold (CSV)
    # ==========================
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
    metrics_csv_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_table_v2.csv")

    # 👉 guardar con save_csv para que vaya a S3 + log
    save_csv(
        df_metrics,
        metrics_csv_path,
        resource_type="results",
        purpose="rf_cv_metrics_v2",
        script_name="comportamiento_rf_v2.py",
    )

    print("\nTabla de métricas por fold guardada en:")
    print(metrics_csv_path)
    print(df_metrics, "\n")

    # ==========================
    # 5) Gráfica: barra media ± std
    # ==========================
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest - 3-fold CV (métrica promedio ± std)")
    fig.tight_layout()

    bar_png_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_bar_v2.png")
    fig.savefig(bar_png_path, dpi=300)
    plt.close(fig)
    print(f"Gráfica de métricas promedio guardada en: {bar_png_path}")
    # (si quieres también mandar PNGs a S3, luego hacemos un helper save_artifact)

    # ==========================
    # 6) Gráfica: métricas por fold
    # ==========================
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

    per_fold_png_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_per_fold_v2.png")
    fig.savefig(per_fold_png_path, dpi=300)
    plt.close(fig)
    print(f"Gráfica de métricas por fold guardada en: {per_fold_png_path}")

    # ==========================
    # 7) Matriz de confusión
    # ==========================
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

    cm_png_path = os.path.join(RESULTS_DIR, "rf_cv_confusion_matrix.png")
    fig.savefig(cm_png_path, dpi=300)
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {cm_png_path}")

    # ==========================
    # 8) Entrenar modelo final y guardar (S3 + log)
    # ==========================
    print("\nEntrenando modelo final con todos los datos...")
    rf_pipeline.fit(X, y)
    print("Modelo entrenado.")

    model_path = os.path.join(MODELS_DIR, "comportamiento_rf_pipeline_v2.joblib")

    save_model(
        rf_pipeline,
        model_path,
        resource_type="model",
        purpose="rf_comportamiento_final_v2",
        script_name="comportamiento_rf_v2.py",
    )

    print(f"Pipeline de Random Forest guardado en: {model_path}")


if __name__ == "__main__":
    main()