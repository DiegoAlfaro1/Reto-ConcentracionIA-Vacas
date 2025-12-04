# models/ranking_rf_v1.py
# Uso desde consola (desde la raíz del repo):
#   python3 models/ranking_rf_v1.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# --- asegurar raíz del proyecto en sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from util.load_dataframes import (
    load_dataframe_vacas,
    load_dataframe_ranking,
)
from util.generate_data import fill_missing_data
from util.generate_summary import generate_summary, save_dataframe
from util.storage import save_csv, save_model

# Rutas de datos (ya en /data)
CSV_REGISTROS = "data/registros_sesiones_merged.csv"
CSV_RANKING = "data/ranking_vacas_df_final.csv"

RESULTS_DIR = "results/meritoProductivo/"
DF_RESUMEN_PATH = "data/resumen_vacas.csv"

# Artefactos estilo v2 (pero para métrica de clasificación)
RF_CV_METRICS_TABLE_PATH = os.path.join(
    RESULTS_DIR, "rf_merito_cv_metrics_table_v1.csv"
)
RF_CV_BAR_PNG = os.path.join(
    RESULTS_DIR, "rf_merito_cv_metrics_bar_v1.png"
)
RF_CV_PER_FOLD_PNG = os.path.join(
    RESULTS_DIR, "rf_merito_cv_metrics_per_fold_v1.png"
)

# Modelo final
RF_MODEL_PATH = "trained_models/meritoProductivo/random_forest_merito_model.joblib"


def build_df_final():
    """Mismas transformaciones del notebook para obtener df_final."""
    print(f"Leyendo registros de sesiones: {CSV_REGISTROS}")
    df = load_dataframe_vacas(CSV_REGISTROS)

    print(f"Leyendo ranking de vacas: {CSV_RANKING}")
    df_ranking = load_dataframe_ranking(CSV_RANKING)[["ID Vaca", "Puntaje_final"]]

    # MultiIndex en columnas del ranking
    df_ranking.columns = pd.MultiIndex.from_tuples(
        [
            ("Ranking", "ID Vaca"),
            ("Ranking", "PuntajeFinal"),
        ]
    )

    print("Rellenando datos faltantes...")
    df_filled = fill_missing_data(df)

    print("Generando resumen por vaca...")
    df_summary = generate_summary(df_filled)

    # Guardar resumen (comportamiento original)
    save_dataframe(df_summary, DF_RESUMEN_PATH)
    # Guardar también con logs
    save_csv(
        df_summary,
        DF_RESUMEN_PATH,
        resource_type="data",
        purpose="resumen_vacas_merito",
        script_name="ranking_rf_v1.py",
    )

    df_ranking_indexed = df_ranking.set_index(("Ranking", "ID Vaca"))
    df_final = (
        df_summary.set_index(("ID", "ID Vaca"))
        .join(df_ranking_indexed, how="left")
        .reset_index()
    )

    print("Shape df_final:", df_final.shape)
    return df_final


def main():
    # Crear directorios
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)

    # ==========================
    # 1) Construir df_final
    # ==========================
    df_final = build_df_final()

    # ==========================
    # 2) Preparar datos
    # ==========================
    exclude_cols = [("ID", "ID Vaca"), ("Ranking", "PuntajeFinal")]
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]

    X = df_final[feature_cols].copy()
    y_cont = df_final[("Ranking", "PuntajeFinal")].copy()

    X = X.fillna(0)
    y_cont = y_cont.fillna(0)

    print("Shape X:", X.shape)
    print("Descripción y (PuntajeFinal):")
    print(y_cont.describe(), "\n")

    # ==========================
    # 2.1 Definir etiqueta binaria para clasificación
    # ==========================
    # Definimos "bajo mérito" como el 10% con peor PuntajeFinal
    contamination = 0.10
    threshold = y_cont.quantile(contamination)
    y = (y_cont <= threshold).astype(int)  # 1 = bajo mérito, 0 = resto

    print(f"Umbral de bajo mérito (quantile {contamination:.2f}): {threshold:.4f}")
    print("Distribución de la etiqueta binaria (1 = bajo mérito):")
    print(y.value_counts(), "\n")

    # ==========================
    # 3) Definir modelo RF clasificador
    # ==========================
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Stratified K-Fold (3 folds, como comportamiento_rf_v2)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Métricas de clasificación
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    # ==========================
    # 4) Cross-validation
    # ==========================
    print("Ejecutando 3-fold cross-validation (Random Forest Classifier - mérito)...")
    cv_results = cross_validate(
        rf_model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    print("\n=== Resultados Random Forest (mérito productivo, clasificación) - 3 folds ===")
    metric_names = []
    per_fold_dict = {}
    means = []
    stds = []

    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        metric_names.append(metric)
        per_fold_dict[metric] = scores
        means.append(scores.mean())
        stds.append(scores.std())

        print(
            f"{metric:9s}: "
            f"mean={scores.mean():.4f}  std={scores.std():.4f}  "
            f"folds={np.round(scores, 4)}"
        )

    # ==========================
    # 5) Tabla de métricas por fold (CSV)
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

    save_csv(
        df_metrics,
        RF_CV_METRICS_TABLE_PATH,
        resource_type="results",
        purpose="rf_merito_cv_metrics_table_v1",
        script_name="ranking_rf_v1.py",
    )

    print("\nTabla de métricas por fold guardada en:")
    print(RF_CV_METRICS_TABLE_PATH)
    print(df_metrics, "\n")

    # ==========================
    # 6) Gráfica: barra media ± std (igual estilo que comportamiento_rf_v2)
    # ==========================
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest (mérito, clasificación)\n3-fold CV (métrica promedio ± std)")
    fig.tight_layout()

    fig.savefig(RF_CV_BAR_PNG, dpi=300)
    plt.close(fig)
    print(f"Gráfica de métricas promedio guardada en: {RF_CV_BAR_PNG}")

    # ==========================
    # 7) Gráfica: métricas por fold
    # ==========================
    folds = np.arange(1, n_folds + 1)

    fig, ax = plt.subplots()
    for metric in metric_names:
        scores = per_fold_dict[metric]
        ax.plot(folds, scores, marker="o", label=metric)

    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Random Forest (mérito, clasificación) - 3-fold CV por métrica")
    ax.legend()
    fig.tight_layout()

    fig.savefig(RF_CV_PER_FOLD_PNG, dpi=300)
    plt.close(fig)
    print(f"Gráfica de métricas por fold guardada en: {RF_CV_PER_FOLD_PNG}")

    # ==========================
    # 8) Entrenar modelo final y guardar (S3 + log)
    # ==========================
    print("\nEntrenando modelo final (clasificador) con todos los datos...")
    rf_model.fit(X, y)
    print("Modelo entrenado.")

    save_model(
        rf_model,
        RF_MODEL_PATH,
        resource_type="model",
        purpose="rf_merito_classifier_final_v1",
        script_name="ranking_rf_v1.py",
    )

    print(f"Modelo RandomForestClassifier (mérito) guardado en: {RF_MODEL_PATH}")


if __name__ == "__main__":
    main()
