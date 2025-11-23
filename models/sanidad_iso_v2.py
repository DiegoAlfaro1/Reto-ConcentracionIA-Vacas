# sanidad_iso_v2.py
# Uso desde consola:
#   python3 models/sanidad_iso_v2.py
#

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib 

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold


CSV_HEALTH = "data/sessions_health.csv"
RESULTS_DIR = "results/"
MODELS_DIR = "models/trained_models/"


def main():
    # Crear directorios de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Leyendo dataset de sanidad: {CSV_HEALTH}")
    X = pd.read_csv(CSV_HEALTH)

    print("Shape X_health:", X.shape)

    # Pipeline: imputar -> escalar -> IsolationForest
    iso_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "iso",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.05,  # ajusta según lo que consideres normal
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ------------------------
    # "K-Fold" no supervisado
    # ------------------------
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    anomaly_rates = []

    print("\nEjecutando 3-fold 'CV' para ver estabilidad de anomalías...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        iso_pipeline.fit(X_train)

        # Transform test
        X_test_imputed = iso_pipeline.named_steps["imputer"].transform(X_test)
        X_test_scaled = iso_pipeline.named_steps["scaler"].transform(X_test_imputed)
        iso_model = iso_pipeline.named_steps["iso"]

        labels = iso_model.predict(X_test_scaled)  # 1 normal, -1 anómalo
        anomaly_rate = (labels == -1).mean()
        anomaly_rates.append(anomaly_rate)

        print(f"Fold {fold}: porcentaje de sesiones anómalas = {anomaly_rate:.3f}")

    mean_rate = np.mean(anomaly_rates)
    std_rate = np.std(anomaly_rates)
    print(
        "\nPromedio de porcentaje de anomalías entre folds: "
        f"{mean_rate:.3f} ± {std_rate:.3f}"
    )

    # ---------------------------------------------------
    # Gráfica 1: porcentaje de anomalías por fold (barra)
    # ---------------------------------------------------
    folds = np.arange(1, 4)

    fig, ax = plt.subplots()
    ax.bar(folds, anomaly_rates)
    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Porcentaje de sesiones anómalas")
    ax.set_title("Isolation Forest - porcentaje de anomalías por fold")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "iso_anomaly_rates_per_fold.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Gráfica de porcentaje de anomalías guardada en: {out_path}")

    # ------------------------
    # Entrenar IsolationForest final con todos los datos
    # ------------------------
    print("\nEntrenando IsolationForest final con todos los datos...")
    iso_pipeline.fit(X)

    # Obtener scores y labels para TODO el dataset
    X_imp = iso_pipeline.named_steps["imputer"].transform(X)
    X_scaled = iso_pipeline.named_steps["scaler"].transform(X_imp)
    iso_model = iso_pipeline.named_steps["iso"]

    anomaly_scores = -iso_model.score_samples(X_scaled)  # mayor = más raro
    anomaly_labels = iso_model.predict(X_scaled)         # -1 = anómalo, 1 = normal

    df_results = X.copy()
    df_results["health_anomaly_score"] = anomaly_scores
    df_results["health_anomaly_label"] = anomaly_labels

    print("\nEjemplo de filas con score y label:")
    print(
        df_results[
            [
                "health_anomaly_score",
                "health_anomaly_label",
            ]
        ].head()
    )

    # ---------------------------------------------------
    # Gráfica 2: histograma de anomaly_score
    # ---------------------------------------------------
    fig, ax = plt.subplots()
    ax.hist(anomaly_scores, bins=30)
    ax.set_xlabel("health_anomaly_score (mayor = más raro)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de anomaly_score del Isolation Forest")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "iso_anomaly_score_hist.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Histograma de anomaly_score guardado en: {out_path}")

    # Guardar el pipeline completo (imputer + scaler + IsolationForest)
    model_path = os.path.join(MODELS_DIR, "iso_sanidad_pipeline.joblib")
    joblib.dump(iso_pipeline, model_path)
    print(f"Pipeline de Isolation Forest guardado en: {model_path}")

    # Opcional: guardar dataset enriquecido con scores
    # df_results.to_csv("../data/sessions_health_with_iso.csv", index=False)
    print("\nModelo IsolationForest entrenado y aplicado a todas las sesiones.")


if __name__ == "__main__":
    main()