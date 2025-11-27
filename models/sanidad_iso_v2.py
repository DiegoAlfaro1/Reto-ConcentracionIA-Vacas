# sanidad_iso_v2.py
# Uso desde consola (desde la raíz del repo):
#   python3 models/sanidad_iso_v2.py
#

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

# Configurar estilo de gráficos
plt.style.use("ggplot")

# --- asegurar raíz del proyecto en sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- helpers de almacenamiento (S3 + logs) ---
from util.storage import load_csv, save_csv, save_model

CSV_HEALTH = "data/sessions_health.csv"
RESULTS_DIR = "results/isolationForest/"
MODELS_DIR = "trained_models/isolationForest/"


def plot_pca_clusters(X_transformed, labels, output_dir):
    """Métrica Extra: Visualiza la separación en espacio 2D."""
    print("Generando Visualización PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transformed)

    plt.figure(figsize=(10, 6))
    label_text = np.where(labels == 1, "Normal", "Anomalía")
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=label_text,
        palette={"Normal": "gray", "Anomalía": "red"},
        alpha=0.6,
    )
    plt.title("Proyección PCA (Modelo Final)")

    out_path = os.path.join(output_dir, "iso_2.0_pca_clusters.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfica PCA guardada en: {out_path}")


def main():
    # Crear directorios de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ==========================
    # 1) Cargar dataset (con logs / S3)
    # ==========================
    print(f"Leyendo dataset de sanidad: {CSV_HEALTH}")
    X = load_csv(
        CSV_HEALTH,
        resource_type="data",
        purpose="iso_sanidad_train_v2",
        script_name="sanidad_iso_v2.py",
    )

    print("Shape X_health:", X.shape)

    # ==========================
    # 2) Pipeline IsolationForest
    # ==========================
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
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    anomaly_rates = []

    print("\nEjecutando 6-fold 'CV' para ver estabilidad de anomalías...")
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
    folds = np.arange(1, 7)
    fold_metrics = [rate * 100 for rate in anomaly_rates]  # porcentaje

    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, fold_metrics, color="steelblue", alpha=0.8)
    plt.axhline(
        mean_rate * 100,
        color="red",
        linestyle="--",
        label=f"Media ({mean_rate*100:.2f}%)",
    )
    plt.title("Tasa de Anomalías por Pliegue (Verificación de Estabilidad)")
    plt.xlabel("Número de Pliegue")
    plt.ylabel("Porcentaje de Anomalías Detectadas")
    plt.legend()
    plt.ylim(0, max(fold_metrics) * 1.25)

    out_path = os.path.join(RESULTS_DIR, "iso_2.0_stability_folds.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfica de porcentaje de anomalías guardada en: {out_path}")

    # ------------------------
    # 3) Entrenar IsolationForest final con todos los datos
    # ------------------------
    print("\nEntrenando IsolationForest final con todos los datos...")
    iso_pipeline.fit(X)

    # Obtener scores y labels para TODO el dataset
    X_imp = iso_pipeline.named_steps["imputer"].transform(X)
    X_scaled = iso_pipeline.named_steps["scaler"].transform(X_imp)
    iso_model = iso_pipeline.named_steps["iso"]

    anomaly_scores = -iso_model.score_samples(X_scaled)  # mayor = más raro
    anomaly_labels = iso_model.predict(X_scaled)  # -1 = anómalo, 1 = normal

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
    print("Generando Histograma de Puntuaciones de Anomalías...")
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores, bins=50, kde=True, color="purple")
    plt.title("Distribución de Puntuaciones de Anomalías (Modelo Final)")
    plt.xlabel("Puntuación de Anomalía (Mayor = Más Raro)")
    plt.ylabel("Frecuencia")

    contamination = 0.05
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Umbral Top {contamination*100:.2f}%",
    )
    plt.legend()

    out_path = os.path.join(RESULTS_DIR, "iso_2.0_score_histogram.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Histograma de anomaly_score guardado en: {out_path}")

    # ---------------------------------------------------
    # Métrica Extra: Visualización PCA
    # ---------------------------------------------------
    plot_pca_clusters(X_scaled, anomaly_labels, RESULTS_DIR)

    # ---------------------------------------------------
    # Métrica Extra: SHAP para explicabilidad
    # ---------------------------------------------------
    try:
        print("Generando gráfica SHAP...")
        explainer = shap.TreeExplainer(iso_model)
        shap_values = explainer.shap_values(X_scaled)
        plt.figure()
        shap.summary_plot(shap_values, X_scaled, show=False)
        shap_path = os.path.join(RESULTS_DIR, "iso_2.0_shap.png")
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()
        print(f"Gráfica SHAP guardada en: {shap_path}")
    except Exception as e:
        print(f"No se pudo generar gráfica SHAP: {e}")

    # ---------------------------------------------------
    # 4) Guardar modelo final (local + S3 + log)
    # ---------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "iso_sanidad_pipeline_v2.joblib")
    save_model(
        iso_pipeline,
        model_path,
        resource_type="model",
        purpose="iso_sanidad_final_v2",
        script_name="sanidad_iso_v2.py",
    )
    print(f"Pipeline de Isolation Forest guardado en: {model_path}")
    print("\nModelo IsolationForest entrenado y aplicado a todas las sesiones.")


if __name__ == "__main__":
    main()
