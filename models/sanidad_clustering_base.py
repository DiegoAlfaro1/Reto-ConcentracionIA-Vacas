# sanidad_clustering_base.py
# Uso desde consola:
#   python models/sanidad_clustering_base.py

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Fix para evitar errores de threading con matplotlib en Windows
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from util.storage import load_csv, save_csv, save_model
plt.style.use("ggplot")

CSV_HEALTH = "data/sessions_health.csv"
RESULTS_DIR = "results/dbscan/"
MODELS_DIR = "trained_models/dbscan/"
RESULTS_SCORES_CSV = os.path.join(RESULTS_DIR, "dbscan_health_scores.csv")
RESULTS_FOLDS_CSV = os.path.join(RESULTS_DIR, "dbscan_kfold_anomaly_rates.csv")


def plot_pca_clusters(X_transformed, labels, output_dir):
    """Métrica Extra: Visualiza la separación en espacio 2D."""
    print("[DBSCAN] Generando visualización PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transformed)

    plt.figure(figsize=(10, 6))
    label_text = np.where(labels == -1, "Anomalía", "Normal")
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=label_text,
        palette={"Normal": "gray", "Anomalía": "red"},
        alpha=0.6,
    )
    plt.title("Proyección PCA (Modelo DBSCAN Final)")

    out_path = os.path.join(output_dir, "dbscan_pca_clusters.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[DBSCAN] Gráfica PCA guardada en: {out_path}")


def find_optimal_eps(X_transformed, min_samples):
    """
    Encuentra el valor óptimo de eps usando el método del codo (k-distance graph).
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X_transformed)
    distances, _ = neighbors.kneighbors(X_transformed)
    k_distances = np.sort(distances[:, -1])
    optimal_eps = np.percentile(k_distances, 90)
    return optimal_eps


def assign_labels_to_new_points(X_train, train_labels, X_new, eps):
    """
    Asigna etiquetas a nuevos puntos basándose en el clustering existente.
    """
    core_mask = train_labels != -1
    if not core_mask.any():
        return np.full(len(X_new), -1)

    X_core = X_train[core_mask]
    labels_core = train_labels[core_mask]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_core)
    distances, indices = nn.kneighbors(X_new)

    new_labels = np.where(
        distances.flatten() <= eps,
        labels_core[indices.flatten()],
        -1,
    )
    return new_labels


def main():
    # Crear directorios de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[DBSCAN] Leyendo dataset de sanidad: {CSV_HEALTH}")
    X = load_csv(
        CSV_HEALTH,
        resource_type="data",
        purpose="dbscan_health_input",
        script_name="sanidad_clustering_base.py",
    )

    print("[DBSCAN] Shape X_health:", X.shape)

    # Parámetros DBSCAN
    min_samples = 5

    # Pipeline: imputar -> escalar
    preprocessing_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocesar datos completos para eps y modelo final
    X_scaled = preprocessing_pipeline.fit_transform(X)

    # Calcular eps óptimo
    eps = find_optimal_eps(X_scaled, min_samples)
    print(f"[DBSCAN] Eps óptimo calculado: {eps:.4f}")

    # ------------------------
    # "K-Fold" no supervisado
    # ------------------------
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    anomaly_rates = []

    print("\n[DBSCAN] Ejecutando 6-fold 'CV' para ver estabilidad de anomalías...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        # Preprocesar por fold (evita data leakage)
        fold_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        X_train_scaled = fold_pipeline.fit_transform(X_train)
        X_test_scaled = fold_pipeline.transform(X_test)

        # Entrenar DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        dbscan.fit(X_train_scaled)

        # Asignar etiquetas a test
        labels = assign_labels_to_new_points(
            X_train_scaled,
            dbscan.labels_,
            X_test_scaled,
            eps,
        )
        anomaly_rate = (labels == -1).mean()
        anomaly_rates.append(anomaly_rate)

        print(
            f"[DBSCAN] Fold {fold}: porcentaje de sesiones anómalas = {anomaly_rate:.3f}"
        )

    mean_rate = np.mean(anomaly_rates)
    std_rate = np.std(anomaly_rates)
    print(
        "\n[DBSCAN] Promedio de porcentaje de anomalías entre folds: "
        f"{mean_rate:.3f} ± {std_rate:.3f}"
    )

    # Guardar métricas por fold en CSV (log)
    folds = np.arange(1, 7)
    df_folds = pd.DataFrame(
        {
            "fold": folds,
            "anomaly_rate": anomaly_rates,
            "anomaly_rate_pct": [r * 100 for r in anomaly_rates],
            "mean_rate": mean_rate,
            "std_rate": std_rate,
        }
    )
    save_csv(
        df_folds,
        RESULTS_FOLDS_CSV,
        resource_type="data",
        purpose="dbscan_kfold_anomaly_rates",
        script_name="sanidad_clustering_base.py",
    )

    # ---------------------------------------------------
    # Gráfica 1: porcentaje de anomalías por fold (barra)
    # ---------------------------------------------------
    fold_metrics = [rate * 100 for rate in anomaly_rates]

    plt.figure(figsize=(8, 5))
    plt.bar(folds, fold_metrics, color="steelblue", alpha=0.8)
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

    # ------------------------------------------
    # Guardar CSV con métricas comparables
    # ------------------------------------------

    max_diff = (max(anomaly_rates) - min(anomaly_rates)) * 100
    diff_le_5 = max_diff <= 5   # cumple criterio aceptable
    diff_le_3 = max_diff <= 3   # cumple criterio ideal

    report_data = {
        "metric": [
            "mean_anomaly_rate",
            "std_anomaly_rate",
            "max_diff_between_folds",
            "meets_diff<=5pct",
            "meets_diff<=3pct",
        ],
        "value": [
            mean_rate * 100,
            std_rate * 100,
            max_diff,
            diff_le_5,
            diff_le_3
        ]
    }

    df_report = pd.DataFrame(report_data)

    REPORT_PATH = os.path.join(RESULTS_DIR, "dbscan_health_report.csv")
    save_csv(
        df_report,
        REPORT_PATH,
        resource_type="data",
        purpose="dbscan_health_report",
        script_name="sanidad_clustering_base.py",
    )

    print(f"[DBSCAN] Reporte general guardado en: {REPORT_PATH}")
    print(df_report)

    out_path = os.path.join(RESULTS_DIR, "dbscan_stability_folds.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[DBSCAN] Gráfica de porcentaje de anomalías guardada en: {out_path}")

    # ------------------------
    # Entrenar DBSCAN final con todos los datos
    # ------------------------
    print("\n[DBSCAN] Entrenando DBSCAN final con todos los datos...")
    dbscan_final = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    cluster_labels = dbscan_final.fit_predict(X_scaled)

    # Calcular anomaly scores basados en distancia a vecinos
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    anomaly_scores = distances.mean(axis=1)

    # Convertir labels: DBSCAN usa -1 para ruido, queremos -1=anomalía, 1=normal
    anomaly_labels = np.where(cluster_labels == -1, -1, 1)

    df_results = X.copy()
    df_results["health_anomaly_score"] = anomaly_scores
    df_results["health_anomaly_label"] = anomaly_labels

    print("\n[DBSCAN] Ejemplo de filas con score y label:")
    print(
        df_results[
            [
                "health_anomaly_score",
                "health_anomaly_label",
            ]
        ].head()
    )

    # Guardar resultados de scores/labels en CSV (log)
    save_csv(
        df_results,
        RESULTS_SCORES_CSV,
        resource_type="data",
        purpose="dbscan_health_scores",
        script_name="sanidad_clustering_base.py",
    )
    print(f"[DBSCAN] Resultados de scores/labels guardados en: {RESULTS_SCORES_CSV}")

    # ---------------------------------------------------
    # Gráfica 2: histograma de anomaly_score
    # ---------------------------------------------------
    print("[DBSCAN] Generando histograma de puntuaciones de anomalía...")
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
        label=f"Umbral Top {contamination*100:.0f}%",
    )
    plt.legend()

    out_path = os.path.join(RESULTS_DIR, "dbscan_score_histogram.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[DBSCAN] Histograma de anomaly_score guardado en: {out_path}")

    # ---------------------------------------------------
    # Métrica Extra: Visualización PCA
    # ---------------------------------------------------
    plot_pca_clusters(X_scaled, cluster_labels, RESULTS_DIR)

    # Guardar el "modelo" (pipeline + meta) usando util.storage -> S3 + local
    model_data = {
        "preprocessing_pipeline": preprocessing_pipeline,
        "dbscan_params": {"eps": eps, "min_samples": min_samples},
        "training_data": X_scaled,
        "training_labels": cluster_labels,
    }

    model_path = os.path.join(MODELS_DIR, "dbscan_sanidad_pipeline_v1.joblib")

    joblib.dump(model_data, model_path)

    save_model(
        model_data,
        model_path,
        resource_type="model",
        purpose="dbscan_sanidad_pipeline",
        script_name="sanidad_clustering_base.py",
    )

    print(f"[DBSCAN] Pipeline de DBSCAN guardado en: {model_path} (local + S3)")

    print("\n[DBSCAN] Modelo DBSCAN entrenado y aplicado a todas las sesiones.")


if __name__ == "__main__":
    main()