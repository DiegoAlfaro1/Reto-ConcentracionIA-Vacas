# sanidad_clustering_base.py
# Baseline model using DBSCAN (Density-Based Spatial Clustering)
# This is a more archaic approach compared to Isolation Forest for anomaly detection
#
# Uso desde consola:
#   python models/sanidad_clustering_base.py
#
# DBSCAN detecta anomalías como puntos de "ruido" que no pertenecen a ningún cluster
# denso, lo cual es conceptualmente más simple que el enfoque de Isolation Forest.

import os
import numpy as np
import pandas as pd
import joblib

# Fix para evitar errores de threading con matplotlib en Windows
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

# Configurar estilo de gráficos
plt.style.use('ggplot')

CSV_HEALTH = "datos/sessions_health.csv"
RESULTS_DIR = "results/dbscan/"
MODELS_DIR = "trained_models/dbscan/"


def plot_pca_clusters(X_transformed, labels, output_dir):
    """Métrica Extra: Visualiza la separación en espacio 2D."""
    print("Generando Visualización PCA...")
    # Reducir dimensionalidad a 2 componentes principales
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transformed)
    
    plt.figure(figsize=(10, 6))
    # Convertir labels de DBSCAN (-1=anomalía, 0+=normal) a formato visual
    label_text = np.where(labels == -1, 'Anomalía', 'Normal')
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=label_text, palette={'Normal': 'gray', 'Anomalía': 'red'}, alpha=0.6)
    plt.title(f"Proyección PCA (Modelo Final)")
    
    out_path = os.path.join(output_dir, "dbscan_pca_clusters.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfica PCA guardada en: {out_path}")


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
        -1
    )
    return new_labels


def main():
    # Crear directorios de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Leyendo dataset de sanidad: {CSV_HEALTH}")
    X = pd.read_csv(CSV_HEALTH)

    print("Shape X_health:", X.shape)

    # Parámetros DBSCAN
    min_samples = 5
    
    # Pipeline: imputar -> escalar
    preprocessing_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocesar datos
    X_scaled = preprocessing_pipeline.fit_transform(X)
    
    # Calcular eps óptimo
    eps = find_optimal_eps(X_scaled, min_samples)
    print(f"Eps óptimo calculado: {eps:.4f}")

    # ------------------------
    # "K-Fold" no supervisado
    # ------------------------
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    anomaly_rates = []

    print("\nEjecutando 6-fold 'CV' para ver estabilidad de anomalías...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        # Preprocesar
        fold_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_train_scaled = fold_pipeline.fit_transform(X_train)
        X_test_scaled = fold_pipeline.transform(X_test)

        # Entrenar DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        dbscan.fit(X_train_scaled)

        # Asignar etiquetas a test
        labels = assign_labels_to_new_points(X_train_scaled, dbscan.labels_, X_test_scaled, eps)
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
    fold_metrics = [rate * 100 for rate in anomaly_rates]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, fold_metrics, color='steelblue', alpha=0.8)
    plt.axhline(mean_rate * 100, color='red', linestyle='--', label=f'Media ({mean_rate*100:.2f}%)')
    plt.title(f"Tasa de Anomalías por Pliegue (Verificación de Estabilidad)")
    plt.xlabel("Número de Pliegue")
    plt.ylabel("Porcentaje de Anomalías Detectadas")
    plt.legend()
    plt.ylim(0, max(fold_metrics) * 1.25)
    
    out_path = os.path.join(RESULTS_DIR, "dbscan_stability_folds.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfica de porcentaje de anomalías guardada en: {out_path}")

    # ------------------------
    # Entrenar DBSCAN final con todos los datos
    # ------------------------
    print("\nEntrenando DBSCAN final con todos los datos...")
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
    sns.histplot(anomaly_scores, bins=50, kde=True, color='purple')
    plt.title("Distribución de Puntuaciones de Anomalías (Modelo Final)")
    plt.xlabel("Puntuación de Anomalía (Mayor = Más Raro)")
    plt.ylabel("Frecuencia")
    
    # Agregar una línea para el umbral aproximado
    contamination = 0.05
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
    plt.axvline(threshold, color='red', linestyle='--', label=f'Umbral Top {contamination*100}%')
    plt.legend()
    
    out_path = os.path.join(RESULTS_DIR, "dbscan_score_histogram.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Histograma de anomaly_score guardado en: {out_path}")

    # ---------------------------------------------------
    # Métrica Extra: Visualización PCA
    # ---------------------------------------------------
    plot_pca_clusters(X_scaled, cluster_labels, RESULTS_DIR)

    # Guardar el modelo
    model_data = {
        'preprocessing_pipeline': preprocessing_pipeline,
        'dbscan_params': {'eps': eps, 'min_samples': min_samples},
        'training_data': X_scaled,
        'training_labels': cluster_labels
    }
    
    model_path = os.path.join(MODELS_DIR, "dbscan_sanidad_pipeline_v1.joblib")
    joblib.dump(model_data, model_path)
    print(f"Pipeline de DBSCAN guardado en: {model_path}")

    print("\nModelo DBSCAN entrenado y aplicado a todas las sesiones.")


if __name__ == "__main__":
    main()
