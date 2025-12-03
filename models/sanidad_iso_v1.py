# models/ranking_iso_v1.py
# Uso desde consola (desde la raíz del repo):
#   python3 models/ranking_iso_v1.py

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

from util.load_dataframes import (
    load_dataframe_vacas,
    load_dataframe_ranking,
)
from util.generate_data import fill_missing_data
from util.generate_summary import generate_summary, save_dataframe
from util.storage import save_csv, save_model  # usamos logs para salidas

CSV_REGISTROS = "data/registros_sesiones_merged.csv"
CSV_RANKING = "data/ranking_vacas_df_final.csv"

RESULTS_DIR = "results/meritoProductivo/"
DF_RESUMEN_PATH = "data/resumen_vacas.csv"
DF_FINAL_ANOMALIAS_PATH = "data/df_final_con_anomalias.csv"

# Resultados tipo v2.0 pero para mérito productivo
RESULTS_FOLDS_CSV = os.path.join(RESULTS_DIR, "iso_merito_kfold_anomaly_rates_v1.csv")
RESULTS_STABILITY_CSV = os.path.join(RESULTS_DIR, "iso_merito_stability_summary_v1.csv")
PLOT_STABILITY_FOLDS = os.path.join(RESULTS_DIR, "iso_merito_stability_folds_v1.png")
PLOT_SCORE_HIST = os.path.join(RESULTS_DIR, "iso_merito_score_histogram_v1.png")
PLOT_PCA_CLUSTERS = os.path.join(RESULTS_DIR, "iso_merito_pca_clusters_v1.png")
PLOT_SHAP = os.path.join(RESULTS_DIR, "iso_merito_shap_v1.png")

MODEL_PATH = "trained_models/meritoProductivo/iso_merito_pipeline_v1.joblib"


def build_df_final():
    """Replica exactamente las transformaciones del notebook para construir df_final."""
    print(f"Leyendo registros de sesiones: {CSV_REGISTROS}")
    df = load_dataframe_vacas(CSV_REGISTROS)

    print(f"Leyendo ranking de vacas: {CSV_RANKING}")
    df_ranking = load_dataframe_ranking(CSV_RANKING)[["ID Vaca", "Puntaje_final"]]

    # Renombrar columnas del ranking a MultiIndex
    df_ranking.columns = pd.MultiIndex.from_tuples(
        [
            ("Ranking", "ID Vaca"),
            ("Ranking", "PuntajeFinal"),
        ]
    )

    # Llenar datos faltantes
    print("Rellenando datos faltantes...")
    df_filled = fill_missing_data(df)

    # Generar resumen por vaca
    print("Generando resumen por vaca...")
    df_summary = generate_summary(df_filled)

    # Guardar resumen (comportamiento original del notebook)
    save_dataframe(df_summary, DF_RESUMEN_PATH)
    # Además guardamos vía sistema de logs
    save_csv(
        df_summary,
        DF_RESUMEN_PATH,
        resource_type="data",
        purpose="resumen_vacas_merito",
        script_name="ranking_iso_v1.py",
    )

    # Construir df_final
    df_ranking_indexed = df_ranking.set_index(("Ranking", "ID Vaca"))
    df_final = (
        df_summary.set_index(("ID", "ID Vaca"))
        .join(df_ranking_indexed, how="left")
        .reset_index()
    )

    print("Shape df_final:", df_final.shape)
    return df_final


def plot_pca_clusters(X_scaled, labels, output_path):
    """Métrica Extra: Visualiza la separación en espacio 2D (similar a sanidad_iso_v2)."""
    print("Generando Visualización PCA (mérito productivo)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    label_text = np.where(labels == 1, "Normal", "Anomalía")
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=label_text,
        palette={"Normal": "gray", "Anomalía": "red"},
        alpha=0.6,
    )
    plt.title("Proyección PCA - Isolation Forest (Mérito Productivo)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfica PCA guardada en: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # ==========================
    # 1) Construir df_final
    # ==========================
    df_final = build_df_final()

    # ==========================
    # 2) Preparar datos para IsolationForest
    # ==========================
    # Excluir ID Vaca y PuntajeFinal de las features
    exclude_cols = [("ID", "ID Vaca"), ("Ranking", "PuntajeFinal")]
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]

    X = df_final[feature_cols].copy()
    y_ranking = df_final[("Ranking", "PuntajeFinal")].copy()

    # Rellenar NaN con 0 (igual que notebook)
    X = X.fillna(0)
    y_ranking = y_ranking.fillna(0)

    print("Shape X (features mérito):", X.shape)
    print("Descripción PuntajeFinal:")
    print(y_ranking.describe(), "\n")

    # ==========================
    # 3) Definir pipeline IsolationForest
    # ==========================
    iso_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "iso",
                IsolationForest(
                    n_estimators=100,
                    contamination=0.1,  # igual que el notebook original de mérito
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ------------------------
    # "K-Fold" no supervisado (estabilidad de anomalías)
    # ------------------------
    k_splits = 6
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    anomaly_rates = []

    print(f"\nEjecutando {k_splits}-fold 'CV' para ver estabilidad de anomalías (mérito)...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        iso_pipeline.fit(X_train)

        # Transform test
        X_test_imp = iso_pipeline.named_steps["imputer"].transform(X_test)
        X_test_scaled = iso_pipeline.named_steps["scaler"].transform(X_test_imp)
        iso_model = iso_pipeline.named_steps["iso"]

        labels = iso_model.predict(X_test_scaled)  # 1 normal, -1 anómalo
        anomaly_rate = (labels == -1).mean()
        anomaly_rates.append(anomaly_rate)

        print(f"Fold {fold}: porcentaje de vacas anómalas = {anomaly_rate:.3f}")

    mean_rate = np.mean(anomaly_rates)
    std_rate = np.std(anomaly_rates)
    print(
        "\nPromedio de porcentaje de anomalías entre folds: "
        f"{mean_rate:.3f} ± {std_rate:.3f}"
    )

    # ---------------------------------------------------
    # Guardar métricas por fold en CSV (log)
    # ---------------------------------------------------
    folds = np.arange(1, k_splits + 1)
    fold_metrics_pct = [r * 100 for r in anomaly_rates]
    max_diff = max(fold_metrics_pct) - min(fold_metrics_pct)

    meets_diff_5 = max_diff <= 5.0
    meets_diff_3 = max_diff <= 3.0

    df_folds = pd.DataFrame(
        {
            "fold": folds,
            "anomaly_rate": anomaly_rates,
            "anomaly_rate_pct": fold_metrics_pct,
            "mean_rate": mean_rate,
            "std_rate": std_rate,
        }
    )

    save_csv(
        df_folds,
        RESULTS_FOLDS_CSV,
        resource_type="data",
        purpose="iso_merito_kfold_anomaly_rates_v1",
        script_name="ranking_iso_v1.py",
    )

    # CSV de resumen de estabilidad
    df_summary = pd.DataFrame(
        [
            {
                "metric": "mean_anomaly_rate",
                "value": mean_rate * 100,
            },
            {
                "metric": "std_anomaly_rate",
                "value": std_rate * 100,
            },
            {
                "metric": "max_diff_between_folds",
                "value": max_diff,
            },
            {
                "metric": "meets_diff<=5pct",
                "value": meets_diff_5,
            },
            {
                "metric": "meets_diff<=3pct",
                "value": meets_diff_3,
            },
        ]
    )

    save_csv(
        df_summary,
        RESULTS_STABILITY_CSV,
        resource_type="data",
        purpose="iso_merito_kfold_stability_summary_v1",
        script_name="ranking_iso_v1.py",
    )

    print(
        f"[ISO MÉRITO] CSV de tasas por fold guardado en: {RESULTS_FOLDS_CSV}\n"
        f"[ISO MÉRITO] CSV de resumen de estabilidad guardado en: {RESULTS_STABILITY_CSV}"
    )

    # ---------------------------------------------------
    # Gráfica 1: porcentaje de anomalías por fold (barra)
    # ---------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(folds, fold_metrics_pct, alpha=0.8)
    plt.axhline(
        mean_rate * 100,
        color="red",
        linestyle="--",
        label=f"Media ({mean_rate*100:.2f}%)",
    )
    plt.title("Tasa de Anomalías por Pliegue (Verificación de Estabilidad - Mérito)")
    plt.xlabel("Número de Pliegue")
    plt.ylabel("Porcentaje de Anomalías Detectadas")
    plt.legend()
    plt.ylim(0, max(fold_metrics_pct) * 1.25)

    plt.savefig(PLOT_STABILITY_FOLDS, dpi=300)
    plt.close()
    print(f"Gráfica de porcentaje de anomalías guardada en: {PLOT_STABILITY_FOLDS}")

    # ------------------------
    # 3) Entrenar IsolationForest final con todos los datos
    # ------------------------
    print("\nEntrenando IsolationForest final con todos los datos de mérito...")
    iso_pipeline.fit(X)

    # Obtener scores y labels para TODO el dataset
    X_imp_all = iso_pipeline.named_steps["imputer"].transform(X)
    X_scaled_all = iso_pipeline.named_steps["scaler"].transform(X_imp_all)
    iso_model = iso_pipeline.named_steps["iso"]

    # Nota: usamos score_samples sin signo y aplicamos el signo aquí,
    # igual que sanidad_iso_v2 pero adaptado a mérito
    anomaly_scores = -iso_model.score_samples(X_scaled_all)  # mayor = más raro
    anomaly_labels = iso_model.predict(X_scaled_all)  # -1 = anómalo, 1 = normal

    # Añadir resultados a df_final con MultiIndex como antes
    df_results = df_final.copy()
    df_results[("IsolationForest", "Anomalia")] = anomaly_labels
    df_results[("IsolationForest", "Score")] = anomaly_scores

    print("\nEjemplo de filas con score y label (mérito):")
    print(
        df_results[
            [
                ("ID", "ID Vaca"),
                ("Ranking", "PuntajeFinal"),
                ("IsolationForest", "Score"),
                ("IsolationForest", "Anomalia"),
            ]
        ].head()
    )

    # Guardar df_final con anomalías (con logs)
    save_csv(
        df_results,
        DF_FINAL_ANOMALIAS_PATH,
        resource_type="data",
        purpose="df_final_con_anomalias_merito",
        script_name="ranking_iso_v1.py",
    )
    print(
        f"\n✓ Archivo df_final con anomalías (mérito) guardado en: {DF_FINAL_ANOMALIAS_PATH}"
    )

    # ---------------------------------------------------
    # Gráfica 2: histograma de anomaly_score
    # ---------------------------------------------------
    print("Generando Histograma de Puntuaciones de Anomalías (mérito)...")
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores, bins=50, kde=True)
    plt.title("Distribución de Puntuaciones de Anomalías (Modelo Final - Mérito)")
    plt.xlabel("Puntuación de Anomalía (Mayor = Más Raro)")
    plt.ylabel("Frecuencia")

    contamination = 0.1  # coherente con el modelo definido arriba
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Umbral Top {contamination*100:.0f}%",
    )
    plt.legend()

    plt.savefig(PLOT_SCORE_HIST, dpi=300)
    plt.close()
    print(f"Histograma de anomaly_score guardado en: {PLOT_SCORE_HIST}")

    # ---------------------------------------------------
    # Métrica Extra: Visualización PCA
    # ---------------------------------------------------
    plot_pca_clusters(X_scaled_all, anomaly_labels, PLOT_PCA_CLUSTERS)

    # ---------------------------------------------------
    # Métrica Extra: SHAP para explicabilidad
    # ---------------------------------------------------
    try:
        print("Generando gráfica SHAP (mérito)...")
        explainer = shap.TreeExplainer(iso_model)
        shap_values = explainer.shap_values(X_scaled_all)
        plt.figure()
        shap.summary_plot(shap_values, X_scaled_all, show=False)
        plt.savefig(PLOT_SHAP, bbox_inches="tight")
        plt.close()
        print(f"Gráfica SHAP guardada en: {PLOT_SHAP}")
    except Exception as e:
        print(f"No se pudo generar gráfica SHAP: {e}")

    # ---------------------------------------------------
    # 4) Guardar modelo final (local + S3 + log)
    # ---------------------------------------------------
    save_model(
        iso_pipeline,
        MODEL_PATH,
        resource_type="model",
        purpose="iso_merito_final_v1",
        script_name="ranking_iso_v1.py",
    )
    print(f"Pipeline de Isolation Forest (mérito) guardado en: {MODEL_PATH}")
    print("\nModelo IsolationForest de mérito entrenado y aplicado a todas las vacas.")


if __name__ == "__main__":
    main()