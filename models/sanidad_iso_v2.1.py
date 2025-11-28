# sanidad_iso_v2.1.py
# Uso desde consola (desde la raíz del repo):
#   python3 models/sanidad_iso_v2.1.py
#

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Fix para evitar errores de threading con matplotlib en Windows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# --- asegurar raíz del proyecto en sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# helpers de almacenamiento (S3 + logs)
from util.storage import load_csv, save_model, save_csv

# Configurar estilo de gráficos
plt.style.use('ggplot')


def parse_args():
    """Configuración de argumentos de línea de comandos para el script"""
    parser = argparse.ArgumentParser(
        description="Isolation Forest: Verificación de Estabilidad + Modelo Final de Producción"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sessions_health.csv",
        help="Ruta al archivo CSV de entrada",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/isolationForest/",
        help="Directorio para imágenes de salida",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="trained_models/isolationForest/",
        help="Directorio para guardar modelos",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Proporción esperada de valores atípicos",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="Número de árboles",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Número de pliegues para verificación de estabilidad",
    )
    return parser.parse_args()


def preprocess_data(df):
    """
    Detecta automáticamente columnas numéricas y categóricas.
    Devuelve un pipeline de preprocesamiento y el dataframe limpio.
    """
    # Excluir columnas de ID si existen (heurística)
    cols_to_drop = [c for c in df.columns if "id" in c.lower() or "date" in c.lower()]
    if cols_to_drop:
        print(f"Eliminando posibles columnas identificadoras: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Identificar tipos de datos
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return df, preprocessor


def evaluate_stability(df, preprocessor, args):
    """
    ARTEFACTO REQUERIDO #1: Gráfico de Anomalías por Pliegue.
    Ejecuta Validación Cruzada K-Fold para verificar si la tasa de anomalías es estable.
    Además:
      - Guarda iso_2.1_kfold_anomaly_rates.csv
      - Guarda iso_2.1_stability_summary.csv
    """
    print(f"\n--- Fase 1: Evaluación de Estabilidad ({args.k_folds}-Fold CV) ---")

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    fold_metrics = []
    fold_indices = []

    for fold_i, (train_index, test_index) in enumerate(kf.split(df)):
        X_train = df.iloc[train_index]
        X_test = df.iloc[test_index]

        iso_fold = IsolationForest(
            n_estimators=args.n_estimators,
            contamination=args.contamination,
            n_jobs=-1,
            random_state=42,
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("iso", iso_fold)])

        pipeline.fit(X_train)

        test_preds = pipeline.predict(X_test)  # -1 anomalía, 1 normal

        n_anomalies = (test_preds == -1).sum()
        n_total = len(test_preds)
        pct_anomalies = (n_anomalies / n_total) * 100

        print(
            f"Pliegue {fold_i+1}: Detectadas {n_anomalies}/{n_total} anomalías "
            f"({pct_anomalies:.2f}%)"
        )

        fold_metrics.append(pct_anomalies)
        fold_indices.append(fold_i + 1)

    mean_rate = np.mean(fold_metrics)
    std_rate = np.std(fold_metrics)
    max_diff = max(fold_metrics) - min(fold_metrics)
    meets_diff_5 = max_diff <= 5.0
    meets_diff_3 = max_diff <= 3.0

    # --- ARTEFACTO #1: gráfica de barras por fold ---
    plt.figure(figsize=(8, 5))
    plt.bar(fold_indices, fold_metrics, color="steelblue", alpha=0.8)
    plt.axhline(
        mean_rate, color="red", linestyle="--", label=f"Media ({mean_rate:.2f}%)"
    )
    plt.title("Tasa de Anomalías por Pliegue (Verificación de Estabilidad)")
    plt.xlabel("Número de Pliegue")
    plt.ylabel("Porcentaje de Anomalías Detectadas")
    plt.legend()
    plt.ylim(0, max(fold_metrics) * 1.25)

    out_path = os.path.join(args.results_dir, "iso_2.1_stability_folds.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Artefacto #1 (Gráfico de Estabilidad) guardado en {out_path}")

    # ==========================
    # Guardar métricas por fold en CSV (log)
    # ==========================
    folds_arr = np.arange(1, args.k_folds + 1)

    df_folds = pd.DataFrame(
        {
            "fold": folds_arr,
            "anomaly_rate_pct": fold_metrics,
            "mean_rate_pct": mean_rate,
            "std_rate_pct": std_rate,
            "max_diff_between_folds_pct": max_diff,
        }
    )

    results_folds_csv = os.path.join(args.results_dir, "iso_2.1_kfold_anomaly_rates.csv")
    save_csv(
        df_folds,
        results_folds_csv,
        resource_type="data",
        purpose="iso_2_1_kfold_anomaly_rates",
        script_name="sanidad_iso_v2.1.py",
    )

    # CSV de resumen de estabilidad
    df_summary = pd.DataFrame(
        [
            {
                "metric": "mean_anomaly_rate_pct",
                "value": mean_rate,
            },
            {
                "metric": "std_anomaly_rate_pct",
                "value": std_rate,
            },
            {
                "metric": "max_diff_between_folds_pct",
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

    results_stability_csv = os.path.join(args.results_dir, "iso_2.1_stability_summary.csv")
    save_csv(
        df_summary,
        results_stability_csv,
        resource_type="data",
        purpose="iso_2_1_kfold_stability_summary",
        script_name="sanidad_iso_v2.1.py",
    )

    print(
        f"[ISO 2.1] CSV de tasas por fold guardado en: {results_folds_csv}\n"
        f"[ISO 2.1] CSV de resumen de estabilidad guardado en: {results_stability_csv}"
    )

    return mean_rate


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

    out_path = os.path.join(output_dir, "iso_2.1_pca_clusters.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    print(f"Leyendo dataset: {args.input}")
    try:
        # load_csv → soporta local/S3 + logs
        df_raw = load_csv(
            args.input,
            resource_type="data",
            purpose="iso_sanidad_input_v2.1",
            script_name="sanidad_iso_v2.1.py",
        )
    except FileNotFoundError:
        print(f"Error: Archivo {args.input} no encontrado (ni local ni en S3).")
        return

    # 1. Preparar Datos
    df_clean, preprocessor = preprocess_data(df_raw)

    # 2. Generar Artefacto 1 (Gráfico de Estabilidad + CSVs métricas)
    evaluate_stability(df_clean, preprocessor, args)

    # 3. Entrenar Modelo Final (Dataset Completo)
    print("\n--- Fase 2: Entrenamiento del Modelo Final (Datos Completos) ---")
    iso_forest = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("iso", iso_forest),
        ]
    )

    pipeline.fit(df_clean)

    # 4. Inferencia y Puntuación
    X_transformed = pipeline.named_steps["preprocessor"].transform(df_clean)
    model = pipeline.named_steps["iso"]

    anomaly_labels = model.predict(X_transformed)
    anomaly_scores = -model.score_samples(X_transformed)  # positivo = más raro

    # --- ARTEFACTO #2: Histograma ---
    print("Generando Histograma de Puntuaciones de Anomalías...")
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores, bins=50, kde=True, color="purple")
    plt.title("Distribución de Puntuaciones de Anomalías (Modelo Final)")
    plt.xlabel("Puntuación de Anomalía (Mayor = Más Raro)")
    plt.ylabel("Frecuencia")

    threshold = np.percentile(anomaly_scores, 100 * (1 - args.contamination))
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Umbral Top {args.contamination*100:.1f}%",
    )
    plt.legend()

    hist_path = os.path.join(args.results_dir, "iso_2.1_score_histogram.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Artefacto #2 (Histograma) guardado en {hist_path}")

    # 5. Métricas Extras (PCA y SHAP)
    plot_pca_clusters(X_transformed, anomaly_labels, args.results_dir)

    print("Generando gráfica SHAP...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, show=False)

        shap_path = os.path.join(args.results_dir, "iso_2.1_shap.png")
        plt.savefig(shap_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Gráfica SHAP guardada en: {shap_path}")
    except Exception as e:
        print(f"No se pudo generar gráfica SHAP: {e}")

    # 6. Guardar Resultados mínimos (modelo)
    df_results = df_raw.copy()
    df_results["anomaly_score"] = anomaly_scores
    df_results["is_anomaly"] = anomaly_labels

    model_path = os.path.join(args.models_dir, "iso_sanidad_pipeline_v2.1.joblib")

    # Guardar modelo usando save_model (local + S3 + logs)
    save_model(
        pipeline,
        model_path,
        resource_type="model",
        purpose="iso_sanidad_pipeline_v2.1",
        script_name="sanidad_iso_v2.1.py",
    )
    print(f"\nProceso Completo. Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()