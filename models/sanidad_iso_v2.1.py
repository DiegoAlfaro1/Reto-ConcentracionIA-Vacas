import argparse
import os
import joblib
import numpy as np
import pandas as pd

# Fix para evitar errores de threading con matplotlib en Windows
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
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

# Configurar estilo de gráficos
plt.style.use('ggplot')

def parse_args():
    """Configuración de argumentos de línea de comandos para el script"""
    parser = argparse.ArgumentParser(description="Isolation Forest: Verificación de Estabilidad + Modelo Final de Producción")
    parser.add_argument("--input", type=str, default="datos/sessions_health.csv", help="Ruta al archivo CSV de entrada")
    parser.add_argument("--results_dir", type=str, default="results", help="Directorio para imágenes de salida")
    parser.add_argument("--models_dir", type=str, default="models", help="Directorio para guardar modelos")
    parser.add_argument("--contamination", type=float, default=0.05, help="Proporción esperada de valores atípicos")
    parser.add_argument("--n_estimators", type=int, default=200, help="Número de árboles")
    parser.add_argument("--k_folds", type=int, default=5, help="Número de pliegues para verificación de estabilidad")
    return parser.parse_args()

def preprocess_data(df):
    """
    Detecta automáticamente columnas numéricas y categóricas.
    Devuelve un pipeline de preprocesamiento y el dataframe limpio.
    """
    # Excluir columnas de ID si existen (heurística)
    cols_to_drop = [c for c in df.columns if 'id' in c.lower() or 'date' in c.lower()]
    if cols_to_drop:
        print(f"Eliminando posibles columnas identificadoras: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Identificar tipos de datos
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    # RobustScaler es mejor para detección de anomalías que StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Imputar valores faltantes con la mediana
        ('scaler', StandardScaler())  # Escalar usando estadísticas robustas
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputar valores faltantes
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Codificación one-hot
    ])

    # Crear transformador de columnas que combina transformadores numéricos y categóricos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return df, preprocessor

def evaluate_stability(df, preprocessor, args):
    """
    ARTEFACTO REQUERIDO #1: Gráfico de Anomalías por Pliegue.
    Ejecuta Validación Cruzada K-Fold para verificar si la tasa de anomalías es estable.
    """
    print(f"\n--- Fase 1: Evaluación de Estabilidad ({args.k_folds}-Fold CV) ---")
    
    # Inicializar K-Fold para validación cruzada
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    fold_indices = []

    # Iterar sobre cada pliegue
    for fold_i, (train_index, test_index) in enumerate(kf.split(df)):
        X_train = df.iloc[train_index]
        X_test = df.iloc[test_index]
        
        # Modelo fresco por pliegue para evitar fuga de datos
        iso_fold = IsolationForest(
            n_estimators=args.n_estimators,
            contamination=args.contamination,
            n_jobs=-1,
            random_state=42
        )
        
        # Crear pipeline que combina preprocesamiento y modelo
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('iso', iso_fold)])
        
        # Entrenar con datos de entrenamiento
        pipeline.fit(X_train)
        
        # Predecir en conjunto de prueba
        test_preds = pipeline.predict(X_test) # -1 para anomalía, 1 para normal
        
        # Calcular métricas
        n_anomalies = (test_preds == -1).sum()
        n_total = len(test_preds)
        pct_anomalies = (n_anomalies / n_total) * 100
        
        print(f"Pliegue {fold_i+1}: Detectadas {n_anomalies}/{n_total} anomalías ({pct_anomalies:.2f}%)")
        
        fold_metrics.append(pct_anomalies)
        fold_indices.append(fold_i + 1)

    # Métricas generales
    mean_rate = np.mean(fold_metrics)
    
    # --- GENERAR ARTEFACTO REQUERIDO #1 ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(fold_indices, fold_metrics, color='steelblue', alpha=0.8)
    plt.axhline(mean_rate, color='red', linestyle='--', label=f'Media ({mean_rate:.2f}%)')
    plt.title(f"Tasa de Anomalías por Pliegue (Verificación de Estabilidad)")
    plt.xlabel("Número de Pliegue")
    plt.ylabel("Porcentaje de Anomalías Detectadas")
    plt.legend()
    plt.ylim(0, max(fold_metrics) * 1.25) 
    
    out_path = os.path.join(args.results_dir, "iso_2.1_stability_folds.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Artefacto #1 (Gráfico de Estabilidad) guardado en {out_path}")
    
    return mean_rate

def plot_pca_clusters(X_transformed, labels, output_dir):
    """Métrica Extra: Visualiza la separación en espacio 2D."""
    print("Generando Visualización PCA...")
    # Reducir dimensionalidad a 2 componentes principales
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transformed)
    
    plt.figure(figsize=(10, 6))
    label_text = np.where(labels == 1, 'Normal', 'Anomalía')
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=label_text, palette={'Normal': 'gray', 'Anomalía': 'red'}, alpha=0.6)
    plt.title(f"Proyección PCA (Modelo Final)")
    
    out_path = os.path.join(output_dir, "iso_2.1_pca_clusters.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    print(f"Leyendo dataset: {args.input}")
    try:
        df_raw = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Archivo {args.input} no encontrado.")
        return

    # 1. Preparar Datos
    df_clean, preprocessor = preprocess_data(df_raw)

    # 2. Generar Artefacto 1 (Gráfico de Estabilidad)
    evaluate_stability(df_clean, preprocessor, args)

    # 3. Entrenar Modelo Final (Dataset Completo)
    print("\n--- Fase 2: Entrenamiento del Modelo Final (Datos Completos) ---")
    iso_forest = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1
    )

    # Crear pipeline completo con preprocesamiento y modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('iso', iso_forest)
    ])

    # Entrenar el modelo con todos los datos
    pipeline.fit(df_clean)

    # 4. Inferencia y Puntuación
    X_transformed = pipeline.named_steps['preprocessor'].transform(df_clean)
    model = pipeline.named_steps['iso']
    
    anomaly_labels = model.predict(X_transformed)
    anomaly_scores = -model.score_samples(X_transformed) # Convertir a puntuaciones positivas

    # --- GENERAR ARTEFACTO REQUERIDO #2: Histograma ---
    print("Generando Histograma de Puntuaciones de Anomalías...")
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores, bins=50, kde=True, color='purple')
    plt.title("Distribución de Puntuaciones de Anomalías (Modelo Final)")
    plt.xlabel("Puntuación de Anomalía (Mayor = Más Raro)")
    plt.ylabel("Frecuencia")
    
    # Agregar una línea para el umbral aproximado
    threshold = np.percentile(anomaly_scores, 100 * (1 - args.contamination))
    plt.axvline(threshold, color='red', linestyle='--', label=f'Umbral Top {args.contamination*100}%')
    plt.legend()
    
    hist_path = os.path.join(args.results_dir, "iso_2.1_score_histogram.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Artefacto #2 (Histograma) guardado en {hist_path}")

    # 5. Métricas Extras (PCA y SHAP)
    plot_pca_clusters(X_transformed, anomaly_labels, args.results_dir)
    
    # Opcional: SHAP para explicabilidad
    print("Generando gráfica SHAP...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        
        # Crear figura sin mostrarla
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, show=False)
        
        shap_path = os.path.join(args.results_dir, "iso_2.1_shap.png")
        plt.savefig(shap_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Gráfica SHAP guardada en: {shap_path}")
    except Exception as e:
        print(f"No se pudo generar gráfica SHAP: {e}")

    # 6. Guardar Resultados
    df_results = df_raw.copy()
    df_results['anomaly_score'] = anomaly_scores
    df_results['is_anomaly'] = anomaly_labels 
    
    # Guardar Modelo entrenado
    model_path = os.path.join(args.models_dir, "trained_models/iso_sanidad_pipeline.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nProceso Completo. Modelo guardado en: {model_path}")

if __name__ == "__main__":
    main()