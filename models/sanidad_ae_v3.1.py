# sanidad_autoencoder_v3.1.py
# Uso desde consola:
#   python3 sanidad_autoencoder_v3.1.py --k_folds 5 --contamination 0.05
#   python3 sanidad_autoencoder_v3.1.py --k_folds 3  (auto-optimiza contamination)
#

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

# Fix para Windows
import matplotlib
matplotlib.use('Agg')

# Configurar estilo de graficos
plt.style.use('ggplot')

CSV_HEALTH = "datos/sessions_health.csv"
RESULTS_DIR = "results/autoencoder/"
MODELS_DIR = "trained_models/autoencoder/"


def parse_args():
    """Configuracion de argumentos CLI"""
    parser = argparse.ArgumentParser(description="Autoencoder para deteccion de anomalias en salud del ganado")
    parser.add_argument("--contamination", type=float, default=0.05, 
                        help="Proporcion esperada de anomalias (None = auto-optimizar)")
    parser.add_argument("--k_folds", type=int, default=3, 
                        help="Numero de pliegues para validacion cruzada")
    return parser.parse_args()


def preprocess_data(df):
    """
    Detecta automaticamente columnas numericas y categoricas.
    Devuelve un preprocessor y el dataframe limpio.
    """
    # Excluir columnas de ID si existen
    cols_to_drop = [c for c in df.columns if 'id' in c.lower() or 'date' in c.lower()]
    if cols_to_drop:
        print(f"Eliminando posibles columnas identificadoras: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Identificar tipos de datos
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Features numericas: {len(numeric_features)}")
    print(f"Features categoricas: {len(categorical_features)}")

    # Transformadores
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Crear transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return df, preprocessor


def optimize_contamination(anomaly_scores, candidate_values=[0.01, 0.03, 0.05, 0.07, 0.10]):
    """
    Auto-optimiza el parametro contamination usando el metodo del codo.
    Busca el punto donde la tasa de cambio en el threshold se estabiliza.
    """
    print("\n--- Optimizando parametro de contaminacion ---")
    
    thresholds = []
    for cont in candidate_values:
        threshold = np.percentile(anomaly_scores, 100 * (1 - cont))
        thresholds.append(threshold)
        print(f"Contamination {cont:.2f} -> Threshold {threshold:.6f}")
    
    # Calcular diferencias (tasa de cambio)
    diffs = np.diff(thresholds)
    
    # Metodo del codo: encontrar donde la tasa de cambio se reduce significativamente
    # Usamos la segunda derivada o el punto donde diff se estabiliza
    if len(diffs) > 1:
        second_diffs = np.diff(diffs)
        # El punto optimo es donde second_diff es mas pequeno (curva se aplana)
        elbow_idx = np.argmin(np.abs(second_diffs)) + 1
    else:
        elbow_idx = len(candidate_values) // 2  # Usar valor medio si no hay suficientes puntos
    
    optimal_contamination = candidate_values[elbow_idx]
    print(f"\nContaminacion optima seleccionada: {optimal_contamination:.2f}")
    
    return optimal_contamination


# ============================================
# AUTOENCODER ARCHITECTURE
# ============================================

class HealthAutoencoder(nn.Module):
    """Autoencoder simple para deteccion de anomalias en salud del ganado"""
    
    def __init__(self, input_dim, encoding_dim=16):
        super(HealthAutoencoder, self).__init__()
        
        # Encoder: comprime los datos
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        
        # Decoder: reconstruye los datos
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Obtener representacion codificada"""
        return self.encoder(x)


def train_autoencoder(model, train_loader, epochs=50, lr=0.001, device='cpu'):
    """Entrena el autoencoder"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, in train_loader:
            batch_X = batch_X.to(device)
            
            # Forward pass
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'    Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return model


def get_anomaly_scores(model, X, device='cpu'):
    """
    Calcula scores de anomalia basados en error de reconstruccion.
    Mayor error = mas anomalo
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        reconstructed = model(X_tensor)
        
        # Error de reconstruccion por muestra (MSE)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
    
    return errors


def predict_anomalies(anomaly_scores, contamination=0.05):
    """
    Clasifica anomalias basado en threshold de contamination.
    Retorna 1 para normal, -1 para anomalia (compatible con Isolation Forest)
    """
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
    predictions = np.where(anomaly_scores > threshold, -1, 1)
    return predictions


# ============================================
# VISUALIZATIONS
# ============================================

def plot_pca_clusters(X_scaled, labels, output_dir):
    """Metrica Extra: Visualiza la separacion en espacio 2D"""
    print("Generando Visualizacion PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    label_text = np.where(labels == 1, 'Normal', 'Anomalia')
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=label_text, 
                    palette={'Normal': 'gray', 'Anomalia': 'red'}, alpha=0.6)
    plt.title(f"Proyeccion PCA (Modelo Final)")
    
    out_path = os.path.join(output_dir, "ae_pca_clusters.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Grafica PCA guardada en: {out_path}")


def plot_feature_reconstruction_errors(model, X, feature_names, output_dir, top_n=10, device='cpu'):
    """
    Metrica Extra: Muestra que features tienen mayor error de reconstruccion.
    Similar a SHAP pero especifico para autoencoders.
    """
    print("Generando analisis de error por feature...")
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        reconstructed = model(X_tensor).cpu().numpy()
    
    # Calcular error por feature (promedio del MSE de cada feature)
    feature_errors = np.mean((X - reconstructed) ** 2, axis=0)
    
    # Crear DataFrame y ordenar
    error_df = pd.DataFrame({
        'Feature': feature_names[:len(feature_errors)],
        'Reconstruction_Error': feature_errors
    })
    error_df = error_df.sort_values('Reconstruction_Error', ascending=False)
    
    # Plotear top N features
    plt.figure(figsize=(10, 6))
    top_errors = error_df.head(top_n)
    plt.barh(range(len(top_errors)), top_errors['Reconstruction_Error'], color='steelblue')
    plt.yticks(range(len(top_errors)), top_errors['Feature'])
    plt.xlabel('Error de Reconstruccion Promedio')
    plt.title(f'Top {top_n} Features con Mayor Error de Reconstruccion')
    plt.gca().invert_yaxis()
    
    out_path = os.path.join(output_dir, "ae_feature_errors.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafica de errores por feature guardada en: {out_path}")


def main():
    # Parse argumentos CLI
    args = parse_args()
    
    # Crear directorios de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Leyendo dataset de sanidad: {CSV_HEALTH}")
    df_raw = pd.read_csv(CSV_HEALTH)
    print("Shape original:", df_raw.shape)
    
    # Preprocesar datos (detecta numericas y categoricas)
    df_clean, preprocessor = preprocess_data(df_raw)
    
    # Aplicar transformaciones
    X_transformed = preprocessor.fit_transform(df_clean)
    print(f"Shape despues de preprocesamiento: {X_transformed.shape}")
    
    # Obtener nombres de features para visualizacion
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            if len(cols) > 0:
                onehot = trans.named_steps['onehot']
                feature_names.extend(onehot.get_feature_names_out(cols))
    
    input_dim = X_transformed.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    # ------------------------
    # K-Fold Cross Validation
    # ------------------------
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    anomaly_rates = []

    print(f"\nEjecutando {args.k_folds}-fold CV para ver estabilidad de anomalias...")
    
    # Usar contamination temporal para CV (se optimizara despues si es None)
    temp_contamination = args.contamination if args.contamination else 0.05
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_transformed), start=1):
        print(f"\nFold {fold}:")
        X_train = X_transformed[train_idx]
        X_test = X_transformed[test_idx]
        
        # Preparar DataLoader
        X_train_tensor = torch.FloatTensor(X_train)
        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Crear y entrenar modelo
        model_fold = HealthAutoencoder(input_dim=input_dim, encoding_dim=16)
        model_fold = train_autoencoder(model_fold, train_loader, epochs=30, lr=0.001, device=device)
        
        # Obtener scores en test
        test_scores = get_anomaly_scores(model_fold, X_test, device=device)
        test_labels = predict_anomalies(test_scores, contamination=temp_contamination)
        
        anomaly_rate = (test_labels == -1).mean()
        anomaly_rates.append(anomaly_rate)
        
        print(f"  Porcentaje de sesiones anomalas = {anomaly_rate:.3f}")

    mean_rate = np.mean(anomaly_rates)
    std_rate = np.std(anomaly_rates)
    print(
        f"\nPromedio de porcentaje de anomalias entre folds: "
        f"{mean_rate:.3f} +/- {std_rate:.3f}"
    )

    # ---------------------------------------------------
    # Grafica 1: porcentaje de anomalias por fold (barra)
    # ---------------------------------------------------
    folds = np.arange(1, args.k_folds + 1)
    fold_metrics = [rate * 100 for rate in anomaly_rates]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, fold_metrics, color='steelblue', alpha=0.8)
    plt.axhline(mean_rate * 100, color='red', linestyle='--', 
                label=f'Media ({mean_rate*100:.2f}%)')
    plt.title(f"Tasa de Anomalias por Pliegue (Verificacion de Estabilidad)")
    plt.xlabel("Numero de Pliegue")
    plt.ylabel("Porcentaje de Anomalias Detectadas")
    plt.legend()
    plt.ylim(0, max(fold_metrics) * 1.25)
    
    out_path = os.path.join(RESULTS_DIR, "ae_anomaly_rates_per_fold.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nGrafica de porcentaje de anomalias guardada en: {out_path}")

    # ------------------------
    # Entrenar Autoencoder final con todos los datos
    # ------------------------
    print("\nEntrenando Autoencoder final con todos los datos...")
    
    X_full_tensor = torch.FloatTensor(X_transformed)
    full_dataset = TensorDataset(X_full_tensor)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    
    final_model = HealthAutoencoder(input_dim=input_dim, encoding_dim=16)
    final_model = train_autoencoder(final_model, full_loader, epochs=50, lr=0.001, device=device)

    # Obtener scores para TODO el dataset
    anomaly_scores = get_anomaly_scores(final_model, X_transformed, device=device)
    
    # ------------------------
    # Optimizar contamination si no se proporciono
    # ------------------------
    if args.contamination is None:
        final_contamination = optimize_contamination(anomaly_scores)
    else:
        final_contamination = args.contamination
        print(f"\nUsando contamination proporcionado: {final_contamination:.2f}")
    
    # Aplicar threshold final
    anomaly_labels = predict_anomalies(anomaly_scores, contamination=final_contamination)

    df_results = df_raw.copy()
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
    # Grafica 2: histograma de anomaly_score
    # ---------------------------------------------------
    print("\nGenerando Histograma de Puntuaciones de Anomalias...")
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores, bins=50, kde=True, color='purple')
    plt.title("Distribucion de Puntuaciones de Anomalias (Modelo Final)")
    plt.xlabel("Puntuacion de Anomalia (Mayor = Mas Raro)")
    plt.ylabel("Frecuencia")
    
    # Agregar una linea para el umbral
    threshold = np.percentile(anomaly_scores, 100 * (1 - final_contamination))
    plt.axvline(threshold, color='red', linestyle='--', 
                label=f'Umbral Top {final_contamination*100}%')
    plt.legend()
    
    out_path = os.path.join(RESULTS_DIR, "ae_anomaly_score_hist.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Histograma de anomaly_score guardado en: {out_path}")

    # ---------------------------------------------------
    # Grafica 3: PCA Visualization
    # ---------------------------------------------------
    plot_pca_clusters(X_transformed, anomaly_labels, RESULTS_DIR)
    
    # ---------------------------------------------------
    # Grafica 4: Feature Reconstruction Errors (equivalente a SHAP)
    # ---------------------------------------------------
    plot_feature_reconstruction_errors(final_model, X_transformed, feature_names, 
                                      RESULTS_DIR, top_n=15, device=device)

    # Guardar el modelo completo
    model_save_dict = {
        'model_state_dict': final_model.state_dict(),
        'input_dim': input_dim,
        'encoding_dim': 16,
        'preprocessor': preprocessor,
        'contamination': final_contamination
    }
    
    model_path = os.path.join(MODELS_DIR, "ae_sanidad_v3.1_model.joblib")
    joblib.dump(model_save_dict, model_path)
    print(f"\nModelo Autoencoder guardado en: {model_path}")
    print(f"Contamination final usado: {final_contamination:.2f}")

    print("\nModelo Autoencoder entrenado y aplicado a todas las sesiones.")


if __name__ == "__main__":
    main()