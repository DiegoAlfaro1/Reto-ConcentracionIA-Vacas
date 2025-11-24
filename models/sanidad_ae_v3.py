# sanidad_autoencoder_v3.py
# Uso desde consola:
#   python3 models/sanidad_autoencoder_v3.py
#

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

# Fix para Windows
import matplotlib
matplotlib.use('Agg')

# Configurar estilo de gráficos
plt.style.use('ggplot')

CSV_HEALTH = "datos/sessions_health.csv"
RESULTS_DIR = "results/autoencoder/"
MODELS_DIR = "trained_models/autoencoder/"


# ============================================
# AUTOENCODER ARCHITECTURE
# ============================================

class HealthAutoencoder(nn.Module):
    """Autoencoder simple para detección de anomalías en salud del ganado"""
    
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
        """Obtener representación codificada"""
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
    Calcula scores de anomalía basados en error de reconstrucción.
    Mayor error = más anómalo
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        reconstructed = model(X_tensor)
        
        # Error de reconstrucción por muestra (MSE)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
    
    return errors


def predict_anomalies(anomaly_scores, contamination=0.05):
    """
    Clasifica anomalías basado en threshold de contamination.
    Retorna 1 para normal, -1 para anomalía (compatible con Isolation Forest)
    """
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
    predictions = np.where(anomaly_scores > threshold, -1, 1)
    return predictions


# ============================================
# VISUALIZATIONS
# ============================================

def plot_pca_clusters(X_scaled, labels, output_dir):
    """Métrica Extra: Visualiza la separación en espacio 2D"""
    print("Generando Visualización PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    label_text = np.where(labels == 1, 'Normal', 'Anomalía')
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=label_text, 
                    palette={'Normal': 'gray', 'Anomalía': 'red'}, alpha=0.6)
    plt.title(f"Proyección PCA (Modelo Final)")
    
    out_path = os.path.join(output_dir, "ae_3.0_pca_clusters.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfica PCA guardada en: {out_path}")


def plot_feature_reconstruction_errors(model, X, feature_names, output_dir, top_n=10, device='cpu'):
    """
    Métrica Extra: Muestra qué features tienen mayor error de reconstrucción.
    Similar a SHAP pero específico para autoencoders.
    """
    print("Generando análisis de error por feature...")
    
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
    plt.xlabel('Error de Reconstrucción Promedio')
    plt.title(f'Top {top_n} Features con Mayor Error de Reconstrucción')
    plt.gca().invert_yaxis()
    
    out_path = os.path.join(output_dir, "ae_3.0_feature_errors.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfica de errores por feature guardada en: {out_path}")


def main():
    # Crear directorios de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Leyendo dataset de sanidad: {CSV_HEALTH}")
    X = pd.read_csv(CSV_HEALTH)

    print("Shape X_health:", X.shape)
    
    # Guardar nombres de features para visualización
    feature_names = X.columns.tolist()

    # Preprocesamiento
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    input_dim = X_scaled.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    # ------------------------
    # K-Fold Cross Validation
    # ------------------------
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    anomaly_rates = []

    print("\nEjecutando 3-fold CV para ver estabilidad de anomalías...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), start=1):
        print(f"\nFold {fold}:")
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        
        # Preparar DataLoader
        X_train_tensor = torch.FloatTensor(X_train)
        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Crear y entrenar modelo
        model_fold = HealthAutoencoder(input_dim=input_dim, encoding_dim=16)
        model_fold = train_autoencoder(model_fold, train_loader, epochs=30, lr=0.001, device=device)
        
        # Obtener scores en test
        test_scores = get_anomaly_scores(model_fold, X_test, device=device)
        test_labels = predict_anomalies(test_scores, contamination=0.05)
        
        anomaly_rate = (test_labels == -1).mean()
        anomaly_rates.append(anomaly_rate)
        
        print(f"  Porcentaje de sesiones anómalas = {anomaly_rate:.3f}")

    mean_rate = np.mean(anomaly_rates)
    std_rate = np.std(anomaly_rates)
    print(
        f"\nPromedio de porcentaje de anomalías entre folds: "
        f"{mean_rate:.3f} ± {std_rate:.3f}"
    )

    # ---------------------------------------------------
    # Gráfica 1: porcentaje de anomalías por fold (barra)
    # ---------------------------------------------------
    folds = np.arange(1, 4)
    fold_metrics = [rate * 100 for rate in anomaly_rates]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, fold_metrics, color='steelblue', alpha=0.8)
    plt.axhline(mean_rate * 100, color='red', linestyle='--', 
                label=f'Media ({mean_rate*100:.2f}%)')
    plt.title(f"Tasa de Anomalías por Pliegue (Verificación de Estabilidad)")
    plt.xlabel("Número de Pliegue")
    plt.ylabel("Porcentaje de Anomalías Detectadas")
    plt.legend()
    plt.ylim(0, max(fold_metrics) * 1.25)
    
    out_path = os.path.join(RESULTS_DIR, "ae_3.0_anomaly_rates_per_fold.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nGráfica de porcentaje de anomalías guardada en: {out_path}")

    # ------------------------
    # Entrenar Autoencoder final con todos los datos
    # ------------------------
    print("\nEntrenando Autoencoder final con todos los datos...")
    
    X_full_tensor = torch.FloatTensor(X_scaled)
    full_dataset = TensorDataset(X_full_tensor)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    
    final_model = HealthAutoencoder(input_dim=input_dim, encoding_dim=16)
    final_model = train_autoencoder(final_model, full_loader, epochs=50, lr=0.001, device=device)

    # Obtener scores y labels para TODO el dataset
    anomaly_scores = get_anomaly_scores(final_model, X_scaled, device=device)
    anomaly_labels = predict_anomalies(anomaly_scores, contamination=0.05)

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
    print("\nGenerando Histograma de Puntuaciones de Anomalías...")
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores, bins=50, kde=True, color='purple')
    plt.title("Distribución de Puntuaciones de Anomalías (Modelo Final)")
    plt.xlabel("Puntuación de Anomalía (Mayor = Más Raro)")
    plt.ylabel("Frecuencia")
    
    # Agregar una línea para el umbral
    contamination = 0.05
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
    plt.axvline(threshold, color='red', linestyle='--', 
                label=f'Umbral Top {contamination*100}%')
    plt.legend()
    
    out_path = os.path.join(RESULTS_DIR, "ae_3.0_anomaly_score_hist.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Histograma de anomaly_score guardado en: {out_path}")

    # ---------------------------------------------------
    # Gráfica 3: PCA Visualization
    # ---------------------------------------------------
    plot_pca_clusters(X_scaled, anomaly_labels, RESULTS_DIR)
    
    # ---------------------------------------------------
    # Gráfica 4: Feature Reconstruction Errors (equivalente a SHAP)
    # ---------------------------------------------------
    plot_feature_reconstruction_errors(final_model, X_scaled, feature_names, 
                                      RESULTS_DIR, top_n=15, device=device)

    # Guardar el modelo completo
    model_save_dict = {
        'model_state_dict': final_model.state_dict(),
        'input_dim': input_dim,
        'encoding_dim': 16,
        'imputer': imputer,
        'scaler': scaler,
        'contamination': 0.05
    }
    
    model_path = os.path.join(MODELS_DIR, "ae_sanidad_v3.0_model.joblib")
    joblib.dump(model_save_dict, model_path)
    print(f"\nModelo Autoencoder guardado en: {model_path}")

    # Opcional: guardar dataset enriquecido con scores
    # df_results.to_csv("data/sessions_health_with_ae.csv", index=False)
    print("\nModelo Autoencoder entrenado y aplicado a todas las sesiones.")


if __name__ == "__main__":
    main()