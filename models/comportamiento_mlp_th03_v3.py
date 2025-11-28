# comportamiento_mlp_th03_v3.py
# Uso desde consola:
#   python3 models/comportamiento_mlp_th03_v3.py
#
# Red neuronal MLP para clasificar el comportamiento de las vacas
# Umbral de clasificación: se optimiza a partir de F1-score

import os
import sys
import csv  # ya casi no lo usamos, pero lo dejamos por si quieres algo manual

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --- asegurar raíz del proyecto en sys.path ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# helpers de almacenamiento (S3 + logs)
from util.storage import load_csv, save_csv, save_model

# Variables globales
CSV_BEHAVIOR = "data/sessions_behavior.csv"      # CSV con los datos
RESULTS_DIR = "results/MLP/"                     # Directorio para guardar resultados
MODELS_DIR = "trained_models/MLP/"               # Directorio para guardar modelos

# CSVs de logs
METRICS_CSV_PATH        = os.path.join(RESULTS_DIR, "mlp_cv_metrics.csv")
CV_FOLDS_CSV_PATH       = os.path.join(RESULTS_DIR, "mlp_cv_folds_metrics.csv")
THRESHOLDS_CSV_PATH     = os.path.join(RESULTS_DIR, "mlp_threshold_search.csv")
METRICS_TABLE_CSV_PATH  = os.path.join(RESULTS_DIR, "mlp_cv_metrics_table.csv")  # tabla tipo RF


def genModel(X_train, y_train):
    """
    Función para generar el modelo MLP.

    Parámetros:
    - X_train: matriz de características para entrenamiento.
    - y_train: vector de etiquetas para entrenamiento.

    Retorna:
    - model: modelo MLP compilado.
    - cw_dict: diccionario con pesos para clases no balanceadas.
    """
    data_shape = X_train.shape

    # Declarar modelo
    model = keras.Sequential(
        [
            # Entrada
            layers.Input(shape=(data_shape[1],)),
            # Capas ocultas
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            # Salida
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Calcular peso para clases no balanceadas
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    cw_dict = dict(zip(classes, cw))

    # Es un problema de clasificación binaria
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model, cw_dict


def crossVal(X_train, y_train):
    """
    Función para realizar validación cruzada estratificada.
    Las clases no están balanceadas.

    Parámetros:
    - X_train: matriz de características para entrenamiento.
    - y_train: vector de etiquetas para entrenamiento.

    Retorna:
    - fold_scores: lista de diccionarios con métricas por fold.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    print("[MLP] Iniciando validación cruzada...")

    # Iterar por los folds
    for fold_idx, (train_index, val_index) in enumerate(
        skf.split(X_train, y_train), start=1
    ):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = (
            y_train.iloc[train_index],
            y_train.iloc[val_index],
        )

        # Crear y entrenar el modelo con cada fold
        mlp, W = genModel(X_train_fold, y_train_fold)
        mlp.fit(
            X_train_fold,
            y_train_fold,
            epochs=20,
            batch_size=32,
            verbose=0,
            class_weight=W,
        )

        # Evaluar por fold
        y_val_prob = mlp.predict(X_val_fold, verbose=0)
        y_val_pred = (y_val_prob >= 0.5).astype(int)

        acc = accuracy_score(y_val_fold, y_val_pred)
        prec = precision_score(y_val_fold, y_val_pred)
        rec = recall_score(y_val_fold, y_val_pred)
        f1 = f1_score(y_val_fold, y_val_pred)

        fold_scores.append(
            {
                "fold": fold_idx,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )

        print(
            f"[MLP] Fold {fold_idx}: "
            f"acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}"
        )

    print(
        "[MLP] Precisión promedio de validación:",
        np.mean([fs["accuracy"] for fs in fold_scores]),
    )
    return fold_scores


def best_threshold(model, X_train, y_train):
    """
    Función para encontrar el mejor umbral de clasificación basado en F1-score.

    Parámetros:
    - model: modelo entrenado.
    - X_train: matriz de características para entrenamiento.
    - y_train: vector de etiquetas para entrenamiento.

    Retorna:
    - best_threshold: umbral que maximiza el F1-score.
    - threshold_history: lista de dicts con threshold y F1 obtenido.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    threshold_history = []

    y_pred_prob = model.predict(X_train)  # Probabilidades predichas
    print("[MLP] Calculando mejores thresholds...")

    # Probar cada posible valor de umbral
    for threshold in np.arange(0.1, 0.9, 0.1):
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_train, y_pred)

        threshold_history.append(
            {
                "threshold": float(threshold),
                "f1": float(f1),
            }
        )

        # Guarda los mejores resultados
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            print(f"[MLP] Threshold: {threshold}, F1: {f1}")

    print("[MLP] Resultados finales de búsqueda de threshold:")
    print(f"[MLP] Mejor threshold: {best_threshold} para mayor F1: {best_f1}")

    return float(best_threshold), threshold_history


def metrics(model, X_test, y_test, threshold):
    """
    Función para calcular métricas de evaluación del modelo,
    así como la matriz de confusión.

    Parámetros:
    - model: modelo entrenado.
    - X_test: matriz de características para prueba.
    - y_test: vector de etiquetas para prueba.
    - threshold: umbral para clasificación binaria.

    Retorna:
    - scores: diccionario con las métricas calculadas.
    - confusion: matriz de confusión.
    """
    scores = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "threshold": float(threshold),
    }

    print("[MLP] Calculando métricas en test...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Matriz de confusión
    confusion = confusion_matrix(y_test, y_pred)

    # Puntuaciones
    scores["accuracy"] = float(accuracy_score(y_test, y_pred))
    scores["precision"] = float(precision_score(y_test, y_pred))
    scores["recall"] = float(recall_score(y_test, y_pred))
    scores["f1"] = float(f1_score(y_test, y_pred))

    print("[MLP] Resultados finales en test:")
    print(f"[MLP] Matriz de confusión:\n{confusion}")
    print(f"[MLP] Puntuaciones:\n{scores}")

    return scores, confusion


def main():
    # Crear carpetas de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Cargar datos
    print(f"[MLP] Leyendo dataset de comportamiento: {CSV_BEHAVIOR}")
    try:
        data = load_csv(
            CSV_BEHAVIOR,
            resource_type="data",
            purpose="mlp_behavior_input",
            script_name="comportamiento_mlp_th03_v3.py",
        )
    except FileNotFoundError:
        print(f"[MLP][ERROR] No se encontró el archivo {CSV_BEHAVIOR}")
        return

    # Separar características y objetivo
    X = data.drop("label_inquieta", axis=1)
    y = data["label_inquieta"]
    print("[MLP] Forma de X:", X.shape)
    print(f"[MLP] Distribución de y:\n{y.value_counts()}")

    # Separar set de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Preprocesar
    preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train_proc = preprocess.fit_transform(X_train)
    # Prevenir fuga de datos al preprocesar con los datos de entrenamiento
    X_test_proc = preprocess.transform(X_test)

    # Validación cruzada (logs por fold)
    fold_scores = crossVal(X_train_proc, y_train)

    # Guardar métricas de CV en CSV (log tipo "un fold por fila")
    df_folds = pd.DataFrame(fold_scores)
    save_csv(
        df_folds,
        CV_FOLDS_CSV_PATH,
        resource_type="data",
        purpose="mlp_cv_folds_metrics",
        script_name="comportamiento_mlp_th03_v3.py",
    )

    # ==========================
    # Tabla de métricas por fold
    # ==========================
    metric_names = ["accuracy", "precision", "recall", "f1"]
    per_fold_dict = {
        metric: np.array([fs[metric] for fs in fold_scores])
        for metric in metric_names
    }

    means = [per_fold_dict[m].mean() for m in metric_names]
    stds = [per_fold_dict[m].std() for m in metric_names]

    n_folds = len(fold_scores)
    table_data = {"metric": metric_names}

    # columnas fold_1, fold_2, fold_3 ...
    for fold_idx in range(n_folds):
        col_name = f"fold_{fold_idx + 1}"
        table_data[col_name] = [
            per_fold_dict[m][fold_idx] for m in metric_names
        ]

    table_data["mean"] = means
    table_data["std"] = stds

    df_metrics_table = pd.DataFrame(table_data)
    save_csv(
        df_metrics_table,
        METRICS_TABLE_CSV_PATH,
        resource_type="results",
        purpose="mlp_cv_metrics_table",
        script_name="comportamiento_mlp_th03_v3.py",
    )
    print("[MLP] Tabla de métricas por fold guardada en:")
    print(METRICS_TABLE_CSV_PATH)
    print(df_metrics_table)

       # ======================================================
    # 📊 Gráfica — MLP Optimizado - 3-fold CV por métrica
    # ======================================================
    folds = np.arange(1, n_folds + 1)  # = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metric_names:
        scores = per_fold_dict[metric]
        ax.plot(folds, scores, marker="o", label=metric)

    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("MLP Optimizado - 3-fold CV por métrica")
    ax.legend()
    fig.tight_layout()

    mlp_folds_png = os.path.join(RESULTS_DIR, "mlp_cv_metrics_per_fold.png")
    fig.savefig(mlp_folds_png, dpi=300)
    plt.close(fig)
    print(f"[MLP] Gráfica por fold guardada en: {mlp_folds_png}")

    # Entrenamiento final
    print("[MLP] Entrenando modelo final...")
    mlp, cw = genModel(X_train_proc, y_train)
    mlp.fit(
        X_train_proc,
        y_train,
        epochs=40,
        batch_size=32,
        verbose=1,
        class_weight=cw,
    )
    best_th, threshold_history = best_threshold(mlp, X_train_proc, y_train)
    print("[MLP] Entrenamiento completado.")

    # Guardar búsqueda de thresholds en CSV
    df_th = pd.DataFrame(threshold_history)
    save_csv(
        df_th,
        THRESHOLDS_CSV_PATH,
        resource_type="data",
        purpose="mlp_threshold_search",
        script_name="comportamiento_mlp_th03_v3.py",
    )

    # Métricas finales en test
    scores, matrix = metrics(mlp, X_test_proc, y_test, best_th)

    # Graficar resultados
    print("[MLP] Exportando resultados visuales...")
    # Matriz de confusión
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.title("Matriz de confusión (MLP)")
    cm_path = os.path.join(RESULTS_DIR, "mlp_cv_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[MLP] Matriz de confusión guardada en: {cm_path}")

    # Gráfico de barras con métricas finales (test)
    plt.figure(figsize=(8, 6))
    plt.bar(
        ["accuracy", "precision", "recall", "f1"],
        [
            scores["accuracy"],
            scores["precision"],
            scores["recall"],
            scores["f1"],
        ],
    )
    plt.title("Métricas del modelo MLP (test)")
    metrics_img_path = os.path.join(RESULTS_DIR, "mlp_cv_metrics.png")
    plt.savefig(metrics_img_path, dpi=300)
    plt.close()
    print(f"[MLP] Gráfico de métricas guardado en: {metrics_img_path}")

    # Exportar métricas finales como CSV (via save_csv)
    df_metrics = pd.DataFrame(
        [
            {"metric": "accuracy",  "value": scores["accuracy"]},
            {"metric": "precision", "value": scores["precision"]},
            {"metric": "recall",    "value": scores["recall"]},
            {"metric": "f1",        "value": scores["f1"]},
            {"metric": "threshold", "value": scores["threshold"]},
        ]
    )
    save_csv(
        df_metrics,
        METRICS_CSV_PATH,
        resource_type="data",
        purpose="mlp_cv_metrics",
        script_name="comportamiento_mlp_th03_v3.py",
    )
    print(f"[MLP] Métricas finales guardadas en: {METRICS_CSV_PATH}")

    # Guardar modelo (local + S3/logs)
    print("[MLP] Guardando modelo...")
    model_path = os.path.join(MODELS_DIR, "comportamiento_mlp_th03_v3.h5")

    # Guardado local estándar de Keras
    mlp.save(model_path)

    # Además, registrar/subir el modelo vía util.storage
    save_model(
        mlp,
        model_path,
        resource_type="model",
        purpose="comportamiento_mlp_th03_v3",
        script_name="comportamiento_mlp_th03_v3.py",
    )

    print(f"[MLP] Modelo guardado en: {model_path} (local + S3)")
    print("[MLP] Pipeline completado correctamente.")


if __name__ == "__main__":
    main()