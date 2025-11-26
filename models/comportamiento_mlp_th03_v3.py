# comportamiento_mlp_th03_v3.py
# Uso desde consola:
# python3 models/comportamiento_mlp_th03_v3.py
# Red neuronal MLP para clasificar el comportamiento de las vacas
# Umbral de clasificación: 0.3 (optimizado para F!-score)

# Librerías
import os
import csv

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Variables globales
CSV_BEHAVIOR = "data/sessions_behavior.csv"  # CSV con los datos
RESULTS_DIR = "results/MLP/"  # Directorio para guardar resultados
MODELS_DIR = "trained_models/MLP/"  # Directorio para guardar modelos


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
    model = keras.Sequential([
        # Entrada
        layers.Input(shape=(data_shape[1],)),
        # Capas ocultas
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        # Salida
        layers.Dense(1, activation='sigmoid')
    ])

    # Calcular peso para clases no balanceadas
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
        )

    cw_dict = dict(zip(classes, cw))

    # Es un problema de clasificación binaria
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, cw_dict


def crossVal(X_train, y_train):
    """
    Función para realizar validación cruzada estratificada.
    Las clases no están balanceadas.

    Parámetros:
    - X_train: matriz de características para entrenamiento.
    - y_train: vector de etiquetas para entrenamiento.

    No retorna nada.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    print("Iniciando validación cruzada...")

    # Iterar por los folds
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Crear y entrenar el modelo con cada fold
        mlp, W = genModel(X_train_fold, y_train_fold)
        mlp.fit(
            X_train_fold,
            y_train_fold,
            epochs=20,
            batch_size=32,
            verbose=0,
            class_weight=W
            )
        # Evaluar la périda por fold y promedio
        _, val_acc = mlp.evaluate(X_val_fold, y_val_fold, verbose=0)
        scores.append(val_acc)
        print(f"Precisión de validación: {val_acc}")

    print(f"Precisión promedio de validación: {np.mean(scores)}")


def best_threshold(model, X_train, y_train):
    """
    Función para encontrar el mejor umbral de clasificación basado en F1-score.
    Parámetros:
    - model: modelo entrenado.
    - X_train: matriz de características para entrenamiento.
    - y_train: vector de etiquetas para entrenamiento.

    Retorna:
    - best_threshold: umbral que maximiza el F1-score.
    """
    best_threshold = 0.5
    best_f1 = 0
    y_pred_prob = model.predict(X_train)  # Probabilidades predichas
    print("Calculando mejores thresholds...")

    # Probar cada posible valor de umbral
    for threshold in np.arange(0.1, 0.9, 0.1):
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_train, y_pred)  # Se optimiza hacia F1

        # Guarda los mejores resultados
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            print(f"Threshold: {threshold}, F1: {f1}")

    print("Resultados finales:")
    print(f"Mejor threshold: {best_threshold} para mayor F1: {best_f1}")

    return best_threshold


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
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
    }
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= threshold).astype(int)
    print("Calculando métricas...")

    # Matriz de confusión
    confusion = confusion_matrix(y_test, y_pred)

    # Puntuaciones
    scores["accuracy"] = accuracy_score(y_test, y_pred)
    scores["precision"] = precision_score(y_test, y_pred)
    scores["recall"] = recall_score(y_test, y_pred)
    scores["f1"] = f1_score(y_test, y_pred)

    print("Resultados finales:")
    print(f"Matriz de confusión:\n{confusion}")
    print(f"Puntuaciones:\n{scores}")

    return scores, confusion


def main():
    # Crear carpetas de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Cargar datos
    print(f"Leyendo dataset de comportamiento: {CSV_BEHAVIOR}")
    if not os.path.exists(CSV_BEHAVIOR):
        print(f"Error: No se encontró el archivo {CSV_BEHAVIOR}")
        return
    data = pd.read_csv(CSV_BEHAVIOR)

    # Separar características y objetivo
    X = data.drop("label_inquieta", axis=1)
    y = data["label_inquieta"]
    print("Forma de X:", X.shape)
    print(f"Distribución de y:\n{y.value_counts()}")

    # Separar set de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
       X, y,
       test_size=0.2,
       random_state=42,
       stratify=y
       )

    # Preprocesar
    preprocess = Pipeline([
        # Rellenar valores faltantes
        ("imputer", SimpleImputer(strategy="median")),
        # Normalizar todo por z-scores
        ("scaler", StandardScaler())
    ])
    X_train = preprocess.fit_transform(X_train)
    # Prevenir fuga de datos al preprocesar con los datos de entrenamiento
    X_test = preprocess.transform(X_test)

    # Validación cruzada
    crossVal(X_train, y_train)

    # Entrenamiento
    mlp, cw = genModel(X_train, y_train)
    mlp.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        verbose=1,
        class_weight=cw
    )
    threshold = best_threshold(mlp, X_train, y_train)
    print("Entrenamiento completado.")

    # Métricas
    scores, matrix = metrics(mlp, X_test, y_test, threshold)

    # Graficar resultados
    print("Exportando resultados...")
    # Matriz de confusión
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.title("Matriz de confusión")
    plt.savefig(
       os.path.join(RESULTS_DIR, "mlp_cv_confusion_matrix.png"),
       dpi=300
       )
    plt.show()

    # Gráfico de barras con métricas
    plt.figure(figsize=(8, 6))
    plt.bar(scores.keys(), scores.values())
    plt.title("Métricas del modelo")
    plt.savefig(os.path.join(RESULTS_DIR, "mlp_cv_metrics.png"), dpi=300)
    plt.show()

    # Exportar métricas como CSV
    with open(os.path.join(RESULTS_DIR, "mlp_cv_metrics.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in scores.items():
            writer.writerow([key, value])

    # Guardar modelo
    print("Guardando modelo...")
    mlp.save(os.path.join(MODELS_DIR, "comportamiento_mlp__th03_v3.h5"))
    print("Modelo guardado.")


if __name__ == "__main__":
    main()
