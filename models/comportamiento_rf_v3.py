# comportamiento_rf_v3.py
# Uso desde consola:
# python3 models/comportamiento_rf_v3.py
# Random Forest con Hyperparameter Tuning (GridSearchCV)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate

CSV_BEHAVIOR = "../data/sessions_behavior.csv"
RESULTS_DIR = "../results"
MODELS_DIR = "../models"


def main():
    # Crear directorios de resultados y modelos
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Leyendo dataset de comportamiento: {CSV_BEHAVIOR}")
    if not os.path.exists(CSV_BEHAVIOR):
        print(f"Error: No se encontró el archivo {CSV_BEHAVIOR}")
        return
        
    df = pd.read_csv(CSV_BEHAVIOR)

    # Separar features y target
    X = df.drop(columns=["label_inquieta"])
    y = df["label_inquieta"]

    print("Shape X:", X.shape)
    print("Distribución de y:")
    print(y.value_counts(), "\n")

    # Pipeline base: imputar -> escalar -> Random Forest
    # Nota: Los parámetros del RF se definirán en el grid
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
        ]
    )

    # Definir el grid de hiperparámetros
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__class_weight': ['balanced', None]
    }

    # K-Fold (3 folds estratificado)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scoring = "f1" # Optimizamos por F1-score, pero GridSearchCV calcula el score principal

    print("Iniciando GridSearchCV...")
    print(f"Grid: {param_grid}")
    
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y)

    print("\n=== Resultados GridSearchCV ===")
    print(f"Mejor Score ({scoring}): {grid_search.best_score_:.4f}")
    print("Mejores Parámetros:")
    print(grid_search.best_params_)

    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_

    # Guardar el mejor modelo
    model_path = os.path.join(MODELS_DIR, "comportamiento_rf_best_pipeline_v3.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nMejor Pipeline de Random Forest guardado en: {model_path}")

    # Opcional: Mostrar resultados detallados de los top 5
    results_df = pd.DataFrame(grid_search.cv_results_)
    print("\nTop 5 configuraciones:")
    print(results_df.sort_values(by="rank_test_score").head(5)[["params", "mean_test_score", "std_test_score"]])

    # ---------------------------------------------------
    # Validación cruzada con las mejores métricas (para gráficas)
    # ---------------------------------------------------
    scoring_metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    print("\nEjecutando 3-fold cross-validation con el mejor modelo para generar gráficas...")
    cv_results = cross_validate(
        best_model,
        X,
        y,
        cv=cv,
        scoring=scoring_metrics,
        n_jobs=-1,
    )

    print("\n=== Resultados Random Forest (Optimizado) - 3 folds ===")
    metric_names = []
    means = []
    stds = []
    per_fold_dict = {}

    for metric in scoring_metrics.keys():
        scores = cv_results[f"test_{metric}"]
        metric_names.append(metric)
        means.append(scores.mean())
        stds.append(scores.std())
        per_fold_dict[metric] = scores

        print(
            f"{metric:9s}: "
            f"mean={scores.mean():.3f}  std={scores.std():.3f}  "
            f"folds={np.round(scores, 3)}"
        )

    # ---------------------------------------------------
    # Gráfica 1: barra con media y desviación estándar
    # ---------------------------------------------------
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest Optimizado - 3-fold CV (métrica promedio ± std)")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_bar_v3.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nGráfica de métricas promedio guardada en: {out_path}")

    # ---------------------------------------------------
    # Gráfica 2: métricas por fold (líneas)
    # ---------------------------------------------------
    folds = np.arange(1, 4)

    fig, ax = plt.subplots()
    for metric in metric_names:
        scores = per_fold_dict[metric]
        ax.plot(folds, scores, marker="o", label=metric)

    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Random Forest Optimizado - 3-fold CV por métrica")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "rf_cv_metrics_per_fold_v3.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Gráfica de métricas por fold guardada en: {out_path}")

if __name__ == "__main__":
    main()
