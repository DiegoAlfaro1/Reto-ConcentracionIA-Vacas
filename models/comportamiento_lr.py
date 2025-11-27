# comportamiento_lr.py
# Regresión Logística para predicción de comportamiento (inquieta/tranquila)
# Uso desde consola:
#   python3 models/comportamiento_lr.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CSV_BEHAVIOR = "sessions_behavior.csv"
RESULTS_DIR = "results/logisticRegression/"
MODELS_DIR = "trained_models/logisticRegression/"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Leyendo dataset de comportamiento: {CSV_BEHAVIOR}")
    df = pd.read_csv(CSV_BEHAVIOR)

    # Separar features y target
    X = df.drop(columns=["label_inquieta"])
    y = df["label_inquieta"]

    print("Shape X:", X.shape)
    print("Distribución de y:")
    print(y.value_counts(), "\n")

    # Pipeline: imputar -> escalar -> Logistic Regression
    lr_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # K-Fold (3 folds estratificado)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    print("Ejecutando 3-fold cross-validation...")
    cv_results = cross_validate(lr_pipeline, X, y, cv=cv, scoring=scoring)

    print("\n=== Resultados Regresión Logística (comportamiento) - 3 folds ===")
    for metric in scoring.keys():
        scores = cv_results[f"test_{metric}"]
        print(f"{metric:9s}: mean={scores.mean():.3f}  std={scores.std():.3f}  folds={np.round(scores, 3)}")

    # Matriz de confusión
    y_pred_cv = cross_val_predict(lr_pipeline, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred_cv)
    
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión - Regresión Logística (3-fold CV)")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "lr_cv_confusion_matrix.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nMatriz de confusión guardada en: {out_path}")

    # Entrenar modelo final
    print("\nEntrenando modelo final con todos los datos...")
    lr_pipeline.fit(X, y)

    # Guardar pipeline
    model_path = os.path.join(MODELS_DIR, "comportamiento_lr_pipeline.joblib")
    joblib.dump(lr_pipeline, model_path)
    print(f"Pipeline guardado en: {model_path}")


if __name__ == "__main__":
    main()
