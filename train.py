import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. CONSTANTES
# ============================================================

COLUMNAS = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class"
]


# ============================================================
# 2. ARGUMENTOS DE LÍNEA DE COMANDOS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Entrenamiento parametrizado del modelo Car Evaluation"
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la regularización de Logistic Regression. Default: 1.0"
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "newton-cg", "saga"],
        help="Algoritmo utilizado para optimizar Logistic Regression."
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Tratamiento del desbalance de clases."
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones del entrenamiento."
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=0.0001,
        help="Tolerancia utilizada para determinar convergencia."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para garantizar reproducibilidad."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/car+evaluation.zip",
        help="Ruta del dataset."
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="TC5061_Car_Evaluation_Experimentacion",
        help="Nombre del experimento de MLflow."
    )

    return parser.parse_args()


# ============================================================
# 3. CARGA DE DATOS
# ============================================================

def load_data(data_path):

    with zipfile.ZipFile(data_path, "r") as archivo_zip:

        with archivo_zip.open("car.data") as archivo:

            df = pd.read_csv(
                archivo,
                header=None,
                names=COLUMNAS
            )

    # Eliminar posibles espacios
    for columna in df.columns:
        df[columna] = df[columna].astype(str).str.strip()

    return df


# ============================================================
# 4. DIVISIÓN TRAIN / TEST
# ============================================================

def split_data(df, seed):

    X = df.drop(columns="class")
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 5. CREACIÓN DEL MODELO
# ============================================================

def build_model(args, columnas):

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                columnas
            )
        ]
    )

    if args.class_weight == "none":
        class_weight = None
    else:
        class_weight = "balanced"

    clasificador = LogisticRegression(
        C=args.C,
        solver=args.solver,
        class_weight=class_weight,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed
    )

    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("clasificador", clasificador)
        ]
    )

    return modelo


# ============================================================
# 6. EVALUACIÓN
# ============================================================

def evaluate_model(modelo, X_test, y_test):

    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    reporte = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    metricas = {
        "accuracy": accuracy,
        "f1_macro": f1_macro
    }

    return metricas, y_pred, reporte


# ============================================================
# 7. GUARDAR ARTEFACTOS
# ============================================================

def save_artifacts(
    y_test,
    y_pred,
    metricas,
    reporte,
    output_dir
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------
    # Matriz de confusión
    # ----------------------

    ruta_matriz = output_dir / "confusion_matrix.png"

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[
            "unacc",
            "acc",
            "good",
            "vgood"
        ]
    )

    plt.title("Matriz de confusión")

    plt.tight_layout()

    plt.savefig(
        ruta_matriz,
        dpi=150
    )

    plt.close()

    # ----------------------
    # Archivo de métricas
    # ----------------------

    ruta_metricas = output_dir / "metrics.json"

    with open(
        ruta_metricas,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            metricas,
            archivo,
            indent=4
        )

    # ----------------------
    # Classification report
    # ----------------------

    ruta_reporte = output_dir / "classification_report.json"

    with open(
        ruta_reporte,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            reporte,
            archivo,
            indent=4
        )

    return (
        ruta_matriz,
        ruta_metricas,
        ruta_reporte
    )


# ============================================================
# 8. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Reproducibilidad
    # --------------------------------------------------------

    np.random.seed(args.seed)

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        args.experiment_name
    )

    # --------------------------------------------------------
    # Datos
    # --------------------------------------------------------

    df = load_data(
        args.data_path
    )

    X_train, X_test, y_train, y_test = split_data(
        df,
        args.seed
    )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    modelo = build_model(
        args,
        X_train.columns.tolist()
    )

    modelo.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluación
    # --------------------------------------------------------

    metricas, y_pred, reporte = evaluate_model(
        modelo,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Nombre descriptivo de la corrida
    # --------------------------------------------------------

    run_name = (
        f"LR_"
        f"C{args.C}_"
        f"{args.solver}_"
        f"{args.class_weight}_"
        f"seed{args.seed}"
    )

    output_dir = Path("outputs") / run_name

    rutas_artefactos = save_artifacts(
        y_test,
        y_pred,
        metricas,
        reporte,
        output_dir
    )

    # --------------------------------------------------------
    # Firma del modelo
    # --------------------------------------------------------

    input_example = X_train.head(5).astype(str)

    pred_example = modelo.predict(
        input_example
    )

    signature = infer_signature(
        input_example,
        pred_example
    )

    # --------------------------------------------------------
    # MLflow Run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        # Parámetros
        mlflow.log_params({
            "C": args.C,
            "solver": args.solver,
            "class_weight": args.class_weight,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "seed": args.seed,
            "test_size": 0.20,
            "modelo": "LogisticRegression",
            "encoding": "OneHotEncoder"
        })

        # Métricas
        mlflow.log_metric(
            "accuracy",
            metricas["accuracy"]
        )

        mlflow.log_metric(
            "f1_macro",
            metricas["f1_macro"]
        )

        # Artefactos
        for ruta in rutas_artefactos:
            mlflow.log_artifact(
                str(ruta),
                artifact_path="evaluation"
            )

        # Modelo
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo",
            input_example=input_example,
            signature=signature
        )

        # Tags
        mlflow.set_tags({
            "dataset": "Car Evaluation",
            "task": "multiclass_classification",
            "stage": "experimentation"
        })

        print("\n==============================")
        print("ENTRENAMIENTO COMPLETADO")
        print("==============================")

        print(
            f"Accuracy: {metricas['accuracy']:.4f}"
        )

        print(
            f"F1 Macro: {metricas['f1_macro']:.4f}"
        )

        print(
            f"Run ID: {run.info.run_id}"
        )


# ============================================================
# 9. PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()