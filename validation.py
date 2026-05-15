"""
validation.py
=============
Validación del dataset y reporte de métricas del modelo.

Responsabilidades:
  - Validar la calidad del dataset de pares etiquetados
  - Generar reportes de métricas detallados para la tesis
  - Detectar problemas: clase desbalanceada, pares duplicados, usuarios huérfanos

Nota de tesis:
  Las métricas aquí calculadas son sobre labels REALES de MongoDB
  (matches aceptados = positivos, dislikes = negativos), no sobre
  labels sintéticas generadas por percentil de similitud como en v1.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

logger = logging.getLogger("studysync.validation")


# ---------------------------------------------------------------------------
# Informe de validación del dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetReport:
    """Reporte de calidad del dataset de entrenamiento."""
    n_pairs: int = 0
    n_positives: int = 0
    n_negatives: int = 0
    class_balance_ratio: float = 0.0   # positives / total
    n_unique_users: int = 0
    has_enough_data: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    min_pairs: int = 50,
) -> DatasetReport:
    """
    Valida el dataset de entrenamiento y retorna un informe de calidad.

    Detecta:
      - Insuficiencia de datos (< min_pairs)
      - Desbalance severo de clases (< 10% de positivos o negativos)
      - Nan o infinitos en la matriz de features
    """
    report = DatasetReport()
    report.n_pairs = len(y)
    report.n_positives = int(y.sum())
    report.n_negatives = int((y == 0).sum())
    report.has_enough_data = report.n_pairs >= min_pairs

    if report.n_pairs > 0:
        report.class_balance_ratio = round(report.n_positives / report.n_pairs, 4)

    # --- Advertencias ---
    if not report.has_enough_data:
        report.warnings.append(
            f"⚠️  Solo {report.n_pairs} pares (mínimo {min_pairs}). "
            "Se usará NearestNeighbors como fallback."
        )

    if report.n_pairs > 0:
        balance = report.class_balance_ratio
        if balance < 0.10:
            report.warnings.append(
                f"⚠️  Desbalance severo: solo {balance*100:.1f}% positivos. "
                "Considera usar class_weight='balanced' o oversampling."
            )
        elif balance > 0.90:
            report.warnings.append(
                f"⚠️  Desbalance severo: {balance*100:.1f}% positivos. "
                "El modelo puede sesgarse hacia la clase mayoritaria."
            )

    if X.shape[0] > 0:
        if np.any(np.isnan(X)):
            report.warnings.append("❌ Hay NaN en la matriz de features.")
        if np.any(np.isinf(X)):
            report.warnings.append("❌ Hay valores infinitos en la matriz de features.")

    for w in report.warnings:
        logger.warning(w)

    return report


# ---------------------------------------------------------------------------
# Evaluación completa del modelo
# ---------------------------------------------------------------------------

@dataclass
class ModelEvaluationReport:
    """Reporte completo de evaluación del modelo."""
    model_name: str = ""
    n_pairs: int = 0
    # Cross-validation (generalización)
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    cv_precision_mean: float = 0.0
    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0
    cv_recall_mean: float = 0.0
    # Métricas in-sample (entrenamiento completo)
    train_accuracy: float = 0.0
    train_precision: float = 0.0
    train_recall: float = 0.0
    train_f1: float = 0.0
    train_roc_auc: float = 0.0
    # Matriz de confusión
    confusion_matrix: list[list[int]] = field(default_factory=list)
    # Informe por clase
    classification_report: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "model",
    cv_folds: int = 5,
) -> ModelEvaluationReport:
    """
    Evaluación completa de un modelo sklearn sobre el dataset.

    Calcula métricas de validación cruzada (generalización) e in-sample.
    Las métricas de CV son las relevantes para la tesis, ya que reflejan
    la capacidad de generalización del modelo a nuevos pares de usuarios.

    Args:
        model:      Pipeline sklearn ya entrenado
        X:          Matriz de features (n_pairs, 6)
        y:          Vector de labels binarias (n_pairs,)
        model_name: Nombre del modelo para el reporte
        cv_folds:   Número de folds para validación cruzada

    Returns:
        ModelEvaluationReport con todas las métricas
    """
    report = ModelEvaluationReport(model_name=model_name, n_pairs=len(y))

    if len(y) == 0:
        logger.error("❌ Dataset vacío, no se puede evaluar")
        return report

    # --- Cross-validation ---
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        n_jobs=-1,
    )

    report.cv_accuracy_mean = round(float(cv_results["test_accuracy"].mean()), 4)
    report.cv_accuracy_std = round(float(cv_results["test_accuracy"].std()), 4)
    report.cv_precision_mean = round(float(cv_results["test_precision"].mean()), 4)
    report.cv_recall_mean = round(float(cv_results["test_recall"].mean()), 4)
    report.cv_f1_mean = round(float(cv_results["test_f1"].mean()), 4)
    report.cv_f1_std = round(float(cv_results["test_f1"].std()), 4)

    # --- Métricas in-sample ---
    y_pred = model.predict(X)
    report.train_accuracy = round(float(accuracy_score(y, y_pred)), 4)
    report.train_precision = round(float(precision_score(y, y_pred, zero_division=0)), 4)
    report.train_recall = round(float(recall_score(y, y_pred, zero_division=0)), 4)
    report.train_f1 = round(float(f1_score(y, y_pred, zero_division=0)), 4)

    # ROC-AUC (requiere predict_proba)
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
            report.train_roc_auc = round(float(roc_auc_score(y, y_proba)), 4)
        except Exception:
            report.train_roc_auc = 0.0

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    report.confusion_matrix = cm.tolist()

    # Reporte por clase (string para incluir en la tesis)
    report.classification_report = classification_report(
        y, y_pred,
        target_names=["dislike (0)", "match (1)"],
        zero_division=0,
    )

    _log_report(report)
    return report


def _log_report(report: ModelEvaluationReport) -> None:
    """Loguea el reporte de evaluación de forma legible."""
    logger.info("=" * 60)
    logger.info(f"📊 EVALUACIÓN: {report.model_name}")
    logger.info(f"   Dataset: {report.n_pairs} pares")
    logger.info(f"   CV {5}-fold:")
    logger.info(f"     Accuracy:  {report.cv_accuracy_mean:.4f} ± {report.cv_accuracy_std:.4f}")
    logger.info(f"     Precision: {report.cv_precision_mean:.4f}")
    logger.info(f"     Recall:    {report.cv_recall_mean:.4f}")
    logger.info(f"     F1-score:  {report.cv_f1_mean:.4f} ± {report.cv_f1_std:.4f}")
    logger.info(f"   In-sample:")
    logger.info(f"     Accuracy:  {report.train_accuracy:.4f}")
    logger.info(f"     F1-score:  {report.train_f1:.4f}")
    logger.info(f"     ROC-AUC:   {report.train_roc_auc:.4f}")
    logger.info(f"   Matriz de confusión: {report.confusion_matrix}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Comparación de modelos
# ---------------------------------------------------------------------------

def compare_models(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
) -> tuple[str, dict[str, ModelEvaluationReport]]:
    """
    Compara múltiples modelos y retorna el nombre del ganador (mayor F1 CV).

    Args:
        models:   dict {nombre: pipeline_sklearn}
        X, y:     Dataset de entrenamiento
        cv_folds: Folds para validación cruzada

    Returns:
        (winner_name, {nombre: ModelEvaluationReport})
    """
    reports: dict[str, ModelEvaluationReport] = {}

    for name, model in models.items():
        logger.info(f"🔬 Evaluando modelo: {name}")
        model.fit(X, y)
        reports[name] = evaluate_model(model, X, y, model_name=name, cv_folds=cv_folds)

    winner = max(reports, key=lambda n: reports[n].cv_f1_mean)
    logger.info(
        f"🏆 Ganador: {winner} "
        f"(F1-CV={reports[winner].cv_f1_mean:.4f})"
    )

    return winner, reports
