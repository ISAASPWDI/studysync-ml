"""
matcher.py
==========
Pipeline de entrenamiento supervisado para StudySync.

Selecciona entre KNeighborsClassifier y RandomForestClassifier
según F1-score en validación cruzada. Si no hay suficientes pares
etiquetados, activa el fallback NearestNeighbors (content-based).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from preprocessing import AcademicVectorizers
from core.database import fetch_all_users, fetch_labeled_pairs
from core.config import MIN_PAIRS, CV_FOLDS   # ← desde config, no hardcodeado

logger = logging.getLogger("studysync.matcher")

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "knn_model.pkl"
METRICS_PATH = Path("metrics.json")


def _build_candidates(n_samples: int = 10) -> dict[str, Any]:
    """
    Retorna los dos pipelines candidatos.
    n_samples: total de pares de entrenamiento (para ajustar k del KNN).
    """
    knn_k = max(1, min(3, n_samples - 1))  # k < n_samples siempre
    return {
        "knn": Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=knn_k,
                metric="cosine",
                weights="distance",
                algorithm="brute",
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=5,            # poco profundo para evitar overfitting
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


async def train_supervised_model(
    vectorizers: AcademicVectorizers,
) -> dict[str, Any] | None:
    """
    Entrena el modelo supervisado.
    Retorna None si hay menos de MIN_PAIRS pares etiquetados → activa fallback.
    """
    logger.info("🏋️  Iniciando entrenamiento supervisado...")

    # 1. Ajustar vectorizadores
    users = await fetch_all_users()
    if not users:
        logger.error("❌ No hay usuarios en MongoDB")
        return None

    vectorizers.fit(users)
    logger.info(f"👥 {len(users)} usuarios cargados y vectorizados")

    # 2. Pares etiquetados
    raw_pairs = await fetch_labeled_pairs()
    X, y, valid_pairs = vectorizers.build_feature_matrix(raw_pairs)
    n_pairs = len(valid_pairs)

    if n_pairs < MIN_PAIRS:
        logger.warning(
            f"⚠️  Solo {n_pairs} pares etiquetados (mínimo configurado: {MIN_PAIRS}). "
            f"Ajusta MIN_PAIRS_FOR_TRAINING en el .env si quieres entrenar con menos datos. "
            "Activando fallback a NearestNeighbors."
        )
        return None

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    logger.info(f"📊 Dataset: {n_pairs} pares ({n_pos} positivos, {n_neg} negativos)")

    # Necesitamos ejemplos de AMBAS clases para entrenar cualquier clasificador
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            f"⚠️  Solo una clase presente ({n_pos} positivos, {n_neg} negativos). "
            "Se necesitan dislikes Y matches aceptados. Activando fallback."
        )
        return None

    # CV: no puede tener más folds que ejemplos de la clase minoritaria
    n_minority = min(n_pos, n_neg)
    actual_folds = min(CV_FOLDS, max(2, n_minority))
    # KNN: k no puede ser mayor que n_samples
    knn_k = min(3, n_pairs)
    if actual_folds < CV_FOLDS:
        logger.warning(f"⚠️  CV reducida a {actual_folds} folds (pocos datos)")

    # 3. Validación cruzada
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    candidates = _build_candidates(n_samples=n_pairs)
    cv_results: dict[str, float] = {}

    for name, pipeline in candidates.items():
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
            mean_f1 = float(scores.mean())
            cv_results[name] = mean_f1
            logger.info(f"   {name}: F1={mean_f1:.4f} ± {scores.std():.4f}")
        except Exception as e:
            logger.warning(f"   {name}: Error en CV ({e}), asignando F1=0")
            cv_results[name] = 0.0

    # 4. Ganador
    winner_name = max(cv_results, key=cv_results.get)
    winner_pipeline = candidates[winner_name]
    logger.info(f"🏆 Modelo ganador: {winner_name} (F1={cv_results[winner_name]:.4f})")

    # 5. Entrenar sobre todos los datos
    winner_pipeline.fit(X, y)

    # 6. Métricas finales
    y_pred = winner_pipeline.predict(X)
    metrics = {
        "model": winner_name,
        "n_pairs": n_pairs,
        "n_users": len(users),
        "positives": n_pos,
        "negatives": n_neg,
        "min_pairs_threshold": MIN_PAIRS,
        "cv_folds_used": actual_folds,
        "cv_f1_scores": cv_results,
        "train_metrics": {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        },
    }

    _save_metrics(metrics)
    _save_model(winner_pipeline, vectorizers)

    logger.info(
        f"✅ Entrenamiento completo: "
        f"accuracy={metrics['train_metrics']['accuracy']}, "
        f"f1={metrics['train_metrics']['f1']}"
    )

    return {
        "model": winner_pipeline,
        "model_name": winner_name,
        "metrics": metrics,
        "n_pairs": n_pairs,
        "vectorizers": vectorizers,
    }


def build_fallback_nn(vectorizers: AcademicVectorizers):
    """
    NearestNeighbors sobre TF-IDF concatenado (content-based filtering).
    Fallback transparente cuando no hay suficientes datos supervisados.
    """
    if not vectorizers.is_fitted:
        return None

    user_ids = vectorizers.get_user_ids()
    if not user_ids:
        return None

    cache = vectorizers._user_cache
    texts = [
        f"{cache[uid]['skills_text']} {cache[uid]['interests_text']} {cache[uid]['goals_text']}"
        for uid in user_ids
    ]

    from sklearn.feature_extraction.text import TfidfVectorizer as _TF
    tfidf = _TF(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_fallback = tfidf.fit_transform(texts)

    nn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
    nn.fit(X_fallback)

    logger.info(f"🔄 Fallback NearestNeighbors ajustado sobre {len(user_ids)} usuarios")
    return nn, tfidf, user_ids


def score_candidates(model: Any, X_candidates: np.ndarray) -> np.ndarray:
    """P(match=1) para cada candidato."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_candidates)
        pos_idx = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
        return proba[:, pos_idx]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_candidates)
        return 1.0 / (1.0 + np.exp(-scores))
    else:
        raise ValueError(f"El modelo {type(model)} no soporta probabilidades")


def _save_metrics(metrics: dict) -> None:
    import datetime
    metrics["trained_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"📝 Métricas guardadas en {METRICS_PATH}")


def _save_model(model: Any, vectorizers: AcademicVectorizers) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "vectorizers": vectorizers}, MODEL_PATH)
    logger.info(f"💾 Modelo guardado en {MODEL_PATH}")


def load_model_from_disk() -> dict | None:
    if not MODEL_PATH.exists():
        logger.info("📂 No se encontró modelo previo en disco")
        return None
    try:
        payload = joblib.load(MODEL_PATH)
        logger.info(f"📂 Modelo cargado desde {MODEL_PATH}")
        return payload
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        return None