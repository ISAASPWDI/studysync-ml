"""
routers/admin.py
================
Router de administración del modelo.

Endpoint: POST /retrain
Re-entrena el modelo supervisado con los datos actuales de MongoDB,
guarda el modelo en models/knn_model.pkl y actualiza metrics.json.

Este endpoint NO está expuesto en el SwipeService de NestJS original,
pero se puede invocar manualmente o desde un cron job en producción.
"""

import logging
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from core.model_registry import ModelRegistry

logger = logging.getLogger("studysync.router.admin")
router = APIRouter()

_retraining_in_progress = False


class RetrainResponse(BaseModel):
    success: bool
    message: str
    metrics: dict = {}


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Dispara el re-entrenamiento del modelo supervisado.

    Flujo:
      1. Carga todos los usuarios de MongoDB y re-ajusta los vectorizadores
      2. Carga los pares etiquetados (matches aceptados + dislikes)
      3. Si ≥ 50 pares: entrena KNN y RandomForest, selecciona el mejor por F1
      4. Si < 50 pares: activa fallback NearestNeighbors y retorna warning
      5. Guarda modelo en models/knn_model.pkl y métricas en metrics.json

    El re-entrenamiento es sincrónico en este endpoint para simplicidad.
    En producción con datasets grandes, considera ejecutarlo como background task.
    """
    global _retraining_in_progress

    if _retraining_in_progress:
        return RetrainResponse(
            success=False,
            message="Ya hay un re-entrenamiento en curso. Intenta más tarde.",
        )

    _retraining_in_progress = True
    logger.info("🚀 /retrain - Iniciando re-entrenamiento...")

    try:
        registry = ModelRegistry.get_instance()
        metrics = await registry.retrain()

        is_fallback = metrics.get("model") in (None, "fallback_nn")
        warning = metrics.get("warning", "")

        if is_fallback:
            logger.warning(f"⚠️  Re-entrenamiento: fallback activo. {warning}")
            return RetrainResponse(
                success=True,
                message=f"Fallback activo: {warning}",
                metrics=metrics,
            )

        logger.info(f"✅ /retrain completado: {metrics}")
        return RetrainResponse(
            success=True,
            message=(
                f"Modelo {metrics.get('model', 'unknown')} re-entrenado con "
                f"{metrics.get('n_pairs', 0)} pares. "
                f"F1={metrics.get('train_metrics', {}).get('f1', 0):.4f}"
            ),
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"❌ Error en re-entrenamiento: {e}", exc_info=True)
        return RetrainResponse(
            success=False,
            message=f"Error durante el re-entrenamiento: {str(e)}",
        )
    finally:
        _retraining_in_progress = False
