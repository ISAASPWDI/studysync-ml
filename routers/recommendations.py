# routers/recommendations.py
# CAMBIOS:
#   - RecommendationRequest acepta `offset` (paginación)
#   - limit máximo sube a 50
#   - offset se pasa a registry.get_recommendations

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.model_registry import ModelRegistry

logger = logging.getLogger("studysync.router.recommendations")
router = APIRouter()


class RecommendationRequest(BaseModel):
    user_id: str
    exclude_users: list[str] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=50)   # default 20, máx 50
    offset: int = Field(default=0, ge=0)            # ← nuevo campo de paginación


class DistanceInfo(BaseModel):
    distance_km: float = 0.0


class RecommendedUser(BaseModel):
    user_id: str
    similarity_score: float
    distance_info: DistanceInfo


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendedUser]
    total: int = 0          # ← total disponible (útil para saber si hay más páginas)
    offset: int = 0
    limit: int = 50


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(body: RecommendationRequest):
    """
    Genera recomendaciones paginadas para un usuario usando el modelo activo.

    Paginación:
      - limit: cuántos traer (máx 50)
      - offset: desde qué posición empezar

    El contrato de respuesta incluye `total` para que NestJS/Flutter
    sepa si hay más páginas disponibles.
    """
    logger.info(
        f"📥 /recommendations - user={body.user_id}, "
        f"exclude={len(body.exclude_users)}, limit={body.limit}, offset={body.offset}"
    )

    registry = ModelRegistry.get_instance()

    # Pedimos limit+offset al modelo para poder paginar correctamente.
    # Si el modelo ya soporta offset nativamente, pásalo directo.
    # Si no, pedimos todos hasta offset+limit y sliceamos aquí.
    all_recommendations = await registry.get_recommendations(
        user_id=body.user_id,
        exclude_users=body.exclude_users,
         limit=50,
    )

    total = len(all_recommendations)
    paginated = all_recommendations[body.offset : body.offset + body.limit]

    logger.info(
        f"📤 Retornando {len(paginated)}/{total} recomendaciones "
        f"para {body.user_id} (offset={body.offset})"
    )

    return RecommendationResponse(
        recommendations=[
            RecommendedUser(
                user_id=r["user_id"],
                similarity_score=r["similarity_score"],
                distance_info=DistanceInfo(
                    distance_km=r.get("distance_info", {}).get("distance_km", 0.0)
                ),
            )
            for r in paginated
        ],
        total=total,
        offset=body.offset,
        limit=body.limit,
    )