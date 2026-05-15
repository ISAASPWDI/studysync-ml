"""
routers/recommendations.py
===========================
Router de recomendaciones.

Endpoint: POST /recommendations
Contrato de entrada (desde SwipeService.getRecommendations en NestJS):
  {
    "user_id":       str,
    "exclude_users": list[str],
    "limit":         int
  }

Contrato de salida esperado por NestJS:
  {
    "recommendations": [
      {
        "user_id":        str,
        "similarity_score": float,   // proba(match=1) en modelo supervisado
        "distance_info":  { "distance_km": float }
      }
    ]
  }

NestJS procesa mlResponse.data.recommendations y accede a:
  - rec.user_id          → para enrichRecommendations
  - rec.similarity_score → mapeado a matchScore en RecommendedUserDTO
  - rec.distance_info.distance_km → mapeado a distance en RecommendedUserDTO
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.model_registry import ModelRegistry

logger = logging.getLogger("studysync.router.recommendations")
router = APIRouter()


class RecommendationRequest(BaseModel):
    user_id: str
    exclude_users: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)


class DistanceInfo(BaseModel):
    distance_km: float = 0.0


class RecommendedUser(BaseModel):
    user_id: str
    similarity_score: float
    distance_info: DistanceInfo


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendedUser]


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(body: RecommendationRequest):
    """
    Genera recomendaciones para un usuario usando el modelo activo.

    Comportamiento:
      - Si hay modelo supervisado (≥50 pares etiquetados):
          ordena candidatos por P(match=1) descendente
      - Si no hay suficientes datos:
          usa NearestNeighbors sobre TF-IDF (filtrado por contenido)
          y loguea un warning (transparente para NestJS)

    En ambos casos el contrato de respuesta es idéntico.
    La lógica de reintentar con syncUserToMLService en NestJS se activa
    cuando este endpoint retorna 404; aquí retornamos siempre 200 con
    lista vacía en caso de error para evitar ese flujo innecesariamente.
    """
    logger.info(
        f"📥 /recommendations - user={body.user_id}, "
        f"exclude={len(body.exclude_users)}, limit={body.limit}"
    )

    registry = ModelRegistry.get_instance()

    recommendations = await registry.get_recommendations(
        user_id=body.user_id,
        exclude_users=body.exclude_users,
        limit=body.limit,
    )

    logger.info(f"📤 Retornando {len(recommendations)} recomendaciones para {body.user_id}")

    return RecommendationResponse(
        recommendations=[
            RecommendedUser(
                user_id=r["user_id"],
                similarity_score=r["similarity_score"],
                distance_info=DistanceInfo(
                    distance_km=r.get("distance_info", {}).get("distance_km", 0.0)
                ),
            )
            for r in recommendations
        ]
    )
