"""
routers/users.py
================
Router de sincronización de usuarios.

Endpoints consumidos por SwipeService en NestJS:
  POST /users/sync          → syncUserToMLService(userId)
  POST /users/sync-all      → syncAllUsersToMLService()
  GET  /users/{userId}/exists → checkUserExistsInML(userId)

La sincronización mantiene el caché de vectorizadores actualizado
sin necesidad de re-entrenar el modelo completo.
"""

import logging
from fastapi import APIRouter
from pydantic import BaseModel

from core.model_registry import ModelRegistry

logger = logging.getLogger("studysync.router.users")
router = APIRouter(prefix="/users")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SyncUserRequest(BaseModel):
    user_id: str
    force_reload: bool = False


class SyncUserResponse(BaseModel):
    success: bool
    message: str
    user_id: str


class SyncAllResponse(BaseModel):
    users_synced: int
    users_failed: int
    message: str


class ExistsResponse(BaseModel):
    exists: bool
    user_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/sync", response_model=SyncUserResponse)
async def sync_user(body: SyncUserRequest):
    """
    Sincroniza un usuario desde MongoDB al caché de vectorizadores.

    Llamado por NestJS cuando el endpoint /recommendations retorna 404
    (usuario no encontrado en el caché). Tras la sincronización, NestJS
    reintenta la petición de recomendaciones.

    force_reload=True: recarga el documento aunque ya esté en caché.
    """
    logger.info(f"🔄 /users/sync - user={body.user_id}, force={body.force_reload}")

    registry = ModelRegistry.get_instance()
    result = await registry.sync_user(body.user_id, body.force_reload)

    status = result.get("status", "unknown")
    success = status in ("synced", "already_synced")

    message_map = {
        "synced": f"Usuario {body.user_id} sincronizado correctamente",
        "already_synced": f"Usuario {body.user_id} ya estaba en caché",
        "not_found": f"Usuario {body.user_id} no encontrado en MongoDB",
    }

    return SyncUserResponse(
        success=success,
        message=message_map.get(status, "Estado desconocido"),
        user_id=body.user_id,
    )


@router.post("/sync-all", response_model=SyncAllResponse)
async def sync_all_users():
    """
    Re-sincroniza todos los usuarios de MongoDB al caché.
    No re-entrena el modelo; usar POST /retrain para eso.

    Llamado por NestJS en syncAllUsersToMLService().
    """
    logger.info("🔄 /users/sync-all iniciado")

    registry = ModelRegistry.get_instance()
    result = await registry.sync_all_users()

    return SyncAllResponse(
        users_synced=result["users_synced"],
        users_failed=result["users_failed"],
        message=(
            f"Sincronización completada: "
            f"{result['users_synced']} OK, {result['users_failed']} errores"
        ),
    )


@router.get("/{user_id}/exists", response_model=ExistsResponse)
async def user_exists(user_id: str):
    """
    Verifica si un usuario está en el caché de vectorizadores.
    Llamado por NestJS en checkUserExistsInML(userId).
    """
    registry = ModelRegistry.get_instance()
    exists = registry.user_exists(user_id)

    if exists:
        logger.info(f"✅ /users/{user_id}/exists → true")
    else:
        logger.warning(f"⚠️  /users/{user_id}/exists → false")

    return ExistsResponse(exists=exists, user_id=user_id)


@router.post("/webhook/user-updated")
async def webhook_user_updated(body: dict):
    """
    Webhook para actualizar un usuario individual cuando su perfil cambia.
    Mantiene el caché sincronizado sin necesidad de un sync masivo.

    Body esperado: { "user_id": str, ... }
    """
    user_id = body.get("user_id") or body.get("userId", "")
    if not user_id:
        return {"success": False, "message": "user_id requerido"}

    logger.info(f"🔔 /webhook/user-updated - user={user_id}")

    registry = ModelRegistry.get_instance()
    result = await registry.sync_user(user_id, force_reload=True)

    return {
        "success": result.get("status") in ("synced", "already_synced"),
        "message": f"Usuario {user_id} actualizado en caché",
        "user_id": user_id,
    }
