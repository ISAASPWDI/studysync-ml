"""
core/database.py
================
Capa de acceso a datos MongoDB usando motor (async).
"""

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.config import MONGODB_URI, DATABASE_NAME  # ← lee el .env de forma segura

logger = logging.getLogger("studysync.database")

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_db() -> AsyncIOMotorDatabase:
    """Retorna la instancia singleton de la base de datos."""
    global _client, _db
    if _db is None:
        logger.info(f"📦 Conectando a MongoDB: {DATABASE_NAME}")
        logger.info(f"   URI: {MONGODB_URI[:40]}...")
        _client = AsyncIOMotorClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            maxPoolSize=1,
            retryWrites=True,
        )
        _db = _client[DATABASE_NAME]
    return _db


async def fetch_user(user_id: str) -> dict | None:
    """Obtiene el documento completo de un usuario por su _id string."""
    db = get_db()
    from bson import ObjectId
    try:
        doc = await db.users.find_one({"_id": ObjectId(user_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception:
        doc = await db.users.find_one({"_id": user_id})
        return doc


async def fetch_all_users(fields: list[str] | None = None) -> list[dict]:
    """
    Carga todos los usuarios con los campos necesarios para vectorización.
    """
    db = get_db()
    projection = {f: 1 for f in (fields or [])} if fields else None
    cursor = db.users.find({}, projection)
    users = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        users.append(doc)
    logger.info(f"👥 {len(users)} usuarios cargados desde MongoDB")
    return users


async def fetch_labeled_pairs() -> list[dict[str, Any]]:
    """
    Construye el dataset de pares etiquetados para entrenamiento supervisado.

    label=1 → match con status 'accepted' (match mutuo real)
    label=0 → swipe con action 'dislike'
    """
    db = get_db()
    pairs: list[dict] = []
    seen: set[frozenset] = set()

    # --- Positivos: matches aceptados ---
    async for match in db.matches.find({"status": "accepted"}):
        u1 = str(match.get("user1") or match.get("user1Id", ""))
        u2 = str(match.get("user2") or match.get("user2Id", ""))
        if not u1 or not u2:
            continue
        key = frozenset({u1, u2})
        if key not in seen:
            seen.add(key)
            pairs.append({"user_a": u1, "user_b": u2, "label": 1})

    # --- Negativos: swipes con dislike ---
    async for swipe in db.swipes.find({"action": "dislike"}):
        u1 = str(swipe.get("swiperId", ""))
        u2 = str(swipe.get("swipedId", ""))
        if not u1 or not u2:
            continue
        key = frozenset({u1, u2})
        if key not in seen:
            seen.add(key)
            pairs.append({"user_a": u1, "user_b": u2, "label": 0})

    logger.info(
        f"📊 Dataset: {len(pairs)} pares "
        f"({sum(1 for p in pairs if p['label']==1)} positivos, "
        f"{sum(1 for p in pairs if p['label']==0)} negativos)"
    )
    return pairs


async def fetch_match_count() -> int:
    db = get_db()
    return await db.matches.count_documents({"status": "accepted"})