"""
seed_synthetic_pairs.py
========================
Genera pares sintéticos de matches (positivos) y swipes (negativos)
para activar el modelo supervisado de StudySync.

Configuración generada:
  - 500 matches positivos (status='accepted')
  - 1000 swipes negativos (action='dislike') → ratio 1:2

Lógica de compatibilidad (simula comportamiento real):
  Un par se considera POSITIVO si tiene alta similitud en:
    - skills.technical  (jaccard ≥ umbral)
    - skills.interests  (jaccard ≥ umbral)
    - objectives.primary (jaccard ≥ umbral)
    - profile.faculty   (mismo facultad = bonus)
  Un par se considera NEGATIVO si tiene baja similitud en todos los campos.

Uso:
    pip install motor pymongo python-dotenv
    python seed_synthetic_pairs.py
    python seed_synthetic_pairs.py --dry-run        # solo muestra stats, no inserta
    python seed_synthetic_pairs.py --clear          # borra datos sintéticos previos
"""

import asyncio
import argparse
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

MONGODB_URI: str = (
    os.getenv("MONGODB_URI")
    or os.getenv("MONGO_URI")
    or "mongodb://localhost:27017"
)
DATABASE_NAME: str = (
    os.getenv("DATABASE_NAME")
    or os.getenv("MONGO_DB")
    or "studysync"
)

TARGET_POSITIVE = 500   # matches aceptados
TARGET_NEGATIVE = 1000  # swipes dislike (ratio 1:2)

# Umbrales de similitud Jaccard para clasificar pares
POSITIVE_MIN_SCORE = 0.30   # score ≥ esto → candidato positivo
NEGATIVE_MAX_SCORE = 0.10   # score ≤ esto → candidato negativo

SYNTHETIC_TAG = "synthetic_seed"   # marca para poder borrar fácilmente

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("seed")

# ---------------------------------------------------------------------------
# Similitud
# ---------------------------------------------------------------------------

def jaccard(a: list, b: list) -> float:
    """Similitud de Jaccard entre dos listas (case-insensitive)."""
    sa = {x.lower().strip() for x in a if x}
    sb = {x.lower().strip() for x in b if x}
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def compatibility_score(u1: dict, u2: dict) -> float:
    """
    Score de compatibilidad entre 0 y 1 basado en los campos reales del schema:
      - skills.technical   (peso 0.35)
      - skills.interests   (peso 0.30)
      - objectives.primary (peso 0.25)
      - misma facultad     (peso 0.10)
    """
    s1 = u1.get("skills") or {}
    s2 = u2.get("skills") or {}
    o1 = u1.get("objectives") or {}
    o2 = u2.get("objectives") or {}
    p1 = u1.get("profile") or {}
    p2 = u2.get("profile") or {}

    tech_sim      = jaccard(s1.get("technical", []),  s2.get("technical", []))
    interest_sim  = jaccard(s1.get("interests", []),  s2.get("interests", []))
    objective_sim = jaccard(o1.get("primary", []),    o2.get("primary", []))
    same_faculty  = 1.0 if (
        p1.get("faculty") and p2.get("faculty")
        and p1["faculty"].lower() == p2["faculty"].lower()
    ) else 0.0

    score = (
        tech_sim      * 0.35 +
        interest_sim  * 0.30 +
        objective_sim * 0.25 +
        same_faculty  * 0.10
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# Builders de documentos MongoDB
# ---------------------------------------------------------------------------

def _random_date_in_last_n_days(n: int = 180) -> datetime:
    """Fecha aleatoria en los últimos N días."""
    return datetime.utcnow() - timedelta(days=random.randint(0, n))


def build_match_doc(u1_id: str, u2_id: str, score: float) -> dict:
    """
    Documento para la colección `matches` compatible con match.schema.ts.
    status='accepted' → par positivo para el modelo supervisado.
    """
    created = _random_date_in_last_n_days(180)
    return {
        "_id": ObjectId(),
        "user1": ObjectId(u1_id),
        "user2": ObjectId(u2_id),
        "status": "accepted",
        "matchScore": round(score, 4),
        "initiatedBy": ObjectId(u1_id),
        "chatId": None,
        "createdAt": created,
        "updatedAt": created + timedelta(minutes=random.randint(1, 60)),
        "_synthetic": SYNTHETIC_TAG,   # marca interna para limpiar después
    }


def build_swipe_doc(swiper_id: str, swiped_id: str) -> dict:
    """
    Documento para la colección `swipes` con action='dislike'.
    Par negativo para el modelo supervisado.
    """
    created = _random_date_in_last_n_days(180)
    return {
        "_id": ObjectId(),
        "swiperId": ObjectId(swiper_id),
        "swipedId": ObjectId(swiped_id),
        "action": "dislike",
        "createdAt": created,
        "_synthetic": SYNTHETIC_TAG,
    }


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

async def load_users(db) -> list[dict]:
    """Carga todos los usuarios con los campos necesarios para el score."""
    projection = {
        "_id": 1,
        "skills.technical": 1,
        "skills.interests": 1,
        "objectives.primary": 1,
        "profile.faculty": 1,
        "profile.university": 1,
    }
    users = []
    async for doc in db.users.find({}, projection):
        doc["_id"] = str(doc["_id"])
        users.append(doc)
    logger.info(f"👥 {len(users)} usuarios cargados")
    return users


def sample_candidate_pairs(
    users: list[dict],
    target_pos: int,
    target_neg: int,
    max_candidates: int = 50_000,
    pos_min_score: float = POSITIVE_MIN_SCORE,
    neg_max_score: float = NEGATIVE_MAX_SCORE,
) -> tuple[list[tuple], list[tuple]]:
    """
    Muestrea pares aleatorios y los clasifica en positivos / negativos
    según su compatibility_score.

    Retorna (positive_pairs, negative_pairs) — listas de (u1_id, u2_id, score).
    Se detiene cuando alcanza los targets o agota max_candidates intentos.
    """
    ids = [u["_id"] for u in users]
    user_map = {u["_id"]: u for u in users}

    positives: list[tuple] = []
    negatives: list[tuple] = []
    seen: set[frozenset] = set()
    attempts = 0

    # Mezclar para no sesgar siempre los mismos usuarios
    random.shuffle(ids)

    while (
        (len(positives) < target_pos or len(negatives) < target_neg)
        and attempts < max_candidates
    ):
        attempts += 1
        u1_id, u2_id = random.sample(ids, 2)
        key = frozenset({u1_id, u2_id})
        if key in seen:
            continue
        seen.add(key)

        score = compatibility_score(user_map[u1_id], user_map[u2_id])

        if score >= pos_min_score and len(positives) < target_pos:
            positives.append((u1_id, u2_id, score))
        elif score <= neg_max_score and len(negatives) < target_neg:
            negatives.append((u1_id, u2_id, score))

    logger.info(
        f"📊 Muestreo completado en {attempts} intentos → "
        f"{len(positives)} positivos / {len(negatives)} negativos"
    )
    return positives, negatives


async def clear_synthetic(db) -> None:
    """Elimina todos los documentos marcados como sintéticos."""
    r_matches = await db.matches.delete_many({"_synthetic": SYNTHETIC_TAG})
    r_swipes  = await db.swipes.delete_many({"_synthetic": SYNTHETIC_TAG})
    logger.info(
        f"🗑️  Borrados: {r_matches.deleted_count} matches + "
        f"{r_swipes.deleted_count} swipes sintéticos"
    )


async def insert_pairs(
    db,
    positives: list[tuple],
    negatives: list[tuple],
    dry_run: bool = False,
) -> dict:
    """Inserta los documentos en MongoDB (o solo los muestra en dry-run)."""

    match_docs = [build_match_doc(u1, u2, score) for u1, u2, score in positives]
    swipe_docs = [build_swipe_doc(u1, u2) for u1, u2, _ in negatives]

    if dry_run:
        logger.info("🔍 DRY-RUN — no se insertará nada")
        logger.info(f"   Matches a insertar : {len(match_docs)}")
        logger.info(f"   Swipes a insertar  : {len(swipe_docs)}")
        if match_docs:
            sample = match_docs[0].copy()
            sample["_id"] = str(sample["_id"])
            sample["user1"] = str(sample["user1"])
            sample["user2"] = str(sample["user2"])
            sample["initiatedBy"] = str(sample["initiatedBy"])
            logger.info(f"   Ejemplo match      : {sample}")
        return {"dry_run": True, "matches": len(match_docs), "swipes": len(swipe_docs)}

    inserted_matches = 0
    inserted_swipes  = 0

    if match_docs:
        res = await db.matches.insert_many(match_docs, ordered=False)
        inserted_matches = len(res.inserted_ids)
        logger.info(f"✅ {inserted_matches} matches insertados")

    if swipe_docs:
        res = await db.swipes.insert_many(swipe_docs, ordered=False)
        inserted_swipes = len(res.inserted_ids)
        logger.info(f"✅ {inserted_swipes} swipes insertados")

    return {"matches_inserted": inserted_matches, "swipes_inserted": inserted_swipes}


# ---------------------------------------------------------------------------
# Verificación post-inserción
# ---------------------------------------------------------------------------

async def verify(db) -> None:
    """Muestra un resumen del estado de la BD después de la inserción."""
    total_matches = await db.matches.count_documents({"status": "accepted"})
    synth_matches = await db.matches.count_documents({"_synthetic": SYNTHETIC_TAG})
    total_swipes  = await db.swipes.count_documents({"action": "dislike"})
    synth_swipes  = await db.swipes.count_documents({"_synthetic": SYNTHETIC_TAG})

    logger.info("─" * 50)
    logger.info("📈 ESTADO FINAL DE LA BD")
    logger.info(f"   matches accepted : {total_matches} total ({synth_matches} sintéticos)")
    logger.info(f"   swipes dislike   : {total_swipes} total ({synth_swipes} sintéticos)")
    logger.info("─" * 50)

    if total_matches >= 10:
        logger.info("🎉 El modelo supervisado debería activarse en el próximo arranque.")
        logger.info("   Llama a POST /retrain en tu ML service para re-entrenar sin reiniciar.")
    else:
        logger.warning(
            f"⚠️  Solo {total_matches} matches. "
            "Revisa si los usuarios tienen skills/objectives poblados."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(dry_run: bool = False, clear: bool = False) -> None:
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DATABASE_NAME]

    try:
        # Ping para verificar conexión antes de hacer nada
        await client.admin.command("ping")
        logger.info(f"🔌 Conectado a MongoDB: {DATABASE_NAME}")

        if clear:
            await clear_synthetic(db)
            if not dry_run:
                return

        users = await load_users(db)
        if len(users) < 2:
            logger.error("❌ Se necesitan al menos 2 usuarios en la BD")
            return

        logger.info(
            f"🎯 Objetivo: {TARGET_POSITIVE} positivos / {TARGET_NEGATIVE} negativos"
        )
        logger.info(
            f"   Umbrales → positivo ≥ {POSITIVE_MIN_SCORE} | negativo ≤ {NEGATIVE_MAX_SCORE}"
        )

        positives, negatives = sample_candidate_pairs(
            users, TARGET_POSITIVE, TARGET_NEGATIVE
        )

        if len(positives) < 10:
            logger.warning(
                f"⚠️  Solo se encontraron {len(positives)} pares positivos. "
                "Puede que los usuarios tengan pocos skills/interests en común. "
                "Bajando umbral pos_min_score a 0.10 y reintentando..."
            )
            positives, negatives = sample_candidate_pairs(
                users, TARGET_POSITIVE, TARGET_NEGATIVE,
                pos_min_score=0.10,
            )

        result = await insert_pairs(db, positives, negatives, dry_run=dry_run)
        logger.info(f"📦 Resultado: {result}")

        if not dry_run:
            await verify(db)

    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed sintético para StudySync ML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra qué se insertaría sin modificar la BD",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Borra los datos sintéticos previos (marcados con _synthetic='synthetic_seed')",
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, clear=args.clear))