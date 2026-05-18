"""
core/model_registry.py
=======================
Singleton que gestiona el estado global del modelo activo en producción.

Responsabilidades:
  - Cargar el modelo serializado al arrancar la aplicación
  - Exponer el modelo actual (supervisado o fallback) a los routers
  - Ejecutar re-entrenamiento y actualizar el estado en caliente
  - Gestionar el caché de usuarios para inferencia rápida

Estados posibles:
  SUPERVISED  → modelo KNN/RF entrenado con ≥ MIN_PAIRS pares etiquetados
  FALLBACK    → NearestNeighbors sobre TF-IDF (datos insuficientes)
  UNINITIALIZED → aún no se ha cargado ningún modelo
"""

import asyncio
import logging
from enum import Enum
from typing import Any

import numpy as np

from preprocessing import AcademicVectorizers
from matcher import (
    train_supervised_model,
    build_fallback_nn,
    load_model_from_disk,
    score_candidates,
    MIN_PAIRS,
)
from core.database import fetch_all_users, fetch_user

logger = logging.getLogger("studysync.model_registry")


class ModelState(str, Enum):
    SUPERVISED = "supervised"
    FALLBACK = "fallback"
    UNINITIALIZED = "uninitialized"


class ModelRegistry:
    """
    Singleton que encapsula el modelo activo y su estado.

    Uso:
        registry = ModelRegistry.get_instance()
        await registry.load_from_disk()
        recs = await registry.get_recommendations(user_id, exclude, limit)
    """

    _instance: "ModelRegistry | None" = None

    def __init__(self):
        self.state: ModelState = ModelState.UNINITIALIZED
        self.model: Any = None
        self.model_name: str = "none"
        self.vectorizers: AcademicVectorizers = AcademicVectorizers()
        self.fallback_nn: Any = None
        self.fallback_tfidf: Any = None
        self.fallback_user_ids: list[str] = []
        self.metrics: dict = {}
        self.n_pairs: int = 0
        self._lock = asyncio.Lock()
        self._cache: dict = {}
        self._cache_ttl: int = 300 

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -----------------------------------------------------------------------
    # Inicialización
    # -----------------------------------------------------------------------

    async def load_from_disk(self) -> None:
        """
        Intenta cargar un modelo previo desde disco.
        Si no existe, entrena uno nuevo o activa el fallback.
        """
        async with self._lock:
            payload = load_model_from_disk()

            if payload:
                self.model = payload["model"]
                self.vectorizers = payload["vectorizers"]
                self.model_name = getattr(self.model, "name", "loaded")
                self.state = ModelState.SUPERVISED
                logger.info("✅ Modelo supervisado cargado desde disco")
            else:
                # Primera vez: intentar entrenar
                await self._train_and_update()

    async def retrain(self) -> dict:
        """
        Re-entrena el modelo con los datos actuales de MongoDB.
        Thread-safe vía asyncio.Lock.
        Retorna las métricas del nuevo modelo.
        """
        async with self._lock:
            logger.info("🔄 Re-entrenamiento solicitado...")
            result = await self._train_and_update()
            return result

    async def _train_and_update(self) -> dict:
        """
        Lógica interna de (re)entrenamiento.
        Actualiza el estado del registry según el resultado.
        """
        self._cache.clear() 
        new_vecs = AcademicVectorizers()
        result = await train_supervised_model(new_vecs)

        if result is not None:
            # Modelo supervisado exitoso
            self.model = result["model"]
            self.model_name = result["model_name"]
            self.vectorizers = result["vectorizers"]
            self.metrics = result["metrics"]
            self.n_pairs = result["n_pairs"]
            self.state = ModelState.SUPERVISED
            self.fallback_nn = None
            self.fallback_tfidf = None
            self.fallback_user_ids = []
            logger.info(f"✅ Modelo supervisado activo: {self.model_name}")
            return result["metrics"]
        else:
            # Datos insuficientes → fallback
            await self._activate_fallback(new_vecs)
            return {
                "model": "fallback_nn",
                "warning": f"Menos de {MIN_PAIRS} pares etiquetados. Usando NearestNeighbors.",
                "n_pairs": self.n_pairs,
            }

    async def _activate_fallback(self, vectorizers: AcademicVectorizers) -> None:
        """
        Construye el modelo de fallback NearestNeighbors.
        Requiere que los vectorizadores ya estén ajustados.
        """
        if not vectorizers.is_fitted:
            users = await fetch_all_users()
            if users:
                vectorizers.fit(users)

        self.vectorizers = vectorizers
        result = build_fallback_nn(vectorizers)

        if result:
            nn, tfidf, user_ids = result
            self.fallback_nn = nn
            self.fallback_tfidf = tfidf
            self.fallback_user_ids = user_ids
            self.state = ModelState.FALLBACK
            logger.warning(
                f"⚠️  Fallback NearestNeighbors activo "
                f"({len(user_ids)} usuarios). "
                f"Re-entrenar cuando haya ≥{MIN_PAIRS} pares etiquetados."
            )
        else:
            self.state = ModelState.UNINITIALIZED
            logger.error("❌ No se pudo inicializar ningún modelo")

    # -----------------------------------------------------------------------
    # Inferencia
    # -----------------------------------------------------------------------

    async def get_recommendations(
        self,
        user_id: str,
        exclude_users: list[str],
        limit: int,
    ) -> list[dict]:
        import time

        cache_key = f"{user_id}:{limit}"
        cached = self._cache.get(cache_key)

        if cached and (time.time() - cached["ts"]) < self._cache_ttl:
            logger.debug(f"💾 Cache hit para {user_id}")
            excluded_set = set(exclude_users) | {user_id}
            filtered = [r for r in cached["data"] if r["user_id"] not in excluded_set]
            return filtered[:limit]

        # Cache miss → computar
        if self.state == ModelState.SUPERVISED:
            results = await self._supervised_recommendations(user_id, exclude_users, limit)
        elif self.state == ModelState.FALLBACK:
            results = await self._fallback_recommendations(user_id, exclude_users, limit)
        else:
            logger.error("❌ Modelo no inicializado")
            return []

        self._cache[cache_key] = {"data": results, "ts": time.time()}
        return results

    async def _supervised_recommendations(
        self,
        user_id: str,
        exclude_users: list[str],
        limit: int,
    ) -> list[dict]:
        """
        Recomendaciones usando el modelo supervisado.

        1. Obtiene candidatos: usuarios en caché que no están excluidos
        2. Calcula features (user_id, candidato) para cada par
        3. Predice P(match=1) con el modelo
        4. Ordena descendente y retorna top-N

        Nota: si el usuario no está en el caché de vectorizadores (usuario
        nuevo), intenta sincronizarlo desde MongoDB antes de proceder.
        """
        excluded_set = set(exclude_users)
        excluded_set.add(user_id)

        # Verificar si el usuario está en el caché
        if user_id not in self.vectorizers._user_cache:
            await self._sync_single_user(user_id)
            if user_id not in self.vectorizers._user_cache:
                logger.warning(f"⚠️  Usuario {user_id} no encontrado, usando fallback")
                return await self._fallback_recommendations(user_id, list(excluded_set), limit)

        candidates = [
            uid for uid in self.vectorizers.get_user_ids()
            if uid not in excluded_set
        ]

        if not candidates:
            return []

        X_candidates, valid_uids = self.vectorizers.compute_candidate_features(
            user_id, candidates
        )

        if len(valid_uids) == 0:
            return []

        proba = score_candidates(self.model, X_candidates)

        # Ordenar por probabilidad descendente
        sorted_indices = np.argsort(proba)[::-1][:limit]

        results = []
        for idx in sorted_indices:
            uid = valid_uids[idx]
            score = float(proba[idx])
            results.append({
                "user_id": uid,
                "similarity_score": round(score, 4),
                "distance_info": {"distance_km": 0.0},  # campo requerido por NestJS
            })

        return results

    async def _fallback_recommendations(
        self,
        user_id: str,
        exclude_users: list[str],
        limit: int,
    ) -> list[dict]:
        """
        Recomendaciones usando NearestNeighbors (filtrado por contenido).

        Equivalente al comportamiento original del servicio.
        Transparente para el backend NestJS: el contrato de respuesta es idéntico.
        """
        if self.fallback_nn is None or self.fallback_tfidf is None:
            logger.error("❌ Fallback NN no disponible")
            return []

        excluded_set = set(exclude_users)
        excluded_set.add(user_id)

        # Índice del usuario target en la lista del fallback
        if user_id not in self.fallback_user_ids:
            await self._sync_single_user(user_id)
            if user_id not in self.fallback_user_ids:
                return []

        target_idx = self.fallback_user_ids.index(user_id)

        # Obtener vecinos
        cache = self.vectorizers._user_cache
        all_texts = [
            f"{cache[uid]['skills_text']} {cache[uid]['interests_text']} {cache[uid]['goals_text']}"
            for uid in self.fallback_user_ids
        ]
        target_vec = self.fallback_tfidf.transform([all_texts[target_idx]])

        n_neighbors = min(limit * 3 + len(excluded_set), len(self.fallback_user_ids))
        distances, indices = self.fallback_nn.kneighbors(target_vec, n_neighbors=n_neighbors)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            uid = self.fallback_user_ids[idx]
            if uid in excluded_set:
                continue
            similarity = float(1.0 - dist)  # cosine distance → similarity
            results.append({
                "user_id": uid,
                "similarity_score": round(similarity, 4),
                "distance_info": {"distance_km": 0.0},
            })
            if len(results) >= limit:
                break

        return results

    # -----------------------------------------------------------------------
    # Sincronización de usuarios
    # -----------------------------------------------------------------------

    async def sync_user(self, user_id: str, force_reload: bool = False) -> dict:
        """
        Sincroniza un usuario desde MongoDB al caché de vectorizadores.
        Si force_reload=True, recarga el documento desde DB aunque ya exista.
        """
        if not force_reload and user_id in self.vectorizers._user_cache:
            return {"status": "already_synced", "user_id": user_id}

        await self._sync_single_user(user_id)

        if user_id in self.vectorizers._user_cache:
            return {"status": "synced", "user_id": user_id}
        else:
            return {"status": "not_found", "user_id": user_id}

    async def _sync_single_user(self, user_id: str) -> None:
        """Carga un usuario desde MongoDB y lo añade al caché de vectorizadores."""
        if not self.vectorizers.is_fitted:
            logger.warning("⚠️  Vectorizadores no ajustados; no se puede sincronizar usuario")
            return

        user = await fetch_user(user_id)
        if user:
            self.vectorizers.update_user(user)
            # Actualizar también la lista del fallback si aplica
            if self.state == ModelState.FALLBACK and user_id not in self.fallback_user_ids:
                self.fallback_user_ids.append(user_id)
        else:
            logger.warning(f"⚠️  Usuario {user_id} no encontrado en MongoDB")

    async def sync_all_users(self) -> dict:
        """
        Re-sincroniza todos los usuarios desde MongoDB al caché.
        No re-entrena el modelo; solo actualiza el caché de vectorizadores.
        """
        users = await fetch_all_users()
        synced = 0
        failed = 0

        for user in users:
            try:
                self.vectorizers.update_user(user)
                synced += 1
            except Exception as e:
                logger.error(f"❌ Error sincronizando {user.get('_id')}: {e}")
                failed += 1

        # Reconstruir fallback si está activo
        if self.state == ModelState.FALLBACK:
            result = build_fallback_nn(self.vectorizers)
            if result:
                self.fallback_nn, self.fallback_tfidf, self.fallback_user_ids = result

        logger.info(f"✅ Sync masivo: {synced} OK, {failed} errores")
        return {"users_synced": synced, "users_failed": failed}

    def user_exists(self, user_id: str) -> bool:
        """Verifica si el usuario está en el caché del vectorizador."""
        return user_id in self.vectorizers._user_cache

    # -----------------------------------------------------------------------
    # Estadísticas
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Retorna estadísticas del servicio.
        Consumido por GET /stats → SwipeService.getMLServiceStats() en NestJS.
        """
        return {
            "available": self.state != ModelState.UNINITIALIZED,
            "model_state": self.state.value,
            "model_name": self.model_name,
            "n_users_cached": len(self.vectorizers.get_user_ids()),
            "n_labeled_pairs": self.n_pairs,
            "min_pairs_for_supervised": MIN_PAIRS,
            "metrics": self.metrics,
            "vectorizers_fitted": self.vectorizers.is_fitted,
        }
