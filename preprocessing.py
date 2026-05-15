"""
preprocessing.py
================
Preprocesamiento de perfiles académicos y construcción de features
para el pipeline de ML supervisado.

Responsabilidades:
  1. Extraer texto plano de los campos anidados del documento MongoDB
  2. Ajustar y transformar vectorizadores TF-IDF (skills, interests, goals)
  3. Calcular las 6 features de similitud para cada par (user_a, user_b)

Las 6 features del vector de entrada al clasificador:
  [0] cosine_skills     – similitud coseno entre TF-IDF de skills
  [1] cosine_interests  – similitud coseno entre TF-IDF de interests
  [2] cosine_goals      – similitud coseno entre TF-IDF de goals
  [3] jaccard_skills    – Jaccard entre arrays de skills
  [4] common_skills     – número de skills en común (intersección)
  [5] academic_diff     – |semester_a - semester_b| normalizado (0-1)
"""

import logging
import re
from typing import Any

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("studysync.preprocessing")


# ---------------------------------------------------------------------------
# Extracción de texto desde documentos MongoDB
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Limpia y normaliza texto: minúsculas, sin caracteres especiales."""
    text = text.lower().strip()
    text = re.sub(r"[^a-záéíóúñüa-z0-9\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_skills_text(user: dict) -> str:
    """
    Extrae skills técnicos del usuario como texto plano.
    Soporta tanto array de strings como array de objetos {name: ...}.
    """
    raw = user.get("skills", {})
    items: list = []

    if isinstance(raw, dict):
        items = raw.get("technical", []) or raw.get("skills", []) or []
    elif isinstance(raw, list):
        items = raw

    tokens = []
    for item in items:
        if isinstance(item, str):
            tokens.append(item)
        elif isinstance(item, dict):
            tokens.append(item.get("name", "") or item.get("skill", ""))

    return _normalize_text(" ".join(filter(None, tokens))) or "none"


def extract_interests_text(user: dict) -> str:
    """
    Extrae intereses del usuario como texto plano.
    Busca en skills.interests, profile.interests e interests directamente.
    """
    skills_dict = user.get("skills", {})
    items: list = []

    if isinstance(skills_dict, dict):
        items = skills_dict.get("interests", []) or []
    if not items:
        items = user.get("interests", []) or []
    if not items:
        profile = user.get("profile", {}) or {}
        items = profile.get("interests", []) or []

    tokens = [i if isinstance(i, str) else i.get("name", "") for i in items]
    return _normalize_text(" ".join(filter(None, tokens))) or "none"


def extract_goals_text(user: dict) -> str:
    """
    Extrae objetivos académicos del usuario como texto plano.
    Combina objectives.primary, objectives.secondary y objectives.description.
    """
    obj = user.get("objectives", {}) or {}
    parts = []

    primary = obj.get("primary", []) or []
    secondary = obj.get("secondary", []) or []

    for lst in [primary, secondary]:
        for item in lst:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("goal", "") or item.get("name", ""))

    description = obj.get("description", "") or ""
    if description:
        parts.append(description)

    return _normalize_text(" ".join(filter(None, parts))) or "none"


def extract_skills_set(user: dict) -> set[str]:
    """Retorna el conjunto de skills técnicos para cálculo de Jaccard."""
    raw = user.get("skills", {})
    items: list = []

    if isinstance(raw, dict):
        items = raw.get("technical", []) or []
    elif isinstance(raw, list):
        items = raw

    result = set()
    for item in items:
        if isinstance(item, str) and item.strip():
            result.add(item.lower().strip())
        elif isinstance(item, dict):
            name = item.get("name", "") or item.get("skill", "")
            if name:
                result.add(name.lower().strip())
    return result


def extract_semester(user: dict) -> float:
    """Extrae el semestre académico como float. Retorna 0.0 si no existe."""
    profile = user.get("profile", {}) or {}
    sem = profile.get("semester") or user.get("semester")
    try:
        return float(sem)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Vectorizadores TF-IDF
# ---------------------------------------------------------------------------

class AcademicVectorizers:
    """
    Encapsula los tres vectorizadores TF-IDF (skills, interests, goals).
    Se ajustan con fit() sobre el corpus completo de usuarios y se persisten
    junto al modelo principal en el ModelRegistry.

    Parámetros TF-IDF:
      - ngram_range=(1,2): unigramas + bigramas para capturar frases cortas
      - min_df=1:          incluir términos que aparezcan al menos 1 vez
      - sublinear_tf=True: aplicar log en frecuencia de término
    """

    TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=1, sublinear_tf=True)

    def __init__(self):
        self.skills_vec = TfidfVectorizer(**self.TFIDF_PARAMS)
        self.interests_vec = TfidfVectorizer(**self.TFIDF_PARAMS)
        self.goals_vec = TfidfVectorizer(**self.TFIDF_PARAMS)
        self.is_fitted = False
        self._user_cache: dict[str, dict] = {}  # user_id → feature dict

    def fit(self, users: list[dict]) -> None:
        """
        Ajusta los tres vectorizadores sobre el corpus de usuarios.
        Construye y cachea el diccionario de features por usuario.
        """
        skills_corpus = [extract_skills_text(u) for u in users]
        interests_corpus = [extract_interests_text(u) for u in users]
        goals_corpus = [extract_goals_text(u) for u in users]

        self.skills_vec.fit(skills_corpus)
        self.interests_vec.fit(interests_corpus)
        self.goals_vec.fit(goals_corpus)
        self.is_fitted = True

        # Cachear vectores transformados por user_id
        self._user_cache.clear()
        for user, sk, in_, go in zip(users, skills_corpus, interests_corpus, goals_corpus):
            uid = str(user["_id"])
            self._user_cache[uid] = {
                "skills_text": sk,
                "interests_text": in_,
                "goals_text": go,
                "skills_set": extract_skills_set(user),
                "semester": extract_semester(user),
            }

        logger.info(
            f"✅ Vectorizadores ajustados sobre {len(users)} usuarios. "
            f"Vocabulario: skills={len(self.skills_vec.vocabulary_)}, "
            f"interests={len(self.interests_vec.vocabulary_)}, "
            f"goals={len(self.goals_vec.vocabulary_)}"
        )

    def update_user(self, user: dict) -> None:
        """
        Actualiza el caché para un usuario individual (después de sync).
        No re-ajusta los vectorizadores; usa transform sobre el corpus existente.
        """
        uid = str(user["_id"])
        self._user_cache[uid] = {
            "skills_text": extract_skills_text(user),
            "interests_text": extract_interests_text(user),
            "goals_text": extract_goals_text(user),
            "skills_set": extract_skills_set(user),
            "semester": extract_semester(user),
        }

    def get_user_ids(self) -> list[str]:
        """Retorna la lista de user_ids actualmente en caché."""
        return list(self._user_cache.keys())

    # -----------------------------------------------------------------------
    # Cálculo de features para pares
    # -----------------------------------------------------------------------

    def compute_pair_features(
        self,
        uid_a: str,
        uid_b: str,
    ) -> np.ndarray | None:
        """
        Calcula el vector de 6 features para el par (uid_a, uid_b).

        Retorna None si alguno de los usuarios no está en el caché.

        Features:
          [0] cosine_skills
          [1] cosine_interests
          [2] cosine_goals
          [3] jaccard_skills
          [4] common_skills (normalizado al máximo observado; se normaliza
                             externamente con MinMaxScaler antes del modelo)
          [5] academic_diff  = |sem_a - sem_b| / 12  (máx ~12 semestres)
        """
        if not self.is_fitted:
            raise RuntimeError("AcademicVectorizers no está ajustado. Llama fit() primero.")

        cache_a = self._user_cache.get(uid_a)
        cache_b = self._user_cache.get(uid_b)

        if cache_a is None or cache_b is None:
            return None

        # --- TF-IDF cosine similarities ---
        def _cosine(vec: TfidfVectorizer, text_a: str, text_b: str) -> float:
            mat = vec.transform([text_a, text_b])
            sim = cosine_similarity(mat[0:1], mat[1:2])
            return float(sim[0, 0])

        cos_skills = _cosine(self.skills_vec, cache_a["skills_text"], cache_b["skills_text"])
        cos_interests = _cosine(self.interests_vec, cache_a["interests_text"], cache_b["interests_text"])
        cos_goals = _cosine(self.goals_vec, cache_a["goals_text"], cache_b["goals_text"])

        # --- Jaccard de skills ---
        set_a: set = cache_a["skills_set"]
        set_b: set = cache_b["skills_set"]
        union = set_a | set_b
        intersection = set_a & set_b
        jaccard = len(intersection) / len(union) if union else 0.0
        common = float(len(intersection))

        # --- Diferencia académica normalizada ---
        sem_a = cache_a["semester"]
        sem_b = cache_b["semester"]
        academic_diff = abs(sem_a - sem_b) / 12.0  # normalizado 0-1

        return np.array(
            [cos_skills, cos_interests, cos_goals, jaccard, common, academic_diff],
            dtype=np.float32,
        )

    def build_feature_matrix(
        self,
        pairs: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """
        Construye la matriz X y el vector y para todos los pares etiquetados.

        Filtra pares donde algún usuario no esté en el caché.
        Retorna (X, y, valid_pairs).
        """
        X_rows = []
        y_rows = []
        valid_pairs = []

        for pair in pairs:
            features = self.compute_pair_features(pair["user_a"], pair["user_b"])
            if features is None:
                continue
            X_rows.append(features)
            y_rows.append(pair["label"])
            valid_pairs.append(pair)

        if not X_rows:
            return np.empty((0, 6)), np.empty(0), []

        X = np.vstack(X_rows)
        y = np.array(y_rows, dtype=int)

        logger.info(f"📐 Matriz de features: {X.shape}, labels: {y.shape}")
        return X, y, valid_pairs

    def compute_candidate_features(
        self,
        target_uid: str,
        candidate_uids: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Calcula features entre target_uid y cada candidato.
        Retorna (X_candidates, valid_candidate_uids).
        Usado en el endpoint /recommendations para scoring.
        """
        rows = []
        valid_uids = []

        for cuid in candidate_uids:
            features = self.compute_pair_features(target_uid, cuid)
            if features is not None:
                rows.append(features)
                valid_uids.append(cuid)

        if not rows:
            return np.empty((0, 6)), []

        return np.vstack(rows), valid_uids
