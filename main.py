"""
StudySync ML Service - main.py
"""

# config.py carga el .env ANTES que cualquier otra cosa
from core.config import MONGODB_URI  # noqa: F401 — importar dispara load_dotenv

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import recommendations, users, admin
from core.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("studysync.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Iniciando StudySync ML Service...")
    registry = ModelRegistry.get_instance()
    await registry.load_from_disk()
    logger.info("✅ ML Service listo")
    yield
    logger.info("🛑 Apagando ML Service...")


app = FastAPI(
    title="StudySync ML Service",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
raw_origins = os.getenv(
    "CORS_ORIGINS",
    '["https://studysync-backend-81wi.onrender.com", "http://localhost:3000"]',
)
try:
    cors_origins = json.loads(raw_origins)
except Exception:
    cors_origins = [raw_origins]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations.router)
app.include_router(users.router)
app.include_router(admin.router)


@app.get("/stats")
async def get_stats():
    return ModelRegistry.get_instance().get_stats()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "studysync-ml", "version": "2.0.0"}