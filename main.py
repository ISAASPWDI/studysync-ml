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
import asyncio
from core.database import get_db

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
    db = get_db()
    try:
        await db.command("ping")
        logger.info("✅ MongoDB conectado")
    except Exception as e:
        logger.warning(f"⚠️ MongoDB ping falló al inicio: {e}")
    async def keep_mongo_alive():
        while True:
            await asyncio.sleep(240)
            try:
                await db.command("ping")
                logger.info("💓 MongoDB keep-alive OK")
            except Exception as e:
                logger.warning(f"⚠️ MongoDB keep-alive falló: {e}")
    
    task = asyncio.create_task(keep_mongo_alive())
    
    logger.info("✅ ML Service listo")
    yield
    
    # Shutdown
    task.cancel()
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


@app.get("/")
async def health():
    return {"status": "ok", "service": "studysync-ml", "version": "2.0.0"}

@app.head("/")  
async def health_head():
    return {}