"""
core/config.py
==============
Carga el .env UNA SOLA VEZ al importar este módulo.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

MONGODB_URI: str = (
    os.getenv("MONGODB_URI")
    or os.getenv("MONGO_URI")
    or "mongodb://localhost:27017"
)
DATABASE_NAME: str = os.getenv("DATABASE_NAME") or os.getenv("MONGO_DB") or "studysync"

# Mínimo de pares etiquetados para activar el modelo supervisado.
# Configurable desde .env: MIN_PAIRS_FOR_TRAINING=10
# Recomendado: sube esto a 50+ cuando tengas más historial de matches reales.
MIN_PAIRS: int = int(os.getenv("MIN_PAIRS_FOR_TRAINING", "10"))

# Folds de validación cruzada — con pocos datos, 3 es más estable que 5
CV_FOLDS: int = int(os.getenv("CV_FOLDS", "3"))