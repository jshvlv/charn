from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import dataset, health, model, predict
from app.model.registry import init_from_disk as init_model_from_disk

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "service.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()
register_exception_handlers(app)


@app.on_event("startup")
def _startup_load_model() -> None:
    init_model_from_disk()


app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(predict.router)
app.include_router(model.router)

