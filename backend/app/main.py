from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import dashboard, datasets, experiments, meta, models, projects, registry_api, runs, system
from .core import db
from .training.orchestrator import orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    if db.engine is None:
        db.init_db()
    orchestrator.start()
    yield


app = FastAPI(title="VoxTune Fine-Tuning Studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
for r in (system.router, meta.router, dashboard.router, models.router, projects.router, datasets.router, runs.router,
          experiments.router, registry_api.router):
    app.include_router(r)
