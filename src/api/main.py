from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import health, ingest, query
from src.observability.prometheus import start_metrics_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запускается один раз при старте FastAPI
    start_metrics_server(port=8001)
    yield
    # Здесь можно добавить cleanup при остановке


app = FastAPI(title="CRAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["chat"])
app.include_router(ingest.router, tags=["ingest"])
