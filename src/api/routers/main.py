from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import health, ingest, query

app = FastAPI(title="CRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["chat"])
app.include_router(ingest.router, tags=["ingest"])
