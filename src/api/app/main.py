# src/api/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from packages.core_logic.clients import setup_clients, shutdown_clients
from .v1.routes import chat as chat_v1_router 

@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_clients()
    yield
    await shutdown_clients()

app = FastAPI(title="AI Hybrid Travel Assistant API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_v1_router.router, prefix="/api/v1", tags=["v1 - Chat"])

@app.get("/", tags=["Root"])
def read_root():
    return {"status": "Welcome to Vietnam Travel endpoint"}