from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.core.db import create_db_and_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(title="Coin Detection API", version="1.0", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")