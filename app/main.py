from fastapi import FastAPI
from app.routers import upload, rag

app = FastAPI()

app.include_router(upload.router)
app.include_router(rag.router)
