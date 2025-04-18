"""
Main FastAPI application for the Document Intelligence Search System (DISS).
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.models.database import Base, engine
from app.api.routes import documents, search, conversations, indexes, frontend

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="Document Intelligence Search System (DISS)",
    description="An AI-powered solution for retrieving information from documents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files
app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")

# Include routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(conversations.router)
app.include_router(indexes.router)
app.include_router(frontend.router)


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
