"""
Main FastAPI application for the Document Intelligence Search System (DISS).
"""
import os
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.api.models.database import Base, engine, get_db
from app.api.routes import documents, search, conversations, indexes

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="Document Intelligence Search System (DISS)",
    description="An AI-powered solution for retrieving information from documents",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/frontend/templates")

# Include routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(conversations.router)
app.include_router(indexes.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    """
    Render the index page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request, conversation_id: int = None, db: Session = Depends(get_db)):
    """
    Render the chat page.
    """
    return templates.TemplateResponse("chat.html", {"request": request, "conversation_id": conversation_id})


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
