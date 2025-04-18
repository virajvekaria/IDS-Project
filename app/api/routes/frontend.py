"""
Frontend routes for the Document Intelligence Search System (DISS).
"""

import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Set up templates
templates = Jinja2Templates(directory="app/frontend/templates")

# Path to React build directory
REACT_BUILD_DIR = "app/frontend/react/dist"
REACT_INDEX_HTML = os.path.join(REACT_BUILD_DIR, "index.html")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the React frontend or fallback to the template.
    """
    if os.path.exists(REACT_INDEX_HTML):
        return FileResponse(REACT_INDEX_HTML)
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/chat", response_class=HTMLResponse)
async def chat(request: Request, conversation_id: int = None):
    """
    Serve the React frontend or fallback to the template.
    """
    if os.path.exists(REACT_INDEX_HTML):
        return FileResponse(REACT_INDEX_HTML)
    return templates.TemplateResponse(
        "chat.html", {"request": request, "conversation_id": conversation_id}
    )

@router.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    """
    Serve the React frontend or fallback to the template.
    """
    if os.path.exists(REACT_INDEX_HTML):
        return FileResponse(REACT_INDEX_HTML)
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/bundle.js")
async def serve_bundle():
    """
    Serve the React bundle.js file.
    """
    bundle_path = os.path.join(REACT_BUILD_DIR, "bundle.js")
    if os.path.exists(bundle_path):
        return FileResponse(bundle_path)
    raise HTTPException(status_code=404, detail="Bundle not found")
