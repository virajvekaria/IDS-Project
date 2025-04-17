"""
Index routes for the API.
"""

from typing import List
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.models.database import get_db
from app.api.models.models import IndexStore
from app.api.schemas import schemas
from app.api.services.document_service import DocumentService
import config

router = APIRouter(
    prefix="/indexes",
    tags=["indexes"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=List[schemas.IndexStoreResponse])
def get_indexes(db: Session = Depends(get_db)):
    """
    Get all indexes.
    """
    indexes = DocumentService.get_all_index_stores(db)
    return indexes


@router.get("/{name}", response_model=schemas.IndexStoreResponse)
def get_index(name: str, db: Session = Depends(get_db)):
    """
    Get an index by name.
    """
    index = DocumentService.get_index_store(db, name=name)
    if index is None:
        raise HTTPException(status_code=404, detail="Index not found")
    return index


@router.post("/", response_model=schemas.IndexStoreResponse)
def create_index(index_data: schemas.IndexStoreCreate, db: Session = Depends(get_db)):
    """
    Create a new index.
    """
    # Check if index with the same name already exists
    existing_index = (
        db.query(IndexStore).filter(IndexStore.name == index_data.name).first()
    )
    if existing_index:
        raise HTTPException(
            status_code=400, detail="Index with this name already exists"
        )

    # Create directory structure if it doesn't exist
    indexes_dir = config.INDEXES_DIR
    os.makedirs(indexes_dir, exist_ok=True)

    # Create index paths
    index_prefix = os.path.join(indexes_dir, index_data.name)
    index_path = f"{index_prefix}.index"
    metadata_path = f"{index_prefix}_metadata.json"

    # Create empty metadata file
    with open(metadata_path, "w") as f:
        f.write("[]")

    # Create index store in database
    index_store = IndexStore(
        name=index_data.name,
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_model=index_data.embedding_model,
        index_type="flat",  # Default to flat index
        use_hybrid=True,
        chunk_count=0,
    )

    db.add(index_store)
    db.commit()
    db.refresh(index_store)

    return index_store
