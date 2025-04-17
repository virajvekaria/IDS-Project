"""
Run script for the Document Intelligence Search System (DISS).
"""

import uvicorn
import config
import threading
import time
from init_documents import init_documents


def initialize_documents_async():
    # Wait a few seconds for the application to start
    time.sleep(5)
    # Initialize documents
    init_documents()


if __name__ == "__main__":
    # Start document initialization in a separate thread
    init_thread = threading.Thread(target=initialize_documents_async)
    init_thread.daemon = True
    init_thread.start()

    # Start the application
    print("Starting Document Intelligence Search System (DISS)...")
    print("The application will automatically process documents from the PDFs folder.")
    uvicorn.run(
        "app.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=True,
    )
