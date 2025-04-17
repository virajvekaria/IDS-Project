# Document Intelligence Search System (DISS)

A comprehensive document intelligence system that processes PDF documents, extracts text, creates vector embeddings, and provides a conversational interface for querying document content.

## Features

- **Document Processing**: Automatically processes PDF documents, extracting text and metadata
- **Vector Search**: Creates embeddings for document chunks and enables semantic search
- **Conversational Interface**: Chat with your documents using a natural language interface
- **GPU Acceleration**: Utilizes GPU for faster processing and inference
- **Adaptive Similarity Thresholds**: Dynamically adjusts similarity thresholds based on query complexity
- **Hybrid Retrieval**: Combines semantic and keyword-based search for better results
- **Chunk Overlap**: Ensures context continuity between document chunks

## System Architecture

The system consists of the following components:

1. **Document Processor**: Extracts text from PDF documents
2. **Chunking Engine**: Splits documents into manageable chunks with overlap
3. **Indexing Service**: Creates vector embeddings for document chunks
4. **Retrieval Engine**: Performs semantic search on document embeddings
5. **Chat Interface**: Provides a conversational interface for document queries
6. **Web API**: RESTful API for document management and search

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- Ollama installed for LLM support

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/diss.git
   cd diss
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Ollama and pull the deepseek-r1:7b model:
   ```
   ollama pull deepseek-r1:7b
   ```

4. Initialize the system with default documents:
   ```
   python init_documents.py
   ```

5. Run the application:
   ```
   python run.py
   ```

6. Access the web interface at http://localhost:8000

## Usage

### Adding Documents

1. Place PDF documents in the `PDFs` folder
2. Run `python init_documents.py` to process and index the documents
3. Alternatively, use the web interface to upload documents

### Querying Documents

1. Navigate to http://localhost:8000/chat
2. Enter your query in the chat interface
3. View the response generated based on the document content

## Configuration

The system can be configured by modifying the `config.py` file:

- `DEFAULT_LLM_MODEL`: The default Ollama model to use (default: "deepseek-r1:7b")
- `CHUNK_SIZE`: The size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: The overlap between chunks (default: 200)
- `SIMILARITY_THRESHOLD`: The default similarity threshold (default: 0.7)
- `TOP_K`: The number of chunks to retrieve (default: 5)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project uses [Ollama](https://github.com/ollama/ollama) for LLM integration
- Vector search is powered by [FAISS](https://github.com/facebookresearch/faiss)
- Document processing uses [PyPDF](https://github.com/py-pdf/pypdf)
