from mcp.server.fastmcp import FastMCP
from typing import List, Optional
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from typing import List, Optional
import json
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings

import os
from dotenv import load_dotenv


# env_path = Path(__file__).parent.parent / '.env'
# load_dotenv(env_path)

load_dotenv("C:/Tredence/pdf-extraction-mcp/.env")

try:
    from vector_store import VectorStore
except ImportError:
    print("[WARNING] Could not import VectorStore, using dummy implementation")
    class VectorStore:
        def store_document(self, doc_id, chunks, vectors, metadata=None):
            return f"Document '{doc_id}' stored with {len(chunks)} chunks."
        def query_similar(self, doc_id, embedding, top_k):
            return ["No vector store available"]
        def list_documents(self):
            return []

# Import your PDF extractor class
from pdf_extractor import PDFExtractor

mcp = FastMCP(name="combined_document_processor")

# Initialize PDF extractor
extractor = PDFExtractor()
vector_store = VectorStore()  # Instantiate the vector store


@mcp.tool()
def extract_pdf_contents(pdf_path: str, pages: Optional[str] = None) -> str:
    """
    Extracts text from a PDF file.
    Args:
        pdf_path: Path to the PDF file.
        pages: Comma-separated page numbers (optional).
    Returns:
        Extracted text as a string.
    """
    print(f"[DEBUG] extract_pdf_contents called with: {pdf_path}")
    try:
        result = extractor.extract_content(pdf_path, pages)
        print(f"[DEBUG] PDF extraction successful, length: {len(result)}")
        return result
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        raise

@mcp.tool()
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Splits the input text into chunks of approximately chunk_size characters.
    Args:
        text: The input text to chunk.
        chunk_size: The maximum size of each chunk (default: 500).
    Returns:
        List of text chunks (as strings).
    """
    return [str(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size)]

@mcp.tool()
def embed_chunks(text_chunks: List[str], doc_id: str = None) -> List[str]:
    """
    Generates vector embeddings for a list of text chunks using Azure OpenAI embeddings.
    If doc_id is provided, also stores the embeddings and chunks in the vector DB.
    Args:
        text_chunks: List of text chunks.
        doc_id: Optional document ID for storage.
    Returns:
        List of embedding vectors as JSON strings (one per chunk), or a confirmation message if stored.
    """
    try:
        embedder = AzureOpenAIEmbeddings(
        model="text-embedding-3-small" ,  # Example: "text-embedding-3-large"
        deployment = "embedding-model",
        azure_endpoint="https://llm-mlops-openai.openai.azure.com/",
        api_key="90858d0b603c4323a2df07d8064dbcf6",
        openai_api_version= "2024-02-01",
    )

        vectors = embedder.embed_documents(text_chunks)
        
        if doc_id:
            # Store in vector DB using the store_embeddings function
            store_result = store_embeddings(doc_id, text_chunks, vectors)
            return [store_result]
        
        return [json.dumps(vec) for vec in vectors]
    except Exception as e:
        return [f"Error: {str(e)}"]

@mcp.tool()
def search_embeddings(doc_id: str, query_embedding: List[float], top_k: int = 5) -> List[str]:
    """
    Search for similar text chunks using embedding similarity.
    Args:
        doc_id: Document identifier
        query_embedding: Query embedding vector
        top_k: Number of similar chunks to return
    Returns:
        List of most similar text chunks
    """
    try:
        # Use your existing VectorStore class
        results = vector_store.query_similar(doc_id, query_embedding, top_k)
        return results if results else []
    except Exception as e:
        print(f"[ERROR] Vector search failed: {e}")
        return [f"Error searching embeddings: {str(e)}"]

@mcp.tool()
def store_embeddings(doc_id: str, chunks: List[str], vectors: List[List[float]], metadata: dict = None) -> str:
    """Store document chunks and their embeddings."""
    try:
        # Fix: Provide default metadata if None
        if metadata is None:
            metadata = {"source": "pdf_processing", "doc_id": doc_id}
        
        vector_store.store_document(doc_id, chunks, vectors, metadata)
        return f"Document '{doc_id}' stored with {len(chunks)} chunks."
    except Exception as e:
        print(f"[ERROR] Vector storage failed: {e}")
        return f"Error storing embeddings: {str(e)}"

@mcp.tool()
def list_stored_documents() -> List[str]:
    """
    List all documents stored in the vector database.
    Returns:
        List of document IDs
    """
    try:
        return vector_store.list_documents()
    except Exception as e:
        print(f"[ERROR] Failed to list documents: {e}")
        return [f"Error listing documents: {str(e)}"]
    


@mcp.tool()
def process_pdf_to_embeddings(pdf_path: str, chunk_size: int = 500, pages: Optional[str] = None) -> dict:
    """
    Complete pipeline: Extract PDF content, chunk it, and generate embeddings.
    Args:
        pdf_path: Path to the PDF file.
        chunk_size: The maximum size of each chunk (default: 500).
        pages: Comma-separated page numbers (optional).
    Returns:
        Dictionary containing chunks and their embeddings.
    """
    try:
        # Step 1: Extract PDF content
        text = extract_pdf_contents(pdf_path, pages)
        
        # Step 2: Chunk the text
        chunks = chunk_text(text, chunk_size)
        
        # Step 3: Generate embeddings
        embeddings = embed_chunks(chunks)
        
        return {
            "original_text": text,
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("[DEBUG] Starting PDF processing server...")
    
    

    
    mcp.run(transport="stdio")