
from mcp.server.fastmcp import FastMCP
from typing import List, Optional, Dict, Any, Union
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2
import time
import uuid

# from phoenix.otel import register

# tracer_provider = register(
#     project_name="pdf-assistant-streamlit", 
#     endpoint="https://agentopsacc-backend-app.yellowriver-a22b4385.westus.azurecontainerapps.io/v1/traces",
#     auto_instrument=True,
#     headers={
#         "api-key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6OSJ9.DF1jhTBxOcI7cfrL84JE82PHtymsypKCUZ5gUlI0BtY",
#         "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6OSJ9.DF1jhTBxOcI7cfrL84JE82PHtymsypKCUZ5gUlI0BtY"
#     }
# )

##---LANG CHAIN TRACE ##---
# tracer = tracer_provider.get_tracer("pdf-processing-server")
# from openinference.instrumentation.langchain import LangChainInstrumentor
# instrumentor = LangChainInstrumentor()
# instrumentor.instrument(tracer_provider=tracer_provider)

####-----MCP INSTRUMENTATION ----- ##

# from openinference.instrumentation.mcp import MCPInstrumentor
# instrumentor = MCPInstrumentor()
# instrumentor.instrument(tracer_provider=tracer_provider)

mcp = FastMCP(name="pdf_processing_server")

load_dotenv("C:/Tredence/pdf-extraction-mcp/.env")



from vector_store import VectorStore
vector_store = VectorStore()

def clean_text(text: str) -> str:
    """text cleanning function"""
    if not text:
        return ""
    text = text.replace('\uffff', '').replace('\x00', '').replace('\ufeff', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = ' '.join(text.split())
    return text.strip()

from langchain_openai import AzureOpenAIEmbeddings
embedder = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2023-05-15",
)




@mcp.tool()
#@tracer.tool(name="MCP.pdf_loader")
def pdf_loader(pdf_path: str) -> Dict[str, Any]:
    """
    PDF Loader Tool
    Loads all text from a PDF and returns per-page + combined metadata.
    """
    try:
        if not os.path.exists(pdf_path):
            return {"success": False, "error": f"PDF not found: {pdf_path}"}

        reader = PyPDF2.PdfReader(open(pdf_path, 'rb'))
        if not reader.pages:
            return {"success": False, "error": "PDF contains no pages"}

        pages_data, combined_text_list = [], []
        for i, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages_data.append({
                    "page_number": i,
                    "raw_text": text,
                    "char_count": len(text),
                    "word_count": len(text.split())
                })
                combined_text_list.append(text)

        if not pages_data:
            return {"success": False, "error": "No readable text in PDF"}

        total_text = "\n\n".join(combined_text_list)
        return {
            "success": True,
            "document_path": pdf_path,
            "total_pages_processed": len(pages_data),
            "pages_data": pages_data,
            "total_chars": len(total_text),
            "total_words": len(total_text.split()),
            "combined_text": total_text
        }

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}

    
from langchain.text_splitter import RecursiveCharacterTextSplitter

@mcp.tool()
#@tracer.tool(name="MCP.text_chunker")
def text_chunker(document_text: str, chunk_size: int=1000 , chunk_overlap: int=200 , doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Text Chunker Tool
    Splits cleaned document text into chunks with overlap using LangChain's RecursiveCharacterTextSplitter.
    """
    try:
        text = clean_text(document_text)
        if not text:
            return {"success": False, "error": "No valid text provided"}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

        text_chunks = splitter.split_text(text)
        chunks_data = []

        for idx, chunk in enumerate(text_chunks):
            chunk_id = f"{doc_id or 'doc'}_{idx:04d}"
            chunks_data.append({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "text": chunk.strip(),
                "char_count": len(chunk),
                "word_count": len(chunk.split()),
                "start_pos": None, 
                "end_pos": None,    
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "timestamp": str(time.time())
                }
            })

        return {
            "success": True,
            "total_chunks": len(chunks_data),
            "chunks": chunks_data,
            "source_text_length": len(text),
            "chunk_config": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        }

    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}

@mcp.tool()
def embedding_generator(text_chunks: List[str], chunk_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Embedding Generator Tool
    Input: List of text chunks (with optional chunk IDs)
    Output: List of embeddings (vectors) using Azure OpenAI
    """
    try:
        if not text_chunks:
            return {"error": "No chunks provided", "success": False}
    
        valid_chunks = []
        valid_ids = []
        
        for i, chunk in enumerate(text_chunks):
            if isinstance(chunk, str):
                cleaned = clean_text(chunk)
                if len(cleaned) > 10:
                    valid_chunks.append(cleaned)
                    chunk_id = chunk_ids[i] if chunk_ids and i < len(chunk_ids) else f"chunk_{i:04d}"
                    valid_ids.append(chunk_id)
        
        if not valid_chunks:
            return {"error": "No valid chunks after cleaning", "success": False}
        
        print(f"[DEBUG] Generating embeddings for {len(valid_chunks)} chunks using Azure OpenAI")
        
      
        vectors = embedder.embed_documents(valid_chunks)
        
        if not vectors:
            return {"error": "Failed to generate embeddings", "success": False}
        
    
        embeddings_data = []
        for i, (chunk_id, chunk, vector) in enumerate(zip(valid_ids, valid_chunks, vectors)):
            embeddings_data.append({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "text": chunk,
                "embedding": vector,
                "vector_dimension": len(vector),
                "text_length": len(chunk)
            })
        
        result = {
            "success": True,
            "embeddings_count": len(embeddings_data),
            "vector_dimension": len(vectors[0]) if vectors else 0,
            "model_used": os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            "embeddings": embeddings_data
        }
        
        print(f"[DEBUG] Generated {len(vectors)} embeddings with dimension {len(vectors[0])}")
        return result
        
    except Exception as e:
        return {"error": f"Failed to generate embeddings: {str(e)}", "success": False}

@mcp.tool()
def vectordb_writer(doc_id: str, embeddings_data: List[Dict[str, Any]], metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    VectorDB Writer Tool
    Input: Document ID, embeddings data (chunk IDs, text, embeddings), and metadata
    Output: Indexes/stores into ChromaDB with success status
    """
    try:
        if not doc_id or not embeddings_data:
            return {"error": "Missing doc_id or embeddings_data", "success": False}
        
        # Extract data for ChromaDB storage
        chunk_ids = []
        texts = []
        vectors = []
        chunk_metadatas = []
        
        for item in embeddings_data:
            if all(key in item for key in ["chunk_id", "text", "embedding"]):
                chunk_ids.append(item["chunk_id"])
                texts.append(item["text"])
                vectors.append(item["embedding"])
                
               
                chunk_metadata = {
                    "doc_id": doc_id,
                    "chunk_id": item["chunk_id"],
                    "chunk_index": item.get("chunk_index", 0),
                    "text_length": item.get("text_length", len(item["text"])),
                    "vector_dimension": item.get("vector_dimension", len(item["embedding"])),
                    "timestamp": str(time.time()),
                    "source": "pdf_processing",
                    **(metadata or {})
                }
                chunk_metadatas.append(chunk_metadata)
        
        if not chunk_ids:
            return {"error": "No valid embeddings data found", "success": False}
        
        print(f"[DEBUG] Storing {len(chunk_ids)} chunks in ChromaDB for document '{doc_id}'")
        
        success = vector_store.store_document(doc_id, texts, vectors, metadata or {})
        
        if success:
            result = {
                "success": True,
                "doc_id": doc_id,
                "chunks_stored": len(chunk_ids),
                "vector_dimension": len(vectors[0]) if vectors else 0,
                "storage_backend": "ChromaDB",
                "stored_chunk_ids": chunk_ids
            }
            print(f"[DEBUG] Successfully stored document '{doc_id}' with {len(chunk_ids)} chunks")
            return result
        else:
            return {"error": f"Failed to store document '{doc_id}' in ChromaDB", "success": False}
            
    except Exception as e:
        return {"error": f"VectorDB storage failed: {str(e)}", "success": False}


@mcp.tool()
def list_stored_documents() -> List[str]:
    """List all documents stored in ChromaDB"""
    return vector_store.list_documents()

@mcp.tool()
def search_similar_chunks(doc_id: str, query_embedding: List[float], top_k: int = 5) -> List[str]:
    """Search for similar chunks using query embedding"""
    if not doc_id or not query_embedding:
        return ["Error: Missing doc_id or query_embedding"]
    
    top_k = max(1, min(top_k, 50))
    results = vector_store.query_similar(doc_id, query_embedding, top_k)
    return results if results else [f"No results found for document '{doc_id}'"]

@mcp.tool()
def get_query_embedding(query_text: str) -> Dict[str, Any]:
    """Generate embedding for a single query text"""
    if not query_text:
        return {"error": "No query text provided", "success": False}
    
    cleaned_query = clean_text(query_text)
    if not cleaned_query:
        return {"error": "Invalid query text", "success": False}
    
    try:
        embedding = embedder.embed_query(cleaned_query)
        return {
            "success": True,
            "query": cleaned_query,
            "embedding": embedding,
            "dimension": len(embedding)
        }
    except Exception as e:
        return {"error": f"Failed to generate query embedding: {str(e)}", "success": False}

if __name__ == "__main__":
    print("[DEBUG] Starting PDF processing")
    print("[DEBUG] Tools: pdf_loader, text_chunker, embedding_generator, vectordb_writer")
    mcp.run(transport="stdio")
