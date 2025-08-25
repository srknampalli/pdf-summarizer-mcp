# vector_store.py - Shared ChromaDB Vector Store for both servers
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import os

class VectorStore:
    """
    Shared ChromaDB vector store that can be used by both MCP servers.
    This ensures consistent data access across the entire RAG system.
    """
    
    _instance = None
    _client = None
    
    def __new__(cls):
        """Singleton pattern to ensure single ChromaDB instance"""
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistent storage"""
        if self._client is None:
            # Create vector_db directory if it doesn't exist
            db_path = "./vector_db"
            os.makedirs(db_path, exist_ok=True)
            
            # Initialize ChromaDB with persistent storage
            self._client = chromadb.PersistentClient(path=db_path)
            print(f"[DEBUG] ChromaDB initialized with persistent storage at {db_path}")
    
    @property
    def client(self):
        """Get ChromaDB client instance"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def store_document(self, doc_id: str, chunks: List[str], vectors: List[List[float]], metadata: Dict = None) -> bool:
        """
        Store document chunks and embeddings in ChromaDB
        
        Args:
            doc_id: Unique document identifier
            chunks: List of text chunks
            vectors: List of embedding vectors
            metadata: Additional metadata for the document
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"[DEBUG] Storing document '{doc_id}' with {len(chunks)} chunks")
            
            # Validate inputs
            if not doc_id or not chunks or not vectors:
                print("[ERROR] Missing required parameters for document storage")
                return False
            
            if len(chunks) != len(vectors):
                print(f"[ERROR] Chunks count ({len(chunks)}) doesn't match vectors count ({len(vectors)})")
                return False
            
            # Create or get collection for this document
            collection_name = f"doc_{doc_id}"
            collection = self.client.get_or_create_collection(name=collection_name)
            
            # Prepare data for ChromaDB
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Prepare metadata for each chunk
            chunk_metadatas = []
            base_metadata = metadata or {}
            for i in range(len(chunks)):
                chunk_metadata = {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_length": len(chunks[i]),
                    **base_metadata
                }
                chunk_metadatas.append(chunk_metadata)
            
            # Clear existing data for this document (if any)
            try:
                existing_ids = collection.get()['ids']
                if existing_ids:
                    collection.delete(ids=existing_ids)
                    print(f"[DEBUG] Cleared {len(existing_ids)} existing chunks for document '{doc_id}'")
            except Exception as e:
                print(f"[DEBUG] No existing data to clear: {e}")
            
            # Add new data to collection
            collection.add(
                embeddings=vectors,
                documents=chunks,
                metadatas=chunk_metadatas,
                ids=ids
            )
            
            print(f"[DEBUG] Successfully stored {len(chunks)} chunks in ChromaDB collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"[ERROR] ChromaDB storage failed: {e}")
            return False
    
    def query_similar(self, doc_id: str, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        Query similar chunks from ChromaDB
        
        Args:
            doc_id: Document identifier to search within
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List[str]: List of similar document chunks
        """
        try:
            collection_name = f"doc_{doc_id}"
            
            # Get collection
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception as e:
                print(f"[ERROR] Collection '{collection_name}' not found: {e}")
                return []
            
            # Query for similar chunks
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 50)  # Limit to reasonable number
            )
            
            # Extract documents from results
            documents = results['documents'][0] if results['documents'] and len(results['documents']) > 0 else []
            
            print(f"[DEBUG] ChromaDB query for '{doc_id}' returned {len(documents)} results")
            return documents
            
        except Exception as e:
            print(f"[ERROR] ChromaDB query failed: {e}")
            return []
    
    def list_documents(self) -> List[str]:
        """
        List all stored documents
        
        Returns:
            List[str]: List of document information strings
        """
        try:
            collections = self.client.list_collections()
            docs = []
            
            for collection in collections:
                if collection.name.startswith("doc_"):
                    doc_id = collection.name[4:]  # Remove "doc_" prefix
                    
                    try:
                        # Get collection info
                        coll = self.client.get_collection(name=collection.name)
                        count = coll.count()
                        
                        # Try to get vector dimension from first embedding
                        if count > 0:
                            sample = coll.peek(limit=1)
                            if sample['embeddings'] and len(sample['embeddings']) > 0:
                                dimension = len(sample['embeddings'][0])
                                docs.append(f"{doc_id} ({count} chunks, {dimension}D vectors)")
                            else:
                                docs.append(f"{doc_id} ({count} chunks)")
                        else:
                            docs.append(f"{doc_id} (empty)")
                            
                    except Exception as e:
                        docs.append(f"{doc_id} (error: {str(e)})")
            
            print(f"[DEBUG] Found {len(docs)} documents in ChromaDB")
            return docs
            
        except Exception as e:
            print(f"[ERROR] ChromaDB list documents failed: {e}")
            return [f"Error listing documents: {str(e)}"]
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks
        
        Args:
            doc_id: Document identifier to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection_name = f"doc_{doc_id}"
            
            try:
                collection = self.client.get_collection(name=collection_name)
                # Delete the entire collection
                self.client.delete_collection(name=collection_name)
                print(f"[DEBUG] Successfully deleted document '{doc_id}' and its collection")
                return True
                
            except Exception as e:
                print(f"[ERROR] Document '{doc_id}' not found for deletion: {e}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to delete document '{doc_id}': {e}")
            return False
    
    def get_document_stats(self, doc_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific document
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dict: Document statistics
        """
        try:
            collection_name = f"doc_{doc_id}"
            
            try:
                collection = self.client.get_collection(name=collection_name)
                count = collection.count()
                
                if count > 0:
                    # Get sample to determine vector dimension
                    sample = collection.peek(limit=1)
                    dimension = len(sample['embeddings'][0]) if sample['embeddings'] else 0
                    
                    return {
                        "doc_id": doc_id,
                        "chunk_count": count,
                        "vector_dimension": dimension,
                        "status": "available"
                    }
                else:
                    return {
                        "doc_id": doc_id,
                        "chunk_count": 0,
                        "status": "empty"
                    }
                    
            except Exception as e:
                return {
                    "doc_id": doc_id,
                    "status": "not_found",
                    "error": str(e)
                }
                
        except Exception as e:
            return {
                "doc_id": doc_id,
                "status": "error",
                "error": str(e)
            }

# Create global instance for import
vector_store = VectorStore()

# Convenience functions for direct use
def store_document_embeddings(doc_id: str, chunks: List[str], vectors: List[List[float]], metadata: Dict = None) -> bool:
    """Convenience function to store document embeddings"""
    return vector_store.store_document(doc_id, chunks, vectors, metadata)

def search_similar_chunks(doc_id: str, query_embedding: List[float], top_k: int = 5) -> List[str]:
    """Convenience function to search similar chunks"""
    return vector_store.query_similar(doc_id, query_embedding, top_k)

def list_all_documents() -> List[str]:
    """Convenience function to list all documents"""
    return vector_store.list_documents()

def get_document_info(doc_id: str) -> Dict[str, Any]:
    """Convenience function to get document info"""
    return vector_store.get_document_stats(doc_id)

if __name__ == "__main__":
    """Test the vector store functionality"""
    print("🧪 Testing ChromaDB Vector Store...")
    
    # Test initialization
    vs = VectorStore()
    print("✅ Vector store initialized")
    
    # Test listing documents
    docs = vs.list_documents()
    print(f"📋 Found {len(docs)} existing documents:")
    for doc in docs:
        print(f"  - {doc}")
    
    print("🏁 Vector store test completed")