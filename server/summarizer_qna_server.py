from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Union
import os
import json
from dotenv import load_dotenv


from phoenix.otel import register
tracer_provider = register(
    project_name="pdf-assistant-streamlit",  # SAME project name
    endpoint="https://agentopsacc-backend-app.yellowriver-a22b4385.westus.azurecontainerapps.io/v1/traces",
    auto_instrument=True,
    headers={
        "api-key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6OSJ9.DF1jhTBxOcI7cfrL84JE82PHtymsypKCUZ5gUlI0BtY",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6OSJ9.DF1jhTBxOcI7cfrL84JE82PHtymsypKCUZ5gUlI0BtY"
    }
)



from openinference.instrumentation.mcp import MCPInstrumentor
tracer = tracer_provider.get_tracer("mcp-qas-server")

instrumentor = MCPInstrumentor()

instrumentor.instrument(tracer_provider=tracer_provider)


load_dotenv("C:/Tredence/pdf-extraction-mcp/.env")

from vector_store import VectorStore
vector_store = VectorStore()

from langchain_openai import AzureOpenAIEmbeddings

embedder = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2023-05-15",
)

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    return ' '.join(text.split()).strip()

mcp = FastMCP(name="summarizer_qna_server")

@mcp.tool()
#@tracer.tool(name="MCP.retriever")  
def retriever(query_text: str, doc_id: str, top_k: int ) -> Dict[str, Any]:
    """
    Retriever Tool
    Input: Query/question text and document ID
    Output: List of most similar chunks (IDs & content) from ChromaDB
    Uses Azure OpenAI to embed the query, ChromaDB to search for nearest neighbors
    """
    try:
        if not query_text or not doc_id:
            return {"error": "Missing query_text or doc_id", "success": False}
        
    
        cleaned_query = clean_text(query_text)
        if not cleaned_query:
            return {"error": "Invalid query text", "success": False}
        
        print(f"[DEBUG] Retrieving similar chunks for query: '{cleaned_query[:50]}...'")
        query_embedding = embedder.embed_query(cleaned_query)
        
        if not query_embedding:
            return {"error": "Failed to generate query embedding", "success": False}
        
        print(f"[DEBUG] Generated query embedding with dimension {len(query_embedding)}")
        similar_chunks = vector_store.query_similar(doc_id, query_embedding, top_k)
        
        if not similar_chunks:
            return {
                "success": True,
                "query": cleaned_query,
                "doc_id": doc_id,
                "chunks_found": 0,
                "retrieved_chunks": [],
                "message": f"No similar chunks found for document '{doc_id}'"
            }
        
        retrieved_data = []
        for i, chunk_text in enumerate(similar_chunks):
            chunk_data = {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "chunk_index": i,
                "content": chunk_text,
                "content_length": len(chunk_text),
                "similarity_rank": i + 1
            }
            retrieved_data.append(chunk_data)
        
        result = {
            "success": True,
            "query": cleaned_query,
            "doc_id": doc_id,
            "top_k_requested": top_k,
            "chunks_found": len(retrieved_data),
            "retrieved_chunks": retrieved_data,
            "query_embedding_dimension": len(query_embedding)
        }
        
        print(f"[DEBUG] Retrieved {len(retrieved_data)} similar chunks")
        return result
        
    except Exception as e:
        return {"error": f"Retrieval failed: {str(e)}", "success": False}

@mcp.tool()
#@tracer.tool(name="MCP.summarization")  
def summarization(context_chunks: Union[List[str], List[Dict], str], doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Summarization Tool
    Input: Retrieved chunks or complete context text
    Output: Summary generated with Azure OpenAI LLM
    Typically called after Retriever for "summarize" requests
    """
    try:
     
        if isinstance(context_chunks, str):
       
            context_text = context_chunks
        elif isinstance(context_chunks, list):
            text_parts = []
            for chunk in context_chunks:
                if isinstance(chunk, dict):
  
                    text_parts.append(chunk.get('content', '') or chunk.get('text', '') or str(chunk))
                else:
                    text_parts.append(str(chunk))
            context_text = "\n\n---\n\n".join(text_parts)
        else:
            return {"error": "Invalid context_chunks format", "success": False}
        
        context_text = clean_text(context_text)
        if not context_text:
            return {"error": "No valid context text provided", "success": False}
        
        max_context_chars = 12000  
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars] + "..."
            print(f"[DEBUG] Truncated context to {max_context_chars} characters")
        
        print(f"[DEBUG] Generating summary for {len(context_text)} characters")
        
        prompt = f"""Summarize the text covering main topics, key findings, important facts, and core arguments in one response 
        {context_text}  Summary:"""
        
        response = llm.invoke(prompt)
        summary_text = response.content if hasattr(response, 'content') else str(response)
        
        if not summary_text:
            return {"error": "Failed to generate summary", "success": False}
        
        result = {
            "success": True,
            "doc_id": doc_id,
            "summary": summary_text.strip(),
            "summary_length": len(summary_text.strip()),
            "source_context_length": len(context_text),
            "model_used": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
            "context_truncated": len(context_text) >= max_context_chars
        }
        
        print(f"[DEBUG] Generated summary: {len(summary_text)} characters")
        return result
        
    except Exception as e:
        return {"error": f"Summarization failed: {str(e)}", "success": False}

@mcp.tool()
#@tracer.tool(name="MCP.qa_tool")
def qa_tool(question: str, context_chunks: Union[List[str], List[Dict], str], doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Q&A Tool
    Input: User question and retrieved context/chunks
    Output: Generated answer using only retrieved content
    LLM is prompted with context and instructed to use only retrieved content
    """
    try:
        if not question:
            return {"error": "No question provided", "success": False}
        
        question = clean_text(question)
        if not question:
            return {"error": "Invalid question text", "success": False}
        
        if isinstance(context_chunks, str):
            context_text = context_chunks
        elif isinstance(context_chunks, list):
            text_parts = []
            for chunk in context_chunks:
                if isinstance(chunk, dict):
                    text_parts.append(chunk.get('content', '') or chunk.get('text', '') or str(chunk))
                else:
                    text_parts.append(str(chunk))
            context_text = "\n\n---\n\n".join(text_parts)
        else:
            return {"error": "Invalid context_chunks format", "success": False}
        
    
        context_text = clean_text(context_text)
        if not context_text:
            return {"error": "No valid context provided", "success": False}
        
        max_context_chars = 10000  
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars] + "..."
            print(f"[DEBUG] Truncated context to {max_context_chars} characters")
        
        print(f"[DEBUG] Answering question: '{question[:50]}...' using {len(context_text)} chars context")
        
        prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context. 

IMPORTANT INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context below
- If the context doesn't contain enough information to answer the question, state that you cannot answer the question.
- Do not use external knowledge or make assumptions beyond what's in the context
- Be specific and cite relevant parts of the context when possible.
- If the context provides a general overview but lacks a specific, step-by-step answer, you may provide a high-level summary of the relevant concepts found.

Context:
{context_text}

Question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, 'content') else str(response)
        
        if not answer_text:
            return {"error": "Failed to generate answer", "success": False}
        
        result = {
            "success": True,
            "question": question,
            "answer": answer_text.strip(),
            "doc_id": doc_id,
            "answer_length": len(answer_text.strip()),
            "context_length": len(context_text),
            "model_used": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
            "context_truncated": len(context_text) >= max_context_chars
        }
        
        print(f"[DEBUG] Generated answer: {len(answer_text)} characters")
        return result
        
    except Exception as e:
        return {"error": f"Q&A failed: {str(e)}", "success": False}


@mcp.tool()
def list_available_documents() -> List[str]:
    """List all documents available for retrieval"""
    return vector_store.list_documents()

@mcp.tool()
def get_document_info(doc_id: str) -> Dict[str, Any]:
    """Get information about a specific document"""
    return vector_store.get_document_stats(doc_id)

if __name__ == "__main__":
    print("[DEBUG] Starting Summarizer Q&A server with 3 core tools...")
    print("[DEBUG] Tools: retriever, summarization, qa_tool")
    mcp.run(transport="stdio")



