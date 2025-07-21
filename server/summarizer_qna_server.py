from mcp.server.fastmcp import FastMCP
from typing import List, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from pathlib import Path
from langchain_openai import AzureChatOpenAI

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix: Import VectorStore correctly
try:
    from vector_store import VectorStore
except ImportError:
    # If import fails, create a dummy class
    class VectorStore:
        def query_similar(self, doc_id, embedding, top_k):
            return ["No vector store available"]

mcp = FastMCP(name="summarizer_qna_server")




          

@mcp.tool()
def summarize_text(text: Union[str, List[str]]) -> str:
    """
    Generates a summary for the input text or list of text chunks using an LLM (OpenAI).
    Args:
        text: A string or list of text chunks to summarize.
    Returns:
        A summary string.
    """
    try:
        if isinstance(text, dict) and 'text' in text:
            text = text['text']
        
        llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.7,
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

        

        
        if isinstance(text, list):
            text = "\n".join(text)
        
        prompt = f"Summarize the following document or text chunks as concisely as possible:\n\n{text}"
        response = llm.invoke(prompt)
        
        # Fix: Handle response properly
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

# Initialize vector store
vector_store = VectorStore()

@mcp.tool()
def answer_question(doc_id: str, question: str, context: str) -> str:
    """
    Answers a user question based on provided context.
    Args:
        doc_id: The document ID
        question: The user's question  
        context: Context from retrieved chunks
    Returns:
        The answer string
    """
    try:
        if not api_key:
            return "Error: Could not find AZURE_OPENAI_API_KEY"

        # Use LLM to answer based on provided context
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
        prompt = f"""Answer the following question based on the provided context from document '{doc_id}'.

Context:
{context}

Question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        
        # Fix: Handle response properly  
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        return f"Error answering question: {str(e)}"

# @mcp.resource("summarizer_qna://status")
# def summarizer_qna_status_resource() -> str:
#     """Get status of summarizer and QnA services"""
#     return "Summarization and QnA services are active"

if __name__ == "__main__":
    print("[DEBUG] Starting summarizer QnA server...")
    mcp.run(transport="stdio")