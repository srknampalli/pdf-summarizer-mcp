from mcp.server.fastmcp import FastMCP
from typing import List, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from pathlib import Path
from langchain_openai import AzureChatOpenAI

import os
from dotenv import load_dotenv


# env_path = Path(__file__).parent.parent / '.env'
# load_dotenv(env_path)

load_dotenv("C:/Tredence/pdf-extraction-mcp/.env")

# Fix: Import VectorStore correctly
try:
    from vector_store import VectorStore
except ImportError:
    # If import fails, create a dummy class
    class VectorStore:
        def query_similar(self, doc_id, embedding, top_k):
            return ["No vector store available"]

mcp = FastMCP(name="summarizer_qna_server")

def get_azure_llm():
    """
    Create Azure OpenAI LLM instance from environment variables.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") 
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-preview")
    temperature = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.7"))
    
    if not all([azure_endpoint, azure_deployment, api_key]):
        raise ValueError("Missing Azure OpenAI configuration. Please check your .env file.")
    
    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment, 
        api_key=api_key,
        api_version=api_version,
        temperature=temperature,
        streaming=True,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

@mcp.tool()
def summarize_text(text: Union[str, List[str]]) -> str:
    """
    Generates a summary for the input text or list of text chunks using Azure OpenAI.
    Args:
        text: A string or list of text chunks to summarize.
    Returns:
        A summary string.
    """
    try:
        if isinstance(text, dict) and 'text' in text:
            text = text['text']
        
        llm = get_azure_llm()
        
        if isinstance(text, list):
            text = "\n".join(str(item) for item in text)
        
        prompt = f"Summarize the following document or text chunks as concisely as possible:\n\n{text}"
        response = llm.invoke(prompt)
        
        # Handle response properly
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
    Answers a user question based on provided context using Azure OpenAI.
    Args:
        doc_id: The document ID
        question: The user's question  
        context: Context from retrieved chunks
    Returns:
        The answer string
    """
    try:
        llm = get_azure_llm()

        prompt = f"""Answer the following question based on the provided context from document '{doc_id}'.

Context:
{context}

Question: {question}

Answer:"""
        
        response = llm.invoke(prompt)
        
        # Handle response properly  
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        return f"Error answering question: {str(e)}"

if __name__ == "__main__":
    print("[DEBUG] Starting Azure OpenAI summarizer QnA server...")
    
    # Test configuration
    try:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        print(f"[DEBUG] Azure endpoint: {'Set' if azure_endpoint else 'Not set'}")
        print(f"[DEBUG] Azure deployment: {azure_deployment or 'Not set'}")
        print(f"[DEBUG] API key: {'Set' if api_key else 'Not set'}")
        
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
    
    mcp.run(transport="stdio")