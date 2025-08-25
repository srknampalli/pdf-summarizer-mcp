import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Optional, List

from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver

from modules.langraph import build_agent_graph, AgentState

def load_mcp_config():
    """
    Loads the MCP server configurations from mcp_config.json with detailed error logging.
    """
    try:
        config_path = r"C:\Tredence\pdf-extraction-mcp\server\mcp_config.json"
        config_file = Path(config_path)

        if not config_file.exists():
            print(f"[ERROR] The file '{config_path}' was not found.")
            raise FileNotFoundError(f"mcp_config.json file {config_path} does not exist.")

        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            print(f"[ERROR] The file '{config_path}' is empty or contains only whitespace.")
            raise ValueError("mcp_config.json is empty.")

        mcp_config = json.loads(content)

        print(f"[INFO] Successfully loaded MCP configuration from '{config_path}'.")
        print(f"[DEBUG] Found servers: {list(mcp_config.get('mcpServers', {}).keys())}")
        
        return mcp_config
        
    except json.JSONDecodeError as e:
        print("[CRITICAL ERROR] Failed to parse mcp_config.json. Please check for syntax errors.")
        print(f"[DEBUG] JSONDecodeError: {e}")
        raise
    except FileNotFoundError as e:
        print("[CRITICAL ERROR] File not found. Please verify the path.")
        raise
    except Exception as e:
        print(f"[CRITICAL ERROR] An unexpected error occurred while loading the config: {e}")
        raise


class HybridOrchestrator:
    """
    Hybrid approach: Direct MCP calls for ingestion, LangGraph for all queries
    """
    
    def __init__(self):
        self.client = None
        self.graph = None
        self.tools = None
        self.tools_by_name = {}  # Initialize empty dictionary
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.initialized = False
        self.graph_config = {
            "configurable": {
                "thread_id": "pdf_session"
            }
        }
        
    async def initialize(self):
        """Initialize both direct MCP client and LangGraph agent"""
        if self.initialized:
            return
            
        print("[INFO] Initializing Hybrid Orchestrator...")
        mcp_config = load_mcp_config()
        self.client = MultiServerMCPClient(connections=mcp_config["mcpServers"])
        
        # Get all tools
        self.tools = await self.client.get_tools()
        print(f"[DEBUG] Available tools: {[t.name for t in self.tools]}")
        
     
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        print(f"[DEBUG] Tools by name: {list(self.tools_by_name.keys())}")
        
        
        query_tools = [t for t in self.tools if t.name in ['retriever', 'qa_tool', 'summarization', 'tavily_search', 'get_weather']]
        print(f"[DEBUG] Agent tools: {[t.name for t in query_tools]}")
        
        self.graph = build_agent_graph(tools=query_tools)
        
        self.initialized = True
        print("[INFO] Hybrid Orchestrator initialized successfully.")
        
    async def aclose(self):
        """Closes the MCP client gracefully."""
        if self.client:
            await self.client.aclose()
            self.initialized = False

    def _sanitize_doc_id(self, pdf_path: str) -> str:
        """Generate a clean document ID from a PDF path."""
        base = os.path.basename(pdf_path)
        doc_id = os.path.splitext(base)[0]
        return re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)

    async def _ensure_initialized(self):
        """Ensure the orchestrator is initialized before use"""
        if not self.initialized:
            print("[WARNING] Orchestrator not initialized. Initializing now...")
            await self.initialize()

    async def _call_tool_direct(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Direct MCP tool call using LangChain tool invoke"""
        try:
         
            await self._ensure_initialized()
            
            print(f"[DEBUG] Calling {tool_name} with args: {arguments}")
            
      
            if tool_name not in self.tools_by_name:
                available_tools = list(self.tools_by_name.keys())
                return {
                    "success": False, 
                    "error": f"Tool {tool_name} not found. Available tools: {available_tools}"
                }
            
            tool = self.tools_by_name[tool_name]
            
           
            result = await tool.ainvoke(arguments)
            print(f"[DEBUG] Tool {tool_name} returned: {type(result)} - {str(result)[:200]}...")
            
    
            if isinstance(result, str):
                try:
                   
                    parsed_result = json.loads(result)
                    return parsed_result
                except json.JSONDecodeError:
                  
                    return {"success": True, "result": result}
            elif isinstance(result, dict):
                return result
            else:
                return {"success": True, "result": str(result)}
                
        except Exception as e:
            print(f"[ERROR] Tool call {tool_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Direct PDF ingestion - bypass agent, use sequential tool calls"""
        print(f"[INFO] Starting PDF ingestion for: {pdf_path}")
        
        await self._ensure_initialized()
        doc_id = self._sanitize_doc_id(pdf_path)
        
        try:
          
            print("[INFO] Step 1: Loading PDF...")
            pdf_result = await self._call_tool_direct("pdf_loader", {"pdf_path": pdf_path})
            if not pdf_result.get("success"):
                return pdf_result

       
            print("[INFO] Step 2: Chunking text...")
            chunks_result = await self._call_tool_direct("text_chunker", {
                "document_text": pdf_result["combined_text"],
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "doc_id": doc_id
            })
            if not chunks_result.get("success"):
                return chunks_result

          
            print("[INFO] Step 3: Generating embeddings...")
            text_chunks = [chunk['text'] for chunk in chunks_result['chunks']]
            chunk_ids = [chunk['chunk_id'] for chunk in chunks_result['chunks']]
            
            embeddings_result = await self._call_tool_direct("embedding_generator", {
                "text_chunks": text_chunks,
                "chunk_ids": chunk_ids
            })
            if not embeddings_result.get("success"):
                return embeddings_result

           
            print("[INFO] Step 4: Storing in vector database...")
            storage_result = await self._call_tool_direct("vectordb_writer", {
                "doc_id": doc_id,
                "embeddings_data": embeddings_result["embeddings"],
                "metadata": {"source_file": pdf_path}
            })
            
            if storage_result.get("success"):
                print(f"[SUCCESS] PDF ingestion completed for doc_id: {doc_id}")
                return {
                    "success": True,
                    "doc_id": doc_id,
                    "message": "PDF successfully ingested",
                    "chunks_processed": len(text_chunks)
                }
            else:
                return storage_result
                
        except Exception as e:
            print(f"[ERROR] PDF ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Ingestion failed: {str(e)}"}
    
    async def process_query(self, user_query: str, doc_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Unified query processing - let the agent handle everything"""
        
        print(f"[INFO] Processing query: '{user_query[:50]}...' with doc_id: {doc_id}")    
        await self._ensure_initialized()
        
        try:
            async for chunk in self._run_agent_query(user_query, doc_id):
                yield chunk
        except Exception as e:
            print(f"[ERROR] Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"⚠️ Query processing failed: {str(e)}"
    
    async def _run_agent_query(self, query: str, doc_id: Optional[str]) -> AsyncGenerator[str, None]:
        """Run query through LangGraph agent"""
        initial_state = AgentState(
            user_query=query, 
            doc_id=doc_id, 
            messages=[HumanMessage(content=query)]
        )
        
        try:
            async for s in self.graph.astream(initial_state, config=self.graph_config):
                for node_name, node_output in s.items():
                    if node_name == "__end__":
                        continue
                    
                    message_list = node_output.get("messages", [])
                    
                    for message in message_list:
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                          
                            tool_call = message.tool_calls[0]
                            print(f"[AGENT] 🔧 Calling tool: {tool_call['name']}")
                            print(f"[AGENT] Tool arguments: {tool_call['args']}")
                            
                        elif hasattr(message, 'content') and message.content:                       
                            if hasattr(message, 'tool_call_id'):
                                print(f"[TOOL] Response from tool: {str(message.content)[:100]}...")
                                continue
                            if isinstance(message.content, str):
                                content = message.content.strip()
                                if any(skip_phrase in content.lower() for skip_phrase in [
                                    'using tool:', 'calling tool:', 'tool call', 'arguments:',
                                    '"success": true', '"success":true', 'chunks_found', 
                                    'similarity_rank', 'retrieved_chunks', 'embeddings_data'
                                ]):
                                    print(f"[DEBUG] Filtered tool message: {content[:100]}...")
                                    continue
                                
                               
                                if len(content) < 10:
                                    continue
                                yield content
                                
                            elif isinstance(message.content, dict):
                                print(f"[AGENT] JSON response: {json.dumps(message.content, indent=2)}")
                            else:
                                # Other content types - log to terminal
                                print(f"[AGENT] Other content: {str(message.content)}")
                            
        except Exception as e:
            print(f"[ERROR] Agent query failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"⚠️ Agent processing failed: {str(e)}"

# import os
# import re
# import json
# import asyncio
# from typing import Dict, Any, List
# from mcp.client.stdio import stdio_client, StdioServerParameters
# from mcp.client.session import ClientSession

# class RAGOrchestrator:
#     """
#     MCP Client (Orchestrator) for RAG workflows
#     Handles API calls between frontend (Streamlit) and backend tools
#     Manages workflow orchestration for PDF ingestion, summarization, and Q&A
#     """
    
#     def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         print(f"[INFO] RAG Orchestrator initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

#     def _sanitize_doc_id(self, pdf_path: str) -> str:
#         """Generate a clean document ID from a PDF path."""
#         base = os.path.basename(pdf_path)
#         doc_id = os.path.splitext(base)[0]
#         return re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)

#     def _check_document_exists(self, doc_id: str) -> bool:
#         """Check if a document exists by calling the list_stored_documents tool."""
#         try:
#             docs = self._call_tool("server/pdf_processing_server.py", "list_stored_documents", {})
#             # The list_stored_documents tool returns a list of strings
#             return doc_id in docs
#         except Exception as e:
#             print(f"[WARNING] Error checking document existence: {e}")
#             return False

#     async def _call_tool_async(self, server_script: str, tool_name: str, arguments: Dict[str, Any]):
#         """Internal asynchronous function to make a single MCP tool call."""
#         server_params = StdioServerParameters(command="python", args=[server_script])
#         async with stdio_client(server_params) as (read, write):
#             async with ClientSession(read, write) as session:
#                 await session.initialize()
#                 result = await session.call_tool(tool_name, arguments=arguments)
#                 return result.content

#     def _call_tool(self, server_script: str, tool_name: str, arguments: Dict[str, Any]):
#         """Synchronous wrapper for MCP tool calls, handling response parsing."""
#         print(f"[DEBUG] Calling {tool_name} on {server_script} with args: {arguments}")
        
#         try:
#             import nest_asyncio
#             nest_asyncio.apply()
            
#             mcp_response_content = asyncio.run(self._call_tool_async(server_script, tool_name, arguments))
            
#             # The tool_name determines the expected return type
#             if tool_name in ["list_stored_documents", "search_similar_chunks"]:
#                 # These tools return a simple list of strings
#                 return mcp_response_content
            
#             # All other tools should return a JSON-serializable dictionary
#             # The response content might be a list of TextContent objects
#             if isinstance(mcp_response_content, list) and len(mcp_response_content) > 0:
#                 first_item = mcp_response_content[0]
#                 if hasattr(first_item, 'text'):
#                     # The content is a TextContent object, so extract the text
#                     json_string = first_item.text
#                 else:
#                     # It's a list of something else, but we'll try to parse it
#                     json_string = str(first_item)
#             elif hasattr(mcp_response_content, 'text'):
#                 # Single TextContent object
#                 json_string = mcp_response_content.text
#             elif isinstance(mcp_response_content, (str, dict, list)):
#                 # If it's already a string, dictionary, or list, use it directly
#                 json_string = mcp_response_content
#             else:
#                 # Fallback for other types
#                 json_string = str(mcp_response_content)
                
#             # Attempt to parse the JSON string
#             if isinstance(json_string, dict):
#                 return json_string
#             else:
#                 return json.loads(json_string)

#         except Exception as e:
#             print(f"[ERROR] {tool_name} failed: {e}")
#             return {"success": False, "error": f"Failed to call {tool_name}: {e}"}

#     def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
#         """Complete PDF ingestion pipeline."""
#         print(f"[INFO] Starting PDF ingestion for: {pdf_path}")
        
#         doc_id = self._sanitize_doc_id(pdf_path)
#         if self._check_document_exists(doc_id):
#             return {"success": True, "doc_id": doc_id, "message": "Document already exists", "status": "cached"}
        
#         # Step 1: PDF Loader
#         pdf_data = self._call_tool("server/pdf_processing_server.py", "pdf_loader", {"pdf_path": pdf_path})
#         if not pdf_data.get("success"):
#             return pdf_data

#         # Step 2: Text Chunker  
#         chunks_data = self._call_tool("server/pdf_processing_server.py", "text_chunker", {
#             "document_text": pdf_data["combined_text"],
#             "chunk_size": self.chunk_size,
#             "chunk_overlap": self.chunk_overlap,
#             "doc_id": doc_id
#         })
#         if not chunks_data.get("success"):
#             return chunks_data

#         # Step 3: Embedding Generator
#         text_chunks = [chunk['text'] for chunk in chunks_data['chunks']]
#         chunk_ids = [chunk['chunk_id'] for chunk in chunks_data['chunks']]
#         embeddings_data = self._call_tool("server/pdf_processing_server.py", "embedding_generator", {
#             "text_chunks": text_chunks,
#             "chunk_ids": chunk_ids
#         })
#         if not embeddings_data.get("success"):
#             return embeddings_data

#         # Step 4: Vector DB Writer
#         storage_result = self._call_tool("server/pdf_processing_server.py", "vectordb_writer", {
#             "doc_id": doc_id,
#             "embeddings_data": embeddings_data["embeddings"],
#             "metadata": {"source_file": pdf_path}
#         })
#         return storage_result

#     def get_summary(self, doc_id: str, summary_queries: List[str] = None) -> str:
#         """Summary workflow: Retriever -> Summarization"""
#         print(f"[INFO] Starting summary workflow for document: {doc_id}")
        
#         if not summary_queries:
#             summary_queries = ["main topics", "key findings", "conclusions", "important information"]
        
#         all_retrieved_chunks = []
#         for query in summary_queries:
#             retrieval_result = self._call_tool("server/summarizer_qna_server.py", "retriever", {
#                 "query_text": query, 
#                 "doc_id": doc_id, 
#                 "top_k": 3
#             })
            
#             if retrieval_result.get("success") and retrieval_result.get("chunks_found", 0) > 0:
#                 all_retrieved_chunks.extend([c["content"] for c in retrieval_result["retrieved_chunks"]])
        
#         if not all_retrieved_chunks:
#             return "⚠️ No relevant content found for summarization."
        
#         summary_result = self._call_tool("server/summarizer_qna_server.py", "summarization", {
#             "context_chunks": all_retrieved_chunks, 
#             "doc_id": doc_id
#         })
        
#         return summary_result.get("summary", f"⚠️ Summary generation failed: {summary_result.get('error')}")

#     def ask_question(self, doc_id: str, question: str, top_k: int = 5) -> str:
#         """Q&A workflow: Retriever -> Q&A"""
#         print(f"[INFO] Starting Q&A workflow for question: '{question[:50]}...'")
        
#         retrieval_result = self._call_tool("server/summarizer_qna_server.py", "retriever", {
#             "query_text": question, 
#             "doc_id": doc_id, 
#             "top_k": top_k
#         })
        
#         if not retrieval_result.get("success") or retrieval_result.get("chunks_found", 0) == 0:
#             return retrieval_result.get("error", "⚠️ No relevant information found for your question.")
        
#         retrieved_chunks = retrieval_result["retrieved_chunks"]
#         qa_result = self._call_tool("server/summarizer_qna_server.py", "qa_tool", {
#             "question": question,
#             "context_chunks": retrieved_chunks,
#             "doc_id": doc_id
#         })
        
#         return qa_result.get("answer", f"⚠️ Answer generation failed: {qa_result.get('error')}")

#     def list_documents(self) -> List[str]:
#         """List all available documents in the system."""
#         try:
#             return self._call_tool("server/pdf_processing_server.py", "list_stored_documents", {})
#         except Exception as e:
#             print(f"[ERROR] Failed to list documents: {e}")
#             return []