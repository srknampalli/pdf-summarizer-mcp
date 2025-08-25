# modules/pipeline.py - CLEAN ORCHESTRATION ONLY
import os
import re
import time
import json
import asyncio
import threading
from typing import Dict, Any, List
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

class DocumentProcessingPipeline:
    def __init__(self, pdf_path: str, chunk_size: int = 600, chunk_overlap: int = 60):
        print(f"[DEBUG] Starting pipeline for: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc_id = self._sanitize_doc_id(pdf_path)
        
        print(f"[DEBUG] Document ID: {self.doc_id}")
        
        try:
            if not self._check_document_exists():
                print(f"[DEBUG] Processing new document...")
                self._process_document()
            else:
                print(f"[DEBUG] Document already exists")
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
            raise

    def _sanitize_doc_id(self, pdf_path: str) -> str:
        """Only business logic that truly belongs in pipeline"""
        base = os.path.basename(pdf_path)
        doc_id = os.path.splitext(base)[0]
        return re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)

    def _check_document_exists(self) -> bool:
        """Pure orchestration - trust server completely"""
        try:
            print(f"[DEBUG] Checking if document exists...")
            docs = self._run_async_safe("server/pdf_processing_server.py", "list_stored_documents", {})
            
            # Simple check - trust server's response format
            for doc in docs:
                if self.doc_id in str(doc):
                    return True
            return False
        except Exception as e:
            print(f"[WARNING] Error checking document existence: {e}")
            return False

    def _process_document(self):
        """Pure orchestration - call tools, pass data, no logic duplication"""
        print(f"[DEBUG] Extracting PDF content...")
        
        # ✅ Call tool, trust result completely
        extracted_text = self._run_async_safe(
            "server/pdf_processing_server.py",
            "extract_pdf_contents", 
            {"pdf_path": self.pdf_path}
        )
        print(f"[DEBUG] Extracted {len(str(extracted_text))} characters")
        
        print(f"[DEBUG] Chunking text...")
        
        # ✅ Call tool, trust result completely 
        chunks = self._run_async_safe(
            "server/pdf_processing_server.py",
            "chunk_text",
            {"text": extracted_text, "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}
        )
        print(f"[DEBUG] Server returned {len(chunks)} chunks")
        
        print(f"[DEBUG] Generating embeddings...")
        
        # ✅ Call tool, trust result completely
        self._run_async_safe(
            "server/pdf_processing_server.py",
            "embed_chunks",
            {"text_chunks": chunks, "doc_id": self.doc_id}
        )
        print(f"[DEBUG] Document processing completed")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Pure orchestration - trust server's embedding response"""
        print(f"[DEBUG] Getting embedding for: {query[:50]}...")
        
        # ✅ Call tool, trust result
        result = self._run_async_safe(
            "server/pdf_processing_server.py",
            "embed_chunks",
            {"text_chunks": [query]}
        )
        
        # ✅ Minimal response parsing - server should return clean JSON
        embedding_response = json.loads(str(result))
        return embedding_response.get("vectors", [[]])[0]

    def get_summary(self) -> str:
        """Pure orchestration - coordinate multiple tool calls"""
        print(f"[DEBUG] Starting summary generation...")
        
        try:
            # Get relevant chunks using coordinated tool calls
            all_chunks = []
            queries = ["main topics", "key findings", "conclusions"]
            
            for query in queries:
                print(f"[DEBUG] Processing query: {query}")
                
                # Get embedding
                query_embedding = self._get_query_embedding(query)
                
                # Search chunks
                chunks = self._run_async_safe(
                    "server/pdf_processing_server.py",
                    "search_embeddings",
                    {"doc_id": self.doc_id, "query_embedding": query_embedding, "top_k": 3}
                )
                
                # ✅ Trust server completely - no filtering or processing
                all_chunks.extend(chunks)
            
            # Remove duplicates (only orchestration logic)
            unique_chunks = list(dict.fromkeys(all_chunks))
            print(f"[DEBUG] Found {len(unique_chunks)} unique chunks")
            
            # Prepare text for summarization
            combined_text = "\n\n---\n\n".join(unique_chunks[:8])
            print(f"[DEBUG] Calling summarizer with {len(combined_text)} characters")
            
            # ✅ Call summarizer tool, trust result completely
            summary = self._run_async_safe(
                "server/summarizer_qna_server.py",
                "summarize_text",
                {"text": combined_text}
            )
            
            print(f"[DEBUG] Summary generated: {len(str(summary))} characters")
            return str(summary)
            
        except Exception as e:
            error_msg = f"⚠️ Summary generation failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg

    def ask_question(self, question: str, top_k: int = 5) -> str:
        """Pure orchestration - coordinate tools for Q&A"""
        print(f"[DEBUG] Starting Q&A for: {question[:50]}...")
        
        try:
            # Get question embedding
            question_embedding = self._get_query_embedding(question)
            
            print(f"[DEBUG] Searching for relevant chunks...")
            
            # ✅ Search chunks, trust server result
            chunks = self._run_async_safe(
                "server/pdf_processing_server.py",
                "search_embeddings",
                {"doc_id": self.doc_id, "query_embedding": question_embedding, "top_k": top_k}
            )
            
            # ✅ Simple orchestration - combine chunks for context
            context = "\n\n".join(chunks)
            print(f"[DEBUG] Prepared context: {len(context)} characters")
            
            print(f"[DEBUG] Generating answer...")
            
            # ✅ Call Q&A tool, trust result completely
            answer = self._run_async_safe(
                "server/summarizer_qna_server.py",
                "answer_question",
                {"doc_id": self.doc_id, "question": question, "context": context}
            )
            
            print(f"[DEBUG] Answer generated: {len(str(answer))} characters")
            return str(answer)
            
        except Exception as e:
            print(f"[ERROR] Answer generation failed: {e}")
            return f"⚠️ Answer generation failed: {str(e)}"

    def _run_async_safe(self, server_script: str, tool_name: str, arguments: Dict[str, Any]):
        """Thread-safe asyncio execution - infrastructure only"""
        print(f"[DEBUG] Calling {tool_name} on {server_script}")
        
        result_container = {"result": None, "exception": None, "completed": False}
        
        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(
                        self._call_mcp_tool_impl(server_script, tool_name, arguments)
                    )
                    result_container["result"] = result
                    print(f"[DEBUG] {tool_name} completed successfully")
                    
                except Exception as e:
                    print(f"[ERROR] {tool_name} failed: {e}")
                    result_container["exception"] = e
                    
                finally:
                    loop.close()
                    
            except Exception as e:
                print(f"[ERROR] Thread execution failed: {e}")
                result_container["exception"] = e
            finally:
                result_container["completed"] = True
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=120)
        
        if not result_container["completed"]:
            raise TimeoutError(f"Operation {tool_name} timed out after 120 seconds")
            
        if result_container["exception"]:
            raise result_container["exception"]
            
        return result_container["result"]

    async def _call_mcp_tool_impl(self, server_script: str, tool_name: str, arguments: Dict[str, Any]):
        """Core MCP communication - infrastructure only"""
        server_params = StdioServerParameters(command="python", args=[server_script])
        
        async with asyncio.timeout(90):
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)
                    return result.content











# import os
# import re
# import time
# import json
# import asyncio
# import threading
# from typing import Dict, Any, List
# from mcp.client.stdio import stdio_client, StdioServerParameters
# from mcp.client.session import ClientSession

# class DocumentProcessingPipeline:
#     def __init__(self, pdf_path: str, chunk_size: int = 600, chunk_overlap: int = 60):
#         print(f"[DEBUG] Starting pipeline for: {pdf_path}")
        
#         self.pdf_path = pdf_path
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.doc_id = self._sanitize_doc_id(pdf_path)
        
#         print(f"[DEBUG] Document ID: {self.doc_id}")
        
#         try:
#             if not self._check_document_exists():
#                 print(f"[DEBUG] Processing new document...")
#                 self._process_document()
#             else:
#                 print(f"[DEBUG] Document already exists")
#         except Exception as e:
#             print(f"[ERROR] Pipeline failed: {e}")
#             raise

#     def _sanitize_doc_id(self, pdf_path: str) -> str:
#         base = os.path.basename(pdf_path)
#         doc_id = os.path.splitext(base)[0]
#         return re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)

#     def _extract_text_from_response(self, response) -> str:
#         if isinstance(response, str):
#             return response
#         if isinstance(response, dict) and 'text' in response:
#             return str(response['text'])
#         if hasattr(response, 'text'):
#             return str(response.text)
#         return str(response)

#     def _check_document_exists(self) -> bool:
#         try:
#             print(f"[DEBUG] Checking if document exists...")
#             docs = self._run_async_safe(
#                 "server/pdf_processing_server.py", 
#                 "list_stored_documents", 
#                 {}
#             )
            
#             for doc in docs:
#                 if self.doc_id in self._extract_text_from_response(doc):
#                     return True
#             return False
#         except Exception as e:
#             print(f"[WARNING] Error checking document existence: {e}")
#             return False

#     def _process_document(self):
#         print(f"[DEBUG] Extracting PDF content...")
#         # Extract PDF
#         pdf_result = self._run_async_safe(
#             "server/pdf_processing_server.py",
#             "extract_pdf_contents",
#             {"pdf_path": self.pdf_path}
#         )
        
#         extracted_text = self._extract_text_from_response(
#             pdf_result[0] if isinstance(pdf_result, list) else pdf_result
#         )
#         print(f"[DEBUG] Extracted {len(extracted_text)} characters")
        
#         print(f"[DEBUG] Chunking text...")
#         # Chunk text
#         chunks_result = self._run_async_safe(
#             "server/pdf_processing_server.py",
#             "chunk_text",
#             {"text": extracted_text, "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}
#         )
        
#         chunks = []
#         if isinstance(chunks_result, list):
#             for chunk in chunks_result:
#                 chunk_text = self._extract_text_from_response(chunk)
#                 if len(chunk_text.strip()) > 10:
#                     chunks.append(chunk_text.strip())
        
#         print(f"[DEBUG] Created {len(chunks)} chunks")
        
#         print(f"[DEBUG] Generating embeddings...")
#         # Generate and store embeddings
#         self._run_async_safe(
#             "server/pdf_processing_server.py",
#             "embed_chunks",
#             {"text_chunks": chunks, "doc_id": self.doc_id}
#         )
#         print(f"[DEBUG] Document processing completed")

#     def _get_query_embedding(self, query: str) -> List[float]:
#         print(f"[DEBUG] Getting embedding for: {query[:50]}...")
#         result = self._run_async_safe(
#             "server/pdf_processing_server.py",
#             "embed_chunks",
#             {"text_chunks": [query]}
#         )
        
#         response_text = self._extract_text_from_response(
#             result[0] if isinstance(result, list) else result
#         )
        
#         embedding_response = json.loads(response_text)
#         return embedding_response.get("vectors", [[]])[0]

#     def get_summary(self) -> str:
#         print(f"[DEBUG] Starting summary generation...")
        
#         try:
#             # Get relevant chunks
#             all_chunks = []
#             queries = ["main topics", "key findings", "conclusions"]
            
#             for query in queries:
#                 print(f"[DEBUG] Processing query: {query}")
#                 query_embedding = self._get_query_embedding(query)
                
#                 chunks = self._run_async_safe(
#                     "server/pdf_processing_server.py",
#                     "search_embeddings",
#                     {"doc_id": self.doc_id, "query_embedding": query_embedding, "top_k": 3}
#                 )
                
#                 for chunk in chunks:
#                     chunk_text = self._extract_text_from_response(chunk)
#                     if len(chunk_text.strip()) > 50 and chunk_text not in all_chunks:
#                         all_chunks.append(chunk_text)
            
#             print(f"[DEBUG] Found {len(all_chunks)} relevant chunks")
            
#             # Generate summary
#             combined_text = "\n\n---\n\n".join(all_chunks[:8])
#             print(f"[DEBUG] Calling summarizer with {len(combined_text)} characters")
            
#             summary_result = self._run_async_safe(
#                 "server/summarizer_qna_server.py",
#                 "summarize_text",
#                 {"text": combined_text}
#             )
            
#             summary = self._extract_text_from_response(
#                 summary_result[0] if isinstance(summary_result, list) else summary_result
#             )
            
#             print(f"[DEBUG] Summary generated: {len(summary)} characters")
#             return summary
            
#         except Exception as e:
#             error_msg = f"⚠️ Summary generation failed: {str(e)}"
#             print(f"[ERROR] {error_msg}")
#             return error_msg

#     def ask_question(self, question: str, top_k: int = 5) -> str:
#         print(f"[DEBUG] Starting Q&A for: {question[:50]}...")
#         try:
#             # Get question embedding
#             question_embedding = self._get_query_embedding(question)
            
#             print(f"[DEBUG] Searching for relevant chunks...")
#             # Search relevant chunks
#             chunks = self._run_async_safe(
#                 "server/pdf_processing_server.py",
#                 "search_embeddings",
#                 {"doc_id": self.doc_id, "query_embedding": question_embedding, "top_k": top_k}
#             )
            
#             # Prepare context
#             context_parts = []
#             for chunk in chunks:
#                 chunk_text = self._extract_text_from_response(chunk)
#                 if len(chunk_text.strip()) > 10:
#                     context_parts.append(chunk_text)
            
#             context = "\n\n".join(context_parts)
#             print(f"[DEBUG] Prepared context: {len(context)} characters")
            
#             print(f"[DEBUG] Generating answer...")
#             # Generate answer
#             answer_result = self._run_async_safe(
#                 "server/summarizer_qna_server.py",
#                 "answer_question",
#                 {"doc_id": self.doc_id, "question": question, "context": context}
#             )
            
#             answer = self._extract_text_from_response(
#                 answer_result[0] if isinstance(answer_result, list) else answer_result
#             )
            
#             print(f"[DEBUG] Answer generated: {len(answer)} characters")
#             return answer
            
#         except Exception as e:
#             print(f"[ERROR] Answer generation failed: {e}")
#             return f"⚠️ Answer generation failed: {str(e)}"

#     def _run_async_safe(self, server_script: str, tool_name: str, arguments: Dict[str, Any]):
#         """
#         Thread-safe asyncio execution that avoids event loop conflicts
#         """
#         print(f"[DEBUG] Calling {tool_name} on {server_script}")
        
#         # Create a result container for thread communication
#         result_container = {"result": None, "exception": None, "completed": False}
        
#         def run_in_thread():
#             """Run the async operation in a separate thread with its own event loop"""
#             try:
#                 # Create new event loop for this thread
#                 loop = asyncio.new_event_loop()
#                 asyncio.set_event_loop(loop)
                
#                 try:
#                     # Run the async operation
#                     result = loop.run_until_complete(
#                         self._call_mcp_tool_impl(server_script, tool_name, arguments)
#                     )
#                     result_container["result"] = result
#                     print(f"[DEBUG] {tool_name} completed successfully")
                    
#                 except Exception as e:
#                     print(f"[ERROR] {tool_name} failed: {e}")
#                     result_container["exception"] = e
                    
#                 finally:
#                     loop.close()
                    
#             except Exception as e:
#                 print(f"[ERROR] Thread execution failed: {e}")
#                 result_container["exception"] = e
#             finally:
#                 result_container["completed"] = True
        
#         # Start the thread
#         thread = threading.Thread(target=run_in_thread, daemon=True)
#         thread.start()
        
#         # Wait for completion with timeout
#         thread.join(timeout=120)  # 2 minute timeout
        
#         if not result_container["completed"]:
#             raise TimeoutError(f"Operation {tool_name} timed out after 120 seconds")
            
#         if result_container["exception"]:
#             raise result_container["exception"]
            
#         return result_container["result"]

#     async def _call_mcp_tool_impl(self, server_script: str, tool_name: str, arguments: Dict[str, Any]):
#         """
#         Core MCP tool call implementation - runs in isolated thread
#         """
#         server_params = StdioServerParameters(command="python", args=[server_script])
        
#         async with asyncio.timeout(90):  # 90 second timeout for individual calls
#             async with stdio_client(server_params) as (read, write):
#                 async with ClientSession(read, write) as session:
#                     await session.initialize()
#                     result = await session.call_tool(tool_name, arguments=arguments)
#                     return result.content