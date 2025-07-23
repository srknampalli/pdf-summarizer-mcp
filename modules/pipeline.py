import os
import re
import time
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ast
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

class DocumentProcessingPipeline:
    def __init__(self, pdf_path: str, chunk_size: int = 300, chunk_overlap: int = 150):
        print(f"[TIMING] Starting document processing for: {pdf_path}")
        total_start = time.time()
        
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc_id = self._sanitize_doc_id(pdf_path)
        self.text = None
        self.chunks = None
        self.embeddings = None
        self._is_processed = False 
        
        # Document processing with Phoenix attributes
        with tracer.start_as_current_span("Document Processing Pipeline") as span:
            
            span.set_attribute("operation", "document_processing")
            span.set_attribute("pdf_path", pdf_path)
            span.set_attribute("chunk_size", chunk_size)
            span.set_attribute("chunk_overlap", chunk_overlap)
            span.set_attribute("doc_id", self.doc_id)
            span.set_attribute("file_size_mb", self._get_file_size_mb(pdf_path))
            
            try:
                # ✅ CHECK IF DOCUMENT ALREADY EXISTS (avoid reprocessing)
                if self._check_document_exists():
                    print(f"[INFO] ✅ Document {self.doc_id} already processed, skipping reprocessing...")
                    self._is_processed = True
                    span.set_attribute("document_already_processed", True)
                    span.set_attribute("processing_skipped", True)
                else:
                    print(f"[INFO] 🔄 New document, processing...")
                    self._process_document()
                    self._is_processed = True
                    span.set_attribute("document_already_processed", False)
                    span.set_attribute("processing_skipped", False)
                
                total_time = time.time() - total_start
                span.set_attribute("total_processing_time_ms", total_time * 1000)
                span.set_attribute("chunks_created", len(self.chunks) if self.chunks else 0)
                span.set_attribute("success", True)
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error_type", type(e).__name__)
                span.set_attribute("error_message", str(e))
                span.set_attribute("success", False)
                raise
        
        total_time = time.time() - total_start
        print(f"[TIMING] ✅ Total document processing completed in: {total_time:.2f} seconds")

    def _get_file_size_mb(self, file_path):
        try:
            return round(os.path.getsize(file_path) / (1024 * 1024), 2)
        except:
            return 0

    def _sanitize_doc_id(self, pdf_path):
        base = os.path.basename(pdf_path)
        doc_id = os.path.splitext(base)[0]
        doc_id = re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)
        return doc_id

    def _check_document_exists(self) -> bool:
        """✅ CHECK if document already exists in vector store"""
        try:
            print(f"[DEBUG] Checking if document {self.doc_id} already exists...")
            existing_docs = self.run_async(self._call_mcp_tool(
                "server/pdf_processing_server.py", "list_stored_documents", {}
            ))
            
            if isinstance(existing_docs, list):
                doc_list = existing_docs
            elif hasattr(existing_docs, '__iter__'):
                doc_list = list(existing_docs)
            else:
                doc_list = []
            
            exists = self.doc_id in doc_list
            print(f"[DEBUG] Document exists check: {exists} (found {len(doc_list)} existing docs)")
            return exists
            
        except Exception as e:
            print(f"[WARNING] Could not check document existence: {e}")
            return False

    async def _call_mcp_tool(self, server_script, tool_name, arguments):
        with tracer.start_as_current_span(f"MCP Tool: {tool_name}") as span:
            span.set_attribute("operation", "mcp_tool_call")
            span.set_attribute("server_script", server_script)
            span.set_attribute("tool_name", tool_name)
            span.set_attribute("arguments", str(arguments)[:200])
            
            try:
                server_params = StdioServerParameters(command="python", args=[server_script])
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments=arguments)
                        
                        span.set_attribute("result_type", type(result.content).__name__)
                        if hasattr(result.content, '__len__'):
                            span.set_attribute("result_length", len(result.content))
                        
                        span.set_attribute("success", True)
                        span.set_status(Status(StatusCode.OK))
                        return result.content
                        
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error_type", type(e).__name__)
                span.set_attribute("error_message", str(e))
                span.set_attribute("success", False)
                raise e

    def run_async(self, coro):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)

    def _process_document(self):
        # Step 1: PDF Extraction
        print(f"[DEBUG] Starting PDF extraction...")
        with tracer.start_as_current_span("PDF Text Extraction") as span:
            span.set_attribute("operation", "pdf_extraction")
            
            self.text = self.run_async(self._call_mcp_tool(
                "server/pdf_processing_server.py", "extract_pdf_contents", {"pdf_path": self.pdf_path}
            ))[0]
            
            if hasattr(self.text, "text"):
                self.text = self.text.text
                
            # Phoenix attributes
            span.set_attribute("doc_id", self.doc_id)
            span.set_attribute("pdf_path", self.pdf_path)
            span.set_attribute("text_length", len(self.text))
            span.set_attribute("text_preview", str(self.text)[:200])
            span.set_attribute("words_extracted", len(self.text.split()))
            span.set_attribute("pages_processed", self.text.count("Page "))
            
            print(f"[DEBUG] ✅ PDF extraction completed. Text length: {len(self.text)}")

        # Step 2: Text Chunking
        print(f"[DEBUG] Starting text chunking...")
        with tracer.start_as_current_span("Text Chunking") as span:
            span.set_attribute("operation", "text_chunking")
            
            self.chunks = self.run_async(self._call_mcp_tool(
                "server/pdf_processing_server.py", "chunk_text", 
                {"text": self.text, "chunk_size": self.chunk_size}
            ))
            
            # Phoenix attributes
            span.set_attribute("num_chunks", len(self.chunks))
            span.set_attribute("chunk_size", self.chunk_size)
            span.set_attribute("chunk_overlap", self.chunk_overlap)
            span.set_attribute("avg_chunk_length", sum(len(str(c)) for c in self.chunks) / len(self.chunks))
            
            if self.chunks:
                first_chunk = self.chunks[0]
                if hasattr(first_chunk, 'text'):
                    first_chunk_text = first_chunk.text[:200]
                else:
                    first_chunk_text = str(first_chunk)[:200]
                span.set_attribute("first_chunk_preview", first_chunk_text)
                
            print(f"[DEBUG] ✅ Text chunking completed. Created {len(self.chunks)} chunks")

        # Step 3: Embedding and Storage
        print(f"[DEBUG] Starting embedding generation...")
        with tracer.start_as_current_span("Embedding Generation & Storage") as span:
            span.set_attribute("operation", "embedding_generation")
            
            plain_chunks = [
                chunk.text if hasattr(chunk, "text") else
                chunk["text"] if isinstance(chunk, dict) and "text" in chunk else
                str(chunk)
                for chunk in self.chunks
            ]
            
            # Phoenix attributes for embeddings
            span.set_attribute("num_chunks_to_embed", len(plain_chunks))
            span.set_attribute("doc_id", self.doc_id)
            span.set_attribute("total_text_to_embed", sum(len(chunk) for chunk in plain_chunks))
            span.set_attribute("embedding_model", "text-embedding-3-small")
            span.set_attribute("embedding_provider", "azure_openai")
            span.set_attribute("embedding_dimensions", 1536)
            span.set_attribute("vector_db", "chromadb")
            
            print(f"[DEBUG] Embedding {len(plain_chunks)} chunks...")
            
            embed_result = self.run_async(self._call_mcp_tool(
                "server/pdf_processing_server.py", "embed_chunks", 
                {"text_chunks": plain_chunks, "doc_id": self.doc_id}
            ))
            
            span.set_attribute("embed_result", str(embed_result)[:200])
            print(f"[DEBUG] ✅ Embedding completed. Result: {embed_result}")

    def get_summary(self) -> str:
        """
        summary based on vector db
        """
        print(f"[TIMING] 📝 Starting VECTOR-BASED document summarization...")
        summary_start = time.time()
        
        with tracer.start_as_current_span("Document Summarization") as span:
            # Phoenix attributes for summarization
            span.set_attribute("operation", "document_summarization")
            span.set_attribute("method", "vector_based_chunked") 
            span.set_attribute("llm_provider", "azure_openai")
            span.set_attribute("use_case", "document_summary")
            span.set_attribute("doc_id", self.doc_id)
            span.set_attribute("llm.model_name", "gpt-4")
            
            try:
                print(f"[DEBUG] Using vector-based summarization approach...")
                
                # ✅ STRATEGY: Use multiple summary queries to get diverse chunks
                summary_queries = [
                    "main topics and key points",
                    "important findings and conclusions", 
                    "methodology and approach",
                    "results and outcomes",
                    "recommendations and implications"
                ]
                
                all_relevant_chunks = []
                
                for i, query in enumerate(summary_queries):
                    print(f"[DEBUG] Query {i+1}/{len(summary_queries)}: Searching for chunks related to '{query}'")
                    
                    try:
                        # Get embedding for summary query
                        q_embedding_obj = self.run_async(self._call_mcp_tool(
                            "server/pdf_processing_server.py", "embed_chunks", {"text_chunks": [query]}
                        ))[0]
                        
                        # Parse embedding
                        if hasattr(q_embedding_obj, 'text'):
                            q_embedding = ast.literal_eval(q_embedding_obj.text)
                        elif isinstance(q_embedding_obj, list):
                            q_embedding = q_embedding_obj
                        else:
                            print(f"[WARNING] Unexpected embedding format for query '{query}': {type(q_embedding_obj)}")
                            continue
                        
                        # Search for relevant chunks
                        relevant_chunks = self.run_async(self._call_mcp_tool(
                            "server/pdf_processing_server.py", "search_embeddings", {
                                "doc_id": self.doc_id,
                                "query_embedding": q_embedding,
                                "top_k": 4  # Get top 4 chunks per query
                            }
                        ))
                        
                        # Add to collection (avoid duplicates)
                        for chunk in relevant_chunks:
                            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                            if chunk_text not in all_relevant_chunks and len(chunk_text.strip()) > 50:
                                all_relevant_chunks.append(chunk_text)
                                
                        print(f"[DEBUG] Query '{query}' found {len(relevant_chunks)} chunks")
                        
                    except Exception as query_error:
                        print(f"[WARNING] Error processing query '{query}': {query_error}")
                        continue
                
                # ✅ LIMIT total chunks to avoid token limits
                max_chunks = 6  # Conservative limit to avoid 429 errors
                selected_chunks = all_relevant_chunks[:max_chunks]
                
                if not selected_chunks:
                    # Fallback: get first few chunks directly
                    print(f"[WARNING] No chunks found via vector search, using fallback approach")
                    
                    if hasattr(self, 'text') and self.text:
                        # Use first 3000 characters as safe fallback
                        fallback_text = self.text[:3000]
                        if len(self.text) > 3000:
                            fallback_text += "\n\n[Document continues...]"
                        selected_chunks = [fallback_text]
                    else:
                        raise ValueError("No text available for summarization")
                
                # Combine chunks for summarization
                combined_text = "\n\n---CHUNK SEPARATOR---\n\n".join(selected_chunks)
                
                print(f"[DEBUG] ✅ Using {len(selected_chunks)} chunks for summarization")
                print(f"[DEBUG] Combined text length: {len(combined_text)} characters")
                
                # Phoenix span attributes
                span.set_attribute("chunks_used", len(selected_chunks))
                span.set_attribute("combined_text_length", len(combined_text))
                span.set_attribute("source_words", len(combined_text.split()))
                span.set_attribute("text_length", len(combined_text))  # For compatibility
                
                # ✅ Generate summary from selected chunks (not full text!)
                summary = self.run_async(self._call_mcp_tool(
                    "server/summarizer_qna_server.py", "summarize_text", {"text": combined_text}
                ))[0]
                
                if hasattr(summary, 'text'):
                    summary = summary.text
                else:
                    summary = str(summary)
                
                # Phoenix success attributes
                span.set_attribute("summary_length", len(str(summary)))
                span.set_attribute("summary_preview", str(summary)[:200])
                span.set_attribute("summary_words", len(str(summary).split()))
                span.set_attribute("compression_ratio", len(combined_text) / len(str(summary)) if len(str(summary)) > 0 else 0)
                span.set_attribute("success", True)
                span.set_status(Status(StatusCode.OK))
                
                print(f"[DEBUG] ✅ Vector-based summary completed successfully using {len(selected_chunks)} chunks")
                
            except Exception as e:
                print(f"[ERROR] Vector-based summary failed: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error_type", type(e).__name__)
                span.set_attribute("error_message", str(e))
                span.set_attribute("success", False)
                
                # Enhanced error handling
                if "429" in str(e) or "rate limit" in str(e).lower():
                    summary = "⚠️ Rate limit exceeded. The document has been processed and stored. Please try the summary again in a few minutes."
                    span.set_attribute("error_category", "rate_limit")
                elif "token" in str(e).lower():
                    summary = "⚠️ Document too large for summarization. Try asking specific questions about the content instead."
                    span.set_attribute("error_category", "token_limit")
                else:
                    summary = f"❌ Error generating summary: {str(e)}. The document is processed and available for Q&A."
                    span.set_attribute("error_category", "general_error")
        
        summary_time = time.time() - summary_start
        print(f"[TIMING] ✅ Document summarization completed in: {summary_time:.2f} seconds")
        
        return summary

    def ask_question(self, question: str, top_k: int = 5, fallback_to_llm: bool = True) -> str:
        print(f"[TIMING] ❓ Starting Q&A for question: '{question[:50]}...'")
        qa_start = time.time()
        
        with tracer.start_as_current_span("Question & Answer Pipeline") as span:
            # Phoenix attributes for Q&A
            span.set_attribute("operation", "question_answering")
            span.set_attribute("rag_type", "retrieval_augmented_generation")
            span.set_attribute("llm_provider", "azure_openai")
            span.set_attribute("vector_db", "chromadb")
            span.set_attribute("use_case", "document_qa")
            
            span.set_attribute("question", question)
            span.set_attribute("doc_id", self.doc_id)
            span.set_attribute("top_k", top_k)
            span.set_attribute("fallback_enabled", fallback_to_llm)
            span.set_attribute("question_length", len(question))
            span.set_attribute("question_words", len(question.split()))
            span.set_attribute("question_type", self._classify_question(question))
            
            try:
                # Step 1: Question embedding
                print(f"[DEBUG] Generating question embedding...")
                with tracer.start_as_current_span("Question Embedding") as embed_span:
                    embed_span.set_attribute("operation", "question_embedding")
                    embed_span.set_attribute("question", question)
                    embed_span.set_attribute("embedding_model", "text-embedding-3-small")
                    embed_span.set_attribute("embedding_provider", "azure_openai")
                    
                    q_embedding_obj = self.run_async(self._call_mcp_tool(
                        "server/pdf_processing_server.py", "embed_chunks", {"text_chunks": [question]}
                    ))[0]
                    
                    embed_span.set_attribute("embedding_latency_ms", (time.time() - qa_start) * 1000)
                    print(f"[DEBUG] ✅ Question embedding completed")
                
                # Parse embedding
                if hasattr(q_embedding_obj, 'embedding'):
                    q_embedding = q_embedding_obj.embedding
                elif hasattr(q_embedding_obj, 'text'):
                    q_embedding = ast.literal_eval(q_embedding_obj.text)
                elif isinstance(q_embedding_obj, dict) and 'embedding' in q_embedding_obj:
                    q_embedding = q_embedding_obj['embedding']
                elif isinstance(q_embedding_obj, list):
                    q_embedding = q_embedding_obj
                else:
                    raise ValueError(f"Unexpected embedder output: {q_embedding_obj}")
                
                # Step 2: Vector search
                print(f"[DEBUG] Searching for relevant chunks...")
                search_start = time.time()
                
                with tracer.start_as_current_span("Vector Similarity Search") as search_span:
                    search_span.set_attribute("operation", "vector_search")
                    search_span.set_attribute("vector_db", "chromadb")
                    search_span.set_attribute("top_k_requested", top_k)
                    search_span.set_attribute("embedding_dimensions", len(q_embedding))
                    
                    relevant_chunks = self.run_async(self._call_mcp_tool(
                        "server/pdf_processing_server.py", "search_embeddings", {
                            "doc_id": self.doc_id,
                            "query_embedding": q_embedding,
                            "top_k": top_k
                        }
                    ))
                    
                    search_span.set_attribute("chunks_found", len(relevant_chunks))
                    search_span.set_attribute("search_efficiency", len(relevant_chunks) / top_k if top_k > 0 else 0)
                    search_span.set_attribute("search_latency_ms", (time.time() - search_start) * 1000)
                    print(f"[DEBUG] ✅ Found {len(relevant_chunks)} relevant chunks")
                
                # Prepare context
                relevant_chunks = [
                    chunk.text if hasattr(chunk, "text") else chunk
                    for chunk in relevant_chunks
                ]
                context = "\n".join(relevant_chunks)
                
                span.set_attribute("context_length", len(context))
                span.set_attribute("context_preview", context[:200])
                span.set_attribute("context_words", len(context.split()))
                span.set_attribute("chunks_used", len(relevant_chunks))
                
                # Step 3: Answer generation
                print(f"[DEBUG] Generating answer...")
                answer_start = time.time()
                
                with tracer.start_as_current_span("LLM Answer Generation") as answer_span:
                    # Phoenix LLM attributes
                    answer_span.set_attribute("operation", "llm_generation")
                    answer_span.set_attribute("llm_provider", "azure_openai")
                    answer_span.set_attribute("llm_use_case", "rag_answer_generation")
                    answer_span.set_attribute("llm.model_name", "gpt-4")
                    answer_span.set_attribute("llm.temperature", 0.7)
                    answer_span.set_attribute("context_length", len(context))
                    
                    # Create prompt for Phoenix tracking
                    full_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                    answer_span.set_attribute("llm.prompts", [full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt])
                    
                    answer = self.run_async(self._call_mcp_tool(
                        "server/summarizer_qna_server.py", "answer_question", {
                            "doc_id": self.doc_id,
                            "question": question,
                            "context": context
                        }
                    ))[0]
                    
                    if hasattr(answer, 'text'):
                        answer = answer.text
                    else:
                        answer = str(answer)
                    
                    # Phoenix answer attributes
                    answer_span.set_attribute("llm.responses", [answer[:500] + "..." if len(answer) > 500 else answer])
                    answer_span.set_attribute("answer_length", len(answer))
                    answer_span.set_attribute("answer_words", len(answer.split()))
                    answer_span.set_attribute("answer_generation_latency_ms", (time.time() - answer_start) * 1000)
                    
                    # Token usage estimation for Phoenix
                    estimated_prompt_tokens = len(full_prompt.split()) * 1.3
                    estimated_completion_tokens = len(answer.split()) * 1.3
                    answer_span.set_attribute("llm.token_usage.prompt_tokens", int(estimated_prompt_tokens))
                    answer_span.set_attribute("llm.token_usage.completion_tokens", int(estimated_completion_tokens))
                    answer_span.set_attribute("llm.token_usage.total_tokens", int(estimated_prompt_tokens + estimated_completion_tokens))
                    
                    print(f"[DEBUG] ✅ Answer generated successfully")
                
                # Phoenix success attributes
                span.set_attribute("fallback_used", False)
                span.set_attribute("total_qa_time_ms", (time.time() - qa_start) * 1000)
                span.set_attribute("success", True)
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                print(f"[ERROR] Q&A failed: {e}")
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error_type", type(e).__name__)
                span.set_attribute("error_message", str(e))
                span.set_attribute("success", False)
                
                # Handle different error types
                if "429" in str(e) or "rate limit" in str(e).lower():
                    answer = "⚠️ Rate limit exceeded. Please try again in a few minutes."
                    span.set_attribute("error_category", "rate_limit")
                else:
                    answer = f"❌ Error answering question: {str(e)}"
                    span.set_attribute("error_category", "general_error")
        
        total_qa_time = time.time() - qa_start
        print(f"[TIMING] ✅ Total Q&A completed in: {total_qa_time:.2f} seconds")
        
        return answer

    def _classify_question(self, question):
        """Classify question type for Phoenix tracking"""
        question_lower = question.lower()
        if any(word in question_lower for word in ['what', 'define', 'explain']):
            return "factual"
        elif any(word in question_lower for word in ['how', 'why', 'when', 'where']):
            return "procedural"
        elif any(word in question_lower for word in ['compare', 'difference', 'versus']):
            return "comparative"
        elif any(word in question_lower for word in ['summarize', 'overview', 'main points']):
            return "summary"
        else:
            return "general"

    @classmethod
    def view_traces(cls, lines=30):
        print("🔥 Traces are being sent to Phoenix DB")
        print("📊 Check your Phoenix DB dashboard to view traces")
    
    @classmethod
    def clear_traces(cls):
        print("🔥 Use Phoenix DB dashboard to manage traces")
    
    @classmethod
    def analyze_traces(cls):
        print("🔥 Use Phoenix DB dashboard for analysis")
        return {"status": "phoenix_db_connected"}
