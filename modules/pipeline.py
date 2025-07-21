import os
import re
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import ast
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class DocumentProcessingPipeline:
    def __init__(self, pdf_path: str, chunk_size: int = 300, chunk_overlap: int = 150):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc_id = self._sanitize_doc_id(pdf_path)
        self.text = None
        self.chunks = None
        self.embeddings = None
        self._process_document()

    def _sanitize_doc_id(self, pdf_path):
        base = os.path.basename(pdf_path)
        doc_id = os.path.splitext(base)[0]
        doc_id = re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)
        return doc_id
    # functio to call mcp sever
    async def _call_mcp_tool(self, server_script, tool_name, arguments):
        server_params = StdioServerParameters(command="python", args=[server_script])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return result.content

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
        with tracer.start_as_current_span("Extract PDF") as span:
            self.text = self.run_async(self._call_mcp_tool(
                "server/pdf_extractor.py", "extract_pdf_contents", {"pdf_path": self.pdf_path}
            ))[0]
            if hasattr(self.text, "text"):
                self.text = self.text.text
            span.set_attribute("doc_id", self.doc_id)
            span.set_attribute("pdf_path", self.pdf_path)
            span.set_attribute("text_preview", str(self.text)[:200])
        print(f"[DEBUG] Extracted text type: {type(self.text)}, value: {str(self.text)[:200]}")
        with tracer.start_as_current_span("Chunk PDF") as span:
            self.chunks = self.run_async(self._call_mcp_tool(
                "server/pdf_processing_server.py", "chunk_text", {"text": self.text, "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}
            ))
            span.set_attribute("num_chunks", len(self.chunks))
            span.set_attribute("first_chunk", self.chunks[0] if self.chunks else "NO CHUNKS")
        print(f"[DEBUG] Number of chunks: {len(self.chunks)}")
        print(f"[DEBUG] First chunk: {self.chunks[0] if self.chunks else 'NO CHUNKS'}")
        print(f"[DEBUG] All chunks: {self.chunks}")
        with tracer.start_as_current_span("Embed Chunks and Store") as span:
            # Convert all chunks to plain strings
            plain_chunks = [
                chunk.text if hasattr(chunk, "text") else
                chunk["text"] if isinstance(chunk, dict) and "text" in chunk else
                str(chunk)
                for chunk in self.chunks
            ]
            embed_result = self.run_async(self._call_mcp_tool(
                "server/pdf_processing_server.py", "embed_chunks", {"text_chunks": plain_chunks, "doc_id": self.doc_id}
            ))
            span.set_attribute("embed_result", str(embed_result))
        print(f"[DEBUG] Embed and store result: {embed_result}")

    def get_summary(self) -> str:
        text = self.text
        if isinstance(text, dict) and "text" in text:
            text = text["text"]
        elif isinstance(text, list):
            if all(isinstance(t, dict) and "text" in t for t in text):
                text = [t["text"] for t in text]
        with tracer.start_as_current_span("Summarize PDF") as span:
            summary = self.run_async(self._call_mcp_tool(
                "server/summarizer_qna_server.py", "summarize_text", {"text": text}
            ))[0]
            span.set_attribute("summary_preview", str(summary)[:200])
        return summary

    def ask_question(self, question: str, top_k: int = 15, fallback_to_llm: bool = True) -> str:
        with tracer.start_as_current_span("QnA") as span:
            q_embedding_obj = self.run_async(self._call_mcp_tool(
                "server/pdf_processing_server.py", "embed_chunks", {"text_chunks": [question]}
            ))[0]
            print("[DEBUG] Embedder output:", q_embedding_obj)
            if hasattr(q_embedding_obj, 'embedding'):
                q_embedding = q_embedding_obj.embedding
            elif hasattr(q_embedding_obj, 'text'):
                try:
                    q_embedding = ast.literal_eval(q_embedding_obj.text)
                except Exception:
                    raise ValueError(f"Could not parse embedding from text: {q_embedding_obj.text}")
            elif isinstance(q_embedding_obj, dict) and 'embedding' in q_embedding_obj:
                q_embedding = q_embedding_obj['embedding']
            elif isinstance(q_embedding_obj, list) and all(isinstance(x, float) for x in q_embedding_obj):
                q_embedding = q_embedding_obj
            else:
                raise ValueError(f"Unexpected embedder output: {q_embedding_obj}")
            relevant_chunks = self.run_async(self._call_mcp_tool(
                "server/pdf_processing_server.py", "search_embeddings", {
                    "doc_id": self.doc_id,
                    "query_embedding": q_embedding,
                    "top_k": top_k
                }
            ))
            relevant_chunks = [
                chunk.text if hasattr(chunk, "text") else chunk
                for chunk in relevant_chunks
            ]
            context = "\n".join(relevant_chunks)
            print("[QNA DEBUG] Context passed to LLM:\n", context)
            print(f"[QNA DEBUG] Question: {question}")
            print(f"[QNA DEBUG] Top-K: {top_k}")
            print(f"[QNA DEBUG] Chunks used for context: {relevant_chunks}")
            span.set_attribute("question", question)
            span.set_attribute("doc_id", self.doc_id)
            span.set_attribute("context_preview", context[:200])
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
            if fallback_to_llm and (not answer or 'not found' in answer.strip().lower()):
                print("[QNA DEBUG] Falling back to direct LLM answer...")
                from langchain_openai import ChatOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                llm = ChatOpenAI(model="gpt-4-turbo-preview", api_key=api_key)
                prompt = f"Based on the following document, answer the question as specifically as possible.\n\nDocument:\n{self.text}\n\nQuestion: {question}\n\nAnswer:"
                fallback_answer = llm.invoke(prompt)
                if hasattr(fallback_answer, "content"):
                    fallback_answer = fallback_answer.content
                span.set_attribute("fallback_used", True)
                span.set_attribute("fallback_answer_preview", str(fallback_answer)[:200])
                return fallback_answer
            span.set_attribute("fallback_used", False)
            span.set_attribute("answer_preview", str(answer)[:200])
        return answer 

