
import os
from phoenix.otel import register

PHOENIX_ENDPOINT = "https://agentopsacc-backend-app.yellowriver-a22b4385.westus.azurecontainerapps.io/v1/traces"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6OSJ9.DF1jhTBxOcI7cfrL84JE82PHtymsypKCUZ5gUlI0BtY"


os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = PHOENIX_ENDPOINT
tracer_provider = register(
    project_name="pdf-assistant-streamlit",
    endpoint=PHOENIX_ENDPOINT,
    auto_instrument=True,
    batch=True,
    headers={
        "api-key": API_KEY,
        "authorization": f"Bearer {API_KEY}"
    }
)

from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

import streamlit as st
st.set_page_config(
    page_title="📄 PDF RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if "orchestrator" not in st.session_state:
        from modules.rag_orchestrator import HybridOrchestrator
        st.session_state.orchestrator = HybridOrchestrator()
    if "doc_id" not in st.session_state:
        st.session_state.doc_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

initialize_session_state()

import asyncio
import tempfile
import os

async def display_streaming_response(response_generator):
    """Stream agent responses to UI"""
    full_response = ""
    placeholder = st.empty()
    async for chunk in response_generator:
        full_response += chunk
        placeholder.markdown(full_response)
    return full_response

def process_pdf_upload():
    """Handles PDF file upload and ingestion"""
    uploaded_file = st.file_uploader("📄 Choose a PDF", type=["pdf"])
    if st.button("🔄 Process PDF", use_container_width=True, type="primary") and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Processing PDF..."):
            result = asyncio.run(st.session_state.orchestrator.ingest_pdf(pdf_path))
            if result.get("success"):
                st.session_state.doc_id = result["doc_id"]
                st.success("✅ PDF processed successfully!")
            else:
                st.error(f"❌ Failed: {result.get('error')}")

        os.unlink(pdf_path)

def ask_question_ui():
    """Handles asking questions about the document or general queries"""
    question = st.text_input("💬 Ask a question")
    if st.button("🔍 Get Answer", use_container_width=True) and question:
        with st.spinner("Thinking..."):
            response_gen = st.session_state.orchestrator.process_query(
                question, st.session_state.doc_id
            )
            answer = asyncio.run(display_streaming_response(response_gen))
            st.session_state.chat_history.append({"question": question, "answer": answer})

def show_chat_history():
    """Displays chat history in collapsible sections"""
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for chat in reversed(st.session_state.chat_history):
            with st.expander(f"💬 {chat['question']}"):
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")


def main():
    st.title("🤖 PDF RAG System")

    st.subheader("1️⃣ Upload and Process PDF")
    process_pdf_upload()

    if st.session_state.doc_id:
        st.success(f"📄 Current Document ID: {st.session_state.doc_id}")
    else:
        st.info("No document loaded yet.")

    # Ask questions
    if st.session_state.doc_id:
        st.subheader("2️⃣ Ask Questions")
        ask_question_ui()

    # Show chat history
    show_chat_history()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ System Info")
        st.metric("Document", "Loaded" if st.session_state.doc_id else "None")
        st.metric("Questions", len(st.session_state.chat_history))
        st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.clear()
            st.rerun()


if __name__ == "__main__":
    main()





# import streamlit as st
# import asyncio
# import tempfile
# import os
# from modules.rag_orchestrator import HybridOrchestrator

# import nest_asyncio
# nest_asyncio.apply()



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


# from openinference.instrumentation.langchain import LangChainInstrumentor
# instrumentor = LangChainInstrumentor()
# instrumentor.instrument(tracer_provider=tracer_provider)

# tracer = tracer_provider.get_tracer("langgraph_agent")

# st.title("🤖 PDF RAG System")

# # Initialize orchestrator
# if 'orchestrator' not in st.session_state:
#     st.session_state.orchestrator = HybridOrchestrator()

# if 'doc_id' not in st.session_state:
#     st.session_state.doc_id = ""

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []


# async def display_streaming_response(response_generator):
#     full_response = ""
#     placeholder = st.empty()
#     async for chunk in response_generator:
#         full_response += chunk
#         placeholder.markdown(full_response)
#     return full_response


# st.subheader("📄 Upload PDF")
# uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# if st.button("🔄 Process PDF", use_container_width=True, type="primary") and uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(uploaded_file.read())
#         pdf_path = tmp.name
    
#     with st.spinner("Processing PDF..."):
#         result = asyncio.run(st.session_state.orchestrator.ingest_pdf(pdf_path))
        
#         if result.get("success"):
#             st.session_state.doc_id = result["doc_id"]
#             st.success(f"✅ PDF processed successfully!")
#             st.info(f"Document ID: **{result['doc_id']}**")
#         else:
#             st.error(f"❌ Failed: {result.get('error')}")
    
#     os.unlink(pdf_path)


# if st.session_state.doc_id:
#     st.success(f"📄 **Current Document:** {st.session_state.doc_id}")
# else:
#     st.info("📄 No document loaded. Please upload and process a PDF first.")

# # Main interaction area
# st.subheader("💬 Ask Questions")

# if st.session_state.doc_id:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("📋 Summarize Document", use_container_width=True, type="secondary"):
#             with st.spinner("Generating summary..."):
#                 async def get_summary():
#                     response_gen = st.session_state.orchestrator.process_query(
#                         "Please provide a comprehensive summary of this document including main topics, key findings, and important insights.", 
#                         st.session_state.doc_id
#                     )
#                     return await display_streaming_response(response_gen)
                
#                 summary = asyncio.run(get_summary())
#                 st.session_state.chat_history.append({
#                     'question': 'Document Summary', 
#                     'answer': summary
#                 })
    
#     with col2:
#         if st.button("🌤️ Get Weather", use_container_width=True):
#             with st.spinner("Getting weather..."):
#                 async def get_weather():
#                     response_gen = st.session_state.orchestrator.process_query(
#                         "What's the weather like in Chennai?", 
#                         None
#                     )
#                     return await display_streaming_response(response_gen)
                
#                 weather = asyncio.run(get_weather())
#                 st.session_state.chat_history.append({
#                     'question': 'Weather in Chennai', 
#                     'answer': weather
#                 })

# question = st.text_input(
#     "Ask any question:", 
#     placeholder="Ask about the document, weather, or general topics..."
# )

# if st.button("🔍 Ask Question", use_container_width=True) and question:
#     with st.spinner("Getting answer..."):
#         doc_related_keywords = ['document', 'pdf', 'text', 'content', 'paper', 'article', 'findings', 'research', 'study', 'report']
#         weather_keywords = ['weather', 'temperature', 'forecast', 'climate', 'rain', 'sunny', 'cloudy']
        
#         question_lower = question.lower()
        
#         async def get_answer():
#             if st.session_state.doc_id and any(keyword in question_lower for keyword in doc_related_keywords):

#                 response_gen = st.session_state.orchestrator.process_query(question, st.session_state.doc_id)
#             elif any(keyword in question_lower for keyword in weather_keywords):
       
#                 response_gen = st.session_state.orchestrator.process_query(question, None)
#             elif st.session_state.doc_id:

#                 response_gen = st.session_state.orchestrator.process_query(question, st.session_state.doc_id)
#             else:

#                 response_gen = st.session_state.orchestrator.process_query(question, None)
            
#             return await display_streaming_response(response_gen)
        
#         answer = asyncio.run(get_answer())
#         st.session_state.chat_history.append({
#             'question': question, 
#             'answer': answer
#         })

# # Chat History
# if st.session_state.chat_history:
#     st.subheader("📜 Chat History")
    
#     for i, chat in enumerate(reversed(st.session_state.chat_history)):
#         with st.expander(f"💬 {chat['question'][:60]}..."):
#             st.markdown(f"**Q:** {chat['question']}")
#             st.markdown(f"**A:** {chat['answer']}")

# # Sidebar
# with st.sidebar:
#     st.header("ℹ️ System Info")
    
#     if st.session_state.doc_id:
#         st.metric("📄 Document", "Loaded")
#         st.metric("💬 Questions", len(st.session_state.chat_history))
#     else:
#         st.metric("📄 Document", "None")
    
#     st.markdown("---")
    
#     if st.button("🗑️ Clear History", use_container_width=True):
#         st.session_state.chat_history = []
#         st.rerun()
    
#     if st.button("🔄 Reset All", use_container_width=True):
#         st.session_state.clear()
#         st.rerun()
    
#     st.markdown("---")
#     st.markdown("**💡 Tips:**")
  











