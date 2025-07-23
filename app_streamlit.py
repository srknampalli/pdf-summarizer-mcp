import streamlit as st
from modules.pipeline import DocumentProcessingPipeline
import tempfile
import os
import time
import hashlib


from modules.tracing_setup import setup_otel_langchain_tracing, get_phoenix_dashboard_url, check_environment_setup


env_check = check_environment_setup()

# Initialize tracing
tracing_enabled = setup_otel_langchain_tracing() if env_check else False

st.set_page_config(
    page_title="PDF QnA & Summarizer", 
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF QnA & Summarizer")

# Simple tracing status
if tracing_enabled:
    st.success("🔍 OpenTelemetry LangChain tracing active")
else:
    st.warning("⚠️ Tracing disabled")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Document Processing")
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Document caching
        file_content = uploaded_file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        uploaded_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            pdf_path = tmp.name

        # Check cache
        if (st.session_state.get('file_hash') == file_hash and 
            'pipeline' in st.session_state):
            
            st.info("✅ Using cached document")
            pipeline = st.session_state.pipeline
            
        else:
            # Process new document
            with st.spinner("🔄 Processing PDF..."):
                start_time = time.time()
                pipeline = DocumentProcessingPipeline(pdf_path)
                processing_time = time.time() - start_time
                
                # Store in session
                st.session_state.pipeline = pipeline
                st.session_state.file_hash = file_hash
                st.session_state.chat_history = []
                
                st.success(f"✅ Processed in {processing_time:.2f}s")

        # Summary Section
        st.subheader("📝 Summary")
        
        if st.button("🔍 Get Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                start_time = time.time()
                summary = pipeline.get_summary()
                summary_time = time.time() - start_time
            
            st.session_state.last_summary = summary
            st.session_state.summary_time = summary_time
        
        # Display summary
        if hasattr(st.session_state, 'last_summary'):
            if st.session_state.last_summary.startswith(('⚠️', '❌')):
                st.warning(st.session_state.last_summary)
            else:
                st.write(st.session_state.last_summary)
            st.caption(f"Generated in {st.session_state.summary_time:.2f}s")

        # Q&A Section
        st.subheader("❓ Questions")
        
        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            question = st.text_input("Ask a question:", key="question_input")
        with col_q2:
            top_k = st.number_input("Top K", min_value=1, max_value=50, value=5)
        
        if st.button("🚀 Ask", use_container_width=True) and question:
            with st.spinner("Getting answer..."):
                start_time = time.time()
                answer = pipeline.ask_question(question, top_k=top_k)
                qa_time = time.time() - start_time
            
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer, 
                'time': qa_time,
                'timestamp': time.strftime("%H:%M:%S")
            })

        # Chat History
        if st.session_state.chat_history:
            st.subheader("💬 History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:40]}... ({chat['time']:.1f}s)"):
                    st.markdown(f"**Q:** {chat['question']}")
                    
                    if chat['answer'].startswith(('⚠️', '❌')):
                        st.warning(chat['answer'])
                    else:
                        st.markdown(f"**A:** {chat['answer']}")
                    
                    st.caption(f"{chat['time']:.2f}s • {chat['timestamp']}")
    else:
        st.info("📋 Upload a PDF to get started")

# Lightweight sidebar
with col2:
    st.header("📊 Stats")
    
    if 'pipeline' in st.session_state:
        st.metric("📄 Document", st.session_state.pipeline.doc_id[:15] + "...")
        st.metric("💬 Questions", len(st.session_state.chat_history))
        
        if st.session_state.chat_history:
            avg_time = sum(c['time'] for c in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.metric("⚡ Avg Time", f"{avg_time:.2f}s")
    else:
        st.info("No document loaded")
    
    st.markdown("---")
    
    # Phoenix Dashboard link (simple)
    if tracing_enabled:
        st.subheader("🔍 Phoenix DB")
        dashboard_url = get_phoenix_dashboard_url()
        
        if dashboard_url:
            if st.button("🔗 Open Dashboard", use_container_width=True):
                st.markdown(f'<a href="{dashboard_url}" target="_blank">Open Phoenix</a>', unsafe_allow_html=True)
            
            st.caption("All LLM calls traced automatically")
        else:
            st.info("Dashboard URL not available")
    else:
        st.subheader("⚠️ Tracing Setup")
        st.warning("Environment variables not set")
        st.info("""
        
        ```
         restart Streamlit.
        """)
    
    # Document management
    st.markdown("---")
    st.subheader("🛠️ Actions")
    
    if 'pipeline' in st.session_state:
        if st.button("🗑️ Clear", use_container_width=True):
            for key in ['pipeline', 'file_hash', 'chat_history', 'last_summary']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Simple footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.metric("🔍 Tracing", "Active" if tracing_enabled else "Off")

with col_f2:
    if 'pipeline' in st.session_state:
        st.metric("📚 Status", "Loaded")
    else:
        st.metric("📚 Status", "Empty")

with col_f3:
    chat_count = len(st.session_state.chat_history)
    st.metric("💬 Total", chat_count)


