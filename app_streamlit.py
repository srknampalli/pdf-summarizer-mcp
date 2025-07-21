import streamlit as st
from modules.pipeline import DocumentProcessingPipeline
import tempfile
import os
from phoenix.otel import register

register(
    endpoint=os.getenv("PHOENIX_ENDPOINT"),
    project_name="pdf-assistant-streamlit",
    headers={"authorization": f"Bearer {os.getenv('PHOENIX_API_KEY')}"} if os.getenv('PHOENIX_API_KEY') else None
)

st.set_page_config(page_title="PDF QnA & Summarizer (MCP-powered)")
st.title("PDF QnA & Summarizer (MCP-powered)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
pdf_path = None

if uploaded_file is not None:
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    if "pipeline" not in st.session_state or st.session_state.get("pdf_path") != pdf_path:
        st.session_state.pipeline = DocumentProcessingPipeline(pdf_path)
        st.session_state.pdf_path = pdf_path
        st.session_state.chat_history = []

    pipeline = st.session_state.pipeline

    st.subheader("Summary")
    if st.button("Get Summary"):
        with st.spinner("Generating summary..."):
            summary = pipeline.get_summary()
        st.write(summary)

    st.subheader("Ask a Question")
    question = st.text_input("Your question:")
    if st.button("Ask") and question:
        with st.spinner("Getting answer..."):
            answer = pipeline.ask_question(question)
        st.session_state.chat_history.append((question, answer))

    if st.session_state.get("chat_history"):
        st.subheader("Chat History")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
else:
    st.info("Please upload a PDF to get started.") 