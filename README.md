# MCP PDF Assistant

A modular, microservice-based PDF assistant using **Model Context Protocol (MCP)** microservices for PDF extraction, chunking, embedding, vector storage, QnA, and summarization. Supports both CLI and Streamlit UI, with robust OpenTelemetry tracing (Arize Phoenix) and a LangChain-style pipeline abstraction.

## ✨ Features

- 📄 **PDF Extraction & Processing** - Extract text from both regular and scanned PDFs with OCR support
- 🔍 **Intelligent Chunking** - Smart text segmentation for optimal retrieval
- 🧠 **Vector Embeddings** - Generate embeddings using OpenAI or Azure OpenAI
- 🗃️ **Vector Storage** - ChromaDB-based vector database for similarity search
- ❓ **Question & Answer** - RAG-powered Q&A with LLM fallback
- 📝 **Summarization** - AI-powered document summarization
- 🖥️ **Dual Interface** - Streamlit web UI and CLI support
- 📊 **Observability** - OpenTelemetry tracing with Arize Phoenix integration
- 🔧 **Modular Architecture** - MCP-based microservices for scalability

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────────────────────────┐
│   Streamlit UI  │───▶│    Pipeline Client Server            │
│      or CLI     │    │     (modules/pipeline.py)          │
└─────────────────┘    └─────────────────┬───────────────────┘
                                         │
                       ┌─────────────────▼───────────────────┐
                       │            MCP Servers              │
                       ├─────────────────────────────────────┤
                       │ • pdf_extractor.py                  │
                       │ • pdf_processing_server.py          │
                       │ • summarizer_qna_server.py          │
                       │ •          │
                       └─────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js (for MCP Inspector)
- Git

### 1. Clone and Setup

```bash
git clone https://github.com/srknampalli/pdf-summarizer-mcp.git
cd pdf-extraction-mcp
```

### 2. Environment Setup

**Option A: Using uv (Recommended)**
```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies
uv sync

# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

**Option B: Using pip**
```bash or cmd
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or 
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration (Option 1)
OPENAI_API_KEY=your-openai-api-key-here

# Azure OpenAI Configuration (Option 2)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_API_VERSION=2024-02-preview

# Phoenix Tracing 
PHOENIX_ENDPOINT=your-phoenix-endpoint
PHOENIX_API_KEY=your-phoenix-api-key


```

### 4. Test MCP Servers

```bash
# Test PDF processing server
npx @modelcontextprotocol/inspector --cli python server/pdf_processing_server.py --method tools/list

# Test summarizer server
npx @modelcontextprotocol/inspector --cli python server/summarizer_qna_server.py --method tools/list
```

### 5. Run the Application

**Streamlit Web UI:**
```bash
streamlit run app_streamlit.py
```

**CLI Interface:**
```bash
python cli.py --pdf "path/to/your/document.pdf"
```

## 🔧 Usage

### Web Interface

1. Open your browser to `http://localhost:8501`
2. Upload a PDF file
3. Click "Get Summary" for document summarization
4. Ask questions in the Q&A section

### CLI Interface

```bash
# Basic usage
python cli.py --pdf document.pdf

# Custom chunk size
python cli.py --pdf document.pdf --chunk-size 500

# Specific pages only
python cli.py --pdf document.pdf --pages "1,2,3"
```

### Using MCP Inspector (Development)

Test individual MCP servers:

```bash
# Interactive UI testing
npx @modelcontextprotocol/inspector python server/pdf_processing_server.py

# CLI testing
npx @modelcontextprotocol/inspector --cli python server/pdf_processing_server.py --method tools/call --tool-name extract_pdf_contents --tool-arg pdf_path="test.pdf"
```

## 📁 Project Structure

```
pdf-extraction-mcp/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .env.example
├── 📄 app_streamlit.py          # Streamlit web interface
├── 📄 cli.py                    # Command-line interface
├── 📁 modules/
│   └── 📄 pipeline.py           # Main processing pipeline
├── 📁 server/                   # MCP microservices
│   ├── 📄 pdf_extractor.py      # PDF text extraction
│   ├── 📄 pdf_processing_server.py  # Processing & embedding
│   ├── 📄 vector_store.py       # Vector storage class
│   └── 📄 summarizer_qna_server.py  # Summarization & Q&A
└── 📁 tests/                    # Unit tests (if any)
```

## 🛠️ Development

### Adding New MCP Tools

1. Create a new tool in the appropriate server file:
```python
@mcp.tool()
def your_new_tool(param1: str, param2: int) -> str:
    """Tool description"""
    # Tool implementation
    return result
```

2. Test with MCP Inspector:
```bash
npx @modelcontextprotocol/inspector python server/your_server.py
```

### Debugging

Enable debug mode by setting environment variables:
```bash
export DEBUG=true
export OTEL_LOG_LEVEL=debug
```

### Testing MCP Servers

```bash
# List all available tools
npx @modelcontextprotocol/inspector --cli python server/pdf_processing_server.py --method tools/list

# Test specific tool
npx @modelcontextprotocol/inspector --cli python server/pdf_processing_server.py --method tools/call --tool-name extract_pdf_contents --tool-arg pdf_path="sample.pdf"
```

## 🔍 Troubleshooting

### Common Issues

**1. "Connection closed" errors:**
- Ensure MCP servers can start individually
- Check for missing dependencies
- Verify API keys are set correctly

**2. SSL/Certificate errors in corporate environments:**
```bash
export ANONYMIZED_TELEMETRY=True/False
export CHROMA_TELEMETRY= True/False
```

**3. Import errors:**
```bash
# Verify all dependencies are installed
pip list | grep -E "(mcp|streamlit|langchain|chromadb)"
```




## 📋 Requirements

### System Requirements
- Python 3.8+
- Node.js 14+ (for MCP Inspector)
- 2GB+ RAM (for vector operations)
- Storage for vector database

### API Access
- OpenAI API key **OR** Azure OpenAI access
- Phoenix endpoint (optional, for observability)

### Core Dependencies
- `mcp` - Model Context Protocol
- `streamlit` - Web interface
- `langchain-openai` - LLM integration
- `chromadb` - Vector database
- `PyPDF2` - PDF processing

### Optional Dependencies
- `pytesseract` - OCR for scanned PDFs
- `phoenix-otel` - Observability tracing
- `python-dotenv` - Environment management

---

