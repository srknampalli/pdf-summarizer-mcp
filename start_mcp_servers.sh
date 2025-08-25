#!/bin/bash

# Ensure logs directory exists
mkdir -p logs
echo "Ensured 'logs' directory exists."

echo "Attempting to start MCP servers..."

# Server 1: QnA / Summarizer
.venv/Scripts/python -m mcp.server \
  --tool server/summarizer_qna_server.py \
  --host 127.0.0.1 \
  --port 5001 \
  > logs/mcp_server_qna_summarizer.log 2>&1 &

echo "Launched Server 1 (QnA/Summarizer). Log: logs/mcp_server_qna_summarizer.log"

# Server 2: PDF Extractor / Embedder / Chunker
.venv/Scripts/python -m mcp.server \
  --tool server/pdf_extractor.py \
  --tool server/pdf_processing_server.py \
  --host 127.0.0.1 \
  --port 5002 \
  > logs/mcp_server_extract_embed_chunk.log 2>&1 &

echo "Launched Server 2 (Extractor/Embedder). Log: logs/mcp_server_extract_embed_chunk.log"

# Server 3: External Tools
.venv/Scripts/python -m mcp.server \
  --tool server/external_tools_server.py \
  --host 127.0.0.1 \
  --port 5003 \
  > logs/mcp_server_external_tools.log 2>&1 &

echo "Launched Server 3 (External Tools). Log: logs/mcp_server_external_tools.log"

echo "All MCP servers launched. Check logs for details."
