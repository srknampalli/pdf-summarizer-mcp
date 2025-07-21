#!/bin/bash

# This line creates the 'logs' directory if it's missing.
mkdir -p logs
echo "Ensured 'logs' directory exists."

echo "Attempting to start MCP servers..."

# Start MCP Server 1 - using the CORRECT path for Windows
.venv/Scripts/python -m mcp.server \
  --tool server/summarizer_qna_server.py \
  --host 127.0.0.1 \
  --port 5001 \
  > logs/mcp_server_qna_summarizer.log 2>&1 &

echo "Launched Server 1 (QnA/Summarizer). Log is at: logs/mcp_server_qna_summarizer.log"

# Start MCP Server 2 - using the CORRECT path for Windows
.venv/Scripts/python -m mcp.server \
  --tool server/pdf_extractor.py \
  --tool server/pdf_processing_server.py \
  --host 127.0.0.1 \
  --port 5002 \
  > logs/mcp_server_extract_embed_chunk.log 2>&1 &

echo "Launched Server 2 (Extractor/Embedder). Log is at: logs/mcp_server_extract_embed_chunk.log"

echo "Script finished. Check the log files to see server status."