import json
import datetime
import os
from pathlib import Path
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SimpleSpanProcessor, SpanExportResult
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

class EnhancedMCPTraceExporter(SpanExporter):
    """Enhanced trace exporter for comprehensive MCP PDF Pipeline monitoring"""
    
    def __init__(self, file_path="logs/mcp_pdf_traces.txt"):
        self.file_path = Path(file_path)
        
        # Auto-create directory if it doesn't exist
        if self.file_path.parent != Path('.'):
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created trace directory: {self.file_path.parent}")
        
        # Create file with header if it doesn't exist
        if not self.file_path.exists():
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Enhanced MCP PDF Assistant Traces - Started {datetime.datetime.now()} ===\n\n")
            print(f"📄 Created trace file: {self.file_path}")
    
    def export(self, spans):
        with open(self.file_path, "a", encoding="utf-8") as f:
            for span in spans:
                # Extract attributes
                attrs = dict(span.attributes) if span.attributes else {}
                
                # Calculate duration
                duration_ms = 0
                if span.start_time and span.end_time:
                    duration_ms = (span.end_time - span.start_time) / 1000000
                
                # Format timestamp
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                
                # Determine operation type and add emoji
                operation_emoji = self._get_operation_emoji(span.name)
                
                # Create readable trace entry
                trace_line = f"[{timestamp}] {operation_emoji} {span.name} ({duration_ms:.2f}ms)\n"
                
                # Add span details
                if span.context:
                    trace_line += f"  ├── Trace ID: {format(span.context.trace_id, '032x')[:8]}...\n"
                    trace_line += f"  ├── Span ID:  {format(span.context.span_id, '016x')[:8]}...\n"
                
                # Add performance classification
                perf_indicator = self._get_performance_indicator(span.name, duration_ms)
                if perf_indicator:
                    trace_line += f"  ├── Performance: {perf_indicator}\n"
                
                # Add meaningful attributes with enhanced categorization
                for key, value in attrs.items():
                    if key in ['doc_id', 'pdf_path', 'question', 'num_chunks', 'fallback_used', 'text_length', 'chunk_size', 'top_k', 'chunks_found']:
                        trace_line += f"  ├── {key}: {value}\n"
                    elif key in ['text_preview', 'context_preview', 'summary_preview', 'answer_preview', 'first_chunk_preview']:
                        # Show first 100 chars for content previews
                        preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        trace_line += f"  ├── {key}: {preview}\n"
                    elif key.startswith('llm.'):
                        # Enhanced LLM monitoring
                        if key == 'llm.token_usage':
                            # Parse token usage if it's a string
                            try:
                                if isinstance(value, str):
                                    token_data = json.loads(value)
                                else:
                                    token_data = value
                                
                                if isinstance(token_data, dict):
                                    prompt_tokens = token_data.get('prompt_tokens', 0)
                                    completion_tokens = token_data.get('completion_tokens', 0)
                                    total_tokens = token_data.get('total_tokens', prompt_tokens + completion_tokens)
                                    
                                    trace_line += f"  ├── 🪙 Token Usage: {prompt_tokens}p + {completion_tokens}c = {total_tokens} total\n"
                                    
                                    # Add cost estimation (rough OpenAI pricing)
                                    estimated_cost = self._estimate_cost(prompt_tokens, completion_tokens)
                                    if estimated_cost:
                                        trace_line += f"  ├── 💰 Estimated Cost: ~${estimated_cost:.4f}\n"
                                else:
                                    trace_line += f"  ├── 🪙 Token Usage: {value}\n"
                            except:
                                trace_line += f"  ├── 🪙 Token Usage: {value}\n"
                                
                        elif key == 'llm.model_name':
                            trace_line += f"  ├── 🤖 Model: {value}\n"
                        elif key == 'llm.prompts':
                            prompt_preview = str(value)[:150] + "..." if len(str(value)) > 150 else str(value)
                            trace_line += f"  ├── 📝 Prompt: {prompt_preview}\n"
                        elif key == 'llm.responses':
                            response_preview = str(value)[:150] + "..." if len(str(value)) > 150 else str(value)
                            trace_line += f"  ├── 💭 Response: {response_preview}\n"
                        elif key == 'llm.temperature':
                            trace_line += f"  ├── 🌡️  Temperature: {value}\n"
                        elif key == 'llm.max_tokens':
                            trace_line += f"  ├── 📏 Max Tokens: {value}\n"
                
                # Enhanced status reporting
                status = span.status.status_code.name if span.status else "UNSET"
                status_emoji = self._get_status_emoji(status)
                trace_line += f"  └── {status_emoji} Status: {status}"
                
                # Add error details if present
                if status == "ERROR" and span.status and span.status.description:
                    trace_line += f" - {span.status.description}"
                
                trace_line += "\n\n"
                
                f.write(trace_line)
        
        return SpanExportResult.SUCCESS
    
    def _get_operation_emoji(self, operation_name):
        """Get emoji for different operation types"""
        emoji_map = {
            'PDF Text Extraction': '📄',
            'Text Chunking': '✂️',
            'Embedding Generation': '🧠',
            'Vector Similarity Search': '🔍',
            'Question & Answer Pipeline': '❓',
            'Document Summarization': '📝',
            'LLM Answer Generation': '🤖',
            'LLM Fallback': '🔄',
            'Question Embedding': '🎯',
            'MCP Tool Call': '🔧'
        }
        
        for pattern, emoji in emoji_map.items():
            if pattern in operation_name:
                return emoji
        return '⚙️'  # Default emoji
    
    def _get_performance_indicator(self, operation_name, duration_ms):
        """Get performance indicator based on operation type and duration"""
        thresholds = {
            'PDF Text Extraction': {'good': 3000, 'warning': 10000},
            'Text Chunking': {'good': 1000, 'warning': 5000},
            'Embedding Generation': {'good': 5000, 'warning': 20000},
            'Vector Similarity Search': {'good': 500, 'warning': 2000},
            'Document Summarization': {'good': 8000, 'warning': 15000},
            'LLM Answer Generation': {'good': 3000, 'warning': 10000},
            'Question Embedding': {'good': 1000, 'warning': 3000}
        }
        
        for op_type, limits in thresholds.items():
            if op_type in operation_name:
                if duration_ms <= limits['good']:
                    return "🟢 Good"
                elif duration_ms <= limits['warning']:
                    return "🟡 Slow"
                else:
                    return "🔴 Very Slow"
        return None
    
    def _get_status_emoji(self, status):
        """Get emoji for different status types"""
        status_map = {
            'OK': '✅',
            'ERROR': '❌',
            'UNSET': '⚪',
            'CANCELLED': '🚫'
        }
        return status_map.get(status, '❓')
    
    def _estimate_cost(self, prompt_tokens, completion_tokens):
        """Rough cost estimation based on OpenAI pricing"""
        # Rough pricing (as of 2024) - adjust based on your model
        # GPT-4: $0.03/1K prompt tokens, $0.06/1K completion tokens
        # GPT-3.5: $0.001/1K prompt tokens, $0.002/1K completion tokens
        
        # Using GPT-4 pricing as default
        prompt_cost = (prompt_tokens / 1000) * 0.03
        completion_cost = (completion_tokens / 1000) * 0.06
        return prompt_cost + completion_cost

def setup_mcp_tracing(trace_file="logs/mcp_pdf_traces.txt"):
    """
    Set up enhanced OpenTelemetry tracing for MCP PDF Pipeline
    """
    print(f"🔍 Setting up enhanced MCP PDF tracing to: {trace_file}")
    
    # Initialize OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    
    # Add our enhanced exporter
    trace_exporter = EnhancedMCPTraceExporter(trace_file)
    span_processor = SimpleSpanProcessor(trace_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument LangChain with enhanced settings
    LangchainInstrumentor().instrument(
        # Enable content logging (set to False for privacy)
        trace_content=True,
        # Enable token usage tracking
        enrich_token_usage=True
    )
    
    print("✅ Enhanced MCP PDF tracing enabled")
    print("📊 Monitoring: Performance, Token Usage, Costs, Errors")
    return trace_file

def view_traces(trace_file="logs/mcp_pdf_traces.txt", lines=50):
    """View recent traces with enhanced formatting"""
    try:
        with open(trace_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
            print("".join(recent))
    except FileNotFoundError:
        print(f"No trace file found at {trace_file}")

def clear_traces(trace_file="logs/mcp_pdf_traces.txt"):
    """Clear trace file"""
    trace_path = Path(trace_file)
    if trace_path.parent != Path('.'):
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created trace directory: {trace_path.parent}")
    
    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Enhanced MCP PDF Assistant Traces - Cleared {datetime.datetime.now()} ===\n\n")
    print(f"🧹 Cleared traces in {trace_file}")

def analyze_traces(trace_file="logs/mcp_pdf_traces.txt"):
    """Analyze trace file for performance insights"""
    try:
        with open(trace_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract performance data
        operations = {}
        total_cost = 0
        error_count = 0
        
        lines = content.split('\n')
        for line in lines:
            if '] 🔴 Very Slow' in line or '] 🟡 Slow' in line or '] 🟢 Good' in line:
                # Extract operation name and performance
                if 'Very Slow' in line:
                    operations['slow_ops'] = operations.get('slow_ops', 0) + 1
                elif 'Slow' in line:
                    operations['warning_ops'] = operations.get('warning_ops', 0) + 1
                else:
                    operations['good_ops'] = operations.get('good_ops', 0) + 1
            
            if '💰 Estimated Cost:' in line:
                try:
                    cost_str = line.split('$')[1].split()[0]
                    total_cost += float(cost_str)
                except:
                    pass
            
            if '❌ Status: ERROR' in line:
                error_count += 1
        
        print("📊 Trace Analysis Summary:")
        print(f"  🟢 Good Performance: {operations.get('good_ops', 0)} operations")
        print(f"  🟡 Slow Performance: {operations.get('warning_ops', 0)} operations") 
        print(f"  🔴 Very Slow: {operations.get('slow_ops', 0)} operations")
        print(f"  💰 Total Estimated Cost: ${total_cost:.4f}")
        print(f"  ❌ Errors: {error_count}")
        
        return {
            'good_ops': operations.get('good_ops', 0),
            'slow_ops': operations.get('warning_ops', 0) + operations.get('slow_ops', 0),
            'total_cost': total_cost,
            'error_count': error_count
        }
        
    except FileNotFoundError:
        print(f"No trace file found at {trace_file}")
        return {}

# Export for easy import
__all__ = ['setup_mcp_tracing', 'view_traces', 'clear_traces', 'analyze_traces']