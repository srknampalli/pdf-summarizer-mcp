import os

def setup_otel_langchain_tracing():
    """
    Setup OpenTelemetry LangChain instrumentation using environment variables
    Returns: bool - True if successful, False if failed
    """
    
    try:
        
        phoenix_endpoint = os.environ.get('PHOENIX_ENDPOINT')
        phoenix_api_key = os.environ.get('PHOENIX_API_KEY')
        
        # Check if credentials are provided
        if not phoenix_endpoint or not phoenix_api_key:
            print("❌ Phoenix credentials not found in environment variables")
            print("Set them using:")
            print("set PHOENIX_ENDPOINT=https://your-phoenix-endpoint/v1/traces")
            print("set PHOENIX_API_KEY=your-api-key")
            return False
        
    
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        
        # Import instrumentation
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        
        # Create resource

        resource = Resource.create({
        "service.name": "pdf-assistant-streamlit",
        "service.version": "1.0.0", 
        "service.instance.id": "streamlit-1",
        "phoenix.project.name": "pdf-assistant-streamlit",
        "deployment.environment": "production"
            })

        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Configure OTLP exporter using environment variables
        otlp_exporter = OTLPSpanExporter(
            endpoint=phoenix_endpoint,
            headers={
                "Authorization": f"Bearer {phoenix_api_key}"
            },
            timeout=30
        )
        
        # Add batch span processor
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            export_timeout_millis=30000,
            max_export_batch_size=512
        )
        tracer_provider.add_span_processor(span_processor)
        
        # ✅ INSTRUMENT LANGCHAIN
        LangchainInstrumentor().instrument()
        
        # ✅ INSTRUMENT HTTP REQUESTS  
        RequestsInstrumentor().instrument()
        
        print("✅ OpenTelemetry LangChain instrumentation successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing OpenTelemetry packages: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Phoenix setup failed: {e}")
        return False

def get_phoenix_dashboard_url():
    """Get Phoenix dashboard URL from environment"""
    phoenix_endpoint = os.environ.get('PHOENIX_ENDPOINT', '')
    
    # Extract dashboard URL from traces endpoint
    if '/v1/traces' in phoenix_endpoint:
        return phoenix_endpoint.replace('/v1/traces', '')
    elif phoenix_endpoint:
        return phoenix_endpoint
    else:
        return None

def check_environment_setup():
    """Check if environment variables are properly set"""
    
    phoenix_endpoint = os.environ.get('PHOENIX_ENDPOINT')
    phoenix_api_key = os.environ.get('PHOENIX_API_KEY')
    

    
    # if phoenix_endpoint:
    #     print(f"Endpoint: {phoenix_endpoint}")
    # if phoenix_api_key:
    #     print(f"API Key: {phoenix_api_key[:20]}...")
    
    return bool(phoenix_endpoint and phoenix_api_key)