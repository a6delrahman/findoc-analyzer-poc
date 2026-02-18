"""
Shared clients module - initializes Azure OpenAI client with Phoenix tracing.
All modules should import the client from here to ensure proper tracing.
"""
import os
import logging
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from env_config import get_config

# Optional Phoenix imports
try:
    from phoenix.otel import register
    from openinference.instrumentation.openai import OpenAIInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = get_config()
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, config['azure_cog_scope'])

# ==========================================
# PHOENIX TRACING (initialized once)
# ==========================================

_tracer_initialized = False

def init_tracing(project_name: str = "hbl_poc"):
    """Initialize Phoenix tracing. Call this BEFORE creating clients."""
    global _tracer_initialized
    
    if _tracer_initialized:
        return
    
    if not PHOENIX_AVAILABLE:
        logger.warning("Phoenix packages not installed. Tracing disabled.")
        _tracer_initialized = True
        return
    
    phoenix_api_key = config.get('phoenix_api_key')
    phoenix_endpoint = config.get('phoenix_collector_endpoint')
    
    if not phoenix_api_key or not phoenix_endpoint:
        logger.warning("Phoenix credentials not configured. Tracing disabled.")
        _tracer_initialized = True
        return
    
    try:
        # Set environment variables for Phoenix
        os.environ["PHOENIX_API_KEY"] = phoenix_api_key
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint
        
        tracer_provider = register(
            project_name=project_name,
            auto_instrument=True,
            set_global_tracer_provider=False
        )
        
        # Instrument OpenAI - this will trace ALL AzureOpenAI clients created after this
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        
        logger.info(f"Phoenix tracing initialized for project: {project_name}")
        _tracer_initialized = True
        
    except Exception as e:
        logger.error(f"Failed to initialize Phoenix tracing: {e}")
        _tracer_initialized = True


# ==========================================
# SHARED CLIENTS
# ==========================================

# Initialize tracing BEFORE creating clients
init_tracing()

# Shared Azure OpenAI client (traced)
llm_client = AzureOpenAI(
    azure_endpoint=config['azure_openai_endpoint'],
    azure_ad_token_provider=token_provider,
    api_version=config['azure_openai_api_version'],
    timeout=300.0,
)

# Shared Document Intelligence client
doc_intel_client = DocumentIntelligenceClient(
    endpoint=config['azure_doc_intel_endpoint'],
    credential=credential
)

# Model deployment name
MODEL_DEPLOYMENT = config['azure_openai_model']