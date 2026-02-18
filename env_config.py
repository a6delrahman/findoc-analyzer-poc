import os
from dotenv import load_dotenv

def get_config():
    load_dotenv()

    config = {
        "azure_tenant_id": os.getenv("AZURE_TENANT_ID"),
        "azure_openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_openai_model": os.getenv("AZURE_OPENAI_MODEL"),
        "azure_openai_model_4_1": os.getenv("AZURE_OPENAI_MODEL_4_1"),
        "azure_cog_scope": os.getenv("AZURE_COG_SCOPE"),
        "azure_doc_intel_endpoint": os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT"),
        "phoenix_api_key": os.getenv("PHOENIX_API_KEY"),
        "phoenix_collector_endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
        "prompt_split": os.getenv("PROMPT_SPLIT"),
        "prompt_steuererklaerung": os.getenv("PROMPT_STEUERERKLAERUNG"),
        "prompt_doc_extraction": os.getenv("PROMPT_DOC_EXTRACTION"),
        # New API key fields
        "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_doc_intel_key": os.getenv("AZURE_DOC_INTEL_KEY"),
    }

    return config