import os
import logging
import requests
from typing import List, Dict, Any, Optional
from env_config import get_config

logging.basicConfig(level=logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
config = get_config()


class PhoenixPromptService:
    """Service for fetching prompts from Arize Phoenix."""
    
    def __init__(self):
        self.api_key = config.get('phoenix_api_key')
        self.endpoint = config.get('phoenix_collector_endpoint')
        
        if not self.api_key or not self.endpoint:
            logger.warning("Phoenix credentials not configured.")
    
    def _normalize_content(self, content: Any) -> str:
        """Normalize Phoenix message content to a plain string."""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            return ''.join(text_parts)
        
        return str(content) if content else ''
    
    def _normalize_messages(self, messages: List[Dict]) -> List[Dict[str, str]]:
        """Normalize all messages to have string content."""
        return [
            {
                'role': msg.get('role', 'user'),
                'content': self._normalize_content(msg.get('content', ''))
            }
            for msg in messages
        ]
    
    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Fetches a prompt template from Phoenix and returns the system/user content as string.
        Falls back to None if Phoenix is unavailable.
        """
        try:
            if not self.api_key or not self.endpoint:
                return None
                
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Get all prompts
            response = requests.get(f"{self.endpoint}/v1/prompts", headers=headers, timeout=10)
            response.raise_for_status()
            prompts = response.json().get("data", [])
            
            # Find prompt by name
            prompt_id = next((p["id"] for p in prompts if p["name"] == prompt_name), None)
            if not prompt_id:
                logger.warning(f"Prompt '{prompt_name}' not found in Phoenix")
                return None
            
            # Get latest version
            response = requests.get(f"{self.endpoint}/v1/prompts/{prompt_id}/latest", headers=headers, timeout=10)
            response.raise_for_status()
            
            messages = response.json().get("data", {}).get("template", {}).get("messages", [])
            normalized = self._normalize_messages(messages)
            
            # Return combined content (typically system prompt)
            return "\n\n".join(msg['content'] for msg in normalized if msg['content'])
            
        except Exception as e:
            logger.warning(f"Failed to fetch prompt '{prompt_name}' from Phoenix: {e}")
            return None


# Singleton instance
_prompt_service: Optional[PhoenixPromptService] = None

def get_prompt_service() -> PhoenixPromptService:
    """Get or create the Phoenix prompt service singleton."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PhoenixPromptService()
    return _prompt_service