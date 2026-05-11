"""
Semantic caption generation module for RAG results.
Supports both mock and real LLM implementations.
"""

import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import os

logger = logging.getLogger(__name__)


class CaptionGenerator(ABC):
    """Abstract base class for caption generators."""
    
    @abstractmethod
    def generate_caption(self, content: str, max_length: int = 150) -> str:
        """Generate a semantic caption for the given content."""
        pass


class MockLLMCaption(CaptionGenerator):
    """
    Mock LLM that generates placeholder captions without API calls.
    Useful for testing and demos without API keys.
    """
    
    def __init__(self):
        """Initialize mock LLM caption generator."""
        self.model_name = "mock-llm"
        logger.info(f"Initialized {self.model_name} - Using MOCK captions (no API key needed)")
    
    def generate_caption(self, content: str, max_length: int = 150) -> str:
        """
        Generate a mock caption based on content preview.
        
        Args:
            content: The chunk content to caption
            max_length: Maximum length of caption (used for indication)
        
        Returns:
            A mock semantic caption
        """
        # Extract first meaningful sentence
        sentences = content.split('.')
        first_sentence = sentences[0].strip() if sentences else content[:max_length]
        
        # Ensure it fits max length
        if len(first_sentence) > max_length:
            first_sentence = first_sentence[:max_length-3] + "..."
        
        # Add mock indicator (for demo purposes)
        caption = f"{first_sentence} [Mock LLM - Replace with real API key for production]"
        return caption[:max_length]


class AzureOpenAICaption(CaptionGenerator):
    """
    Real semantic caption generator using Azure OpenAI API.
    Requires AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "gpt-35-turbo"
    ):
        """
        Initialize Azure OpenAI caption generator.
        
        Args:
            api_key: Azure OpenAI API key (or from env AZURE_OPENAI_KEY)
            endpoint: Azure OpenAI endpoint (or from env AZURE_OPENAI_ENDPOINT)
            api_version: API version to use
            deployment_name: Deployment model name
        
        Raises:
            ValueError: If API key or endpoint not provided and not in environment
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.model_name = "Azure OpenAI (GPT-3.5-turbo)"
        
        if not self.api_key or not self.endpoint:
            raise ValueError(
                "Azure OpenAI credentials not provided. "
                "Set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables "
                "or pass them as arguments."
            )
        
        try:
            from azure.ai.openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            logger.info(f"Initialized {self.model_name} caption generator")
        except ImportError:
            raise ImportError(
                "azure-ai-openai not installed. "
                "Install with: pip install azure-ai-openai"
            )
    
    def generate_caption(self, content: str, max_length: int = 150) -> str:
        """
        Generate a semantic caption using Azure OpenAI.
        
        Args:
            content: The chunk content to caption
            max_length: Maximum length of caption
        
        Returns:
            AI-generated semantic caption
        """
        try:
            prompt = f"""Summarize this text in {max_length} characters or less:

{content}

Provide only the summary, nothing else."""
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            caption = response.choices[0].message.content.strip()
            
            # Ensure it fits max length
            if len(caption) > max_length:
                caption = caption[:max_length-3] + "..."
            
            return caption
        
        except Exception as e:
            logger.warning(f"Failed to generate caption via Azure OpenAI: {e}")
            # Fallback to mock
            return MockLLMCaption().generate_caption(content, max_length)


class OpenAICaption(CaptionGenerator):
    """
    Real semantic caption generator using OpenAI API.
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize OpenAI caption generator.
        
        Args:
            api_key: OpenAI API key (or from env OPENAI_API_KEY)
            model: Model to use (gpt-3.5-turbo, gpt-4, etc.)
        
        Raises:
            ValueError: If API key not provided and not in environment
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.model_name = f"OpenAI ({model})"
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass it as argument."
            )
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized {self.model_name} caption generator")
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
    
    def generate_caption(self, content: str, max_length: int = 150) -> str:
        """
        Generate a semantic caption using OpenAI.
        
        Args:
            content: The chunk content to caption
            max_length: Maximum length of caption
        
        Returns:
            AI-generated semantic caption
        """
        try:
            prompt = f"""Summarize this text in {max_length} characters or less:

{content}

Provide only the summary, nothing else."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            caption = response.choices[0].message.content.strip()
            
            # Ensure it fits max length
            if len(caption) > max_length:
                caption = caption[:max_length-3] + "..."
            
            return caption
        
        except Exception as e:
            logger.warning(f"Failed to generate caption via OpenAI: {e}")
            # Fallback to mock
            return MockLLMCaption().generate_caption(content, max_length)


def get_caption_generator(
    provider: str = "mock",
    **kwargs
) -> CaptionGenerator:
    """
    Factory function to get a caption generator.
    
    Args:
        provider: "mock", "azure", or "openai"
        **kwargs: Additional arguments for the specific provider
    
    Returns:
        CaptionGenerator instance
    
    Example:
        # Use mock (no API key needed)
        generator = get_caption_generator("mock")
        
        # Use Azure OpenAI (requires API key in env)
        generator = get_caption_generator("azure")
        
        # Use OpenAI (requires API key in env)
        generator = get_caption_generator("openai")
    """
    provider = provider.lower()
    
    if provider == "mock":
        return MockLLMCaption()
    elif provider == "azure":
        return AzureOpenAICaption(**kwargs)
    elif provider == "openai":
        return OpenAICaption(**kwargs)
    else:
        logger.warning(f"Unknown provider '{provider}', using mock")
        return MockLLMCaption()
