"""
AILEE Trust Pipeline - AI Framework Integrations
Version: 1.0.0

Easy-to-use adapters for integrating AILEE with popular AI frameworks
and services including OpenAI, Anthropic, HuggingFace, and LangChain.

These adapters automatically extract confidence scores, handle multi-model
ensembles, and provide seamless trust validation for AI outputs.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Sequence
from dataclasses import dataclass
import warnings


# ===========================
# Base AI Adapter
# ===========================

@dataclass
class AIResponse:
    """
    Standardized AI response format for AILEE integration.
    
    Attributes:
        value: The primary output value (numeric or converted to numeric)
        confidence: Confidence score (0.0-1.0)
        raw_response: Original response from AI system
        metadata: Additional context and metadata
    """
    value: float
    confidence: float
    raw_response: Any
    metadata: Dict[str, Any]


class AIAdapter:
    """
    Base class for AI framework adapters.
    
    Provides common functionality for extracting values and confidence
    scores from various AI frameworks.
    """
    
    def __init__(self, value_extractor: Optional[Callable] = None,
                 confidence_extractor: Optional[Callable] = None):
        """
        Initialize AI adapter with custom extractors.
        
        Args:
            value_extractor: Function to extract numeric value from AI response
            confidence_extractor: Function to extract confidence from AI response
        """
        self.value_extractor = value_extractor
        self.confidence_extractor = confidence_extractor
    
    def extract_response(self, response: Any, context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """
        Extract standardized response from AI output.
        
        Args:
            response: Raw AI response
            context: Additional context for extraction
            
        Returns:
            AIResponse: Standardized response object
        """
        raise NotImplementedError("Subclasses must implement extract_response")


# ===========================
# OpenAI Integration
# ===========================

class OpenAIAdapter(AIAdapter):
    """
    Adapter for OpenAI API responses (GPT-4, GPT-3.5, etc.).
    
    Automatically extracts confidence from logprobs or uses custom extractors.
    Supports both chat completions and legacy completions.
    
    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> adapter = OpenAIAdapter()
        >>> 
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Rate quality 0-100: ..."}],
        ...     logprobs=True,
        ...     top_logprobs=1
        ... )
        >>> ai_response = adapter.extract_response(response)
        >>> # ai_response.value = extracted numeric value
        >>> # ai_response.confidence = computed from logprobs
    """
    
    def __init__(self, value_extractor: Optional[Callable] = None,
                 confidence_extractor: Optional[Callable] = None,
                 use_logprobs: bool = True):
        """
        Initialize OpenAI adapter.
        
        Args:
            value_extractor: Custom function to extract value from response
            confidence_extractor: Custom function to extract confidence
            use_logprobs: Use logprobs for confidence calculation (default: True)
        """
        super().__init__(value_extractor, confidence_extractor)
        self.use_logprobs = use_logprobs
    
    def extract_response(self, response: Any, context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """
        Extract value and confidence from OpenAI response.
        
        Args:
            response: OpenAI API response object
            context: Optional context (e.g., {"extract_pattern": r"\\d+"})
            
        Returns:
            AIResponse: Standardized response with value and confidence
        """
        context = context or {}
        
        # Extract text content
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                content = choice.message.content
            elif hasattr(choice, 'text'):
                content = choice.text
            else:
                content = str(choice)
        else:
            content = str(response)
        
        # Extract value
        if self.value_extractor:
            value = self.value_extractor(response, content, context)
        else:
            value = self._default_value_extraction(content, context)
        
        # Extract confidence
        if self.confidence_extractor:
            confidence = self.confidence_extractor(response, content, context)
        elif self.use_logprobs and hasattr(response, 'choices') and response.choices:
            confidence = self._extract_logprob_confidence(response.choices[0])
        else:
            confidence = 0.8  # Default moderate confidence
        
        metadata = {
            "model": getattr(response, 'model', 'unknown'),
            "content": content,
            "framework": "openai",
        }
        
        if hasattr(response, 'usage'):
            metadata["tokens"] = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }
        
        return AIResponse(
            value=value,
            confidence=confidence,
            raw_response=response,
            metadata=metadata
        )
    
    def _default_value_extraction(self, content: str, context: Dict[str, Any]) -> float:
        """Extract first numeric value from content."""
        import re
        pattern = context.get('extract_pattern', r'-?\d+\.?\d*')
        match = re.search(pattern, content)
        if match:
            return float(match.group(0))
        raise ValueError(f"Could not extract numeric value from: {content[:100]}")
    
    def _extract_logprob_confidence(self, choice: Any) -> float:
        """
        Extract confidence from logprobs.
        Converts log probability to linear probability (0.0-1.0).
        """
        if hasattr(choice, 'logprobs') and choice.logprobs:
            if hasattr(choice.logprobs, 'content') and choice.logprobs.content:
                # Chat completions with logprobs
                logprobs = [token.logprob for token in choice.logprobs.content if hasattr(token, 'logprob')]
                if logprobs:
                    import math
                    # Average probability across tokens
                    avg_prob = sum(math.exp(lp) for lp in logprobs) / len(logprobs)
                    return min(1.0, max(0.0, avg_prob))
        return 0.8  # Default if logprobs unavailable


# ===========================
# Anthropic Integration
# ===========================

class AnthropicAdapter(AIAdapter):
    """
    Adapter for Anthropic Claude API responses.
    
    Extracts confidence from stop_reason and usage patterns.
    
    Example:
        >>> import anthropic
        >>> client = anthropic.Anthropic()
        >>> adapter = AnthropicAdapter()
        >>> 
        >>> response = client.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Rate quality 0-100: ..."}]
        ... )
        >>> ai_response = adapter.extract_response(response)
    """
    
    def extract_response(self, response: Any, context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """
        Extract value and confidence from Anthropic response.
        
        Args:
            response: Anthropic API response object
            context: Optional context for extraction
            
        Returns:
            AIResponse: Standardized response
        """
        context = context or {}
        
        # Extract text content
        if hasattr(response, 'content') and response.content:
            if isinstance(response.content, list):
                content = ' '.join(block.text for block in response.content if hasattr(block, 'text'))
            else:
                content = str(response.content)
        else:
            content = str(response)
        
        # Extract value
        if self.value_extractor:
            value = self.value_extractor(response, content, context)
        else:
            value = self._default_value_extraction(content, context)
        
        # Extract confidence
        if self.confidence_extractor:
            confidence = self.confidence_extractor(response, content, context)
        else:
            confidence = self._estimate_confidence(response)
        
        metadata = {
            "model": getattr(response, 'model', 'unknown'),
            "content": content,
            "framework": "anthropic",
            "stop_reason": getattr(response, 'stop_reason', None),
        }
        
        if hasattr(response, 'usage'):
            metadata["tokens"] = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        
        return AIResponse(
            value=value,
            confidence=confidence,
            raw_response=response,
            metadata=metadata
        )
    
    def _default_value_extraction(self, content: str, context: Dict[str, Any]) -> float:
        """Extract first numeric value from content."""
        import re
        pattern = context.get('extract_pattern', r'-?\d+\.?\d*')
        match = re.search(pattern, content)
        if match:
            return float(match.group(0))
        raise ValueError(f"Could not extract numeric value from: {content[:100]}")
    
    def _estimate_confidence(self, response: Any) -> float:
        """Estimate confidence based on stop_reason and other indicators."""
        if hasattr(response, 'stop_reason'):
            if response.stop_reason == 'end_turn':
                return 0.85  # Completed naturally
            elif response.stop_reason == 'max_tokens':
                return 0.70  # Truncated, less confident
            elif response.stop_reason == 'stop_sequence':
                return 0.85  # Stopped at expected point
        return 0.80  # Default moderate confidence


# ===========================
# HuggingFace Integration
# ===========================

class HuggingFaceAdapter(AIAdapter):
    """
    Adapter for HuggingFace model outputs.
    
    Supports text generation, classification, and other tasks.
    Extracts confidence from model scores when available.
    
    Example:
        >>> from transformers import pipeline
        >>> classifier = pipeline("text-classification")
        >>> adapter = HuggingFaceAdapter(task="classification")
        >>> 
        >>> result = classifier("This is great!")
        >>> ai_response = adapter.extract_response(result[0])
    """
    
    def __init__(self, task: str = "generation",
                 value_extractor: Optional[Callable] = None,
                 confidence_extractor: Optional[Callable] = None):
        """
        Initialize HuggingFace adapter.
        
        Args:
            task: Task type ("generation", "classification", "qa", etc.)
            value_extractor: Custom value extraction function
            confidence_extractor: Custom confidence extraction function
        """
        super().__init__(value_extractor, confidence_extractor)
        self.task = task
    
    def extract_response(self, response: Any, context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """
        Extract value and confidence from HuggingFace model output.
        
        Args:
            response: Model output (dict, list, or object)
            context: Optional context
            
        Returns:
            AIResponse: Standardized response
        """
        context = context or {}
        
        # Handle different response formats
        if isinstance(response, dict):
            content = response.get('generated_text') or response.get('answer') or str(response)
            score = response.get('score', 0.8)
        elif isinstance(response, list) and response:
            # Take first result if list
            first = response[0]
            if isinstance(first, dict):
                content = first.get('generated_text') or first.get('answer') or str(first)
                score = first.get('score', 0.8)
            else:
                content = str(first)
                score = 0.8
        else:
            content = str(response)
            score = 0.8
        
        # Extract value
        if self.value_extractor:
            value = self.value_extractor(response, content, context)
        else:
            if self.task == "classification":
                # For classification, use the score as value if numeric label
                value = score
            else:
                value = self._default_value_extraction(content, context)
        
        # Extract confidence
        if self.confidence_extractor:
            confidence = self.confidence_extractor(response, content, context)
        else:
            confidence = score
        
        metadata = {
            "content": content,
            "framework": "huggingface",
            "task": self.task,
        }
        
        return AIResponse(
            value=value,
            confidence=confidence,
            raw_response=response,
            metadata=metadata
        )
    
    def _default_value_extraction(self, content: str, context: Dict[str, Any]) -> float:
        """Extract first numeric value from content."""
        import re
        pattern = context.get('extract_pattern', r'-?\d+\.?\d*')
        match = re.search(pattern, content)
        if match:
            return float(match.group(0))
        raise ValueError(f"Could not extract numeric value from: {content[:100]}")


# ===========================
# LangChain Integration
# ===========================

class LangChainAdapter(AIAdapter):
    """
    Adapter for LangChain outputs.
    
    Can wrap LangChain chains and extract values from their outputs.
    
    Example:
        >>> from langchain.chains import LLMChain
        >>> from langchain.llms import OpenAI
        >>> 
        >>> adapter = LangChainAdapter()
        >>> chain = LLMChain(llm=OpenAI(), prompt=prompt)
        >>> result = chain.run(input="...")
        >>> ai_response = adapter.extract_response(result)
    """
    
    def extract_response(self, response: Any, context: Optional[Dict[str, Any]] = None) -> AIResponse:
        """
        Extract value and confidence from LangChain output.
        
        Args:
            response: LangChain output (string or dict)
            context: Optional context
            
        Returns:
            AIResponse: Standardized response
        """
        context = context or {}
        
        # Extract content based on response type
        if isinstance(response, dict):
            # Chain output with multiple keys
            content = response.get('output') or response.get('text') or str(response)
            confidence = context.get('default_confidence', 0.8)
        elif isinstance(response, str):
            content = response
            confidence = context.get('default_confidence', 0.8)
        else:
            content = str(response)
            confidence = context.get('default_confidence', 0.8)
        
        # Extract value
        if self.value_extractor:
            value = self.value_extractor(response, content, context)
        else:
            value = self._default_value_extraction(content, context)
        
        # Extract confidence
        if self.confidence_extractor:
            confidence = self.confidence_extractor(response, content, context)
        
        metadata = {
            "content": content,
            "framework": "langchain",
        }
        
        return AIResponse(
            value=value,
            confidence=confidence,
            raw_response=response,
            metadata=metadata
        )
    
    def _default_value_extraction(self, content: str, context: Dict[str, Any]) -> float:
        """Extract first numeric value from content."""
        import re
        pattern = context.get('extract_pattern', r'-?\d+\.?\d*')
        match = re.search(pattern, content)
        if match:
            return float(match.group(0))
        raise ValueError(f"Could not extract numeric value from: {content[:100]}")


# ===========================
# Multi-Model Ensemble Helper
# ===========================

class MultiModelEnsemble:
    """
    Helper for combining multiple AI model outputs with AILEE trust validation.
    
    Automatically aggregates responses from multiple models and provides
    them as peer values for consensus checking.
    
    Example:
        >>> ensemble = MultiModelEnsemble()
        >>> ensemble.add_response("gpt4", gpt4_response, adapter_gpt4)
        >>> ensemble.add_response("claude", claude_response, adapter_claude)
        >>> ensemble.add_response("llama", llama_response, adapter_llama)
        >>> 
        >>> # Get aggregated response for AILEE
        >>> primary, peers, confidences = ensemble.get_consensus_inputs()
    """
    
    def __init__(self):
        """Initialize multi-model ensemble."""
        self.responses: Dict[str, AIResponse] = {}
    
    def add_response(self, model_name: str, raw_response: Any, 
                    adapter: AIAdapter, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a model response to the ensemble.
        
        Args:
            model_name: Identifier for the model
            raw_response: Raw response from the model
            adapter: AIAdapter for extracting value and confidence
            context: Optional context for extraction
        """
        ai_response = adapter.extract_response(raw_response, context)
        self.responses[model_name] = ai_response
    
    def get_consensus_inputs(self, primary_model: Optional[str] = None) -> tuple:
        """
        Get inputs ready for AILEE consensus validation.
        
        Args:
            primary_model: Name of primary model (uses highest confidence if None)
            
        Returns:
            tuple: (primary_value, peer_values, confidence_weights)
        """
        if not self.responses:
            raise ValueError("No responses added to ensemble")
        
        # Select primary model
        if primary_model:
            if primary_model not in self.responses:
                raise ValueError(f"Primary model '{primary_model}' not in responses")
            primary = self.responses[primary_model]
        else:
            # Use highest confidence as primary
            primary = max(self.responses.values(), key=lambda r: r.confidence)
        
        # Get peer values (all values including primary)
        peer_values = [r.value for r in self.responses.values()]
        
        # Get confidence weights
        confidence_weights = {
            name: response.confidence 
            for name, response in self.responses.items()
        }
        
        return primary.value, peer_values, confidence_weights
    
    def get_weighted_average(self) -> tuple:
        """
        Get weighted average value based on confidence scores.
        
        Returns:
            tuple: (weighted_value, average_confidence)
        """
        if not self.responses:
            raise ValueError("No responses added to ensemble")
        
        total_confidence = sum(r.confidence for r in self.responses.values())
        if total_confidence == 0:
            # Fallback to simple average
            values = [r.value for r in self.responses.values()]
            return sum(values) / len(values), 0.0
        
        weighted_value = sum(
            r.value * r.confidence 
            for r in self.responses.values()
        ) / total_confidence
        
        avg_confidence = total_confidence / len(self.responses)
        
        return weighted_value, avg_confidence


# ===========================
# Convenience Functions
# ===========================

def create_openai_adapter(**kwargs) -> OpenAIAdapter:
    """Create OpenAI adapter with optional configuration."""
    return OpenAIAdapter(**kwargs)


def create_anthropic_adapter(**kwargs) -> AnthropicAdapter:
    """Create Anthropic adapter with optional configuration."""
    return AnthropicAdapter(**kwargs)


def create_huggingface_adapter(task: str = "generation", **kwargs) -> HuggingFaceAdapter:
    """Create HuggingFace adapter for specific task."""
    return HuggingFaceAdapter(task=task, **kwargs)


def create_langchain_adapter(**kwargs) -> LangChainAdapter:
    """Create LangChain adapter with optional configuration."""
    return LangChainAdapter(**kwargs)


def create_multi_model_ensemble() -> MultiModelEnsemble:
    """Create a new multi-model ensemble."""
    return MultiModelEnsemble()


__all__ = [
    'AIResponse',
    'AIAdapter',
    'OpenAIAdapter',
    'AnthropicAdapter',
    'HuggingFaceAdapter',
    'LangChainAdapter',
    'MultiModelEnsemble',
    'create_openai_adapter',
    'create_anthropic_adapter',
    'create_huggingface_adapter',
    'create_langchain_adapter',
    'create_multi_model_ensemble',
]
