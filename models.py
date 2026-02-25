"""
models.py - Helper module for Search and Multi-Model Generation

This module encapsulates the logic for:
1. Real internet search via DuckDuckGo.
2. Real multi-model generation via OpenAI, Anthropic, and Gemini.
3. Fallback mock generation if no API keys are available.

It is designed to be imported by app.py to keep the main application logic clean.
"""

import os
import re
import logging
from typing import List, Tuple, Any, Dict, Optional

# Configure logging immediately
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ailee_models")

# Search
from duckduckgo_search import DDGS

# AI Providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenAI library not found: {e}")
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Anthropic library not found: {e}")
    ANTHROPIC_AVAILABLE = False

try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Google GenAI library not found: {e}")
    GEMINI_AVAILABLE = False

# Log availability status
logger.info(f"AI Models Availability: OpenAI={OPENAI_AVAILABLE}, Anthropic={ANTHROPIC_AVAILABLE}, Gemini={GEMINI_AVAILABLE}")

# AILEE Adapters
from ailee.optional.ailee_ai_integrations import (
    AIAdapter,
    create_openai_adapter,
    create_anthropic_adapter,
    create_gemini_adapter,
    create_huggingface_adapter,  # Used for mock/fallback
)

def search_duckduckgo(query: str, max_results: int = 3) -> str:
    """
    Perform a real DuckDuckGo search.
    Returns a concatenated string of snippets.
    """
    logger.info(f"Searching DuckDuckGo for: {query}")
    try:
        results = []
        # Use DUCKDUCKGO_APP_NAME as the User-Agent header if set
        app_name = os.getenv("DUCKDUCKGO_APP_NAME")
        headers = {"User-Agent": app_name} if app_name else None
        with DDGS(headers=headers) as ddgs:
            # text() returns an iterator of dicts: {'title':..., 'href':..., 'body':...}
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"- {r.get('title', 'No Title')}: {r.get('body', '')}")

        if not results:
            return "No relevant search results found."

        return "\n".join(results)
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return f"Search failed: {str(e)}"


def extract_confidence_from_text(response: Any, content: str, context: Dict[str, Any]) -> float:
    """
    Custom extractor to parse 'Confidence: X' from the text.
    Returns the confidence score as a float (0-100).
    """
    # 1. Look for explicit "Confidence: [number]" pattern
    # We use capturing group 1 for the number
    match = re.search(r"Confidence:\s*(\d+(\.\d+)?)", content, re.IGNORECASE)
    if match:
        try:
            val = float(match.group(1))
            return val
        except ValueError:
            pass

    # 2. Fallback: if the text ends with a number, take it
    # (The prompt asks for confidence on a new line at the end)
    fallback = re.search(r"(\d+(\.\d+)?)\s*$", content.strip())
    if fallback:
        try:
            return float(fallback.group(1))
        except ValueError:
            pass

    # 3. Last resort: default to 0.0 (untrusted)
    return 0.0


def generate_with_models(query: str, search_context: str) -> List[Tuple[str, Any, AIAdapter]]:
    """
    Call available AI models (OpenAI, Anthropic, Gemini) with the query and context.
    Returns a list of (model_name, raw_response, adapter_instance).

    If no models are available or all fail, returns a mock fallback.
    """
    responses = []

    # Construct the prompt
    # We ask for a confidence score to feed the AILEE Trust Pipeline (which expects numbers).
    prompt = (
        f"Context from internet search:\n{search_context}\n\n"
        f"User Query: {query}\n\n"
        "Instructions:\n"
        "1. Answer the query concisely based on the context.\n"
        "2. Provide a confidence score (0-100) for your answer on a new line.\n"
        "Format:\n"
        "Answer: [Your answer here]\n"
        "Confidence: [Score]"
    )

    # 1. OpenAI
    # Strip whitespace AND surrounding quotes to handle common copy-paste errors
    openai_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'").strip("“").strip("”").strip("‘").strip("’")
    if OPENAI_AVAILABLE and openai_key:
        try:
            logger.info(f"Calling OpenAI with key: {openai_key[:4]}...{openai_key[-4:] if len(openai_key) > 8 else ''}")
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                top_logprobs=1
            )
            # Pass custom extractor
            adapter = create_openai_adapter(
                use_logprobs=True,
                value_extractor=extract_confidence_from_text
            )
            responses.append(("openai", response, adapter))
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
    else:
        if not OPENAI_AVAILABLE:
            logger.warning("Skipping OpenAI: Library not available.")
        if not openai_key:
            logger.warning("Skipping OpenAI: API Key missing or empty.")

    # 2. Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip().strip('"').strip("'").strip("“").strip("”").strip("‘").strip("’")
    if ANTHROPIC_AVAILABLE and anthropic_key:
        try:
            logger.info(f"Calling Anthropic with key: {anthropic_key[:4]}...{anthropic_key[-4:] if len(anthropic_key) > 8 else ''}")
            client = Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            adapter = create_anthropic_adapter(
                value_extractor=extract_confidence_from_text
            )
            responses.append(("anthropic", response, adapter))
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
    else:
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Skipping Anthropic: Library not available.")
        if not anthropic_key:
            logger.warning("Skipping Anthropic: API Key missing or empty.")

    # 3. Gemini
    google_key = os.getenv("GOOGLE_API_KEY", "").strip().strip('"').strip("'").strip("“").strip("”").strip("‘").strip("’")
    if GEMINI_AVAILABLE and google_key:
        try:
            logger.info(f"Calling Gemini with key: {google_key[:4]}...{google_key[-4:] if len(google_key) > 8 else ''}")
            client = genai.Client(api_key=google_key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            adapter = create_gemini_adapter(
                value_extractor=extract_confidence_from_text
            )
            responses.append(("gemini", response, adapter))
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
    else:
        if not GEMINI_AVAILABLE:
            logger.warning("Skipping Gemini: Library not available.")
        if not google_key:
            logger.warning("Skipping Gemini: API Key missing or empty.")

    # Fallback to Mock if no responses
    if not responses:
        logger.warning("No real models available/successful. Using mock fallback.")
        mock_response = {
            'generated_text': (
                f"Answer: This is a simulated response based on search results for '{query}'. "
                "The actual AI models were not available or failed.\n"
                "Confidence: 85.0"
            ),
            'score': 0.85  # Explicit score for the adapter (though value_extractor will parse the text)
        }
        # Use HuggingFaceAdapter with our custom extractor
        adapter = create_huggingface_adapter(
            value_extractor=extract_confidence_from_text
        )
        responses.append(("mock_fallback", mock_response, adapter))

    return responses
