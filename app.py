"""
AILEE Trust Layer — Render.com Web Service

This module exposes the AILEE Trust Pipeline as an HTTP API.
It routes user queries through an AI backend (OpenAI or Gemini),
then passes every response through the AILEE safety, grace, and
consensus layers before returning it to the caller.

Environment variables (set in Render dashboard):
  AI_PROVIDER       "openai" (default) or "gemini"
  OPENAI_API_KEY    Required when AI_PROVIDER=openai
  OPENAI_MODEL      GPT model name, default "gpt-4o"
  GEMINI_API_KEY    Required when AI_PROVIDER=gemini
  GEMINI_MODEL      Gemini model name, default "gemini-1.5-flash"
  AILEE_PRESET      Config preset: "balanced" (default), "conservative", "permissive"
  PORT              HTTP port (Render sets this automatically)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import ailee
from ailee import AileeTrustPipeline, AileeConfig, DecisionResult

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AILEE Trust Layer API",
    description=(
        "Adaptive Integrity Layer for AI Decision Systems. "
        "Every response is validated through AILEE's safety, grace, "
        "and consensus pipeline before reaching the user."
    ),
    version=ailee.__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AILEE pipeline (initialised once at startup)
# ---------------------------------------------------------------------------

PRESET = os.getenv("AILEE_PRESET", "balanced")

try:
    _pipeline = ailee.create_pipeline(PRESET)
except Exception as exc:
    logger.warning(
        "Could not load AILEE preset %r (%s); falling back to default AileeConfig.",
        PRESET,
        exc,
    )
    _pipeline = AileeTrustPipeline(AileeConfig())

# ---------------------------------------------------------------------------
# AI provider helpers
# ---------------------------------------------------------------------------

AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Safety system prompt that is prepended to every request
_SYSTEM_PROMPT = (
    "You are AILEE, a helpful and honest AI assistant governed by the "
    "AILEE Trust Layer. You provide accurate, well-reasoned, and safe "
    "responses based on your broad knowledge. If you are uncertain, say so "
    "clearly. Never produce harmful, illegal, or deceptive content."
)


def _call_openai(prompt: str, context: Optional[str]) -> tuple[str, float]:
    """Call OpenAI and return (content, raw_confidence 0-1)."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="openai package not installed") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    messages: List[Dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "user", "content": f"[Context]: {context}"})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        logprobs=True,
        top_logprobs=1,
    )

    content: str = response.choices[0].message.content or ""

    # Derive confidence from average token log-probability.
    # 0.80 is used as the fallback when logprobs are unavailable (e.g. the
    # model or endpoint does not support them), representing moderate confidence.
    confidence = 0.80
    try:
        import math
        lp_content = response.choices[0].logprobs.content or []
        if lp_content:
            # Use the geometric mean (exp of the mean log-prob) so that a
            # single very-low-probability token drags the score down more
            # faithfully than a simple arithmetic average would.
            mean_logprob = sum(t.logprob for t in lp_content) / len(lp_content)
            confidence = min(1.0, max(0.0, math.exp(mean_logprob)))
    except Exception:
        pass

    return content, confidence


def _call_gemini(prompt: str, context: Optional[str]) -> tuple[str, float]:
    """Call Google Gemini and return (content, raw_confidence 0-1)."""
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise HTTPException(
            status_code=503, detail="google-generativeai package not installed"
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_SYSTEM_PROMPT,
    )

    parts = []
    if context:
        parts.append(f"[Context]: {context}")
    parts.append(prompt)
    full_prompt = "\n".join(parts)

    response = model.generate_content(full_prompt)
    content: str = response.text or ""
    # Gemini does not expose per-token log-probabilities, so we use a fixed
    # moderate-high value (0.82) as the initial confidence seed passed to AILEE.
    # AILEE's pipeline will refine this signal using its own stability and
    # likelihood scoring as history accumulates.
    confidence = 0.82
    return content, confidence


def _get_ai_response(prompt: str, context: Optional[str]) -> tuple[str, float]:
    if AI_PROVIDER == "gemini":
        return _call_gemini(prompt, context)
    return _call_openai(prompt, context)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    prompt: str = Field(..., description="The user's question or request.")
    context: Optional[str] = Field(
        None,
        description="Optional background context supplied by the caller.",
    )
    peer_values: Optional[List[float]] = Field(
        None,
        description=(
            "Optional list of numeric confidence values from peer models "
            "for consensus scoring."
        ),
    )


class TrustMetadata(BaseModel):
    safety_status: str
    grace_status: str
    consensus_status: str
    used_fallback: bool
    confidence_score: float
    reasons: List[str]
    ailee_version: str
    ai_provider: str


class QueryResponse(BaseModel):
    response: str
    trusted: bool
    trust: TrustMetadata


class HealthResponse(BaseModel):
    status: str
    ailee_version: str
    ai_provider: str
    ailee_preset: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Health check — confirms the service is running."""
    return HealthResponse(
        status="ok",
        ailee_version=ailee.__version__,
        ai_provider=AI_PROVIDER,
        ailee_preset=PRESET,
    )


@app.get("/", response_model=HealthResponse, tags=["System"])
def root() -> HealthResponse:
    """Root endpoint — same as /health."""
    return health()


@app.post("/query", response_model=QueryResponse, tags=["AILEE"])
def query(request: QueryRequest) -> QueryResponse:
    """
    Submit a natural-language prompt.

    The response is generated by the configured AI provider and then
    validated through the full AILEE Trust Pipeline (safety → grace →
    consensus → fallback) before being returned to you.

    A `trusted: true` result means every layer accepted the output.
    A `trusted: false` result means AILEE applied a fallback or
    downgraded the response — inspect `trust.reasons` for details.
    """
    # 1. Call the AI backend
    ai_content, raw_confidence = _get_ai_response(request.prompt, request.context)

    # 2. Use the AI's confidence as the numeric signal passed into AILEE.
    #    raw_value = raw_confidence so the pipeline evaluates whether the
    #    model's self-reported confidence meets the safety threshold.
    raw_value = raw_confidence

    # 3. Run through AILEE Trust Pipeline
    result: DecisionResult = _pipeline.process(
        raw_value=raw_value,
        raw_confidence=raw_confidence,
        peer_values=request.peer_values,
        context={
            "prompt": request.prompt[:200],
            "provider": AI_PROVIDER,
            "content_length": len(ai_content),
        },
    )

    # trusted = True only when AILEE fully accepted the output without any
    # fallback.  BORDERLINE responses are *conditionally* passing (Grace /
    # Consensus layers evaluated them); callers should inspect
    # trust.safety_status for the precise verdict.
    trusted = not result.used_fallback and result.safety_status.value == "ACCEPTED"

    return QueryResponse(
        response=ai_content,
        trusted=trusted,
        trust=TrustMetadata(
            safety_status=result.safety_status.value,
            grace_status=result.grace_status.value,
            consensus_status=result.consensus_status.value,
            used_fallback=result.used_fallback,
            confidence_score=result.confidence_score,
            reasons=result.reasons,
            ailee_version=ailee.__version__,
            ai_provider=AI_PROVIDER,
        ),
    )
