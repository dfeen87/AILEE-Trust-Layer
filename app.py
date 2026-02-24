import logging
import os
from typing import Dict, Any, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# AILEE Core
from ailee import AileeTrustPipeline, AileeConfig

# AILEE Integrations
from ailee.optional.ailee_ai_integrations import MultiModelEnsemble

# Local helpers
import models
import formatters

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ailee_app")

app = FastAPI(
    title="AILEE Trust Layer - Deploy",
    description="Public deployment of the AILEE Trust Layer with real search and multi-model generation.",
    version="1.0.0"
)

# Enable CORS for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AILEE Pipeline
pipeline_config = AileeConfig(
    hard_min=0.0,
    hard_max=100.0,
    consensus_quorum=2,  # Require at least 2 models to agree if possible
    consensus_delta=10.0, # Within 10 points
    grace_peer_delta=15.0,
    fallback_mode="median"
)
trust_pipeline = AileeTrustPipeline(pipeline_config)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the landing page."""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>AILEE Trust Layer</h1><p>index.html not found.</p>"


@app.get("/trust")
def get_trust(
    query: str = Query(..., min_length=1, description="The user query to validate"),
    format: str = Query("json", regex="^(json|text|markdown|html)$", description="Output format: json, text, markdown, or html")
):
    """
    Main endpoint:
    1. Search internet for context.
    2. Generate answers + confidence scores using multiple AI models.
    3. Aggregate results using MultiModelEnsemble.
    4. Run AileeTrustPipeline to validate the confidence scores.
    5. Return the trusted answer and the pipeline metadata.
    """
    logger.info(f"Received query: {query}, format: {format}")

    # 1. Real Search
    try:
        search_context = models.search_duckduckgo(query)
    except Exception as e:
        logger.error(f"Search error: {e}")
        search_context = "Search failed, proceeding with internal knowledge only."

    # 2. Real Multi-Model Generation
    # Returns list of (model_name, raw_response, adapter)
    try:
        model_results = models.generate_with_models(query, search_context)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not model_results:
        raise HTTPException(status_code=500, detail="All AI models failed.")

    # 3. Aggregate into Ensemble
    ensemble = MultiModelEnsemble()

    for name, response, adapter in model_results:
        try:
            ensemble.add_response(
                model_name=name,
                raw_response=response,
                adapter=adapter
                # context is handled by the custom extractor in models.py
            )
        except Exception as e:
            logger.warning(f"Failed to add response from {name}: {e}")

    if not ensemble.responses:
        raise HTTPException(status_code=500, detail="Failed to extract valid responses from models.")

    # 4. Pipeline Execution
    try:
        # Get consensus inputs (primary value, peers, weights)
        primary_score, peer_scores, confidence_weights = ensemble.get_consensus_inputs()

        # Identify which model provided the primary score
        primary_model_name = None
        for name, resp in ensemble.responses.items():
            if resp.value == primary_score:
                primary_model_name = name
                break

        if not primary_model_name:
            primary_model_name = list(ensemble.responses.keys())[0]

        # 5. Run Trust Pipeline
        trust_result = trust_pipeline.process(
            raw_value=primary_score,
            raw_confidence=confidence_weights.get(primary_model_name, 0.8),
            peer_values=peer_scores,
            context={"query": query, "primary_model": primary_model_name}
        )

        # 6. Format Output
        primary_text = ensemble.responses[primary_model_name].metadata.get("content", "")

        response_data = {
            "query": query,
            "trusted_answer": primary_text,
            "trust_score": trust_result.value,
            "safety_status": trust_result.safety_status,
            "consensus_status": trust_result.consensus_status,
            "grace_status": trust_result.grace_status,
            "reasons": trust_result.reasons,
            "metadata": trust_result.metadata
        }

        # Return based on requested format
        if format == "json":
            return JSONResponse(content=response_data)
        elif format == "text":
            return PlainTextResponse(content=formatters.format_text(response_data))
        elif format == "markdown":
            return PlainTextResponse(content=formatters.format_markdown(response_data), media_type="text/markdown")
        elif format == "html":
            return HTMLResponse(content=formatters.format_html(response_data))
        else:
            # Should be caught by regex validation, but for safety:
            return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Pipeline processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
