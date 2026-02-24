from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import random
import time

# AILEE Imports
from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_ai_integrations import (
    MultiModelEnsemble,
    AIAdapter,
    AIResponse,
)

app = FastAPI()

# Input model
class Query(BaseModel):
    query: str

# Mock Search Function
def perform_search(query: str):
    # Simulate search latency
    time.sleep(0.5)
    return f"Search results for '{query}': AILEE is a trust layer for AI systems."

# Mock Model Adapter for simulation
class MockAdapter(AIAdapter):
    def extract_response(self, response, context=None):
        return AIResponse(
            value=response.get('value', 0.0),
            confidence=response.get('confidence', 0.0),
            raw_response=response,
            metadata={"model": response.get('model', 'mock')}
        )

# Generation Logic
def generate_multi_model(query: str, search_context: str):
    ensemble = MultiModelEnsemble()
    adapter = MockAdapter()

    # Simulate 3 models with slightly different outputs based on query hash or length
    # This ensures "deterministic" but varied output for demo purposes
    # We map the query string length to a base value between 50 and 95
    base_val = 50 + (len(query) * 7) % 45

    # Model 1
    ensemble.add_response("model_a", {
        "value": min(100, max(0, base_val + random.uniform(-2, 2))),
        "confidence": 0.95,
        "model": "Model A"
    }, adapter)

    # Model 2
    ensemble.add_response("model_b", {
        "value": min(100, max(0, base_val + random.uniform(-2, 2))),
        "confidence": 0.88,
        "model": "Model B"
    }, adapter)

    # Model 3
    ensemble.add_response("model_c", {
        "value": min(100, max(0, base_val + random.uniform(-2, 2))),
        "confidence": 0.92,
        "model": "Model C"
    }, adapter)

    return ensemble

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "<h1>AILEE Trust Layer - Index Not Found</h1>"

@app.post("/trust")
async def trust_endpoint(q: Query):
    try:
        # 1. Search
        context = perform_search(q.query)

        # 2. Multi-model Generation
        ensemble = generate_multi_model(q.query, context)
        primary_value, peer_values, confidences = ensemble.get_consensus_inputs()

        # 3. Trust Pipeline
        # Configure pipeline (using a custom config for demo purposes)
        config = AileeConfig(
            accept_threshold=0.90,
            consensus_quorum=3,
            consensus_delta=5.0, # Allow small variance
            grace_peer_delta=5.0,
            enable_grace=True,
            enable_consensus=True
        )
        pipeline = AileeTrustPipeline(config)

        result = pipeline.process(
            raw_value=primary_value,
            raw_confidence=max(confidences.values()),
            peer_values=peer_values,
            context={"query": q.query, "search_context": context}
        )

        # 4. Return Result
        return {
            "query": q.query,
            "search_context": context,
            "trusted_output": {
                "value": result.value,
                "safety_status": result.safety_status,
                "reasons": result.reasons,
                "used_fallback": result.used_fallback,
                "consensus_status": result.consensus_status,
                "grace_status": result.grace_status
            },
            "models_data": {
                "primary_value": primary_value,
                "peer_values": peer_values,
                "confidences": confidences
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
