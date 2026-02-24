#!/usr/bin/env python3
"""
AILEE AI - Trust Layer Orchestration Agent

This agent implements the AILEE trust pipeline:
1. Input Handling
2. Web Search Retrieval (RAG)
3. Multi-Model Generation
4. AILEE Trust Layer Validation (API or Local Fallback)
5. Output Formatting
"""

import argparse
import json
import sys
import hashlib
import time
import math
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

# External Libraries
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# =============================================================================
# Trust Scoring Logic (Ported from Rust Core src/trust.rs)
# =============================================================================

class TrustScore:
    def __init__(self, confidence: float, safety: float, consistency: float, determinism: float):
        self.confidence_score = confidence
        self.safety_score = safety
        self.consistency_score = consistency
        self.determinism_score = determinism
        self.aggregate_score = self._compute_aggregate()

    def _compute_aggregate(self) -> float:
        # Weighted average from Rust core
        SAFETY_WEIGHT = 0.35
        CONSISTENCY_WEIGHT = 0.30
        CONFIDENCE_WEIGHT = 0.20
        DETERMINISM_WEIGHT = 0.15

        score = (
            self.safety_score * SAFETY_WEIGHT +
            self.consistency_score * CONSISTENCY_WEIGHT +
            self.confidence_score * CONFIDENCE_WEIGHT +
            self.determinism_score * DETERMINISM_WEIGHT
        )
        return max(0.0, min(1.0, score))

    def to_dict(self) -> Dict[str, float]:
        return {
            "confidence_score": self.confidence_score,
            "safety_score": self.safety_score,
            "consistency_score": self.consistency_score,
            "determinism_score": self.determinism_score,
            "aggregate_score": self.aggregate_score,
        }

class TrustScorer:
    def __init__(self):
        self.hedging_words = {
            "maybe", "perhaps", "possibly", "might", "could",
            "uncertain", "unsure", "probably", "likely"
        }
        self.unsafe_patterns = {
            "error", "exception", "failed", "invalid", "corrupt",
            "unsafe", "dangerous", "harmful", "malicious"
        }
        self.nondeterministic_markers = {
            "random", "varies", "depends on timing"
        }

    def score_output(self, text: str, peer_texts: List[str] = []) -> TrustScore:
        confidence = self._compute_confidence(text)
        safety = self._compute_safety(text)
        consistency = self._compute_consistency(text, peer_texts)
        determinism = self._compute_determinism(text)
        return TrustScore(confidence, safety, consistency, determinism)

    def _compute_confidence(self, text: str) -> float:
        # Length-based heuristic
        length_score = min(1.0, len(text) / 500.0)

        # Hedging penalty
        hedging_count = sum(1 for word in text.lower().split() if word in self.hedging_words)
        penalty = hedging_count * 0.1

        return max(0.0, min(1.0, length_score - penalty))

    def _compute_safety(self, text: str) -> float:
        text_lower = text.lower()
        unsafe_count = sum(1 for pattern in self.unsafe_patterns if pattern in text_lower)
        return max(0.0, min(1.0, 1.0 - (unsafe_count * 0.1)))

    def _compute_consistency(self, text: str, peer_texts: List[str]) -> float:
        if not peer_texts:
            return 0.5  # Neutral if no peers

        similarities = [self._token_similarity(text, peer) for peer in peer_texts]
        if not similarities:
            return 0.5

        return sum(similarities) / len(similarities)

    def _compute_determinism(self, text: str) -> float:
        # Mock latency consistency (assuming consistent here)
        base_score = 0.8

        text_lower = text.lower()
        if any(m in text_lower for m in self.nondeterministic_markers):
            return base_score * 0.5
        return base_score

    def _token_similarity(self, text1: str, text2: str) -> float:
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

# =============================================================================
# Lineage & Verification (Ported from Rust Core src/lineage.rs)
# =============================================================================

class Lineage:
    @staticmethod
    def build(query: str, model_outputs: List[Dict[str, Any]], final_answer: str) -> Dict[str, Any]:
        timestamp = int(datetime.now().timestamp())

        # Sort outputs by model name for deterministic hashing
        sorted_outputs = sorted(model_outputs, key=lambda x: x['model'])

        # Compute SHA-256 hash
        hasher = hashlib.sha256()
        hasher.update(query.encode('utf-8'))

        for output in sorted_outputs:
            hasher.update(output['model'].encode('utf-8'))
            hasher.update(output['answer'].encode('utf-8'))

        hasher.update(final_answer.encode('utf-8'))
        verification_hash = hasher.hexdigest()

        return {
            "timestamp": timestamp,
            "verification_hash": verification_hash,
            "models": [o['model'] for o in sorted_outputs],
            "sources": [] # Populated later
        }

# =============================================================================
# Orchestration Logic
# =============================================================================

class AILEE_AI:
    def __init__(self):
        self.scorer = TrustScorer()

        # API Keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.gemini_key = os.getenv("GOOGLE_API_KEY")

        # Initialize clients if keys exist
        if self.openai_key and openai:
            openai.api_key = self.openai_key

        if self.anthropic_key and anthropic:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
        else:
            self.anthropic_client = None

        if self.gemini_key and genai:
            genai.configure(api_key=self.gemini_key)

    def perform_web_search(self, query: str) -> List[Dict[str, str]]:
        """
        Uses DuckDuckGo Search to retrieve live results.
        """
        results = []
        if DDGS:
            print(f"[*] Searching DuckDuckGo for: {query}...", file=sys.stderr)
            try:
                with DDGS() as ddgs:
                    # Fetch top 5 results
                    ddg_results = list(ddgs.text(query, max_results=5))
                    for r in ddg_results:
                        results.append({
                            "source": r.get('title', 'Unknown Source'),
                            "url": r.get('href', ''),
                            "snippet": r.get('body', '')[:200]
                        })
            except Exception as e:
                 print(f"[!] DuckDuckGo Search failed ({e}). Using mock fallback.", file=sys.stderr)
        else:
             print("[!] duckduckgo-search not installed. Using mock fallback.", file=sys.stderr)

        if not results:
            # Fallback if search fails or no library
            results = [
                {
                    "source": "example.com (Mock)",
                    "url": "https://example.com/topic",
                    "snippet": f"Information about {query} found on example.com."
                },
                {
                    "source": "wikipedia.org (Mock)",
                    "url": "https://en.wikipedia.org/wiki/Topic",
                    "snippet": f"Wikipedia entry discussing {query} in detail."
                }
            ]

        return results

    def generate_answers(self, query: str, context: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Queries available LLMs. If keys are missing, falls back to mocks.
        """
        answers = []
        context_str = "\n".join([f"- {s['snippet']} ({s['url']})" for s in context])
        prompt = f"Question: {query}\n\nContext:\n{context_str}\n\nAnswer concisely based on the context."

        # 1. OpenAI
        if self.openai_key and openai:
            print("[*] Querying OpenAI GPT-4...", file=sys.stderr)
            try:
                start = time.time()
                client = openai.OpenAI(api_key=self.openai_key)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150
                )
                latency = int((time.time() - start) * 1000)
                answers.append({
                    "model": "OpenAI GPT-4",
                    "answer": response.choices[0].message.content.strip(),
                    "metadata": {"latency": latency}
                })
            except Exception as e:
                print(f"[!] OpenAI failed: {e}", file=sys.stderr)

        # 2. Anthropic
        if self.anthropic_client:
            print("[*] Querying Anthropic Claude...", file=sys.stderr)
            try:
                start = time.time()
                response = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}]
                )
                latency = int((time.time() - start) * 1000)
                answers.append({
                    "model": "Anthropic Claude 3",
                    "answer": response.content[0].text.strip(),
                    "metadata": {"latency": latency}
                })
            except Exception as e:
                print(f"[!] Anthropic failed: {e}", file=sys.stderr)

        # 3. Gemini
        if self.gemini_key and genai:
            print("[*] Querying Google Gemini...", file=sys.stderr)
            try:
                start = time.time()
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                latency = int((time.time() - start) * 1000)
                answers.append({
                    "model": "Google Gemini Pro",
                    "answer": response.text.strip(),
                    "metadata": {"latency": latency}
                })
            except Exception as e:
                print(f"[!] Gemini failed: {e}", file=sys.stderr)

        # If no answers (e.g. no keys), fallback to mocks
        if not answers:
            print("[!] No API keys provided or all calls failed. Using mock responses.", file=sys.stderr)
            base_answer = f"The answer to '{query}' involves considering multiple factors."
            answers = [
                {
                    "model": "OpenAI GPT-4 (Mock)",
                    "answer": f"{base_answer} It is generally considered true.",
                    "metadata": {"latency": 120}
                },
                {
                    "model": "Anthropic Claude 3 (Mock)",
                    "answer": f"{base_answer} However, there are nuances to consider.",
                    "metadata": {"latency": 150}
                },
                {
                    "model": "Google Gemini Pro (Mock)",
                    "answer": f"{base_answer} Recent data supports this view.",
                    "metadata": {"latency": 110}
                }
            ]

        return answers

    def call_trust_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempts to call the external AILEE Trust API.
        Falls back to local TrustScorer if unreachable.
        """
        api_url = "https://ailee-api.onrender.com/trust"

        try:
            print(f"[*] Calling AILEE Trust API ({api_url})...", file=sys.stderr)
            response = requests.post(api_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[!] AILEE API unreachable or error ({e}). Using local Trust Core fallback.", file=sys.stderr)
            return self._local_trust_logic(payload)

    def _local_trust_logic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload['query']
        model_outputs = payload['model_answers']

        # Score each output
        scores = []
        answers = [m['answer'] for m in model_outputs]

        for i, output in enumerate(model_outputs):
            # Peer texts are all other answers
            peers = answers[:i] + answers[i+1:]
            score = self.scorer.score_output(output['answer'], peers)
            scores.append({
                "model": output['model'],
                "score": score
            })

        # Consensus: Pick highest aggregate score
        best = max(scores, key=lambda x: x['score'].aggregate_score)
        best_output = next(m for m in model_outputs if m['model'] == best['model'])

        # Build lineage
        lineage = Lineage.build(query, model_outputs, best_output['answer'])
        lineage['sources'] = [s['url'] for s in payload.get('search_snippets', [])]

        return {
            "final_answer": best_output['answer'],
            "trust_score": best['score'].aggregate_score,
            "rationale": f"Selected output from {best['model']} based on highest trust score ({best['score'].aggregate_score:.2f}).",
            "lineage": lineage,
            "all_scores": [s['score'].to_dict() for s in scores] # Debug info
        }

    def process(self, query: str) -> str:
        # 1. Input Handling (Query)

        # 2. Web Search
        snippets = self.perform_web_search(query)

        # 3. Multi-Model Generation
        model_answers = self.generate_answers(query, snippets)

        # 4. Prepare Payload
        payload = {
            "query": query,
            "model_answers": model_answers,
            "search_snippets": snippets
        }

        # 5. Call Trust Layer
        result = self.call_trust_api(payload)

        # 6. Output
        return self._format_output(result)

    def _format_output(self, result: Dict[str, Any]) -> str:
        lineage = result['lineage']

        output = []
        output.append("Final Answer:")
        output.append(result['final_answer'])
        output.append("")
        output.append(f"Trust Score: {result['trust_score']:.2f}")
        output.append("")
        output.append("Why AILEE Chose This:")
        output.append(result['rationale'])
        output.append("")
        output.append("Lineage:")
        output.append(f"Models: {', '.join(lineage['models'])}")
        output.append(f"Sources: {', '.join(lineage['sources'])}")

        return "\n".join(output)

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AILEE AI - Trust Layer Orchestration Agent")
    parser.add_argument("--query", type=str, help="The user question to answer")

    args = parser.parse_args()

    if not args.query:
        # Interactive mode or error
        if sys.stdin.isatty():
            query = input("Enter your question: ")
        else:
            print("Error: No query provided. Use --query or pipe input.", file=sys.stderr)
            sys.exit(1)
    else:
        query = args.query

    agent = AILEE_AI()
    response = agent.process(query)
    print(response)

if __name__ == "__main__":
    main()
