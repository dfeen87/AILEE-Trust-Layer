"""
formatters.py - Helper module for formatting trust results.

This module converts the dictionary output of the AILEE Trust Layer
into human-readable text, Markdown, or HTML.
"""

import json
import html
from typing import Dict, Any

def format_text(data: Dict[str, Any]) -> str:
    """Format the trust result as plain text."""
    lines = [
        "=== AILEE Trust Validation ===",
        f"Query: {data.get('query')}",
        f"Trusted Answer: {data.get('trusted_answer', '').strip()}",
        "",
        "--- Trust Metrics ---",
        f"Trust Score: {data.get('trust_score', 0):.2f}",
        f"Safety Status: {data.get('safety_status')}",
        f"Consensus Status: {data.get('consensus_status')}",
        f"Grace Status: {data.get('grace_status')}",
        "",
        "--- Reasons ---"
    ]
    for reason in data.get('reasons', []):
        lines.append(f"- {reason}")

    return "\n".join(lines)


def format_markdown(data: Dict[str, Any]) -> str:
    """Format the trust result as Markdown."""
    lines = [
        "# AILEE Trust Validation",
        f"**Query:** {data.get('query')}",
        "",
        "## Trusted Answer",
        f">{data.get('trusted_answer', '').strip()}",
        "",
        "## Trust Metrics",
        f"- **Trust Score:** {data.get('trust_score', 0):.2f}",
        f"- **Safety Status:** `{data.get('safety_status')}`",
        f"- **Consensus Status:** `{data.get('consensus_status')}`",
        f"- **Grace Status:** `{data.get('grace_status')}`",
        "",
        "## Reasons"
    ]
    for reason in data.get('reasons', []):
        lines.append(f"- {reason}")

    return "\n".join(lines)


def format_html(data: Dict[str, Any]) -> str:
    """Format the trust result as simple HTML."""
    # Escape all user-provided content to prevent XSS
    query = html.escape(str(data.get('query', '')))
    trusted_answer = html.escape(str(data.get('trusted_answer', ''))).replace('\n', '<br>')

    # Safe to use f-strings for status/scores as they are internal enums/floats
    reasons_html = "".join([f"<li>{html.escape(str(r))}</li>" for r in data.get('reasons', [])])

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>AILEE Trust Result</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }}
        .answer {{ background: #f8f9fa; padding: 1rem; border-left: 5px solid #007bff; margin: 1rem 0; font-style: italic; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; background: #fff; border: 1px solid #ddd; padding: 1rem; border-radius: 8px; }}
        .metric-item {{ padding: 0.5rem; }}
        .metric-label {{ font-weight: bold; color: #666; display: block; font-size: 0.9em; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .reasons ul {{ padding-left: 1.5rem; }}
        .status-accepted {{ color: green; }}
        .status-rejected {{ color: red; }}
        .status-borderline {{ color: orange; }}
    </style>
</head>
<body>
    <h1>AILEE Trust Validation</h1>

    <p><strong>Query:</strong> {query}</p>

    <h2>Trusted Answer</h2>
    <div class="answer">
        {trusted_answer}
    </div>

    <h2>Trust Metrics</h2>
    <div class="metrics">
        <div class="metric-item">
            <span class="metric-label">Trust Score</span>
            <span class="metric-value">{data.get('trust_score', 0):.2f}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Safety Status</span>
            <span class="metric-value">{data.get('safety_status')}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Consensus Status</span>
            <span class="metric-value">{data.get('consensus_status')}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Grace Status</span>
            <span class="metric-value">{data.get('grace_status')}</span>
        </div>
    </div>

    <h2>Reasons</h2>
    <div class="reasons">
        <ul>
            {reasons_html}
        </ul>
    </div>
</body>
</html>
"""
