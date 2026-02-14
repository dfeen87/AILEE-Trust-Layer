# AI Integration Quick Start Guide

## Overview

AILEE Trust Layer makes it easy to add trust validation to any AI system. This guide shows how to integrate AILEE with popular AI frameworks.

## Table of Contents

1. [Why Add Trust Validation to AI?](#why-add-trust-validation-to-ai)
2. [Basic Integration Pattern](#basic-integration-pattern)
3. [Framework-Specific Guides](#framework-specific-guides)
   - [OpenAI / GPT](#openai--gpt)
   - [Anthropic / Claude](#anthropic--claude)
   - [Google Gemini](#google-gemini)
   - [HuggingFace](#huggingface)
   - [LangChain](#langchain)
4. [Multi-Model Ensembles](#multi-model-ensembles)
5. [Advanced Patterns](#advanced-patterns)
6. [Real-World Examples](#real-world-examples)

---

## Why Add Trust Validation to AI?

Modern AI systems produce outputs with varying levels of confidence and reliability. AILEE adds a **governance layer** that:

- ✅ Validates confidence before acting on AI outputs
- ✅ Enables consensus checking across multiple models
- ✅ Provides fallback mechanisms when AI is uncertain
- ✅ Creates full audit trails for compliance
- ✅ Prevents silent failures and hallucinations

**AILEE doesn't replace your AI — it makes it trustworthy.**

---

## Basic Integration Pattern

All AI integrations follow this simple 3-step pattern:

```python
from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_ai_integrations import OpenAIAdapter

# Step 1: Create AILEE pipeline
config = AileeConfig(
    accept_threshold=0.85,  # Require 85% confidence
    borderline_low=0.70,    # Below 70% is rejected
)
pipeline = AileeTrustPipeline(config)

# Step 2: Get AI response (your existing code)
response = your_ai_model.generate(...)  # OpenAI, Claude, etc.

# Step 3: Extract and validate through AILEE
adapter = OpenAIAdapter()
ai_response = adapter.extract_response(response)

result = pipeline.process(
    raw_value=ai_response.value,
    raw_confidence=ai_response.confidence,
    context={"model": "gpt-4"}
)

# Use validated output
if not result.used_fallback:
    safe_value = result.value  # Trusted AI output
else:
    # Handle case where AI wasn't confident enough
    safe_value = result.value  # Fallback value based on history
```

---

## Framework-Specific Guides

### OpenAI / GPT

#### Installation
```bash
pip install ailee-trust-layer openai
```

#### Basic Usage

```python
from openai import OpenAI
from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_ai_integrations import create_openai_adapter

# Initialize
client = OpenAI()
pipeline = AileeTrustPipeline(AileeConfig())
adapter = create_openai_adapter(use_logprobs=True)

# Call OpenAI API
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Rate the quality of this code from 0-100: ..."}
    ],
    logprobs=True,  # Enable for better confidence scores
    top_logprobs=1
)

# Extract and validate
ai_response = adapter.extract_response(response)
result = pipeline.process(
    raw_value=ai_response.value,
    raw_confidence=ai_response.confidence,
    context={"feature": "code_quality"}
)

print(f"Trusted score: {result.value}")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Safe to use: {not result.used_fallback}")
```

#### Custom Value Extraction

If your prompt returns structured data, use a custom extractor:

```python
import json
import re

def extract_json_score(response, content, context):
    """Extract score from JSON response."""
    match = re.search(r'\{[^}]+\}', content)
    if match:
        data = json.loads(match.group(0))
        return float(data.get('score', 0))
    return 0.0

adapter = OpenAIAdapter(
    value_extractor=extract_json_score,
    use_logprobs=True
)
```

#### Best Practices for OpenAI

1. **Enable logprobs** for better confidence estimation
2. **Use structured outputs** (JSON mode) for easier extraction
3. **Set temperature=0** for deterministic outputs
4. **Use system prompts** to request confidence scores

Example prompt:
```python
messages = [
    {"role": "system", "content": "Always provide your confidence (0-100) with each answer."},
    {"role": "user", "content": "Question: ... Please include your confidence."}
]
```

---

### Anthropic / Claude

#### Installation
```bash
pip install ailee-trust-layer anthropic
```

#### Basic Usage

```python
import anthropic
from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_ai_integrations import create_anthropic_adapter

# Initialize
client = anthropic.Anthropic()
pipeline = AileeTrustPipeline(AileeConfig())
adapter = create_anthropic_adapter()

# Call Claude API
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Evaluate this proposal and rate 0-100: ..."}
    ]
)

# Extract and validate
ai_response = adapter.extract_response(response)
result = pipeline.process(
    raw_value=ai_response.value,
    raw_confidence=ai_response.confidence,
    context={"model": "claude-3-5-sonnet"}
)

print(f"Validated rating: {result.value}")
```

#### Claude-Specific Tips

1. **Use thinking tags** to get Claude's reasoning
2. **Request explicit confidence** in your prompts
3. **Monitor stop_reason** (included in metadata)

---

### Google Gemini

#### Installation
```bash
pip install ailee-trust-layer google-generativeai
```

#### Basic Usage

```python
import google.generativeai as genai
import os
from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_ai_integrations import create_gemini_adapter

# Initialize
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')
pipeline = AileeTrustPipeline(AileeConfig())
adapter = create_gemini_adapter()

# Call Gemini API
response = model.generate_content(
    "Analyze this data and provide a quality score from 0-100: ..."
)

# Extract and validate
ai_response = adapter.extract_response(response)
result = pipeline.process(
    raw_value=ai_response.value,
    raw_confidence=ai_response.confidence,
    context={"model": "gemini-pro"}
)

print(f"Validated score: {result.value}")
print(f"Safety ratings: {ai_response.metadata.get('safety_ratings')}")
```

#### Gemini-Specific Features

1. **Automatic safety rating integration** - Confidence adjusts based on content safety checks
2. **Finish reason analysis** - STOP, SAFETY, MAX_TOKENS, RECITATION
3. **Multimodal support** - Works with Gemini Pro Vision for image/video analysis
4. **Token usage tracking** - Included in metadata

Example with Gemini Pro Vision:

```python
import PIL.Image

# Initialize vision model
vision_model = genai.GenerativeModel('gemini-pro-vision')
adapter = create_gemini_adapter()

# Analyze image
image = PIL.Image.open('photo.jpg')
response = vision_model.generate_content([
    "Rate the quality of this image from 0-100",
    image
])

ai_response = adapter.extract_response(response)
# ai_response.metadata includes safety_ratings and finish_reason
```

#### Best Practices for Gemini

1. **Monitor safety ratings** - Gemini provides built-in content safety checks
2. **Handle finish reasons** - Different finish reasons affect confidence
3. **Use appropriate models** - gemini-pro for text, gemini-pro-vision for multimodal
4. **Request explicit scores** - Ask for numeric outputs in prompts

---

### HuggingFace

#### Installation
```bash
pip install ailee-trust-layer transformers
```

#### Basic Usage

```python
from transformers import pipeline as hf_pipeline
from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_ai_integrations import create_huggingface_adapter

# Initialize
classifier = hf_pipeline("text-classification")
pipeline = AileeTrustPipeline(AileeConfig())
adapter = create_huggingface_adapter(task="classification")

# Run inference
result_hf = classifier("This product is excellent!")

# Extract and validate
ai_response = adapter.extract_response(result_hf[0])
result = pipeline.process(
    raw_value=ai_response.value,
    raw_confidence=ai_response.confidence,
    context={"model": "distilbert"}
)

print(f"Validated classification score: {result.value}")
```

#### For Text Generation

```python
generator = hf_pipeline("text-generation", model="gpt2")
adapter = create_huggingface_adapter(task="generation")

output = generator("Once upon a time", max_length=50, return_full_text=False)
ai_response = adapter.extract_response(output[0])

# Process through AILEE...
```

---

### LangChain

#### Installation
```bash
pip install ailee-trust-layer langchain langchain-openai
```

#### Basic Usage

```python
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from ailee import AileeTrustPipeline, AileeConfig
from ailee.optional.ailee_ai_integrations import create_langchain_adapter

# Create LangChain chain
llm = OpenAI(temperature=0)
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer with a score from 0-100: {question}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Initialize AILEE
pipeline = AileeTrustPipeline(AileeConfig())
adapter = create_langchain_adapter()

# Run chain
result_lc = chain.run(question="How good is this approach?")

# Extract and validate
ai_response = adapter.extract_response(result_lc)
result = pipeline.process(
    raw_value=ai_response.value,
    raw_confidence=ai_response.confidence,
    context={"chain": "llm_eval"}
)

print(f"Trusted score: {result.value}")
```

---

## Multi-Model Ensembles

Combine outputs from multiple AI models with consensus validation:

```python
from ailee.optional.ailee_ai_integrations import (
    create_multi_model_ensemble,
    create_openai_adapter,
    create_anthropic_adapter
)

# Initialize ensemble
ensemble = create_multi_model_ensemble()

# Add responses from different models
openai_adapter = create_openai_adapter()
anthropic_adapter = create_anthropic_adapter()

# Get responses (your existing code)
gpt4_response = openai_client.chat.completions.create(...)
claude_response = anthropic_client.messages.create(...)

# Add to ensemble
ensemble.add_response("gpt4", gpt4_response, openai_adapter)
ensemble.add_response("claude", claude_response, anthropic_adapter)

# Get consensus inputs
primary_value, peer_values, confidence_weights = ensemble.get_consensus_inputs()

# Validate with consensus
config = AileeConfig(
    consensus_quorum=2,      # Require 2 models to agree
    consensus_delta=10.0,    # Within 10 points
)
pipeline = AileeTrustPipeline(config)

result = pipeline.process(
    raw_value=primary_value,
    raw_confidence=max(confidence_weights.values()),
    peer_values=peer_values,
    context={"ensemble": "gpt4+claude"}
)

print(f"Ensemble decision: {result.value}")
print(f"Consensus status: {result.consensus_status}")  # PASS or FAIL
```

### Weighted Ensemble

Give more weight to high-confidence models:

```python
weighted_value, avg_confidence = ensemble.get_weighted_average()

result = pipeline.process(
    raw_value=weighted_value,
    raw_confidence=avg_confidence,
    peer_values=peer_values
)
```

---

## Advanced Patterns

### 1. Multi-Turn Conversations with Trust Building

```python
pipeline = AileeTrustPipeline(AileeConfig())
conversation_history = []

for turn in range(5):
    # Get AI response
    response = ai_model.generate(context=conversation_history)
    ai_response = adapter.extract_response(response)
    
    # Validate (trust builds over consistent turns)
    result = pipeline.process(
        raw_value=ai_response.value,
        raw_confidence=ai_response.confidence,
        context={"turn": turn}
    )
    
    conversation_history.append(result.value)
```

### 2. Confidence Thresholds by Use Case

```python
# High-stakes decisions (medical, financial)
high_stakes_config = AileeConfig(
    accept_threshold=0.95,
    borderline_low=0.85,
    consensus_quorum=3  # Require 3 models to agree
)

# General purpose
balanced_config = AileeConfig(
    accept_threshold=0.85,
    borderline_low=0.70
)

# Exploratory / low-risk
permissive_config = AileeConfig(
    accept_threshold=0.70,
    borderline_low=0.50
)
```

### 3. Custom Confidence Extractors

```python
def custom_confidence(response, content, context):
    """
    Custom logic to extract confidence from AI response.
    Could parse explicit confidence statements, analyze
    uncertainty markers, check for hedging language, etc.
    """
    # Example: Look for explicit confidence
    import re
    match = re.search(r'confidence[:\s]+(\d+)', content.lower())
    if match:
        return float(match.group(1)) / 100.0
    
    # Default
    return 0.8

adapter = OpenAIAdapter(confidence_extractor=custom_confidence)
```

---

## Real-World Examples

### Medical Diagnosis Support

```python
# High safety threshold for medical applications
config = AileeConfig(
    accept_threshold=0.95,
    consensus_quorum=3,
    consensus_delta=5.0
)
pipeline = AileeTrustPipeline(config)

# Get diagnoses from multiple AI models
diagnoses = []
for model in [gpt4, claude, medical_specialist_model]:
    response = model.diagnose(patient_data)
    ai_resp = adapter.extract_response(response)
    diagnoses.append(ai_resp)

# Ensemble validation
ensemble = create_multi_model_ensemble()
for i, diag in enumerate(diagnoses):
    ensemble.add_response(f"model_{i}", diag.raw_response, adapter)

primary, peers, _ = ensemble.get_consensus_inputs()
result = pipeline.process(
    raw_value=primary,
    raw_confidence=max(d.confidence for d in diagnoses),
    peer_values=peers,
    context={"domain": "medical", "critical": True}
)

if result.consensus_status == "PASS" and not result.used_fallback:
    print(f"High-confidence diagnosis: {result.value}")
else:
    print("Insufficient confidence - escalate to human expert")
```

### Content Moderation

```python
from ailee.optional.ailee_config_presets import CONSERVATIVE

# Use conservative preset for safety
pipeline = AileeTrustPipeline(CONSERVATIVE)

# Check content safety
safety_score = content_safety_model.evaluate(content)
ai_response = adapter.extract_response(safety_score)

result = pipeline.process(
    raw_value=ai_response.value,
    raw_confidence=ai_response.confidence,
    context={"type": "content_safety"}
)

if result.value > 80 and not result.used_fallback:
    publish_content()
else:
    flag_for_human_review()
```

### Automated Trading Signals

```python
from ailee.optional.ailee_config_presets import TRADING_SIGNAL

# Domain-optimized preset
pipeline = AileeTrustPipeline(TRADING_SIGNAL)

# Get signal from AI
trade_signal = trading_model.predict(market_data)
ai_response = adapter.extract_response(trade_signal)

result = pipeline.process(
    raw_value=ai_response.value,
    raw_confidence=ai_response.confidence,
    peer_values=historical_signals[-10:],  # Compare with recent history
    context={"symbol": "AAPL", "strategy": "momentum"}
)

if result.safety_status == "ACCEPTED" and result.consensus_status == "PASS":
    execute_trade(result.value)
else:
    log_uncertain_signal(result)
```

---

## Next Steps

1. **Install AILEE**: `pip install ailee-trust-layer`
2. **Choose your framework**: Pick the integration guide above
3. **Start simple**: Begin with basic integration
4. **Add consensus**: Use multi-model ensembles for critical decisions
5. **Tune thresholds**: Adjust confidence thresholds for your use case
6. **Monitor**: Use AILEE's monitoring tools to track trust metrics

## Resources

- **Full API Documentation**: [docs/API.md](../docs/API.md)
- **Configuration Presets**: [optional/ailee_config_presets.py](../optional/ailee_config_presets.py)
- **More Examples**: [examples/](../examples/)
- **White Paper**: [docs/whitepaper/](../docs/whitepaper/)

## Support

- GitHub Issues: https://github.com/dfeen87/ailee-trust-layer/issues
- Documentation: https://github.com/dfeen87/ailee-trust-layer

---

**Remember**: AILEE doesn't make AI smarter — it makes it trustworthy.
