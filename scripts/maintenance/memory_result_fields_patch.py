# Licensed under the PolyForm Noncommercial License 1.0.0
with open("ailee/domains/memory/ailee_memory_domain.py", "r") as f:
    content = f.read()

# Fix result.fallback_reason -> result.safety_status
content = content.replace("result.fallback_reason", "result.safety_status")
content = content.replace("reasons=[],", "reasons=result.reasons,")
content = content.replace("result.original_value", "signals.ai_value")

with open("ailee/domains/memory/ailee_memory_domain.py", "w") as f:
    f.write(content)
