with open("ailee/domains/memory/ailee_memory_domain.py", "r") as f:
    content = f.read()

# Fix SafetyStatus strings instead of enums
content = content.replace("SafetyStatus.REJECTED", '"REJECTED"')
content = content.replace("SafetyStatus.ACCEPTED", '"ACCEPTED"')
content = content.replace("SafetyStatus.OUTRIGHT_REJECTED", '"OUTRIGHT_REJECTED"')
content = content.replace("result.safety_status == SafetyStatus.ACCEPTED", 'result.safety_status == "ACCEPTED"')

# Fix result.is_safe -> result.safety_status != "OUTRIGHT_REJECTED" or whatever the condition was
content = content.replace("if not result.is_safe:", 'if result.safety_status == "OUTRIGHT_REJECTED":')

# Fix result.final_value -> result.value
content = content.replace("result.final_value", "result.value")

with open("ailee/domains/memory/ailee_memory_domain.py", "w") as f:
    f.write(content)
