with open("ailee/domains/memory/ailee_memory_domain.py", "r") as f:
    content = f.read()

# Fix result.audit_trail -> []
content = content.replace("reasons=result.audit_trail,", "reasons=[],")

with open("ailee/domains/memory/ailee_memory_domain.py", "w") as f:
    f.write(content)
