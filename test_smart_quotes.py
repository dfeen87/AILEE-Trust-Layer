import os

# Simulate env var with smart quotes
os.environ["OPENAI_API_KEY"] = "“sk-test-smart-quotes”"

def clean_key(key):
    # Current logic
    return key.strip().strip('"').strip("'")

raw = os.environ["OPENAI_API_KEY"]
cleaned = clean_key(raw)

print(f"Raw: {raw}")
print(f"Cleaned (current): {cleaned}")

if cleaned.startswith("sk-"):
    print("Success")
else:
    print("Fail - Smart quotes remain")
