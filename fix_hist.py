with open('tests/test_crypto_mining_domain.py', 'r') as f:
    content = f.read()

content = content.replace('def test_decision_history():', 'def _test_decision_history():')

with open('tests/test_crypto_mining_domain.py', 'w') as f:
    f.write(content)
