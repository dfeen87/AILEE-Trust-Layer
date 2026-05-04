1. **Explore the reference domains (`datacenter`, `crypto_mining`) to understand the exact structure required for `memory` domain.**
   - Enums
   - Dataclasses
   - Config instances
   - Governor class logic
   - Factory functions
   - Validation logic

2. **Implement `ailee/domains/memory/ailee_memory_domain.py`**
   - Config Presets: `RAM_ALLOCATION`, `HEAP_MONITORING`, `SWAP_MANAGEMENT`, `PROCESS_MEMORY`.
     - *Wait*, I need the exact thresholds from the specification! The prompt provided a specification implicitly or requires me to follow standard domain setups. Ah, the prompt says "Use the exact thresholds, weights, and flags from the specification." but the user didn't attach `MEMORY_SPEC.md` or a specification document.
     - *Correction*: Let me read the prompt again. "Your task is to implement the AILEE Memory (RAM) Trust Governance Domain exactly according to the specification provided. ... Use the exact thresholds, weights, and flags from the specification."
     - Wait, was the specification provided in the prompt? No. Let me search the environment to see if there's a file I missed. Let me check the user prompt again. "Your task is to implement the AILEE Memory (RAM) Trust Governance Domain exactly according to the specification provided. You must: ... Follow the exact fields, defaults, and types."
     - If the spec is missing, maybe I should ask for it, or maybe it is hidden in memory or in the issue text? Let me review the system memory and context. I can't find a file. Let me ask the user or search more. Wait, I should request user input to get the spec if it isn't anywhere, but let's check one more time if I can find it in the repository.
