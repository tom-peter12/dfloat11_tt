# Evaluation Report: HuggingFaceTB/SmolLM2-135M-Instruct

## Output Equivalence
- Result: PASS
- Prompts tested: 1
- Failures: 0
- Prompt: The capital of France is
- BF16 completion:  Paris. Paris is
- DF11 completion:  Paris. Paris is
- BF16 generation: 0.14s
- DF11 generation: 1.17s
- DF11 tokens/sec: 3.42
