# Evaluation Report: Qwen/Qwen2.5-0.5B-Instruct

## Output Equivalence
- Result: FAIL
- Prompts tested: 1
- Failures: 1
- Prompt: The capital of France is
- BF16 completion:  Paris. It was founded in 789 AD by Charlemagne,
- DF11 completion:  Paris. It is the largest city in Europe and one of the most important cities
- BF16 generation: 0.64s
- DF11 generation: 238.71s
- DF11 tokens/sec: 0.07
