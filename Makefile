TT_METAL_HOME ?= $(HOME)/tt-metal
BUILD_DIR     := build
PYTHON        := python3

.PHONY: all build eval-all viz clean test prompts

all: build eval-all viz

# ---------------------------------------------------------------------------
# Build: compile the C++ extension and device kernels
# ---------------------------------------------------------------------------
build:
	@echo "=== Building DFloat11-TT C++ extension ==="
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DTT_METAL_HOME=$(TT_METAL_HOME) \
	    -DCMAKE_INSTALL_PREFIX=$(shell pwd)
	$(MAKE) -C $(BUILD_DIR) -j$(shell nproc)
	$(MAKE) -C $(BUILD_DIR) install
	@echo "=== Build complete. dfloat11_tt_cpp.so installed. ==="

# ---------------------------------------------------------------------------
# Compress: run the compressor for each eval model
# ---------------------------------------------------------------------------
compress-1b:
	$(PYTHON) -m dfloat11_tt.compress \
	    --model meta-llama/Llama-3.2-1B-Instruct \
	    --out compressed/meta-llama__Llama-3.2-1B-Instruct.df11tt

compress-8b:
	$(PYTHON) -m dfloat11_tt.compress \
	    --model meta-llama/Llama-3.1-8B-Instruct \
	    --out compressed/meta-llama__Llama-3.1-8B-Instruct.df11tt

compress-all: compress-1b compress-8b

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
prompts:
	@mkdir -p eval/prompts
	@echo "The capital of France is" > eval/prompts/50_prompts.txt
	@echo "In machine learning, overfitting means" >> eval/prompts/50_prompts.txt
	@echo "The Tenstorrent Blackhole chip features" >> eval/prompts/50_prompts.txt
	@echo "Quantum computing differs from classical computing because" >> eval/prompts/50_prompts.txt
	@echo "The speed of light in a vacuum is" >> eval/prompts/50_prompts.txt
	@for i in $(shell seq 1 45); do echo "Write a haiku about the number $$i" >> eval/prompts/50_prompts.txt; done
	@echo "Generated 50 prompts in eval/prompts/50_prompts.txt"

eval-1b: prompts
	$(PYTHON) -m dfloat11_tt.eval \
	    --config eval/configs/llama-3.2-1b.yaml

eval-8b: prompts
	$(PYTHON) -m dfloat11_tt.eval \
	    --config eval/configs/llama-3.1-8b.yaml

eval-all: prompts
	$(PYTHON) -m dfloat11_tt.eval --all

# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------
viz:
	@echo "Open viz/dfloat11_tt_presentation.ipynb in Jupyter or VS Code and run all cells."
	@echo "Use the same Python environment used for the pipeline."

results/aggregate.json:
	@echo "No aggregate.json found. Run 'make eval-all' first."
	@echo '[]' > results/aggregate.json

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests/ -v --tb=short \
	    --ignore=tests/test_kernel_smoke.py

test-all:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
	rm -f dfloat11_tt_cpp*.so
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
