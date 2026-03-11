# PTX-OS Makefile
# Auto-detects CUDA and GPU compute capability

# Directories
SRC_DIR := native/core
INC_DIR := native/core/include
LIB_DIR := lib
BUILD_DIR := build

# Centralized env discovery (scripts/ptx_env.sh)
PTX_ENV_SH := ./tooling/scripts/ptx_env.sh
PTX_ENV_MK := $(BUILD_DIR)/ptx_env.mk
ifeq ($(wildcard $(PTX_ENV_SH)),)
    $(error Missing tooling/scripts/ptx_env.sh)
endif
$(shell mkdir -p $(BUILD_DIR) && rm -f $(PTX_ENV_MK) && $(PTX_ENV_SH) --format make --quiet > $(PTX_ENV_MK) || rm -f $(PTX_ENV_MK))
-include $(PTX_ENV_MK)

# Require CUDA_PATH from the env script or user override
ifeq ($(strip $(CUDA_PATH)),)
    $(error CUDA_PATH not set. Run tooling/scripts/ptx_env.sh or set CUDA_PATH/CUDA_HOME)
endif

NVCC := $(CUDA_PATH)/bin/nvcc
CC := gcc

# Detect GPU compute capability (provided by tooling/scripts/ptx_env.sh)
# Always build fat binary with all common architectures.
# For CUDA 12.8+ add: compute_100/sm_100, compute_120/sm_120
GENCODE_FLAGS := -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_86,code=sm_86 \
                 -gencode arch=compute_87,code=sm_87 \
                 -gencode arch=compute_89,code=sm_89 \
                 -gencode arch=compute_90,code=sm_90
ifneq ($(GPU_SM),)
    GPU_SM_NUM := $(shell echo "$(GPU_SM)" | tr -cd '0-9')
    ifeq ($(shell [ -n "$(GPU_SM_NUM)" ] && [ "$(GPU_SM_NUM)" -lt 75 ] && echo yes),yes)
        $(error GPU_SM=$(GPU_SM_NUM) is unsupported by current kernel profile; sm_75+ required)
    endif
    # Also include detected SM if not already in the standard list
    ifeq ($(filter $(GPU_SM),75 80 86 87 89 90),)
        GENCODE_FLAGS += -gencode arch=compute_$(GPU_SM),code=sm_$(GPU_SM)
    endif
endif

# Compiler flags
NVCC_FLAGS := -O3 -shared -Xcompiler '-fPIC' --compiler-options '-fPIC'
NVCC_FLAGS += $(GENCODE_FLAGS)
NVCC_FLAGS += -rdc=true -lcudadevrt  # Enable dynamic parallelism

# Optional strict free enforcement (abort on non-PTX frees even in release)
ifneq ($(PTX_STRICT_FREE),)
    NVCC_FLAGS += -DPTX_STRICT_FREE
    CFLAGS += -DPTX_STRICT_FREE
endif

# Include paths
INCLUDES := -I$(INC_DIR) -I$(SRC_DIR) -I$(CUDA_PATH)/include

# Libraries
CUDA_LIB ?= $(shell if [ -d "$(CUDA_PATH)/lib64" ]; then echo "$(CUDA_PATH)/lib64"; else echo "$(CUDA_PATH)/lib"; fi)
LIBS := -L$(CUDA_LIB) -lcublas -lcudart -lcuda

# Source files (recursive)
CUDA_SOURCES := $(shell find $(SRC_DIR) -type f -name '*.cu' | sort)

# Output
TARGET := $(LIB_DIR)/libptx_os.so
HOOK_TARGET := $(LIB_DIR)/libptx_hook.so

# C compiler flags for hook
CFLAGS := -O3 -fPIC -shared
HOOK_LIBS := -L$(LIB_DIR) -lptx_os -ldl -lcudart -Wl,-rpath,$(LIB_DIR)

.PHONY: all clean dirs rust rust-debug test test-hook test-stable info check-cuda hook python python-jax

all: check-cuda dirs $(TARGET) $(HOOK_TARGET)

check-cuda:
	@if [ ! -f "$(NVCC)" ]; then \
		echo "Error: nvcc not found at $(NVCC)"; \
		echo "Set CUDA_PATH or CUDA_HOME environment variable"; \
		exit 1; \
	fi
	@echo "CUDA: $(CUDA_PATH)"
	@$(NVCC) --version | head -n 1
	@echo "GPU SM: $(GPU_SM)"

dirs:
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BUILD_DIR)

$(TARGET): $(CUDA_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)
	@echo "Built: $@"

# Build CUDA intercept hook (LD_PRELOAD library) - includes context hook
HOOK_SOURCES := $(SRC_DIR)/hooks/cuda_intercept.c $(SRC_DIR)/hooks/cuda_context_hook.c
$(HOOK_TARGET): $(HOOK_SOURCES) $(TARGET)
	$(CC) $(CFLAGS) $(INCLUDES) -I$(CUDA_PATH)/include $(HOOK_SOURCES) -o $@ -L$(CUDA_LIB) -L$(LIB_DIR) -Wl,-rpath,'$$ORIGIN' -Wl,--no-as-needed -lptx_os -ldl -lcudart
	@echo "Built: $@"

hook: $(HOOK_TARGET)
	@echo "Hook library ready. Usage:"
	@echo "  LD_PRELOAD=$(LIB_DIR)/libptx_hook.so python train.py"

# Build Python PyTorch allocator
python: $(TARGET)
	cd python/ptx_torch && pip install -e .
	@echo "PyTorch allocator installed. Usage:"
	@echo "  import ptx_torch; ptx_torch.enable()"

# Build Python JAX allocator
python-jax: $(TARGET)
	cd python/ptx_jax && pip install pybind11 && pip install -e .
	@echo "JAX allocator installed. Usage:"
	@echo "  import ptx_jax; ptx_jax.enable()  # Call BEFORE importing jax"
	@echo "  import jax"
	@echo ""
	@echo "Or use LD_PRELOAD for zero code changes:"
	@echo "  LD_PRELOAD=python/ptx_jax/lib/libptx_jax_preload.so python my_jax_script.py"

# Build Rust crates
rust: $(TARGET)
	CUDA_PATH=$(CUDA_PATH) cargo build --release

rust-debug: $(TARGET)
	CUDA_PATH=$(CUDA_PATH) cargo build

# Check Rust code compiles
check: $(TARGET)
	CUDA_PATH=$(CUDA_PATH) cargo check

# Run Rust tests
test: $(TARGET)
	CUDA_PATH=$(CUDA_PATH) LD_LIBRARY_PATH=$(abspath $(LIB_DIR)):$$LD_LIBRARY_PATH cargo test --release

# Build and run the CUDA hook interception test (context + memory)
TEST_HOOK_SRC := tests/test_hook.cu
TEST_HOOK_BIN := $(BUILD_DIR)/test_hook
TEST_STABLE_SRC := tests/test_stable_runtime.cu
TEST_STABLE_BIN := $(BUILD_DIR)/test_stable_runtime

$(TEST_HOOK_BIN): $(TEST_HOOK_SRC) $(TARGET) $(HOOK_TARGET)
	$(NVCC) -o $@ $(TEST_HOOK_SRC) -I$(CUDA_PATH)/include -L$(CUDA_LIB) -lcuda -lcudart

test-hook: $(TEST_HOOK_BIN)
	@echo "Running PTX-OS hook interception test..."
	PTX_HOOK_VERBOSE=1 ./tooling/scripts/ferrite-run $(TEST_HOOK_BIN)

$(TEST_STABLE_BIN): $(TEST_STABLE_SRC) $(TARGET)
	$(NVCC) -o $@ $(TEST_STABLE_SRC) -I$(INC_DIR) -I$(CUDA_PATH)/include -L$(LIB_DIR) -lptx_os -L$(CUDA_LIB) -lcuda -lcudart

test-stable: $(TEST_STABLE_BIN)
	@echo "Running PTX-OS stable runtime ABI smoke test..."
	LD_LIBRARY_PATH=$(abspath $(LIB_DIR)):$$LD_LIBRARY_PATH $(TEST_STABLE_BIN)

clean:
	rm -rf $(LIB_DIR) $(BUILD_DIR)
	cargo clean 2>/dev/null || true

info:
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "CUDA Version: $(CUDA_VERSION)"
	@echo "NVCC: $(NVCC)"
	@echo "GPU SM: $(GPU_SM)"
	@echo "PTX GPU SM: $(PTX_GPU_SM)"
	@echo "CUDA Lib: $(CUDA_LIB)"
	@echo "GENCODE: $(GENCODE_FLAGS)"
	@echo "Sources: $(CUDA_SOURCES)"
	@echo "Target: $(TARGET)"
