CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -Iinclude
LDFLAGS :=

NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -Xcompiler "-Wall -Wextra" -Iinclude
NVCC_AVAILABLE := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)
CU_STUB_FLAGS := -x c++ -DCUDA_STUB

SRCS_CPP := $(shell find src -name '*.cpp')
SRCS_CU := $(shell find src -name '*.cu')

OBJ_DIR := build
OBJS_CPP := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(SRCS_CPP))
OBJS_CU := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(SRCS_CU))
OBJS := $(OBJS_CPP) $(OBJS_CU)

all: main

main: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

ifeq ($(NVCC_AVAILABLE),1)
$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CU_STUB_FLAGS) -c $< -o $@
endif

clean:
	rm -rf $(OBJ_DIR) main

.PHONY: all clean
