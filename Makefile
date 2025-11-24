<<<<<<< HEAD
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude
LDFLAGS ?=

CPP_SOURCES := main.cpp \
               src/common.cpp \
               src/single.cpp \
               src/2d.cpp \
               src/openmp.cpp \
               src/avx.cpp \
               src/avx512.cpp \
               src/benchmark.cpp \
               src/visiualize.cpp \
               src/visualize_gl.cpp \
               src/cuda_stub.cpp

CUDA_SOURCES :=

ifeq ($(USE_CUDA),1)
CUDA_SOURCES += src/cuda.cu
endif

SOURCES := $(CPP_SOURCES) $(CUDA_SOURCES)

OBJECTS := $(CPP_SOURCES:.cpp=.o)
OBJECTS += $(CUDA_SOURCES:.cu=.o)
TARGET := life

ifeq ($(USE_OPENMP),0)
NO_OPENMP := 1
else
CXXFLAGS += -fopenmp
endif

ifeq ($(USE_AVX2),1)
CXXFLAGS += -mavx2
endif

ifeq ($(USE_AVX512),1)
CXXFLAGS += -mavx512f -mavx512bw
endif

ifeq ($(USE_CUDA),1)
CPPFLAGS += -DUSE_CUDA
LDFLAGS += -lcudart
NVCC ?= nvcc
CUDAFLAGS ?= -std=c++17
endif

GLFW_CFLAGS ?=
GLFW_LIBS ?= -lglfw -lGL -ldl -lpthread

ifeq ($(USE_GLFW),1)
CPPFLAGS += -DUSE_GLFW $(GLFW_CFLAGS)
LDFLAGS += $(GLFW_LIBS)
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
=======
CXX := g++
# Base flags for C++ compilation
# Use O3 for better optimization, especially for AVX2
CXXFLAGS_BASE := -std=c++17 -Wall -Wextra -O3 -Iinclude
# Default C++ flags for non-CUDA build: define CUDA_STUB so wrappers fall back to CPU
CXXFLAGS := $(CXXFLAGS_BASE) -DCUDA_STUB
# Flags to use when building CUDA target (no CUDA_STUB)
CXXFLAGS_NO_STUB := $(CXXFLAGS_BASE)
LDFLAGS :=

# Check if OpenMP is available
OPENMP_AVAILABLE := $(shell echo | $(CXX) -fopenmp -E - 2>/dev/null && echo 1 || echo 0)
ifeq ($(OPENMP_AVAILABLE),1)
  OPENMP_FLAGS := -fopenmp
  OPENMP_LDFLAGS := -fopenmp
else
  OPENMP_FLAGS :=
  OPENMP_LDFLAGS :=
endif

# AVX2 flags (most modern CPUs support AVX2)
# Use -O3 and -march=native for better optimization
AVX2_FLAGS := -mavx2 -march=native -O3

# Tools linking (SDL2 + OpenGL for 3D viewer)
TOOLS_LDFLAGS := -lSDL2 -lGL -lGLU

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
NVCC_OBJ_DIR := $(OBJ_DIR)/nvcc
# Only compile CUDA sources that have real implementations (avoid compiling empty stubs)
NVCC_SRCS := \
	src/2d/cuda_single_species.cu \
	src/2d/cuda_multi_species.cu \
	src/3d/cuda_single_species.cu \
	src/3d/cuda_multi_species.cu
NVCC_OBJS := $(patsubst src/%.cu,$(NVCC_OBJ_DIR)/%.o,$(NVCC_SRCS))

all: main tools

main: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(OPENMP_LDFLAGS)

# Compile rules with backend-specific flags
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	@if echo "$<" | grep -q "openmp"; then \
		$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -c $< -o $@; \
	elif echo "$<" | grep -q "avx2"; then \
		$(CXX) $(CXXFLAGS) $(AVX2_FLAGS) -c $< -o $@; \
	else \
		$(CXX) $(CXXFLAGS) -c $< -o $@; \
	fi

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
	rm -rf $(OBJ_DIR) main tools/visualize_3d_sdl tools/realtime_viewer

.PHONY: all clean

.PHONY: cuda
cuda:
	@echo "Building with nvcc for CUDA sources (if nvcc available)"
	@mkdir -p $(OBJ_DIR)/2d $(OBJ_DIR)/3d
	@echo "Compiling CUDA sources with nvcc into $(NVCC_OBJ_DIR)"
	@for src in $(NVCC_SRCS); do \
		rel=$${src#src/}; \
		obj=$(NVCC_OBJ_DIR)/$${rel%.cu}.o; \
		mkdir -p $$(dirname $$obj); \
		$(NVCC) $(NVCCFLAGS) -c $$src -o $$obj || true; \
	done
	@echo "Compiling C++ sources into objects"
	@for src in $(SRCS_CPP); do \
		rel=$${src#src/}; \
		obj=$(OBJ_DIR)/$${rel%.cpp}.o; \
		mkdir -p $$(dirname $$obj); \
		$(CXX) $(CXXFLAGS_NO_STUB) -c $$src -o $$obj; \
	done
	@echo "Linking with nvcc to include CUDA runtime"
	@$(NVCC) $(NVCCFLAGS) $(OBJS_CPP) $(NVCC_OBJS) -o main $(LDFLAGS)

.PHONY: tools
tools:
	@echo "Building tools (SDL/OpenGL viewers)"
	@$(CXX) $(CXXFLAGS_BASE) tools/visualize_3d_sdl.cpp -o tools/visualize_3d_sdl $(TOOLS_LDFLAGS)
	$(CXX) $(CXXFLAGS) -o tools/realtime_viewer tools/realtime_viewer_v2.cpp src/species.cpp $(TOOLS_LDFLAGS) $(OPENMP_FLAGS)
>>>>>>> 0.0.2.5
