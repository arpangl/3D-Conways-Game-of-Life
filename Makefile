# CXX := g++
# # Base flags for C++ compilation
# # Use O3 for better optimization, especially for AVX2
# CXXFLAGS_BASE := -std=c++17 -Wall -Wextra -O3 -Iinclude
# # Default C++ flags for non-CUDA build: define CUDA_STUB so wrappers fall back to CPU
# CXXFLAGS := $(CXXFLAGS_BASE) -DCUDA_STUB
# # Flags to use when building CUDA target (no CUDA_STUB)
# CXXFLAGS_NO_STUB := $(CXXFLAGS_BASE)
# LDFLAGS :=

# # Check if OpenMP is available
# OPENMP_AVAILABLE := $(shell echo | $(CXX) -fopenmp -E - 2>/dev/null && echo 1 || echo 0)
# ifeq ($(OPENMP_AVAILABLE),1)
#   OPENMP_FLAGS := -fopenmp
#   OPENMP_LDFLAGS := -fopenmp
# else
#   OPENMP_FLAGS :=
#   OPENMP_LDFLAGS :=
# endif

# # AVX2 flags (most modern CPUs support AVX2)
# # Use -O3 and -march=native for better optimization
# AVX2_FLAGS := -mavx2 -march=native -O3

# # Tools linking (SDL2 + OpenGL for 3D viewer)
# TOOLS_LDFLAGS := -framework OpenGL -lSDL2#-lSDL2 -lGL -lGLU

# NVCC ?= nvcc
# NVCCFLAGS ?= -std=c++17 -O2 -Xcompiler "-Wall -Wextra" -Iinclude
# NVCC_AVAILABLE := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)
# CU_STUB_FLAGS := -x c++ -DCUDA_STUB

# SRCS_CPP := $(shell find src -name '*.cpp')
# SRCS_CU := $(shell find src -name '*.cu')

# OBJ_DIR := build
# OBJS_CPP := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(SRCS_CPP))
# OBJS_CU := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(SRCS_CU))
# OBJS := $(OBJS_CPP) $(OBJS_CU)
# NVCC_OBJ_DIR := $(OBJ_DIR)/nvcc
# # Only compile CUDA sources that have real implementations (avoid compiling empty stubs)
# NVCC_SRCS := \
# 	src/2d/cuda_single_species.cu \
# 	src/2d/cuda_multi_species.cu \
# 	src/3d/cuda_single_species.cu \
# 	src/3d/cuda_multi_species.cu
# NVCC_OBJS := $(patsubst src/%.cu,$(NVCC_OBJ_DIR)/%.o,$(NVCC_SRCS))

# all: main tools

# main: $(OBJS)
# 	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(OPENMP_LDFLAGS)

# # Compile rules with backend-specific flags
# $(OBJ_DIR)/%.o: src/%.cpp
# 	@mkdir -p $(dir $@)
# 	@if echo "$<" | grep -q "openmp"; then \
# 		$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -c $< -o $@; \
# 	elif echo "$<" | grep -q "avx2"; then \
# 		$(CXX) $(CXXFLAGS) $(AVX2_FLAGS) -c $< -o $@; \
# 	else \
# 		$(CXX) $(CXXFLAGS) -c $< -o $@; \
# 	fi

# ifeq ($(NVCC_AVAILABLE),1)
# $(OBJ_DIR)/%.o: src/%.cu
# 	@mkdir -p $(dir $@)
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@
# else
# $(OBJ_DIR)/%.o: src/%.cu
# 	@mkdir -p $(dir $@)
# 	$(CXX) $(CXXFLAGS) $(CU_STUB_FLAGS) -c $< -o $@
# endif

# clean:
# 	rm -rf $(OBJ_DIR) main tools/visualize_3d_sdl tools/realtime_viewer

# .PHONY: all clean

# .PHONY: cuda
# cuda:
# 	@echo "Building with nvcc for CUDA sources (if nvcc available)"
# 	@mkdir -p $(OBJ_DIR)/2d $(OBJ_DIR)/3d
# 	@echo "Compiling CUDA sources with nvcc into $(NVCC_OBJ_DIR)"
# 	@for src in $(NVCC_SRCS); do \
# 		rel=$${src#src/}; \
# 		obj=$(NVCC_OBJ_DIR)/$${rel%.cu}.o; \
# 		mkdir -p $$(dirname $$obj); \
# 		$(NVCC) $(NVCCFLAGS) -c $$src -o $$obj || true; \
# 	done
# 	@echo "Compiling C++ sources into objects"
# 	@for src in $(SRCS_CPP); do \
# 		rel=$${src#src/}; \
# 		obj=$(OBJ_DIR)/$${rel%.cpp}.o; \
# 		mkdir -p $$(dirname $$obj); \
# 		$(CXX) $(CXXFLAGS_NO_STUB) -c $$src -o $$obj; \
# 	done
# 	@echo "Linking with nvcc to include CUDA runtime"
# 	@$(NVCC) $(NVCCFLAGS) $(OBJS_CPP) $(NVCC_OBJS) -o main $(LDFLAGS)

# .PHONY: tools
# tools:
# 	@echo "Building tools (SDL/OpenGL viewers)"
# 	@$(CXX) $(CXXFLAGS_BASE) tools/visualize_3d_sdl.cpp -o tools/visualize_3d_sdl $(TOOLS_LDFLAGS)
# 	$(CXX) $(CXXFLAGS) -o tools/realtime_viewer tools/realtime_viewer_v2.cpp src/species.cpp $(TOOLS_LDFLAGS) $(OPENMP_FLAGS)
CXX := g++

# --- 1. 自動偵測作業系統與路徑設定 ---
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# 預設設定 (適用於 Linux)
# 基礎 Include 路徑
INCLUDES := -Iinclude
# 基礎 Library 搜尋路徑
LIB_DIRS := 
# OpenMP 設定
OMP_FLAGS := -fopenmp
OMP_LIBS := -fopenmp
# 3D/視覺化工具連結設定 (Linux 使用 -lGL)
TOOLS_LDFLAGS := -lGL -lGLU -lSDL2

# macOS 設定 (Darwin)
ifeq ($(UNAME_S), Darwin)
	# 偵測 Homebrew 路徑 (Apple Silicon vs Intel)
	ifeq ($(UNAME_M), arm64)
		BREW_PREFIX := /opt/homebrew
	else
		BREW_PREFIX := /usr/local
	endif

	# 更新 Include 和 Lib 路徑 (加入 Homebrew 和 libomp)
	INCLUDES += -I$(BREW_PREFIX)/include -I$(BREW_PREFIX)/opt/libomp/include
	LIB_DIRS += -L$(BREW_PREFIX)/lib -L$(BREW_PREFIX)/opt/libomp/lib

	# macOS OpenMP 特殊 Flag
	OMP_FLAGS := -Xpreprocessor -fopenmp
	OMP_LIBS := -lomp

	# macOS 視覺化連結 (使用 Framework)
	TOOLS_LDFLAGS := -framework OpenGL -lSDL2
	
	# 加入巨集定義，方便 C++ 程式碼識別 macOS
	CXXFLAGS_EXTRA := -D__APPLE__
	AVX2_FLAGS := -mavx2 -march=haswell -O3 -D__AVX2__
endif

# Windows 設定 (MinGW/MSYS)
ifneq (,$(findstring MINGW,$(UNAME_S)))
	TOOLS_LDFLAGS := -lmingw32 -lSDL2main -lSDL2 -lopengl32 -lglu32
	OMP_FLAGS := -fopenmp
	OMP_LIBS := -fopenmp
	CXXFLAGS_EXTRA := -D_WIN32
	AVX2_FLAGS := -mavx2 -march=native -O3
endif


# --- 2. 編譯參數設定 ---

# 基礎 C++ Flags (加入 INCLUDES 和 OS 定義)
# 使用 -O3 進行優化
CXXFLAGS_BASE := -std=c++17 -Wall -Wextra -O3 $(INCLUDES) $(CXXFLAGS_EXTRA)

# 預設 C++ Flags (非 CUDA 建置): 定義 CUDA_STUB
CXXFLAGS := $(CXXFLAGS_BASE) -DCUDA_STUB

# CUDA 建置專用 Flags (不含 STUB)
CXXFLAGS_NO_STUB := $(CXXFLAGS_BASE)

# AVX2 Flags
#AVX2_FLAGS := -mavx2 -march=native -O3

# NVCC 設定
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -Xcompiler "-Wall -Wextra" -Iinclude
NVCC_AVAILABLE := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1)
CU_STUB_FLAGS := -x c++ -DCUDA_STUB

# --- 3. 檔案搜尋與規則 ---

SRCS_CPP := $(shell find src -name '*.cpp')
SRCS_CU := $(shell find src -name '*.cu')

OBJ_DIR := build
OBJS_CPP := $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(SRCS_CPP))
OBJS_CU := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(SRCS_CU))
OBJS := $(OBJS_CPP) $(OBJS_CU)

NVCC_OBJ_DIR := $(OBJ_DIR)/nvcc
# 只編譯有實作的 CUDA 檔案
NVCC_SRCS := \
	src/2d/cuda_single_species.cu \
	src/2d/cuda_multi_species.cu \
	src/3d/cuda_single_species.cu \
	src/3d/cuda_multi_species.cu
NVCC_OBJS := $(patsubst src/%.cu,$(NVCC_OBJ_DIR)/%.o,$(NVCC_SRCS))

# 主要目標
all: main tools

# Main 程式連結
# 注意：這裡加入了 $(LIB_DIRS) 以確保 Linker 找得到庫
main: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIB_DIRS) $(OMP_LIBS)

# C++ 檔案編譯規則 (根據檔名選擇 OpenMP 或 AVX2)
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	@if echo "$<" | grep -q "openmp"; then \
		$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -c $< -o $@; \
	elif echo "$<" | grep -q "avx2"; then \
		$(CXX) $(CXXFLAGS) $(AVX2_FLAGS) -c $< -o $@; \
	else \
		$(CXX) $(CXXFLAGS) -c $< -o $@; \
	fi

# CUDA 檔案編譯規則
ifeq ($(NVCC_AVAILABLE),1)
$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CU_STUB_FLAGS) -c $< -o $@
endif

# 清理規則
clean:
	rm -rf $(OBJ_DIR) main tools/visualize_3d_sdl tools/realtime_viewer tools/realtime_viewer_avx2
	rm -rf $(OBJ_DIR) main tools/visualize_3d_sdl tools/realtime_viewer
.PHONY: all clean cuda tools

# CUDA 專用建置 (保留原本邏輯)
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
	@$(NVCC) $(NVCCFLAGS) $(OBJS_CPP) $(NVCC_OBJS) -o main $(LIB_DIRS) $(OMP_LIBS)

# Tools 建置規則 (視覺化工具)
# 注意：這裡加入了 $(LIB_DIRS) 和 $(TOOLS_LDFLAGS)
tools:
	@echo "Building tools (SDL/OpenGL viewers)"
	@mkdir -p tools
	$(CXX) $(CXXFLAGS_BASE) tools/visualize_3d_sdl.cpp -o tools/visualize_3d_sdl $(LIB_DIRS) $(TOOLS_LDFLAGS)
	$(CXX) $(CXXFLAGS_BASE) tools/ppm_viewer.cpp -o tools/ppm_viewer $(LIB_DIRS) $(TOOLS_LDFLAGS)
	$(CXX) $(CXXFLAGS_BASE) tools/ppm_viewer_sdl.cpp -o tools/ppm_viewer_sdl $(LIB_DIRS) $(TOOLS_LDFLAGS)
	$(CXX) $(CXXFLAGS_BASE) $(OMP_FLAGS) tools/realtime_viewer_v2.cpp src/species.cpp -o tools/realtime_viewer $(LIB_DIRS) $(TOOLS_LDFLAGS) $(OMP_LIBS)
	$(CXX) $(CXXFLAGS_BASE) $(AVX2_FLAGS) $(OMP_FLAGS) tools/realtime_viewer_avx2.cpp src/species.cpp -o tools/realtime_viewer_avx2 $(LIB_DIRS) $(TOOLS_LDFLAGS) $(OMP_LIBS)
	$(CXX) $(CXXFLAGS_BASE) $(AVX2_FLAGS) $(OMP_FLAGS) tools/realtime_viewer_avx2_prules.cpp src/species.cpp -o tools/realtime_viewer_avx2_prules $(LIB_DIRS) $(TOOLS_LDFLAGS) $(OMP_LIBS)
	$(CXX) $(CXXFLAGS_BASE) $(AVX2_FLAGS) $(OMP_FLAGS) tools/realtime_viewer_terrain.cpp src/species.cpp -o tools/realtime_viewer_terrain $(LIB_DIRS) $(TOOLS_LDFLAGS) $(OMP_LIBS)