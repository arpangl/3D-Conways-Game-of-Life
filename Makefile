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
