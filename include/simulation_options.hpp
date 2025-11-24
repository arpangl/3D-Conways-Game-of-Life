#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

enum class Backend {
    CPU,
    AVX2,
    OpenMP,
    CUDA
};

enum class MultiSpeciesMode {
    FoodChain,
    Symbiosis
};

struct CommonOptions {
    std::size_t iterations = 1000;
    Backend backend = Backend::CPU;
    int species = 1;
    MultiSpeciesMode multi_species_mode = MultiSpeciesMode::FoodChain;
    int parallel_core = 1;           // OpenMP specific
    int cuda_device = 0;             // CUDA specific
    int cuda_streams = 1;            // CUDA specific
    std::size_t avx2_tile = 64;      // AVX2 specific
    std::size_t avx2_alignment = 32; // AVX2 specific
    std::uint64_t seed = 1337;       // RNG seed (default)
    std::size_t output_every = 1;    // how often to write frames
};

struct SimulationOptions2D : public CommonOptions {
    std::size_t width = 256;
    std::size_t height = 256;
};

struct SimulationOptions3D : public CommonOptions {
    std::size_t width = 128;
    std::size_t height = 128;
    std::size_t depth = 128;
    double initial_density = 0.15; // probability [0..1] a cell starts alive
};

struct VisualizationOptions2D {
    std::size_t width;
    std::size_t height;
    int species;
};

struct VisualizationOptions3D {
    std::size_t width;
    std::size_t height;
    std::size_t depth;
    int species;
};

std::string backend_to_string(Backend backend);
