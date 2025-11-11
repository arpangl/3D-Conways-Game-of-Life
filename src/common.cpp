#include "life_common.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace gol3d {

std::string backend_to_string(Backend backend) {
    switch (backend) {
        case Backend::Single:
            return "single";
        case Backend::OpenMP:
            return "openmp";
        case Backend::AVX:
            return "avx";
        case Backend::AVX512:
            return "avx512";
        case Backend::CUDA:
            return "cuda";
        case Backend::Visualize:
            return "visualize";
        case Backend::Benchmark:
            return "benchmark";
        case Backend::CPU2D:
            return "2d";
    }
    return "unknown";
}

StepFunction3D select_step_function(Backend backend) {
    switch (backend) {
        case Backend::Single:
            return &step_single;
        case Backend::OpenMP:
            return &step_openmp;
        case Backend::AVX:
            return &step_avx;
        case Backend::AVX512:
            return &step_avx512;
        case Backend::CUDA:
            return &step_cuda;
        default:
            return nullptr;
    }
}

std::size_t index_3d(const LifeConfig3D& cfg, std::size_t x, std::size_t y, std::size_t z) {
    return (z * cfg.height + y) * cfg.width + x;
}

std::size_t index_2d(const LifeConfig2D& cfg, std::size_t x, std::size_t y) {
    return y * cfg.width + x;
}

void randomize(Grid3D& grid, const LifeConfig3D& cfg) {
    std::mt19937 rng(cfg.seed);
    std::bernoulli_distribution dist(cfg.initialAliveProbability);
    for (auto& cell : grid) {
        cell = dist(rng) ? 1 : 0;
    }
}

void randomize(Grid2D& grid, const LifeConfig2D& cfg) {
    std::mt19937 rng(cfg.seed);
    std::bernoulli_distribution dist(cfg.initialAliveProbability);
    for (auto& cell : grid) {
        cell = dist(rng) ? 1 : 0;
    }
}

std::size_t alive_count_3d(const Grid3D& grid) {
    return static_cast<std::size_t>(std::count(grid.begin(), grid.end(), static_cast<std::uint8_t>(1)));
}

std::size_t alive_count_2d(const Grid2D& grid) {
    return static_cast<std::size_t>(std::count(grid.begin(), grid.end(), static_cast<std::uint8_t>(1)));
}

void print_grid_slice(const Grid3D& grid, const LifeConfig3D& cfg, std::size_t zPlane, std::ostream& os) {
    if (zPlane >= cfg.depth) {
        throw std::out_of_range("z plane out of range");
    }
    for (std::size_t y = 0; y < cfg.height; ++y) {
        for (std::size_t x = 0; x < cfg.width; ++x) {
            const auto idx = index_3d(cfg, x, y, zPlane);
            os << (grid[idx] ? '#' : '.');
        }
        os << '\n';
    }
}

}  // namespace gol3d
