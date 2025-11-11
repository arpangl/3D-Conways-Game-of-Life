#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace gol3d {

struct LifeConfig3D {
    std::size_t width{32};
    std::size_t height{32};
    std::size_t depth{32};
    std::size_t steps{32};
    double initialAliveProbability{0.2};
    std::uint32_t seed{1337};
    bool wrap{false};
};

struct LifeConfig2D {
    std::size_t width{128};
    std::size_t height{128};
    std::size_t steps{128};
    double initialAliveProbability{0.2};
    std::uint32_t seed{42};
    bool wrap{false};
};

using Grid3D = std::vector<std::uint8_t>;
using Grid2D = std::vector<std::uint8_t>;

using StepFunction3D = void (*)(const Grid3D&, Grid3D&, const LifeConfig3D&);

enum class Backend {
    Single,
    OpenMP,
    AVX,
    AVX512,
    CUDA,
    Visualize,
    Benchmark,
    CPU2D
};

struct BenchmarkConfig {
    std::size_t warmupSteps{4};
    std::size_t measuredSteps{64};
    std::size_t repetitions{5};
    bool printIntermediate{false};
};

struct BenchmarkResult {
    Backend backend{Backend::Single};
    double avgMillis{0.0};
    double minMillis{0.0};
    double maxMillis{0.0};
    std::size_t stepsSimulated{0};
    std::size_t cellsPerStep{0};
};

std::string backend_to_string(Backend backend);

StepFunction3D select_step_function(Backend backend);

std::size_t index_3d(const LifeConfig3D& cfg, std::size_t x, std::size_t y, std::size_t z);
std::size_t index_2d(const LifeConfig2D& cfg, std::size_t x, std::size_t y);

void randomize(Grid3D& grid, const LifeConfig3D& cfg);
void randomize(Grid2D& grid, const LifeConfig2D& cfg);

std::size_t alive_count_3d(const Grid3D& grid);
std::size_t alive_count_2d(const Grid2D& grid);

void step_single(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg);
void step_openmp(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg);
void step_avx(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg);
void step_avx512(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg);
void step_cuda(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg);

void step_2d_single(const Grid2D& current, Grid2D& next, const LifeConfig2D& cfg);

void run_visualization(const LifeConfig3D& cfg,
                       StepFunction3D stepper,
                       Backend backend,
                       std::size_t totalSteps,
                       double delaySeconds);
void run_visualization_gl(const LifeConfig3D& cfg,
                          StepFunction3D stepper,
                          Backend backend,
                          std::size_t totalSteps,
                          double secondsPerStep);

BenchmarkResult run_benchmark(const LifeConfig3D& cfg,
                              BenchmarkConfig benchCfg,
                              Backend backend,
                              std::ostream& os);

void print_grid_slice(const Grid3D& grid, const LifeConfig3D& cfg, std::size_t zPlane, std::ostream& os);

}  // namespace gol3d
