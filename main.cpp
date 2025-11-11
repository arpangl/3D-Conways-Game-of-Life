#include "life_common.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace gol3d;

void print_usage() {
    std::cout << "Usage: life <mode> [options]\n"
              << "Modes:\n"
              << "  single       Run single-threaded 3D simulation\n"
              << "  openmp       Run OpenMP accelerated simulation\n"
              << "  avx          Run AVX/AVX2 accelerated simulation\n"
              << "  avx512       Run AVX-512 accelerated simulation\n"
              << "  cuda         Run CUDA accelerated simulation (requires USE_CUDA build)\n"
              << "  visualize    ASCII visualization of a 3D slice\n"
              << "  visualize3d  OpenGL visualization with fixed 3D view (requires USE_GLFW)\n"
              << "  benchmark    Benchmark one or more backends\n"
              << "  2d           Run classic 2D Conway's Game of Life\n"
              << "\nGeneral options:\n"
              << "  --width <n>        Grid width (default 32 for 3D, 128 for 2D)\n"
              << "  --height <n>       Grid height\n"
              << "  --depth <n>        Grid depth (3D only)\n"
              << "  --steps <n>        Simulation steps\n"
              << "  --density <p>      Initial alive probability (0..1)\n"
              << "  --seed <n>         RNG seed\n"
              << "  --wrap             Enable toroidal wrapping\n"
              << "  --no-wrap          Disable wrapping (default)\n"
              << "\nVisualization options:\n"
              << "  --visual-steps <n>   Number of frames to display (default cfg.steps)\n"
              << "  --visual-delay <s>   Delay between frames in seconds (default 0.15)\n"
              << "\nBenchmark options:\n"
              << "  --backend <name>     Backend to benchmark (repeatable, default all)\n"
              << "  --warmup <n>         Warmup steps (default 4)\n"
              << "  --measure <n>        Steps per measurement (default 64)\n"
              << "  --reps <n>           Repetitions (default 5)\n"
              << "  --print-intermediate Print per-run stats\n";
}

Backend parse_backend(const std::string& token) {
    static const std::unordered_map<std::string, Backend> table = {
        {"single", Backend::Single},
        {"openmp", Backend::OpenMP},
        {"avx", Backend::AVX},
        {"avx512", Backend::AVX512},
        {"cuda", Backend::CUDA},
    };
    const auto it = table.find(token);
    if (it == table.end()) {
        throw std::invalid_argument("Unknown backend: " + token);
    }
    return it->second;
}

void simulate_backend(Backend backend, LifeConfig3D cfg) {
    auto* stepper = select_step_function(backend);
    if (!stepper) {
        throw std::runtime_error("Unsupported backend requested");
    }

    const auto totalCells = cfg.width * cfg.height * cfg.depth;
    Grid3D current(totalCells, 0);
    Grid3D next(totalCells, 0);
    randomize(current, cfg);

    const auto beginAlive = alive_count_3d(current);
    const auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t step = 0; step < cfg.steps; ++step) {
        stepper(current, next, cfg);
        current.swap(next);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Backend " << backend_to_string(backend)
              << " completed " << cfg.steps << " steps of "
              << totalCells << " cells in "
              << std::fixed << std::setprecision(3) << elapsed << " ms.\n"
              << "Alive cells before: " << beginAlive
              << ", after: " << alive_count_3d(current) << '\n';
}

void simulate_2d(LifeConfig2D cfg) {
    const auto totalCells = cfg.width * cfg.height;
    Grid2D current(totalCells, 0);
    Grid2D next(totalCells, 0);
    randomize(current, cfg);

    const auto beginAlive = alive_count_2d(current);
    const auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t step = 0; step < cfg.steps; ++step) {
        step_2d_single(current, next, cfg);
        current.swap(next);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "2D simulation completed " << cfg.steps << " steps of "
              << totalCells << " cells in "
              << std::fixed << std::setprecision(3) << elapsed << " ms.\n"
              << "Alive cells before: " << beginAlive
              << ", after: " << alive_count_2d(current) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    using namespace gol3d;

    if (argc < 2) {
        print_usage();
        return EXIT_FAILURE;
    }

    std::string mode = argv[1];
    LifeConfig3D cfg3d;
    LifeConfig2D cfg2d;
    BenchmarkConfig benchCfg;
    double visualDelay = 0.15;
    std::size_t visualSteps = 0;
    std::vector<Backend> benchmarkTargets;
    Backend visualBackend = Backend::Single;

    auto require_value = [&](int& index, const char* name) -> const char* {
        if (index + 1 >= argc) {
            throw std::invalid_argument(std::string("Missing value for ") + name);
        }
        return argv[++index];
    };

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--width") {
            const auto value = std::stoul(require_value(i, "--width"));
            cfg3d.width = value;
            cfg2d.width = value;
        } else if (arg == "--height") {
            const auto value = std::stoul(require_value(i, "--height"));
            cfg3d.height = value;
            cfg2d.height = value;
        } else if (arg == "--depth") {
            cfg3d.depth = std::stoul(require_value(i, "--depth"));
        } else if (arg == "--steps") {
            const auto value = std::stoul(require_value(i, "--steps"));
            cfg3d.steps = value;
            cfg2d.steps = value;
            if (mode == "visualize") {
                visualSteps = value;
            }
        } else if (arg == "--density") {
            const auto value = std::stod(require_value(i, "--density"));
            cfg3d.initialAliveProbability = value;
            cfg2d.initialAliveProbability = value;
        } else if (arg == "--seed") {
            const auto value = static_cast<std::uint32_t>(std::stoul(require_value(i, "--seed")));
            cfg3d.seed = value;
            cfg2d.seed = value;
        } else if (arg == "--wrap") {
            cfg3d.wrap = true;
            cfg2d.wrap = true;
        } else if (arg == "--no-wrap") {
            cfg3d.wrap = false;
            cfg2d.wrap = false;
        } else if (arg == "--visual-steps") {
            visualSteps = std::stoul(require_value(i, "--visual-steps"));
        } else if (arg == "--visual-delay") {
            visualDelay = std::stod(require_value(i, "--visual-delay"));
        } else if (arg == "--backend") {
            const auto backend = parse_backend(require_value(i, "--backend"));
            if (mode == "benchmark") {
                benchmarkTargets.push_back(backend);
            } else if (mode == "visualize" || mode == "visualize3d") {
                visualBackend = backend;
            } else {
                throw std::invalid_argument("--backend is only valid with benchmark or visualize modes");
            }
        } else if (arg == "--warmup") {
            benchCfg.warmupSteps = std::stoul(require_value(i, "--warmup"));
        } else if (arg == "--measure") {
            benchCfg.measuredSteps = std::stoul(require_value(i, "--measure"));
        } else if (arg == "--reps") {
            benchCfg.repetitions = std::stoul(require_value(i, "--reps"));
        } else if (arg == "--print-intermediate") {
            benchCfg.printIntermediate = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return EXIT_SUCCESS;
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    try {
        if (mode == "single") {
            simulate_backend(Backend::Single, cfg3d);
        } else if (mode == "openmp") {
            simulate_backend(Backend::OpenMP, cfg3d);
        } else if (mode == "avx") {
            simulate_backend(Backend::AVX, cfg3d);
        } else if (mode == "avx512") {
            simulate_backend(Backend::AVX512, cfg3d);
        } else if (mode == "cuda") {
            simulate_backend(Backend::CUDA, cfg3d);
        } else if (mode == "visualize") {
            if (visualSteps == 0) {
                visualSteps = cfg3d.steps;
            }
            auto* stepper = select_step_function(visualBackend);
            if (!stepper) {
                std::cerr << "Warning: visualize backend not available; falling back to single.\n";
                visualBackend = Backend::Single;
                stepper = &step_single;
            }
            run_visualization(cfg3d, stepper, visualBackend, visualSteps, visualDelay);
        } else if (mode == "visualize3d") {
            if (visualSteps == 0) {
                visualSteps = cfg3d.steps;
            }
            auto* stepper = select_step_function(visualBackend);
            if (!stepper) {
                std::cerr << "Warning: visualize backend not available; falling back to single.\n";
                visualBackend = Backend::Single;
                stepper = &step_single;
            }
            run_visualization_gl(cfg3d, stepper, visualBackend, visualSteps, visualDelay);
        } else if (mode == "benchmark") {
            if (benchmarkTargets.empty()) {
                benchmarkTargets = {Backend::Single, Backend::OpenMP, Backend::AVX, Backend::AVX512, Backend::CUDA};
            }
            for (const auto backend : benchmarkTargets) {
                run_benchmark(cfg3d, benchCfg, backend, std::cout);
            }
        } else if (mode == "2d") {
            simulate_2d(cfg2d);
        } else {
            std::cerr << "Unknown mode: " << mode << "\n\n";
            print_usage();
            return EXIT_FAILURE;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
