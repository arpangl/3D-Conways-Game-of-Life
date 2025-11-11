#include "life_common.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gol3d {

namespace {

void perform_steps(StepFunction3D stepper,
                   Grid3D& current,
                   Grid3D& next,
                   const LifeConfig3D& cfg,
                   std::size_t steps) {
    for (std::size_t i = 0; i < steps; ++i) {
        stepper(current, next, cfg);
        current.swap(next);
    }
}

}  // namespace

BenchmarkResult run_benchmark(const LifeConfig3D& cfg,
                              BenchmarkConfig benchCfg,
                              Backend backend,
                              std::ostream& os) {
    auto* stepper = select_step_function(backend);
    if (!stepper) {
        throw std::invalid_argument("Benchmark: unsupported backend");
    }

    const auto totalCells = cfg.width * cfg.height * cfg.depth;
    Grid3D current(totalCells, 0);
    Grid3D next(totalCells, 0);
    randomize(current, cfg);

    if (benchCfg.warmupSteps > 0) {
        perform_steps(stepper, current, next, cfg, benchCfg.warmupSteps);
    }

    std::vector<double> durations;
    durations.reserve(benchCfg.repetitions);

    for (std::size_t rep = 0; rep < benchCfg.repetitions; ++rep) {
        auto beforeCount = alive_count_3d(current);
        auto start = std::chrono::high_resolution_clock::now();
        perform_steps(stepper, current, next, cfg, benchCfg.measuredSteps);
        auto end = std::chrono::high_resolution_clock::now();

        auto millis = std::chrono::duration<double, std::milli>(end - start).count();
        durations.push_back(millis);

        if (benchCfg.printIntermediate) {
            os << "Backend=" << backend_to_string(backend) << " run " << rep + 1
               << "/" << benchCfg.repetitions << ": "
               << std::fixed << std::setprecision(3) << millis << " ms, "
               << "alive before=" << beforeCount
               << ", after=" << alive_count_3d(current) << '\n';
        }

        // Re-randomize between repetitions to avoid converging to a fixed point.
        randomize(current, cfg);
    }

    const auto minmax = std::minmax_element(durations.begin(), durations.end());
    const double total = std::accumulate(durations.begin(), durations.end(), 0.0);

    BenchmarkResult result;
    result.backend = backend;
    result.minMillis = durations.empty() ? 0.0 : *minmax.first;
    result.maxMillis = durations.empty() ? 0.0 : *minmax.second;
    result.avgMillis = durations.empty() ? 0.0 : total / static_cast<double>(durations.size());
    result.stepsSimulated = benchCfg.measuredSteps;
    result.cellsPerStep = totalCells;

    if (!durations.empty()) {
        os << backend_to_string(backend) << ": "
           << std::fixed << std::setprecision(3) << result.avgMillis
           << " ms avg (" << result.minMillis << " min / "
           << result.maxMillis << " max) over "
           << benchCfg.measuredSteps << " steps of "
           << totalCells << " cells.\n";
    }

    return result;
}

}  // namespace gol3d
