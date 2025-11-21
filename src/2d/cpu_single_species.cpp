#include "dispatcher_2d.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

void run_2d_cpu_single_species(const SimulationOptions2D &options) {
    const auto width = options.width;
    const auto height = options.height;

    if (width == 0 || height == 0) {
        std::cout << "Width and height must be greater than zero for a valid simulation." << std::endl;
        return;
    }

    std::vector<uint8_t> current(width * height, 0);
    std::vector<uint8_t> next(width * height, 0);
    std::mutex grid_mutex;
    std::atomic_bool running{true};
    std::thread renderer;

    auto index = [width](std::size_t x, std::size_t y) {
        return y * width + x;
    };

    // Deterministic initialization to keep runs reproducible.
    std::mt19937 rng(options.seed);
    std::bernoulli_distribution dist(0.5);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            current[index(x, y)] = static_cast<uint8_t>(dist(rng));
        }
    }

    auto alive_neighbors = [&](std::size_t x, std::size_t y) {
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const std::size_t nx = (x + width + dx) % width;
                const std::size_t ny = (y + height + dy) % height;
                count += current[index(nx, ny)];
            }
        }
        return count;
    };

    if (options.visualize) {
        renderer = std::thread([&]() {
            while (running.load(std::memory_order_relaxed)) {
                std::vector<uint8_t> snapshot;
                {
                    std::lock_guard<std::mutex> lock(grid_mutex);
                    snapshot = current;
                }

                std::ostringstream oss;
                for (std::size_t y = 0; y < height; ++y) {
                    for (std::size_t x = 0; x < width; ++x) {
                        oss << (snapshot[y * width + x] ? '#' : '.') << ' ';
                    }
                    oss << '\n';
                }

                std::cout << "\033[2J\033[H";
                std::cout << "[single species] live preview" << std::endl;
                std::cout << oss.str() << std::flush;

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }

    const auto start_time = std::chrono::steady_clock::now();
    for (std::size_t iter = 0; iter < options.iterations; ++iter) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const int neighbors = alive_neighbors(x, y);
                const bool alive = current[index(x, y)] != 0;
                next[index(x, y)] = static_cast<uint8_t>((alive && (neighbors == 2 || neighbors == 3)) || (!alive && neighbors == 3));
            }
        }
        {
            std::lock_guard<std::mutex> lock(grid_mutex);
            current.swap(next);
        }
    }
    const auto end_time = std::chrono::steady_clock::now();

    running.store(false, std::memory_order_relaxed);
    if (renderer.joinable()) {
        renderer.join();
    }

    std::size_t alive_cells = 0;
    for (const auto cell : current) {
        alive_cells += cell != 0;
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "2D CPU single-species simulation complete." << std::endl;
    std::cout << "Grid: " << width << "x" << height << ", Iterations: " << options.iterations << std::endl;
    std::cout << "Alive cells after final iteration: " << alive_cells << std::endl;
    std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;
}
