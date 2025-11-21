#include "dispatcher_2d.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string render_grid_to_ascii(const std::vector<uint8_t> &grid,
                                 std::size_t width,
                                 std::size_t height,
                                 const std::string &palette) {
    std::ostringstream oss;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const auto value = grid[y * width + x];
            const std::size_t palette_index = std::min<std::size_t>(value, palette.size() - 1);
            oss << palette[palette_index] << ' ';
        }
        oss << '\n';
    }
    return oss.str();
}

std::string render_grid_to_colored_ascii(const std::vector<uint8_t> &grid,
                                         std::size_t width,
                                         std::size_t height,
                                         const std::vector<std::string> &palette) {
    std::ostringstream oss;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const auto value = grid[y * width + x];
            const std::size_t palette_index = std::min<std::size_t>(value, palette.size() - 1);
            oss << palette[palette_index] << ' ';
        }
        oss << '\n';
    }
    return oss.str();
}

std::thread start_single_species_terminal_renderer(std::vector<uint8_t> &shared_grid,
                                                   std::mutex &grid_mutex,
                                                   std::atomic_bool &running,
                                                   std::size_t width,
                                                   std::size_t height,
                                                   std::chrono::milliseconds frame_delay) {
    const std::string palette = ".#"; // dead vs alive
    return std::thread([&, width, height, palette, frame_delay]() {
        while (running.load(std::memory_order_relaxed)) {
            std::vector<uint8_t> snapshot;
            {
                std::lock_guard<std::mutex> lock(grid_mutex);
                snapshot = shared_grid;
            }

            std::cout << "\033[2J\033[H"; // Clear terminal and move cursor to top-left.
            std::cout << "[single species] live preview" << std::endl;
            std::cout << render_grid_to_ascii(snapshot, width, height, palette) << std::flush;

            std::this_thread::sleep_for(frame_delay);
        }
    });
}

std::thread start_multi_species_terminal_renderer(std::vector<std::vector<uint8_t>> &species_grids,
                                                  std::mutex &grid_mutex,
                                                  std::atomic_bool &running,
                                                  std::size_t width,
                                                  std::size_t height,
                                                  std::chrono::milliseconds frame_delay) {
    const std::vector<std::string> palette = [] {
        std::vector<std::string> entries;
        entries.emplace_back(".");

        const std::vector<int> color_codes = {31, 32, 33, 34, 35, 36, 91, 92, 93, 94, 95, 96, 90, 97};
        const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        for (std::size_t i = 0; i < letters.size(); ++i) {
            const int color = color_codes[i % color_codes.size()];
            std::ostringstream entry;
            entry << "\033[" << color << "m" << letters[i] << "\033[0m";
            entries.emplace_back(entry.str());
        }

        return entries;
    }();
    return std::thread([&, width, height, palette, frame_delay]() {
        while (running.load(std::memory_order_relaxed)) {
            std::vector<std::vector<uint8_t>> snapshot;
            {
                std::lock_guard<std::mutex> lock(grid_mutex);
                snapshot = species_grids;
            }

            std::vector<uint8_t> composite(width * height, 0);
            for (std::size_t species = 0; species < snapshot.size(); ++species) {
                const auto &grid = snapshot[species];
                for (std::size_t idx = 0; idx < grid.size(); ++idx) {
                    if (grid[idx] != 0) {
                        // Later species override earlier ones for visualization clarity.
                        composite[idx] = static_cast<uint8_t>(std::min<std::size_t>(species + 1, palette.size() - 1));
                    }
                }
            }

            std::cout << "\033[2J\033[H"; // Clear terminal and move cursor to top-left.
            std::cout << "[multi-species] symbiosis preview (letters denote species)" << std::endl;
            std::cout << render_grid_to_colored_ascii(composite, width, height, palette) << std::flush;

            std::this_thread::sleep_for(frame_delay);
        }
    });
}

} // namespace

void visualize_2d_state(const VisualizationOptions2D &options) {
    if (options.width == 0 || options.height == 0) {
        std::cout << "Width and height must be non-zero to visualize." << std::endl;
        return;
    }

    const std::size_t width = options.width;
    const std::size_t height = options.height;

    constexpr auto frame_delay = std::chrono::milliseconds(100);
    std::atomic_bool running{true};
    std::mutex grid_mutex;

    if (options.species <= 1) {
        std::vector<uint8_t> grid(width * height, 0);
        auto renderer = start_single_species_terminal_renderer(grid, grid_mutex, running, width, height, frame_delay);

        for (std::size_t tick = 0; tick < 60; ++tick) {
            {
                std::lock_guard<std::mutex> lock(grid_mutex);
                for (std::size_t y = 0; y < height; ++y) {
                    for (std::size_t x = 0; x < width; ++x) {
                        grid[y * width + x] = static_cast<uint8_t>((x + y + tick) % 2);
                    }
                }
            }
            std::this_thread::sleep_for(frame_delay);
        }

        running.store(false, std::memory_order_relaxed);
        renderer.join();
    } else {
        const std::size_t species_count = static_cast<std::size_t>(options.species);
        std::vector<std::vector<uint8_t>> species_grids(species_count, std::vector<uint8_t>(width * height, 0));
        auto renderer = start_multi_species_terminal_renderer(species_grids, grid_mutex, running, width, height, frame_delay);

        for (std::size_t tick = 0; tick < 60; ++tick) {
            {
                std::lock_guard<std::mutex> lock(grid_mutex);
                for (std::size_t species = 0; species < species_count; ++species) {
                    for (std::size_t y = 0; y < height; ++y) {
                        for (std::size_t x = 0; x < width; ++x) {
                            const bool active = ((x + tick + species) % (species + 2) == 0) || ((y + species + tick) % (species + 3) == 0);
                            species_grids[species][y * width + x] = static_cast<uint8_t>(active);
                        }
                    }
                }
            }
            std::this_thread::sleep_for(frame_delay);
        }

        running.store(false, std::memory_order_relaxed);
        renderer.join();
    }

    std::cout << "Visualization demo complete." << std::endl;
}
