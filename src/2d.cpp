#include "life_common.h"

#include <stdexcept>

namespace gol3d {

namespace {

inline std::size_t wrap_index(std::size_t coord, std::size_t max) {
    return (coord + max) % max;
}

bool in_bounds(std::size_t coord, std::size_t max) {
    return coord < max;
}

std::uint8_t apply_rule_2d(std::uint8_t current, int neighbors) {
    if (current) {
        return (neighbors == 2 || neighbors == 3) ? 1U : 0U;
    }
    return (neighbors == 3) ? 1U : 0U;
}

}  // namespace

void step_2d_single(const Grid2D& current, Grid2D& next, const LifeConfig2D& cfg) {
    if (current.size() != cfg.width * cfg.height) {
        throw std::invalid_argument("current grid size mismatch (2D)");
    }
    if (next.size() != current.size()) {
        next.resize(current.size());
    }

    for (std::size_t y = 0; y < cfg.height; ++y) {
        for (std::size_t x = 0; x < cfg.width; ++x) {
            int neighbors = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    std::size_t nx;
                    std::size_t ny;
                    if (cfg.wrap) {
                        nx = wrap_index(static_cast<std::size_t>(static_cast<std::ptrdiff_t>(x) + dx), cfg.width);
                        ny = wrap_index(static_cast<std::size_t>(static_cast<std::ptrdiff_t>(y) + dy), cfg.height);
                    } else {
                        const auto sx = static_cast<std::ptrdiff_t>(x) + dx;
                        const auto sy = static_cast<std::ptrdiff_t>(y) + dy;
                        if (sx < 0 || sy < 0) {
                            continue;
                        }
                        nx = static_cast<std::size_t>(sx);
                        ny = static_cast<std::size_t>(sy);
                        if (!in_bounds(nx, cfg.width) || !in_bounds(ny, cfg.height)) {
                            continue;
                        }
                    }
                    neighbors += current[index_2d(cfg, nx, ny)];
                }
            }

            const auto idx = index_2d(cfg, x, y);
            next[idx] = apply_rule_2d(current[idx], neighbors);
        }
    }
}

}  // namespace gol3d

