#include "life_common.h"

#include <algorithm>
#include <stdexcept>

namespace gol3d {

namespace {

inline std::size_t wrap_index(std::size_t coord, std::size_t max) {
    return (coord + max) % max;
}

bool in_bounds(std::size_t coord, std::size_t max) {
    return coord < max;
}

std::uint8_t apply_rule_3d(std::uint8_t current, int neighbors) {
    if (current) {
        return (neighbors == 4 || neighbors == 5) ? 1U : 0U;
    }
    return (neighbors == 5) ? 1U : 0U;
}

}  // namespace

void step_single(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg) {
    if (current.size() != cfg.width * cfg.height * cfg.depth) {
        throw std::invalid_argument("current grid size mismatch");
    }
    if (next.size() != current.size()) {
        next.resize(current.size());
    }

    for (std::size_t z = 0; z < cfg.depth; ++z) {
        for (std::size_t y = 0; y < cfg.height; ++y) {
            for (std::size_t x = 0; x < cfg.width; ++x) {
                int neighbors = 0;
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0 && dz == 0) {
                                continue;
                            }
                            std::size_t nx;
                            std::size_t ny;
                            std::size_t nz;

                            if (cfg.wrap) {
                                nx = wrap_index(static_cast<std::size_t>(static_cast<std::ptrdiff_t>(x) + dx), cfg.width);
                                ny = wrap_index(static_cast<std::size_t>(static_cast<std::ptrdiff_t>(y) + dy), cfg.height);
                                nz = wrap_index(static_cast<std::size_t>(static_cast<std::ptrdiff_t>(z) + dz), cfg.depth);
                            } else {
                                const auto sx = static_cast<std::ptrdiff_t>(x) + dx;
                                const auto sy = static_cast<std::ptrdiff_t>(y) + dy;
                                const auto sz = static_cast<std::ptrdiff_t>(z) + dz;
                                if (sx < 0 || sy < 0 || sz < 0) {
                                    continue;
                                }
                                nx = static_cast<std::size_t>(sx);
                                ny = static_cast<std::size_t>(sy);
                                nz = static_cast<std::size_t>(sz);
                                if (!in_bounds(nx, cfg.width) || !in_bounds(ny, cfg.height) || !in_bounds(nz, cfg.depth)) {
                                    continue;
                                }
                            }

                            neighbors += current[index_3d(cfg, nx, ny, nz)];
                        }
                    }
                }

                const auto idx = index_3d(cfg, x, y, z);
                next[idx] = apply_rule_3d(current[idx], neighbors);
            }
        }
    }
}

}  // namespace gol3d
