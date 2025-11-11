#include "life_common.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>

namespace {

__device__ inline std::size_t device_index_3d(std::size_t width, std::size_t height, std::size_t x, std::size_t y, std::size_t z) {
    return (z * height + y) * width + x;
}

__device__ inline std::uint8_t apply_rule_3d_device(std::uint8_t current, int neighbors) {
    if (current) {
        return (neighbors == 4 || neighbors == 5) ? 1U : 0U;
    }
    return (neighbors == 5) ? 1U : 0U;
}

__global__ void life_step_kernel(const std::uint8_t* current,
                                 std::uint8_t* next,
                                 std::size_t width,
                                 std::size_t height,
                                 std::size_t depth,
                                 bool wrap) {
    const auto x = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const auto y = static_cast<std::size_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const auto z = static_cast<std::size_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= width || y >= height || z >= depth) {
        return;
    }

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

                if (wrap) {
                    auto sx = static_cast<long long>(x) + dx;
                    auto sy = static_cast<long long>(y) + dy;
                    auto sz = static_cast<long long>(z) + dz;
                    if (sx < 0) {
                        sx += static_cast<long long>(width);
                    } else if (sx >= static_cast<long long>(width)) {
                        sx -= static_cast<long long>(width);
                    }
                    if (sy < 0) {
                        sy += static_cast<long long>(height);
                    } else if (sy >= static_cast<long long>(height)) {
                        sy -= static_cast<long long>(height);
                    }
                    if (sz < 0) {
                        sz += static_cast<long long>(depth);
                    } else if (sz >= static_cast<long long>(depth)) {
                        sz -= static_cast<long long>(depth);
                    }
                    nx = static_cast<std::size_t>(sx);
                    ny = static_cast<std::size_t>(sy);
                    nz = static_cast<std::size_t>(sz);
                } else {
                    const auto sx = static_cast<long long>(x) + dx;
                    const auto sy = static_cast<long long>(y) + dy;
                    const auto sz = static_cast<long long>(z) + dz;
                    if (sx < 0 || sy < 0 || sz < 0) {
                        continue;
                    }
                    nx = static_cast<std::size_t>(sx);
                    ny = static_cast<std::size_t>(sy);
                    nz = static_cast<std::size_t>(sz);
                    if (nx >= width || ny >= height || nz >= depth) {
                        continue;
                    }
                }
                neighbors += current[device_index_3d(width, height, nx, ny, nz)];
            }
        }
    }

    const auto idx = device_index_3d(width, height, x, y, z);
    next[idx] = apply_rule_3d_device(current[idx], neighbors);
}

inline void check_cuda(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace
#endif  // USE_CUDA

namespace gol3d {

void step_cuda(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg) {
#ifdef USE_CUDA
    if (current.size() != cfg.width * cfg.height * cfg.depth) {
        throw std::invalid_argument("current grid size mismatch (CUDA)");
    }

    if (next.size() != current.size()) {
        next.resize(current.size());
    }

    const auto totalBytes = current.size() * sizeof(std::uint8_t);
    std::uint8_t* d_current = nullptr;
    std::uint8_t* d_next = nullptr;

    check_cuda(cudaMalloc(&d_current, totalBytes), "cudaMalloc d_current failed");
    check_cuda(cudaMalloc(&d_next, totalBytes), "cudaMalloc d_next failed");

    check_cuda(cudaMemcpy(d_current, current.data(), totalBytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");

    const dim3 block(8, 8, 8);
    const dim3 grid((cfg.width + block.x - 1) / block.x,
                    (cfg.height + block.y - 1) / block.y,
                    (cfg.depth + block.z - 1) / block.z);

    life_step_kernel<<<grid, block>>>(d_current, d_next, cfg.width, cfg.height, cfg.depth, cfg.wrap);
    check_cuda(cudaDeviceSynchronize(), "CUDA kernel execution failed");

    check_cuda(cudaMemcpy(next.data(), d_next, totalBytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");

    cudaFree(d_current);
    cudaFree(d_next);
#else
    static bool warned = false;
    if (!warned) {
        std::cerr << "[CUDA] Backend not built with USE_CUDA; using AVX implementation instead.\n";
        warned = true;
    }
    step_avx(current, next, cfg);
#endif
}

}  // namespace gol3d
