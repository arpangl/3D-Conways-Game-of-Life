#include "dispatcher_2d.hpp"
#include <iostream>

#ifdef CUDA_STUB
void run_2d_cuda_single_species(const SimulationOptions2D &options) {
    (void)options;
    std::cerr << "CUDA not available: compiled without nvcc.\n";
}
#else
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <random>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <chrono>

static inline std::size_t idx(std::size_t x, std::size_t y, std::size_t width) {
    return y * width + x;
}

// Shared-memory tiled kernel: each block loads a tile (blockDim + 2 halo) into shared memory.
// For simplicity and correctness we let each thread load the 3x3 neighborhood into shared
// memory (duplicated writes) then synchronize and compute using the shared tile.
__global__ void gol_step_kernel(const uint8_t *cur, uint8_t *next, std::size_t W, std::size_t H) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x = bx * blockDim.x + tx;
    const int y = by * blockDim.y + ty;

    // shared tile dimensions (including 1-cell halo on each side)
    const int sWidth = blockDim.x + 2;
    const int sHeight = blockDim.y + 2;
    extern __shared__ uint8_t s[];

    // Each thread writes its 3x3 neighborhood into shared memory (simple, correct).
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            int gx = bx * blockDim.x + tx + ox;
            int gy = by * blockDim.y + ty + oy;
            // wrap-around
            int gxr = (gx % (int)W + (int)W) % (int)W;
            int gyr = (gy % (int)H + (int)H) % (int)H;
            int sx = tx + 1 + ox;
            int sy = ty + 1 + oy;
            s[sy * sWidth + sx] = cur[gyr * W + gxr];
        }
    }

    __syncthreads();

    if (x >= (int)W || y >= (int)H) return;

    int live = 0;
    int center_sx = tx + 1;
    int center_sy = ty + 1;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            if (ox == 0 && oy == 0) continue;
            live += s[(center_sy + oy) * sWidth + (center_sx + ox)];
        }
    }

    uint8_t alive = s[center_sy * sWidth + center_sx];
    uint8_t out = alive ? ((live == 2 || live == 3) ? 1 : 0) : ((live == 3) ? 1 : 0);
    next[y * W + x] = out;
}

static void write_ppm(const std::string &path, const std::vector<uint8_t> &grid, std::size_t width, std::size_t height) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror("fopen"); return; }
    std::fprintf(f, "P6\n%zu %zu\n255\n", width, height);
    for (std::size_t i = 0; i < width * height; ++i) {
        unsigned char px = grid[i] ? 255 : 0;
        unsigned char rgb[3] = {px, px, px};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
}

void run_2d_cuda_single_species_impl(const SimulationOptions2D &options) {
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t N = W * H;
    if (W==0 || H==0) { std::cerr << "Invalid dimensions" << std::endl; return; }

    std::vector<uint8_t> host_cur(N);
    std::vector<uint8_t> host_next(N);
    std::mt19937_64 rng(options.seed);
    std::uniform_int_distribution<int> bit(0,1);
    for (std::size_t i=0;i<N;++i) host_cur[i] = static_cast<uint8_t>(bit(rng));

    uint8_t *d_cur = nullptr;
    uint8_t *d_next = nullptr;
    cudaMalloc(&d_cur, N);
    cudaMalloc(&d_next, N);
    cudaMemcpy(d_cur, host_cur.data(), N, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y);

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    mkdir("frames", 0755);
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t iter=0; iter<max_iter; ++iter) {
        // dynamic shared memory size: (block.x+2)*(block.y+2) bytes
        std::size_t shmem = (block.x + 2) * (block.y + 2) * sizeof(uint8_t);
        gol_step_kernel<<<grid, block, shmem>>>(d_cur, d_next, W, H);
        cudaDeviceSynchronize();
        // swap
        std::swap(d_cur, d_next);

        if (iter % output_every == 0 || iter+1==max_iter) {
            cudaMemcpy(host_cur.data(), d_cur, N, cudaMemcpyDeviceToHost);
            char path[256]; std::snprintf(path, sizeof(path), "frames/frame_%04zu.ppm", iter);
            write_ppm(path, host_cur, W, H);
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "CUDA host-run time (including kernel launches & sync): " << dt.count() << " s" << std::endl;

    cudaFree(d_cur);
    cudaFree(d_next);
}
#endif
