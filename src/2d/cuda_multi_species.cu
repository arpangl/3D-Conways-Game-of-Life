#include "dispatcher_2d.hpp"
#include <iostream>

#ifndef CUDA_STUB
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <array>
#include <random>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <chrono>

static inline std::size_t idx(std::size_t x, std::size_t y, std::size_t width) {
    return y * width + x;
}

// limit species to a small compile-time max to keep per-thread arrays on registers
#define MAX_SPECIES 8

__global__ void gol_multi_step_kernel(const uint8_t *cur, uint8_t *next, std::size_t W, std::size_t H, int S) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x = bx * blockDim.x + tx;
    const int y = by * blockDim.y + ty;

    const int sWidth = blockDim.x + 2;
    const int sHeight = blockDim.y + 2;
    extern __shared__ uint8_t s[];

    // each thread writes its 3x3 neighborhood into shared memory
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            int gx = bx * blockDim.x + tx + ox;
            int gy = by * blockDim.y + ty + oy;
            int gxr = (gx % (int)W + (int)W) % (int)W;
            int gyr = (gy % (int)H + (int)H) % (int)H;
            int sx = tx + 1 + ox;
            int sy = ty + 1 + oy;
            s[sy * sWidth + sx] = cur[gyr * W + gxr];
        }
    }

    __syncthreads();

    if (x >= (int)W || y >= (int)H) return;

    int counts[MAX_SPECIES+1];
    for (int i=0;i<=MAX_SPECIES;++i) counts[i]=0;

    int center_sx = tx + 1;
    int center_sy = ty + 1;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            if (ox==0 && oy==0) continue;
            uint8_t v = s[(center_sy + oy) * sWidth + (center_sx + ox)];
            if (v <= (uint8_t)MAX_SPECIES) counts[v]++;
        }
    }

    uint8_t me = s[center_sy * sWidth + center_sx];
    uint8_t out = 0;
    if (me != 0) {
        int c = counts[me];
        out = (c == 2 || c == 3) ? me : 0;
    } else {
        int chosen = 0;
        int best_count = -1;
        for (int sidx=1; sidx<=S && sidx<=MAX_SPECIES; ++sidx) {
            if (counts[sidx] == 3) {
                if (counts[sidx] > best_count) { best_count = counts[sidx]; chosen = sidx; }
                else if (counts[sidx] == best_count && chosen != 0) {
                    if (sidx < chosen) chosen = sidx;
                }
            }
        }
        out = chosen;
    }
    next[y * W + x] = out;
}

static void write_ppm_multi(const std::string &path, const std::vector<uint8_t> &grid, std::size_t width, std::size_t height, int species) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror("fopen"); return; }
    std::fprintf(f, "P6\n%zu %zu\n255\n", width, height);
    std::vector<std::array<unsigned char,3>> pal;
    pal.push_back(std::array<unsigned char,3>{0,0,0});
    const std::vector<std::array<unsigned char,3>> base = {
        {255,0,0}, {0,255,0}, {0,0,255}, {255,255,0}, {255,0,255}, {0,255,255}, {200,100,50}, {100,200,150}
    };
    for (int s=1; s<=species; ++s) {
        pal.push_back(base[(s-1) % base.size()]);
    }
    for (std::size_t i = 0; i < width*height; ++i) {
        int v = grid[i];
        auto c = pal[ (v>=0 && v<(int)pal.size()) ? v : 0 ];
        unsigned char rgb[3] = { c[0], c[1], c[2] };
        std::fwrite(rgb,1,3,f);
    }
    std::fclose(f);
}

void run_2d_cuda_multi_species_impl(const SimulationOptions2D &options) {
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t N = W * H;
    const int S = std::max(1, options.species);
    if (W==0 || H==0) { std::cerr << "Invalid dimensions" << std::endl; return; }

    if (S > MAX_SPECIES) {
        std::cerr << "Requested species (" << S << ") > MAX_SPECIES (" << MAX_SPECIES << "); falling back to CPU." << std::endl;
        extern void run_2d_cpu_multi_species(const SimulationOptions2D &);
        run_2d_cpu_multi_species(options);
        return;
    }

    std::vector<uint8_t> host_cur(N);
    std::mt19937_64 rng(options.seed);
    std::uniform_int_distribution<int> dist(0, S);
    for (std::size_t i=0;i<N;++i) host_cur[i] = static_cast<uint8_t>(dist(rng));

    uint8_t *d_cur = nullptr;
    uint8_t *d_next = nullptr;
    cudaMalloc(&d_cur, N);
    cudaMalloc(&d_next, N);
    cudaMemcpy(d_cur, host_cur.data(), N, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y);
    std::size_t shmem = (block.x + 2) * (block.y + 2) * sizeof(uint8_t);

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    mkdir("frames", 0755);
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t iter=0; iter<max_iter; ++iter) {
        gol_multi_step_kernel<<<grid, block, shmem>>>(d_cur, d_next, W, H, S);
        cudaDeviceSynchronize();
        std::swap(d_cur, d_next);

        if (iter % output_every == 0 || iter+1==max_iter) {
            cudaMemcpy(host_cur.data(), d_cur, N, cudaMemcpyDeviceToHost);
            char path[256]; std::snprintf(path, sizeof(path), "frames/frame_%04zu.ppm", iter);
            write_ppm_multi(path, host_cur, W, H, S);
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "CUDA host-run time (including kernel launches & sync): " << dt.count() << " s" << std::endl;

    cudaFree(d_cur);
    cudaFree(d_next);
}

#else
void run_2d_cuda_multi_species_impl(const SimulationOptions2D &options) {
    (void)options;
    std::cerr << "CUDA_STUB: cuda implementation not compiled." << std::endl;
}
#endif

