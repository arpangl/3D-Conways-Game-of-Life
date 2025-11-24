#include "dispatcher_3d.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel: compute one Game of Life step for a single species layer (multi-species variant)
__global__ void step_kernel_multi(const uint8_t *cur, uint8_t *next, std::size_t W, std::size_t H, std::size_t D) {
    std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t N = W * H * D;
    if (id >= N) return;
    std::size_t x = id % W;
    std::size_t y = (id / W) % H;
    std::size_t z = id / (W * H);
    int live = 0;
    for (int dz = -1; dz <= 1; ++dz) {
        std::size_t zz = (z + D + dz) % D;
        for (int dy = -1; dy <= 1; ++dy) {
            std::size_t yy = (y + H + dy) % H;
            for (int dx = -1; dx <= 1; ++dx) {
                std::size_t xx = (x + W + dx) % W;
                if (dx == 0 && dy == 0 && dz == 0) continue;
                std::size_t nidx = zz * (W * H) + yy * W + xx;
                live += cur[nidx];
            }
        }
    }
    uint8_t alive = cur[id];
    uint8_t out = 0;
    if (alive) out = (live == 4 || live == 5) ? 1 : 0;
    else out = (live == 5) ? 1 : 0;
    next[id] = out;
}

void run_3d_cuda_multi_species_impl(const SimulationOptions3D &options) {
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t D = options.depth;
    const std::size_t S = std::max<int>(1, options.species);
    const std::size_t N = W * H * D;
    if (W == 0 || H == 0 || D == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }

    std::vector<uint8_t> host_buf(S * N);
    std::mt19937_64 rng(options.seed);
    std::bernoulli_distribution bd(options.initial_density);
    for (std::size_t i = 0; i < S * N; ++i) host_buf[i] = bd(rng) ? 1 : 0;

    uint8_t *d_buf = nullptr;
    uint8_t *d_next = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_buf, S * N * sizeof(uint8_t));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_buf failed: " << cudaGetErrorString(err) << std::endl; run_3d_cpu_multi_species(options); return; }
    err = cudaMalloc(&d_next, N * sizeof(uint8_t));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_next failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_buf); run_3d_cpu_multi_species(options); return; }

    err = cudaMemcpy(d_buf, host_buf.data(), S * N * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_buf); cudaFree(d_next); run_3d_cpu_multi_species(options); return; }

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);

    const char *outdir = "frames";
    mkdir(outdir, 0755);

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t s = 0; s < S; ++s) {
            uint8_t *d_cur_s = d_buf + s * N;
            // compute into d_next
            step_kernel_multi<<<blocks, threads>>>(d_cur_s, d_next, W, H, D);
            err = cudaGetLastError();
            if (err != cudaSuccess) { std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl; break; }
            // copy d_next back into d_buf (overwrite species layer)
            err = cudaMemcpy(d_cur_s, d_next, N * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) { std::cerr << "cudaMemcpy D2D failed: " << cudaGetErrorString(err) << std::endl; break; }
        }

        if (output_every > 0 && (iter % output_every == 0 || iter + 1 == max_iter)) {
            // copy whole buffer back
            err = cudaMemcpy(host_buf.data(), d_buf, S * N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl; break; }
            // write projection and slices
            std::vector<uint8_t> proj(W * H);
            for (std::size_t s = 0; s < S; ++s) {
                for (std::size_t zz = 0; zz < D; ++zz) {
                    for (std::size_t i = 0; i < W * H; ++i) {
                        proj[i] = proj[i] || host_buf[s * N + zz * W * H + i];
                    }
                }
            }
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            FILE *f = std::fopen(path, "wb");
            if (f) {
                std::fprintf(f, "P6\\n%zu %zu\\n255\\n", W, H);
                for (std::size_t i = 0; i < W * H; ++i) {
                    unsigned char px = proj[i] ? 255 : 0;
                    unsigned char rgb[3] = {px, px, px};
                    std::fwrite(rgb, 1, 3, f);
                }
                std::fclose(f);
            }
            std::string sdir = "frames_3d";
            int mk = mkdir(sdir.c_str(), 0755);
            if (mk != 0) { /* ignore */ }
            for (std::size_t zz = 0; zz < D; ++zz) {
                char spath[256];
                std::snprintf(spath, sizeof(spath), "%s/frame_%04zu_slice_%03zu.ppm", sdir.c_str(), iter, zz);
                FILE *sf = std::fopen(spath, "wb");
                if (!sf) continue;
                std::fprintf(sf, "P6\\n%zu %zu\\n255\\n", W, H);
                for (std::size_t i = 0; i < W * H; ++i) {
                    unsigned char px = 0;
                    for (std::size_t s = 0; s < S; ++s) px = px || host_buf[s * N + zz * W * H + i];
                    unsigned char rgb[3] = {px ? 255 : 0, px ? 255 : 0, px ? 255 : 0};
                    std::fwrite(rgb, 1, 3, sf);
                }
                std::fclose(sf);
            }
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "3D CUDA multi-species run time: " << dt.count() << " s" << std::endl;

    if (d_buf) cudaFree(d_buf);
    if (d_next) cudaFree(d_next);
}
