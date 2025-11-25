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

static inline std::size_t idx3(std::size_t x, std::size_t y, std::size_t z, std::size_t W, std::size_t H) {
    return z * (W * H) + y * W + x;
}

__global__ void step_kernel(const uint8_t *__restrict__ cur, uint8_t *__restrict__ next, int W, int H, int D) {
    // 3D block/grid: blockDim.x => x, blockDim.y => y, blockIdx.z => z slice
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    if (x >= W || y >= H || z >= D) return;

    int plane = W * H;
    int id = z * plane + y * W + x;

    int x_prev = (x == 0) ? (W - 1) : (x - 1);
    int x_next = (x + 1 == W) ? 0 : (x + 1);
    int y_prev = (y == 0) ? (H - 1) : (y - 1);
    int y_next = (y + 1 == H) ? 0 : (y + 1);
    int z_prev = (z == 0) ? (D - 1) : (z - 1);
    int z_next = (z + 1 == D) ? 0 : (z + 1);

    int live = 0;
    // iterate three z layers
    int zzs[3] = {z_prev, z, z_next};
    int yys[3] = {y_prev, y, y_next};
    for (int iz = 0; iz < 3; ++iz) {
        int zz = zzs[iz];
        int zbase = zz * plane;
        for (int iy = 0; iy < 3; ++iy) {
            int yy = yys[iy];
            int rowbase = zbase + yy * W;
            // accumulate three contiguous x neighbors for better memory pattern
            live += cur[rowbase + x_prev];
            live += cur[rowbase + x];
            live += cur[rowbase + x_next];
        }
    }
    // subtract center cell counted once
    live -= cur[id];

    uint8_t alive = cur[id];
    uint8_t out = 0;
    if (alive) out = (live == 4 || live == 5) ? 1 : 0;
    else out = (live == 5) ? 1 : 0;
    next[id] = out;
}

void run_3d_cuda_single_species_impl(const SimulationOptions3D &options) {
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t D = options.depth;
    const std::size_t N = W * H * D;
    if (W == 0 || H == 0 || D == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }

    std::vector<uint8_t> host_cur(N);
    std::mt19937_64 rng(options.seed);
    std::bernoulli_distribution bd(options.initial_density);
    for (std::size_t i = 0; i < N; ++i) host_cur[i] = bd(rng) ? 1 : 0;

    uint8_t *d_cur = nullptr;
    uint8_t *d_next = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_cur, N * sizeof(uint8_t));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_cur failed: " << cudaGetErrorString(err) << std::endl; run_3d_cpu_single_species(options); return; }
    err = cudaMalloc(&d_next, N * sizeof(uint8_t));
    if (err != cudaSuccess) { std::cerr << "cudaMalloc d_next failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_cur); run_3d_cpu_single_species(options); return; }

    err = cudaMemcpy(d_cur, host_cur.data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl; cudaFree(d_cur); cudaFree(d_next); run_3d_cpu_single_species(options); return; }

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;

    // choose a 2D thread block for x/y and grid.z for depth slices
    dim3 block(16, 8, 1);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + 0));

    const char *outdir = "frames";
    mkdir(outdir, 0755);

    std::string sdir = "frames_3d";
    mkdir(sdir.c_str(), 0755);

    bool skip_output = (std::getenv("SKIP_OUTPUT") != nullptr);
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        step_kernel<<<grid, block>>>(d_cur, d_next, (int)W, (int)H, (int)D);
        err = cudaGetLastError();
        if (err != cudaSuccess) { std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl; break; }
        // ensure kernel finished before swapping / next iteration (and for correct timing)
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl; break; }
        // swap
        std::swap(d_cur, d_next);

        if (!skip_output && output_every > 0 && (iter % output_every == 0 || iter + 1 == max_iter)) {
            // copy back to host for output
            err = cudaMemcpy(host_cur.data(), d_cur, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl; break; }
            // write projection and slices
            std::vector<uint8_t> proj(W * H);
            for (std::size_t zz = 0; zz < D; ++zz) {
                for (std::size_t i = 0; i < W * H; ++i) {
                    proj[i] = proj[i] || host_cur[zz * W * H + i];
                }
            }
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            // write_ppm implementation (simple)
            FILE *f = std::fopen(path, "wb");
            if (f) {
                std::fprintf(f, "P6\n%zu %zu\n255\n", W, H);
                for (std::size_t i = 0; i < W * H; ++i) {
                    unsigned char px = proj[i] ? 255 : 0;
                    unsigned char rgb[3] = {px, px, px};
                    std::fwrite(rgb, 1, 3, f);
                }
                std::fclose(f);
            }
            // frames_3d directory was created once before the loop
            for (std::size_t zz = 0; zz < D; ++zz) {
                char spath[256];
                std::snprintf(spath, sizeof(spath), "%s/frame_%04zu_slice_%03zu.ppm", sdir.c_str(), iter, zz);
                FILE *sf = std::fopen(spath, "wb");
                if (!sf) continue;
                std::fprintf(sf, "P6\n%zu %zu\n255\n", W, H);
                for (std::size_t i = 0; i < W * H; ++i) {
                    unsigned char px = host_cur[zz * W * H + i] ? 255 : 0;
                    unsigned char rgb[3] = {px, px, px};
                    std::fwrite(rgb, 1, 3, sf);
                }
                std::fclose(sf);
            }
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "3D CUDA run time: " << dt.count() << " s" << std::endl;

    if (d_cur) cudaFree(d_cur);
    if (d_next) cudaFree(d_next);
}
