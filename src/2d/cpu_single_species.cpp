#include "dispatcher_2d.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <chrono>

static void write_ppm(const std::string &path, const std::vector<uint8_t> &grid, std::size_t width, std::size_t height) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::perror("fopen");
        return;
    }
    std::fprintf(f, "P6\n%zu %zu\n255\n", width, height);
    for (std::size_t i = 0; i < width * height; ++i) {
        unsigned char px = grid[i] ? 255 : 0;
        unsigned char rgb[3] = {px, px, px};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
}

static inline std::size_t idx(std::size_t x, std::size_t y, std::size_t width) {
    return y * width + x;
}

void run_2d_cpu_single_species(const SimulationOptions2D &options) {
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t N = W * H;
    if (W == 0 || H == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }

    std::vector<uint8_t> cur(N);
    std::vector<uint8_t> next(N);

    std::mt19937_64 rng(options.seed);
    std::uniform_int_distribution<int> bit(0, 1);
    for (std::size_t i = 0; i < N; ++i) cur[i] = static_cast<uint8_t>(bit(rng));

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1; // write every generation
    bool skip_output = (std::getenv("SKIP_OUTPUT") != nullptr);

    const char *outdir = "frames";
    mkdir(outdir, 0755);

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t y = 0; y < H; ++y) {
            for (std::size_t x = 0; x < W; ++x) {
                int live_neighbors = 0;
                for (int oy = -1; oy <= 1; ++oy) {
                    std::size_t yy = (y + H + oy) % H;
                    for (int ox = -1; ox <= 1; ++ox) {
                        std::size_t xx = (x + W + ox) % W;
                        if (ox == 0 && oy == 0) continue;
                        live_neighbors += cur[idx(xx, yy, W)];
                    }
                }
                uint8_t alive = cur[idx(x, y, W)];
                uint8_t out = 0;
                if (alive) {
                    out = (live_neighbors == 2 || live_neighbors == 3) ? 1 : 0;
                } else {
                    out = (live_neighbors == 3) ? 1 : 0;
                }
                next[idx(x, y, W)] = out;
            }
        }

        bool equal = (std::memcmp(cur.data(), next.data(), N) == 0);
        cur.swap(next);

        if (!skip_output && (output_every > 0 && (iter % output_every == 0 || equal || iter + 1 == max_iter))) {
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            write_ppm(path, cur, W, H);
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }

        if (equal) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "CPU run time: " << dt.count() << " s" << std::endl;
}
