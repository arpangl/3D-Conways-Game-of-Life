#include "dispatcher_3d.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <cstdio>
#include <chrono>

#if defined(__GNUC__)
__attribute__((target("avx2")))
#endif
static inline std::size_t idx3(std::size_t x, std::size_t y, std::size_t z, std::size_t W, std::size_t H) {
    return z * (W * H) + y * W + x;
}

static void write_ppm(const std::string &path, const std::vector<uint8_t> &grid2d, std::size_t width, std::size_t height) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror("fopen"); return; }
    std::fprintf(f, "P6\n%zu %zu\n255\n", width, height);
    for (std::size_t i = 0; i < width * height; ++i) {
        unsigned char px = grid2d[i] ? 255 : 0;
        unsigned char rgb[3] = {px, px, px};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
}

void run_3d_avx2_multi_species(const SimulationOptions3D &options) {
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t D = options.depth;
    const std::size_t S = std::max<int>(1, options.species);
    const std::size_t N = W * H * D;
    if (W == 0 || H == 0 || D == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }

    std::vector<uint8_t> cur(S * N);
    std::vector<uint8_t> next(S * N);

    std::mt19937_64 rng(options.seed);
    std::bernoulli_distribution bd(options.initial_density);
    for (std::size_t s = 0; s < S; ++s) for (std::size_t i = 0; i < N; ++i) cur[s * N + i] = bd(rng) ? 1 : 0;

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;

    const char *outdir = "frames";
    mkdir(outdir, 0755);

    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t s = 0; s < S; ++s) {
            uint8_t *cur_s = cur.data() + s * N;
            uint8_t *next_s = next.data() + s * N;
            for (std::size_t z = 0; z < D; ++z) {
                for (std::size_t y = 0; y < H; ++y) {
                    #pragma GCC ivdep
                    for (std::size_t x = 0; x < W; ++x) {
                        int live = 0;
                        for (int dz = -1; dz <= 1; ++dz) {
                            std::size_t zz = (z + D + dz) % D;
                            for (int dy = -1; dy <= 1; ++dy) {
                                std::size_t yy = (y + H + dy) % H;
                                for (int dx = -1; dx <= 1; ++dx) {
                                    std::size_t xx = (x + W + dx) % W;
                                    if (dx == 0 && dy == 0 && dz == 0) continue;
                                    live += cur_s[idx3(xx, yy, zz, W, H)];
                                }
                            }
                        }
                        uint8_t alive = cur_s[idx3(x, y, z, W, H)];
                        uint8_t out = 0;
                        if (alive) out = (live == 4 || live == 5) ? 1 : 0;
                        else out = (live == 5) ? 1 : 0;
                        next_s[idx3(x, y, z, W, H)] = out;
                    }
                }
            }
        }

        bool equal_all = true;
        for (std::size_t s = 0; s < S; ++s) {
            uint8_t *cur_s = cur.data() + s * N;
            uint8_t *next_s = next.data() + s * N;
            bool eq = (std::memcmp(cur_s, next_s, N * sizeof(uint8_t)) == 0);
            if (!eq) equal_all = false;
            std::swap_ranges(cur_s, cur_s + N, next_s);
        }

        if (output_every > 0 && (iter % output_every == 0 || equal_all || iter + 1 == max_iter)) {
            std::vector<uint8_t> proj(W * H);
            for (std::size_t s = 0; s < S; ++s) {
                for (std::size_t zz = 0; zz < D; ++zz) {
                    for (std::size_t i = 0; i < W * H; ++i) {
                        proj[i] = proj[i] || cur[s * N + zz * W * H + i];
                    }
                }
            }
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            write_ppm(path, proj, W, H);
            std::string sdir = "frames_3d";
            int mk = mkdir(sdir.c_str(), 0755);
            if (mk != 0) { std::perror("mkdir frames_3d"); }
            for (std::size_t zz = 0; zz < D; ++zz) {
                std::vector<uint8_t> slice(W * H);
                for (std::size_t s = 0; s < S; ++s) {
                    for (std::size_t i = 0; i < W * H; ++i) slice[i] = slice[i] || cur[s * N + zz * W * H + i];
                }
                char spath[256];
                std::snprintf(spath, sizeof(spath), "%s/frame_%04zu_slice_%03zu.ppm", sdir.c_str(), iter, zz);
                std::printf("Writing slice: %s\n", spath);
                write_ppm(spath, slice, W, H);
            }
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }

        if (equal_all) { std::cout << "Converged at iteration " << iter << std::endl; break; }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "3D AVX2 multi-species run time: " << dt.count() << " s" << std::endl;
}
