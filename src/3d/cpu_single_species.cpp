#include "dispatcher_3d.hpp"
#include <iostream>

#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <cstdio>
#include <chrono>
#include <cerrno>

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

void run_3d_cpu_single_species(const SimulationOptions3D &options) {
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t D = options.depth;
    const std::size_t N = W * H * D;
    if (W == 0 || H == 0 || D == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }

    std::vector<uint8_t> cur(N);
    std::vector<uint8_t> next(N);

    std::mt19937_64 rng(options.seed);
    std::bernoulli_distribution bd(options.initial_density);
    for (std::size_t i = 0; i < N; ++i) cur[i] = bd(rng) ? 1 : 0;

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;

    const char *outdir = "frames";
    mkdir(outdir, 0755);

    std::string sdir = "frames_3d";
    mkdir(sdir.c_str(), 0755);
    bool skip_output = (std::getenv("SKIP_OUTPUT") != nullptr);
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        for (std::size_t z = 0; z < D; ++z) {
            for (std::size_t y = 0; y < H; ++y) {
                for (std::size_t x = 0; x < W; ++x) {
                    int live = 0;
                    for (int dz = -1; dz <= 1; ++dz) {
                        long zl = (long)z + dz;
                        if (zl < 0) zl += (long)D;
                        else if (zl >= (long)D) zl -= (long)D;
                        std::size_t zz = (std::size_t)zl;
                        for (int dy = -1; dy <= 1; ++dy) {
                            long yl = (long)y + dy;
                            if (yl < 0) yl += (long)H;
                            else if (yl >= (long)H) yl -= (long)H;
                            std::size_t yy = (std::size_t)yl;
                            for (int dx = -1; dx <= 1; ++dx) {
                                long xl = (long)x + dx;
                                if (xl < 0) xl += (long)W;
                                else if (xl >= (long)W) xl -= (long)W;
                                std::size_t xx = (std::size_t)xl;
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                live += cur[idx3(xx, yy, zz, W, H)];
                            }
                        }
                    }
                    uint8_t alive = cur[idx3(x, y, z, W, H)];
                    uint8_t out = 0;
                    // Use 3D Life variant: Birth on 5 neighbors (B5), survive on 4 or 5 neighbors (S4/5)
                    if (alive) out = (live == 4 || live == 5) ? 1 : 0;
                    else out = (live == 5) ? 1 : 0;
                    next[idx3(x, y, z, W, H)] = out;
                }
            }
        }

        bool equal = (std::memcmp(cur.data(), next.data(), N * sizeof(uint8_t)) == 0);
        cur.swap(next);

        if (!skip_output && (iter % output_every == 0 || equal || iter + 1 == max_iter)) {
            // produce a simple projection (max over depth) for visualization
            std::vector<uint8_t> proj(W * H);
            for (std::size_t zz = 0; zz < D; ++zz) {
                for (std::size_t i = 0; i < W * H; ++i) {
                    proj[i] = proj[i] || cur[zz * W * H + i];
                }
            }
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            write_ppm(path, proj, W, H);
            // also write per-slice images into frames_3d for 3D viewer
            // frames_3d directory was created once before the loop
            for (std::size_t zz = 0; zz < D; ++zz) {
                std::vector<uint8_t> slice(W * H);
                for (std::size_t i = 0; i < W * H; ++i) slice[i] = cur[zz * W * H + i];
                char spath[256];
                std::snprintf(spath, sizeof(spath), "%s/frame_%04zu_slice_%03zu.ppm", sdir.c_str(), iter, zz);
                std::printf("Writing slice: %s\n", spath);
                write_ppm(spath, slice, W, H);
            }
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }

        if (equal) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "3D CPU run time: " << dt.count() << " s" << std::endl;
}
