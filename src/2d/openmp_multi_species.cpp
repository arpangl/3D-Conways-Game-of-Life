#include "dispatcher_2d.hpp"
#include <iostream>

#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>

static inline std::size_t idx(std::size_t x, std::size_t y, std::size_t width) {
    return y * width + x;
}

static void write_ppm_multi(const std::string &path, const std::vector<int> &grid, std::size_t width, std::size_t height, int species) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return;
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

void run_2d_openmp_multi_species(const SimulationOptions2D &options) {
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t N = W * H;
    const int S = std::max(1, options.species);
    if (W==0 || H==0) { std::cerr << "Invalid dimensions" << std::endl; return; }

    std::vector<int> cur(N);
    std::vector<int> next(N);

    // random init: values 0..S (0 = empty)
    std::mt19937_64 rng(options.seed);
    std::uniform_int_distribution<int> dist(0, S);
    for (std::size_t i=0;i<N;++i) cur[i] = dist(rng);

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    mkdir("frames", 0755);

    for (std::size_t iter=0; iter<max_iter; ++iter) {
        // For each cell, count neighbors per species. Parallelize rows.
#pragma omp parallel for schedule(static)
        for (std::size_t y=0;y<H;++y) {
            std::vector<int> counts(S+1);
            for (std::size_t x=0;x<W;++x) {
                std::fill(counts.begin(), counts.end(), 0);
                for (int oy=-1; oy<=1; ++oy) {
                    std::size_t yy = (y + H + oy) % H;
                    for (int ox=-1; ox<=1; ++ox) {
                        std::size_t xx = (x + W + ox) % W;
                        if (ox==0 && oy==0) continue;
                        int v = cur[idx(xx,yy,W)];
                        if (v>=0 && v<=S) counts[v]++;
                    }
                }
                int me = cur[idx(x,y,W)];
                int out = 0;
                if (me != 0) {
                    if (counts[me] == 2 || counts[me] == 3) out = me;
                    else out = 0;
                } else {
                    int chosen = 0;
                    int best_count = -1;
                    for (int s=1; s<=S; ++s) {
                        if (counts[s] == 3) {
                            if (counts[s] > best_count) { best_count = counts[s]; chosen = s; }
                            else if (counts[s] == best_count && chosen != 0) {
                                if (s < chosen) chosen = s;
                            }
                        }
                    }
                    out = chosen;
                }
                next[idx(x,y,W)] = out;
            }
        }

        bool equal = (std::memcmp(cur.data(), next.data(), N * sizeof(int)) == 0);
        cur.swap(next);

        if (iter % output_every == 0 || equal || iter+1==max_iter) {
            char path[256];
            std::snprintf(path, sizeof(path), "frames/frame_%04zu.ppm", iter);
            write_ppm_multi(path, cur, W, H, S);
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }

        if (equal) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }
}
