#include "dispatcher_3d.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <cstdio>
#include <chrono>
#include <immintrin.h> 

#ifdef _OPENMP
#include <omp.h>
#endif

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

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void process_slice_avx2(const uint8_t* __restrict src, uint8_t* __restrict dst, 
                       size_t count, const std::vector<int>& offsets) {
    size_t i = 0;
    for (; i + 31 < count; i += 32) {
        __m256i sum = _mm256_setzero_si256();
        __m256i self = _mm256_loadu_si256((const __m256i*)(src + i));
        for (int off : offsets) {
            __m256i neighbor = _mm256_loadu_si256((const __m256i*)(src + i + off));
            sum = _mm256_add_epi8(sum, neighbor);
        }
        
        __m256i is_5 = _mm256_cmpeq_epi8(sum, _mm256_set1_epi8(5));
        __m256i is_4 = _mm256_cmpeq_epi8(sum, _mm256_set1_epi8(4));
        
        // Survival logic: (sum == 5) OR ((sum == 4) AND (self == 1))
        __m256i survive_cond = _mm256_and_si256(is_4, self);
        __m256i result_raw = _mm256_or_si256(is_5, survive_cond);
        
        __m256i final_res = _mm256_and_si256(result_raw, _mm256_set1_epi8(1));
        _mm256_storeu_si256((__m256i*)(dst + i), final_res);
    }
    
    //處理剩餘不足32個的部分
    for (; i < count; ++i) {
        int live = 0;
        for (int off : offsets) live += src[i + off];
        uint8_t alive = src[i];
        if (alive) dst[i] = (live == 4 || live == 5) ? 1 : 0;
        else       dst[i] = (live == 5) ? 1 : 0;
    }
}

void update_ghost_cells(std::vector<uint8_t>& grid, int W, int H, int D, int pW, int pH, int pD) {
    size_t plane_sz = (size_t)pW * pH;
    
    #pragma omp parallel
    {
        // 1. 複製 Z 面 (上下)
        #pragma omp for nowait
        for (int y = 0; y < pH; ++y) {
            std::memcpy(&grid[0 * plane_sz + y * pW], &grid[D * plane_sz + y * pW], pW);         // Z=D -> Z=0
            std::memcpy(&grid[(D + 1) * plane_sz + y * pW], &grid[1 * plane_sz + y * pW], pW); // Z=1 -> Z=D+1
        }
        #pragma omp barrier

        // 2. 複製 Y 面 (前後) - 包含剛剛複製好的 Z Ghost
        #pragma omp for collapse(2) nowait
        for (int z = 0; z < pD; ++z) {
            for (int x = 0; x < pW; ++x) {
                 grid[z * plane_sz + 0 * pW + x] = grid[z * plane_sz + H * pW + x];             // Y=H -> Y=0
                 grid[z * plane_sz + (H + 1) * pW + x] = grid[z * plane_sz + 1 * pW + x];     // Y=1 -> Y=H+1
            }
        }
        #pragma omp barrier

        // 3. 複製 X 面 (左右) - 包含 Z 和 Y Ghost
        #pragma omp for collapse(2)
        for (int z = 0; z < pD; ++z) {
            for (int y = 0; y < pH; ++y) {
                grid[z * plane_sz + y * pW + 0] = grid[z * plane_sz + y * pW + W];              // X=W -> X=0
                grid[z * plane_sz + y * pW + (W + 1)] = grid[z * plane_sz + y * pW + 1];      // X=1 -> X=W+1
            }
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void run_3d_avx2_single_species(const SimulationOptions3D &options) {
    const int W = options.width;
    const int H = options.height;
    const int D = options.depth;
    
    const int pW = W + 2;
    const int pH = H + 2;
    const int pD = D + 2;
    const size_t pN = (size_t)pW * pH * pD;

    if (W == 0 || H == 0 || D == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }

    std::vector<uint8_t> cur(pN, 0);
    std::vector<uint8_t> next(pN, 0);
    {
        std::mt19937_64 rng(options.seed);
        std::bernoulli_distribution bd(options.initial_density);
        for (int z = 1; z <= D; ++z) {
            for (int y = 1; y <= H; ++y) {
                for (int x = 1; x <= W; ++x) {
                    size_t idx = z * (pW * pH) + y * pW + x;
                    cur[idx] = bd(rng) ? 1 : 0;
                }
            }
        }
    }

    std::vector<int> offsets;
    offsets.reserve(26);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                offsets.push_back(dz * (pW * pH) + dy * pW + dx);
            }
        }
    }

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    const char *outdir = "frames";
    mkdir(outdir, 0755);

    auto t0 = std::chrono::steady_clock::now();

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // 更新 Ghost Cells (處理邊界 Wrapping)
        update_ghost_cells(cur, W, H, D, pW, pH, pD);

        // 平行化執行核心運算
        #pragma omp parallel for collapse(2)
        for (int z = 1; z <= D; ++z) {
            for (int y = 1; y <= H; ++y) {
                // 指向該列的有效資料起始點 (跳過左邊的)
                size_t start_idx = z * (pW * pH) + y * pW + 1;
                process_slice_avx2(&cur[start_idx], &next[start_idx], W, offsets);
            }
        }

        bool equal = true;
        #pragma omp parallel for collapse(2) reduction(&&:equal)
        for (int z = 1; z <= D; ++z) {
            for (int y = 1; y <= H; ++y) {
                if (!equal) continue;
                size_t start_idx = z * (pW * pH) + y * pW + 1;
                if (std::memcmp(&cur[start_idx], &next[start_idx], W) != 0) {
                    equal = false;
                }
            }
        }

        std::swap(cur, next);

        // 輸出邏輯
        // if (output_every > 0 && (iter % output_every == 0 || equal || iter + 1 == max_iter)) {
        //     // 1.產生投影圖
        //     std::vector<uint8_t> proj(W * H, 0);
        //     for (int z = 1; z <= D; ++z) {
        //         for (int y = 1; y <= H; ++y) {
        //             size_t src_offset = z * (pW * pH) + y * pW + 1;
        //             size_t dst_offset = (y - 1) * W;
        //             for(int x = 0; x < W; ++x) {
        //                 if (cur[src_offset + x]) proj[dst_offset + x] = 1;
        //             }
        //         }
        //     }
            
        //     char path[256];
        //     std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
        //     write_ppm(path, proj, W, H);
            
        //     std::string sdir = "frames_3d";
        //     int mk = mkdir(sdir.c_str(), 0755);
        //     if (mk != 0) { /* ignore existing */ }

        //     // 2.產生切片圖 
        //     // 需要從Padded Grid還原成標準 W*H 寫入
        //     for (int z = 1; z <= D; ++z) {
        //          std::vector<uint8_t> slice(W * H);
        //          for (int y = 1; y <= H; ++y) {
        //              size_t src_idx = z * (pW * pH) + y * pW + 1;
        //              std::memcpy(&slice[(y - 1) * W], &cur[src_idx], W);
        //          }
        //          char spath[256];
        //          std::snprintf(spath, sizeof(spath), "%s/frame_%04zu_slice_%03zu.ppm", sdir.c_str(), iter, (size_t)z-1);
        //          std::printf("Writing slice: %s\n", spath);
        //          write_ppm(spath, slice, W, H);
        //     }
        //     std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        // }

        if (equal) { 
            std::cout << "Converged at iteration " << iter << std::endl; 
            break; 
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "3D AVX2 run time: " << dt.count() << " s" << std::endl;
}