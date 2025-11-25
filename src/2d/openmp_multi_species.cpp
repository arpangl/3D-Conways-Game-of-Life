#include "dispatcher_2d.hpp"
#include "species.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <algorithm>
#include <cstring>
#include <sys/stat.h>
#include <chrono>
#include <omp.h> // OpenMP header

// 定義最大物種數以優化效能 (避免在迴圈內 malloc)
#define MAX_SPECIES_LIMIT 20

static void get_species_color(int id, unsigned char rgb[3]) {
    if (id == 0) { rgb[0]=0; rgb[1]=0; rgb[2]=0; return; }
    // 簡易配色雜湊
    rgb[0] = (id * 123) % 256;
    rgb[1] = (id * 456) % 256;
    rgb[2] = (id * 789) % 256;
    // 特殊指定
    if (id == 1) { rgb[0]=0; rgb[1]=255; rgb[2]=0; } // Bulbasaur Green
    if (id == 2) { rgb[0]=255; rgb[1]=0; rgb[2]=0; } // Ponyta Red
    if (id == 3) { rgb[0]=255; rgb[1]=192; rgb[2]=203; } // Sylveon Pink
}

static void write_ppm_color(const std::string &path, const std::vector<int> &grid, std::size_t width, std::size_t height) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%zu %zu\n255\n", width, height);
    for (int id : grid) {
        unsigned char rgb[3];
        get_species_color(id, rgb);
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
}

static inline std::size_t idx(std::size_t x, std::size_t y, std::size_t width) {
    return y * width + x;
}

// 針對 OpenMP 優化的規則檢查 (使用固定陣列 counts)
static bool check_rule_fast(const Rule& rule, const int* counts) {
    for (const auto& condition : rule) {
        bool match = true;
        for (const auto& req : condition) {
            int target_id = req.first;
            int required_count = req.second;
            int actual_count = (target_id < MAX_SPECIES_LIMIT) ? counts[target_id] : 0;
            
            if (actual_count != required_count) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

void run_2d_openmp_multi_species(const SimulationOptions2D &options) {
    // 1. 讀取 CSV
    SpeciesManager::instance().load_species("species.csv");
    const auto& all_species = SpeciesManager::instance().get_all_species();

    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t N = W * H;

    std::vector<int> cur(N);
    std::vector<int> next(N);

    // 初始化
    std::mt19937_64 rng(options.seed);
    int max_id = 0;
    for (const auto& kv : all_species) if (kv.first > max_id) max_id = kv.first;
    std::uniform_int_distribution<int> dist(0, max_id + 2); // +2 讓空地多一點

    for (std::size_t i = 0; i < N; ++i) {
        int val = dist(rng);
        cur[i] = (val <= max_id && all_species.count(val)) ? val : 0;
    }

    mkdir("frames_omp", 0755);
    auto t0 = std::chrono::steady_clock::now();
    // 主迴圈
    for (std::size_t iter = 0; iter < options.iterations; ++iter) {
        
        // OpenMP 平行化：將 Y 軸的任務分給不同 CPU 核心
        #pragma omp parallel for schedule(static)
        for (std::size_t y = 0; y < H; ++y) {
            
            // 每個執行緒有自己的計數陣列 (避免 map 的開銷)
            int counts[MAX_SPECIES_LIMIT] = {0};

            for (std::size_t x = 0; x < W; ++x) {
                // 清空計數器
                std::memset(counts, 0, sizeof(counts));

                // 統計鄰居
                for (int oy = -1; oy <= 1; ++oy) {
                    std::size_t yy = (y + H + oy) % H;
                    for (int ox = -1; ox <= 1; ++ox) {
                        std::size_t xx = (x + W + ox) % W;
                        if (ox == 0 && oy == 0) continue;

                        int neighbor_id = cur[yy * W + xx];
                        if (neighbor_id > 0 && neighbor_id < MAX_SPECIES_LIMIT) {
                            counts[neighbor_id]++;
                        }
                    }
                }

                int current_id = cur[y * W + x];
                int next_id = 0;

                if (current_id > 0) {
                    // 檢查存活
                    if (all_species.count(current_id)) {
                        const Species* sp = &all_species.at(current_id);
                        if (check_rule_fast(sp->survival_rules, counts)) {
                            next_id = current_id;
                        }
                    }
                } else {
                    // 檢查誕生 (優先順序法)
                    for (const auto& kv : all_species) {
                        const Species* sp = &kv.second;
                        if (check_rule_fast(sp->birth_rules, counts)) {
                            next_id = sp->id;
                            break; 
                        }
                    }
                }
                next[y * W + x] = next_id;
            }
        }

        // 交換與輸出
        bool equal = (std::memcmp(cur.data(), next.data(), N * sizeof(int)) == 0);
        cur = next;

        if (iter % options.output_every == 0 || equal) {
            char path[256];
            std::snprintf(path, sizeof(path), "frames_omp/frame_%04zu.ppm", iter);
            write_ppm_color(path, cur, W, H);
            std::cout << "OMP Iter " << iter << " done." << std::endl;
        }
        if (equal) break;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "CPU run time: " << dt.count() << " s" << std::endl;
}
