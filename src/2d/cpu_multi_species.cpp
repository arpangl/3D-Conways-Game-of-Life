#include "dispatcher_2d.hpp"
#include "species.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <chrono>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <algorithm> // for std::max

static void get_species_color(int id, unsigned char rgb[3]) {
    if (id == 0) { rgb[0]=0;   rgb[1]=0;   rgb[2]=0;   return; } // 黑 (空)
    if (id == 1) { rgb[0]=0;   rgb[1]=255; rgb[2]=0;   return; } // 綠 (Bulbasaur)
    if (id == 2) { rgb[0]=255; rgb[1]=0;   rgb[2]=0;   return; } // 紅 (Ponyta)
    if (id == 3) { rgb[0]=255; rgb[1]=192; rgb[2]=203; return; } // 粉 (Sylveon)
    rgb[0]=255; rgb[1]=255; rgb[2]=255;
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

// --- 2. 新的規則檢查器 (支援多物種交互作用) ---
// neighbors 是一個 map: {物種ID: 數量, 物種ID: 數量...}
static bool check_rule_multi(const Rule& rule, const std::map<int, int>& neighbors) {
    // 遍歷所有 "OR" 條件 (只要有一組符合就回傳 true)
    for (const auto& condition : rule) {
        bool match = true;
        // 遍歷該條件內的所有 "AND" 要求 (例如: ID1要2個 且 ID2要0個)
        for (const auto& req : condition) {
            int target_id = req.first;
            int required_count = req.second;

            // 取得實際鄰居數量 (如果 map 裡沒有這個 key，代表數量是 0)
            int actual_count = 0;
            if (neighbors.find(target_id) != neighbors.end()) {
                actual_count = neighbors.at(target_id);
            }

            if (actual_count != required_count) {
                match = false;
                break; // 這個條件失敗，換下一個
            }
        }
        if (match) return true; // 找到符合的規則了！
    }
    return false;
}

void run_2d_cpu_multi_species(const SimulationOptions2D &options) {
    // 1. 讀取 CSV
    SpeciesManager::instance().load_species("species/species.csv"); // 請確保路徑正確
    const auto& all_species = SpeciesManager::instance().get_all_species();

    if (all_species.empty()) {
        std::cerr << "Error: No species loaded from CSV!" << std::endl;
        return;
    }

    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t N = W * H;

    // 2. 地圖改用 int 儲存物種 ID (0:空, 1:物種1, 2:物種2...)
    std::vector<int> cur(N);
    std::vector<int> next(N);

    // 3. 隨機初始化
    std::mt19937_64 rng(options.seed);
    
    // [FIX 1] 手動尋找最大 ID，因為 unordered_map 沒有 rbegin()
    int max_id = 0;
    for (const auto& kv : all_species) {
        if (kv.first > max_id) max_id = kv.first;
    }

    // 隨機骰 0 ~ max_id
    std::uniform_int_distribution<int> dist(0, max_id); 
    
    // 如果想要一開始空曠一點，可以把範圍加大
    std::uniform_int_distribution<int> dist_sparse(0, max_id + 2); 

    for (std::size_t i = 0; i < N; ++i) {
        int val = dist_sparse(rng);
        // 確認骰到的 ID 是否真的存在於 map 中 (除了 0 以外)
        if (val > 0 && all_species.find(val) != all_species.end()) {
            cur[i] = val;
        } else {
            cur[i] = 0;
        }
    }
    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    bool skip_output = (std::getenv("SKIP_OUTPUT") != nullptr);
    const char *outdir = "frames_multi";
    mkdir(outdir, 0755);
    auto t0 = std::chrono::steady_clock::now();

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // --- CPU 單執行緒運算 ---
        for (std::size_t y = 0; y < H; ++y) {
            for (std::size_t x = 0; x < W; ++x) {
                
                // A. 統計鄰居 (分類統計)
                std::map<int, int> neighbors_count;
                for (int oy = -1; oy <= 1; ++oy) {
                    std::size_t yy = (y + H + oy) % H;
                    for (int ox = -1; ox <= 1; ++ox) {
                        std::size_t xx = (x + W + ox) % W;
                        if (ox == 0 && oy == 0) continue;
                        
                        int neighbor_id = cur[idx(xx, yy, W)];
                        if (neighbor_id > 0) {
                            neighbors_count[neighbor_id]++;
                        }
                    }
                }

                // B. 決定下一代狀態
                int current_id = cur[idx(x, y, W)];
                int next_id = 0; // 預設死亡/維持空

                if (current_id > 0) {
                    // --- Case 1: 已經有生物 (檢查存活規則) ---
                    if (all_species.find(current_id) != all_species.end()) {
                        // [FIX 2] unordered_map 的 value 是物件，直接取址
                        const Species* sp = &all_species.at(current_id);
                        
                        if (check_rule_multi(sp->survival_rules, neighbors_count)) {
                            next_id = current_id; // 存活
                        } else {
                            next_id = 0; // 死亡 (變成空)
                        }
                    } else {
                        next_id = 0; // 異常ID
                    }
                } else {
                    // --- Case 2: 空格子 (檢查誕生規則) ---
                    // 檢查 "所有" 物種，看誰符合誕生條件
                    for (const auto& kv : all_species) {
                        // [FIX 2] 使用 &kv.second 取得指標
                        const Species* sp = &kv.second; 

                        if (check_rule_multi(sp->birth_rules, neighbors_count)) {
                            next_id = sp->id;
                            break; // 找到一個就誕生，並跳出 (First-fit)
                        }
                    }
                }
                
                next[idx(x, y, W)] = next_id;
            }
        }

        // 交換與輸出
        bool equal = (std::memcmp(cur.data(), next.data(), N * sizeof(int)) == 0);
        cur = next; // vector copy assignment

        if (iter % options.output_every == 0 || equal) {
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            write_ppm_color(path, cur, W, H);
            std::cout << "Iter " << iter << " done." << std::endl;
        }
        if (equal) break;
    }
}