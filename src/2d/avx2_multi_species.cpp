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
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif


static inline std::size_t idx(std::size_t x, std::size_t y, std::size_t width) {
    return y * width + x;
}

struct Condition { //給下面的RuleGroup用
    int species_id; // CSV 中的 ID
    int count;      // 所需數量
};

struct RuleGroup {
    std::vector<Condition> conditions;
};

struct SpeciesData {
    int id;           
    int index;          //內部 Index
    std::string name;
    int strength;
    std::string image_path;
    
    // 規則集合(OR關係）
    std::vector<RuleGroup> survival_rules;
    std::vector<RuleGroup> birth_rules;
};

// 字串工具
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// 解析類似 {(1,2),(2,0)}{(1,3)} 的字串
static std::vector<RuleGroup> parse_rule_string(const std::string& raw) {
    std::vector<RuleGroup> groups;
    std::string s = trim(raw);
    if (s.empty()) return groups;

    bool in_group = false;
    std::string current_group_str;
    
    for (char c : s) {
        if (c == '{') {
            in_group = true;
            current_group_str = "";
        } else if (c == '}') {
            in_group = false;
            // 解析 group 內部的 pairs (id, count)
            RuleGroup rg;
            // 簡單 parser: 尋找 (a,b)
            bool in_pair = false;
            std::string pair_str;
            for (char gc : current_group_str) {
                if (gc == '(') { in_pair = true; pair_str = ""; }
                else if (gc == ')') {
                    in_pair = false;
                    // split pair_str by comma
                    size_t comma = pair_str.find(',');
                    if (comma != std::string::npos) {
                        try {
                            int sp_id = std::stoi(trim(pair_str.substr(0, comma)));
                            int cnt = std::stoi(trim(pair_str.substr(comma + 1)));
                            rg.conditions.push_back({sp_id, cnt});
                        } catch (...) {}
                    }
                } else if (in_pair) {
                    pair_str += gc;
                }
            }
            if (!rg.conditions.empty()) {
                groups.push_back(rg);
            }
        } else if (in_group) {
            current_group_str += c;
        }
    }
    return groups;
}

static std::vector<SpeciesData> load_species_rules(const std::string& filename) {
    std::vector<SpeciesData> species_list;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return species_list;
    }

    std::string line;
    // Skip Header (Line 0)
    if(std::getline(file, line)) { 
        // check if strictly header or empty
    }

    int internal_index = 0;
    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;
        
        // Manual split logic because of nested commas in {}
        std::vector<std::string> parts;
        int balance = 0;
        std::string current_part;
        for (char c : line) {
            if (c == '{') balance++;
            else if (c == '}') balance--;
            
            if (c == ',' && balance == 0) {
                parts.push_back(trim(current_part));
                current_part = "";
            } else {
                current_part += c;
            }
        }
        parts.push_back(trim(current_part));

        // Expected: ID, Name, Survival, Birth, Strength, Image
        if (parts.size() < 6) continue;

        SpeciesData sp;
        try {
            sp.id = std::stoi(parts[0]);
            sp.name = parts[1];
            sp.survival_rules = parse_rule_string(parts[2]);
            sp.birth_rules = parse_rule_string(parts[3]);
            sp.strength = std::stoi(parts[4]);
            sp.image_path = parts[5];
            sp.index = internal_index++;
            species_list.push_back(sp);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << " | " << e.what() << std::endl;
        }
    }
    return species_list;
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

#if defined(__GNUC__)
__attribute__((target("avx2")))
#endif
static void count_neighbors_avx2(const uint8_t* __restrict src, uint8_t* __restrict dst_counts, 
                         size_t count, const std::vector<int>& offsets) {
    size_t i = 0;
    for (; i + 31 < count; i += 32) {
        __m256i sum = _mm256_setzero_si256();
        for (int off : offsets) {
            __m256i neighbor = _mm256_loadu_si256((const __m256i*)(src + i + off));
            sum = _mm256_add_epi8(sum, neighbor);
        }
        _mm256_storeu_si256((__m256i*)(dst_counts + i), sum);
    }
    for (; i < count; ++i) {
        int live = 0;
        for (int off : offsets) live += src[i + off];
        dst_counts[i] = (uint8_t)live;
    }
}

// 2D 邊界處理 (Wrap-around)
static void update_ghost_cells_2d(uint8_t* grid, int W, int H, int pW, int pH) {
    // Y borders (上下)
    for (int x = 0; x < pW; ++x) {
        grid[0 * pW + x] = grid[H * pW + x];              // Y=H -> Y=0
        grid[(H + 1) * pW + x] = grid[1 * pW + x];       // Y=1 -> Y=H+1
    }
    // X borders (左右)
    for (int y = 0; y < pH; ++y) {
        grid[y * pW + 0] = grid[y * pW + W];              // X=W -> X=0
        grid[y * pW + (W + 1)] = grid[y * pW + 1];       // X=1 -> X=W+1
    }
}

// 根據物種 ID 生成顏色
static void get_species_color(int species_id, uint8_t& r, uint8_t& g, uint8_t& b) {
    int colors[][3] = {
        {0, 0, 0},      // 0: 黑色（空）
        {0, 255, 0},   // 1: 綠色
        {255, 0, 0},   // 2: 紅色
        {255, 192, 203},   // 3: 藍色
        {255, 255, 0}, // 4: 黃色
        {255, 0, 255}, // 5: 洋紅色
        {0, 255, 255}, // 6: 青色
        {255, 128, 0}, // 7: 橙色
        {128, 0, 255}, // 8: 紫色
        {0, 0, 255}, // 9: 粉色
    };
    
    if (species_id < 0 || species_id >= 10) {
        r = g = b = 128; // 灰色
    } else {
        r = colors[species_id][0];
        g = colors[species_id][1];
        b = colors[species_id][2];
    }
}

// 彩色 PPM 輸出（根據物種 ID）
static void write_ppm_color(const std::string &path, const std::vector<uint8_t> &species_grid, 
                           std::size_t width, std::size_t height) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror("fopen"); return; }
    std::fprintf(f, "P6\n%zu %zu\n255\n", width, height);
    for (std::size_t i = 0; i < width * height; ++i) {
        uint8_t r, g, b;
        get_species_color(species_grid[i], r, g, b);
        unsigned char rgb[3] = {r, g, b};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void run_2d_avx2_multi_species(const SimulationOptions2D &options) {
    // 1. Load Rules from CSV
    std::string csv_path = "species/species.csv";
    auto species_list = load_species_rules(csv_path);
    
    if (species_list.empty()) {
        std::cerr << "No species rules found in " << csv_path << ". Exiting." << std::endl;
        return;
    }
    
    int S = species_list.size();
    // Build ID -> Index Map for fast lookup in rules
    std::map<int, int> id_to_index;
    for(const auto& sp : species_list) id_to_index[sp.id] = sp.index;

    // Convert rules to use internal Indices instead of IDs
    for(auto& sp : species_list) {
        auto convert_rules = [&](std::vector<RuleGroup>& groups) {
            for(auto& rg : groups) {
                for(auto& cond : rg.conditions) {
                    if(id_to_index.count(cond.species_id)) {
                        cond.species_id = id_to_index[cond.species_id]; // Overwrite ID with Index
                    } else {
                        cond.species_id = -1; // Unknown species?
                    }
                }
            }
        };
        convert_rules(sp.survival_rules);
        convert_rules(sp.birth_rules);
    }

    std::cout << "Loaded " << S << " species from CSV." << std::endl;

    // 2. Setup Grid with padding
    const int W = options.width;
    const int H = options.height;
    const int pW = W + 2, pH = H + 2;
    const size_t pN = (size_t)pW * pH;

    if (W == 0 || H == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }

    // cur: Current state (0/1) for each species
    // neighbor_counts: Neighbor counts for each species
    std::vector<uint8_t> cur(S * pN, 0);
    std::vector<uint8_t> next(S * pN, 0);
    std::vector<uint8_t> neighbor_counts(S * pN, 0); // Stores neighbor counts (0-8 for 2D)

    // Init Data
    {
    std::mt19937_64 rng(options.seed);
        double initial_density = 0.15; // Default density for 2D (3D uses options.initial_density)
        std::bernoulli_distribution bd(initial_density);
        for (int s = 0; s < S; ++s) {
            for (int y = 1; y <= H; ++y) {
                for (int x = 1; x <= W; ++x) {
                    // Simple random init
                    if(bd(rng)) cur[s * pN + y*pW + x] = 1;
                }
            }
        }
    }

    // Offsets for neighbor counting (2D: 8 neighbors)
    std::vector<int> offsets;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx|dy) offsets.push_back(dy * pW + dx);
        }
    }

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    bool skip_output = (std::getenv("SKIP_OUTPUT") != nullptr);
    const char *outdir = "frames";
    mkdir(outdir, 0755);

    auto t0 = std::chrono::steady_clock::now();

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        
        // --- Step 1: Count Neighbors (AVX2) ---
        // For each species, update ghosts and count neighbors
        #pragma omp parallel for
        for (int s = 0; s < S; ++s) {
            uint8_t* sp_grid = &cur[s * pN];
            uint8_t* sp_counts = &neighbor_counts[s * pN];
            
            update_ghost_cells_2d(sp_grid, W, H, pW, pH);
            
            for (int y = 1; y <= H; ++y) {
                size_t idx = y * pW + 1;
                count_neighbors_avx2(sp_grid + idx, sp_counts + idx, W, offsets);
            }
        }

        // --- Step 2: Resolve Rules & Competition (Scalar) ---
        #pragma omp parallel for
        for (int y = 1; y <= H; ++y) {
            size_t base_idx = y * pW + 1;
            std::vector<uint8_t> local_n_counts_vec(S);
            std::vector<uint8_t> current_alive_vec(S);
            for (int x = 0; x < W; ++x) {
                size_t idx = base_idx + x;
                
                // Identify candidates who WANT to be here
                int best_species = -1;
                int max_strength = -1;
                
                // Optimization: Read all neighbor counts for this cell once
                // std::vector<uint8_t> local_n_counts_vec(S);
                // std::vector<uint8_t> current_alive_vec(S);
                uint8_t* local_n_counts = local_n_counts_vec.data();
                uint8_t* current_alive = current_alive_vec.data();
                int current_occupant = -1;

                for(int s=0; s<S; ++s) {
                    local_n_counts[s] = neighbor_counts[s * pN + idx];
                    current_alive[s] = cur[s * pN + idx];
                    if (current_alive[s]) current_occupant = s;
                    next[s * pN + idx] = 0; // Reset next state
                }

                // Check rules for each species
                for (int s = 0; s < S; ++s) {
                    const auto& sp = species_list[s];
                    bool wants_to_live = false;
                    
                    // Pick Rule Set: Survival (if self) vs Birth (if not self)
                    const std::vector<RuleGroup>* rules = (current_alive[s]) ? &sp.survival_rules : &sp.birth_rules;

                    // Check if ANY rule group is satisfied (OR logic)
                    for (const auto& group : *rules) {
                        bool group_sat = true;
                        // Check all conditions in group (AND logic)
                        for (const auto& cond : group.conditions) {
                            if (cond.species_id < 0 || cond.species_id >= S) continue;
                            if (local_n_counts[cond.species_id] != cond.count) {
                                group_sat = false; 
                                break; 
                            }
                        }
                        if (group_sat) {
                            wants_to_live = true;
                            break;
                        }
                    }

                    if (wants_to_live) {
                        // Competition Logic
                        if (sp.strength > max_strength) {
                            max_strength = sp.strength;
                            best_species = s;
                        } else if (sp.strength == max_strength) {
                            // Tie-breaker: Incumbent wins, else Highest ID
                            if (s == current_occupant) best_species = s;
                            else if (best_species == -1 || sp.id > species_list[best_species].id) {
                                best_species = s;
                            }
                        }
                    }
                }

                // Apply Winner
                if (best_species != -1) {
                    next[best_species * pN + idx] = 1;
                }
            }
        }

        // Check convergence
        bool equal_all = true;
        for (int s = 0; s < S; ++s) {
            uint8_t *cur_s = cur.data() + s * pN;
            uint8_t *next_s = next.data() + s * pN;
            bool eq = (std::memcmp(cur_s, next_s, pN * sizeof(uint8_t)) == 0);
            if (!eq) equal_all = false;
        }

        std::swap(cur, next);

        // Output logic
        if (!skip_output && (output_every > 0 && (iter % output_every == 0 || equal_all || iter + 1 == max_iter))) {
            // Generate output grid with species IDs
            std::vector<uint8_t> output_grid(W * H, 0);
            for(int y=1; y<=H; ++y) {
                for(int x=1; x<=W; ++x) {
                    size_t pidx = y * pW + x;
                    size_t out_idx = (y-1) * W + (x-1);
                    
                    // Find which species is alive at this position (highest strength wins)
                    int best_id = 0;
                    int max_strength = -1;
                    
                    for (int s = 0; s < S; ++s) {
                        if (cur[s * pN + pidx]) {
                            const auto& sp = species_list[s];
                            if (sp.strength > max_strength) {
                                max_strength = sp.strength;
                                best_id = sp.id;
                            } else if (sp.strength == max_strength && sp.id > best_id) {
                                best_id = sp.id;
                            }
                        }
                    }
                    output_grid[out_idx] = best_id;
                }
            }
            
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            write_ppm_color(path, output_grid, W, H);
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }

        if (equal_all) { 
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "2D AVX2 multi-species (CSV rules) run time: " << dt.count() << " s" << std::endl;
}

