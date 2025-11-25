#include "dispatcher_3d.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <random>
#include <cstring>
#include <sys/stat.h>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline std::size_t idx3(std::size_t x, std::size_t y, std::size_t z, std::size_t W, std::size_t H) {
    return z * (W * H) + y * W + x;
}

struct Condition {
    int species_id;
    int count;
};

struct RuleGroup {
    std::vector<Condition> conditions;
};

struct SpeciesData {
    int id;
    int index;
    std::string name;
    int strength;
    std::string image_path;
    std::vector<RuleGroup> survival_rules;
    std::vector<RuleGroup> birth_rules;
};

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

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
            RuleGroup rg;
            bool in_pair = false;
            std::string pair_str;
            for (char gc : current_group_str) {
                if (gc == '(') { in_pair = true; pair_str = ""; }
                else if (gc == ')') {
                    in_pair = false;
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
            if (!rg.conditions.empty()) groups.push_back(rg);
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
    if (std::getline(file, line)) {}

    int internal_index = 0;
    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;

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

static void get_species_color(int species_id, uint8_t& r, uint8_t& g, uint8_t& b) {
    int colors[][3] = {
        {0, 0, 0},      // 0: 黑色（空）
        {0, 255, 0},   // 1: 綠色
        {255, 0, 0},   // 2: 紅色
        {255, 192, 203},   // 3: 粉色
        {255, 255, 0}, // 4: 黃色
        {255, 0, 255}, // 5: 洋紅色
        {0, 255, 255}, // 6: 青色
        {255, 128, 0}, // 7: 橙色
        {128, 0, 255}, // 8: 紫色
        {0, 0, 255}, // 9: 藍色
    };
    
    if (species_id < 0 || species_id >= 10) {
        // 超出範圍，使用白色
        r = g = b = 255;
    } else {
        r = colors[species_id][0];
        g = colors[species_id][1];
        b = colors[species_id][2];
    }
}

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


static void update_ghost_cells_generic(uint8_t* grid, int W, int H, int D, int pW, int pH, int pD) {
    size_t plane_sz = (size_t)pW * pH;
    for (int y = 0; y < pH; ++y) {
        std::memcpy(&grid[0 * plane_sz + y * pW], &grid[D * plane_sz + y * pW], pW);
        std::memcpy(&grid[(D + 1) * plane_sz + y * pW], &grid[1 * plane_sz + y * pW], pW);
    }
    for (int z = 0; z < pD; ++z) {
        for (int x = 0; x < pW; ++x) {
             grid[z * plane_sz + 0 * pW + x] = grid[z * plane_sz + H * pW + x];
             grid[z * plane_sz + (H + 1) * pW + x] = grid[z * plane_sz + 1 * pW + x];
        }
    }
    for (int z = 0; z < pD; ++z) {
        for (int y = 0; y < pH; ++y) {
            grid[z * plane_sz + y * pW + 0] = grid[z * plane_sz + y * pW + W];
            grid[z * plane_sz + y * pW + (W + 1)] = grid[z * plane_sz + y * pW + 1];
        }
    }
}

void run_3d_openmp_multi_species(const SimulationOptions3D &options) {
    std::string csv_path = "species/species.csv";
    auto species_list = load_species_rules(csv_path);
    if (species_list.empty()) {
        std::cerr << "No species rules found in " << csv_path << ". Exiting." << std::endl;
        return;
    }

    int S = species_list.size();
    std::map<int, int> id_to_index;
    for (const auto& sp : species_list) id_to_index[sp.id] = sp.index;

    for (auto& sp : species_list) {
        auto convert_rules = [&](std::vector<RuleGroup>& groups) {
            for (auto& rg : groups) {
                for (auto& cond : rg.conditions) {
                    if (id_to_index.count(cond.species_id)) {
                        cond.species_id = id_to_index[cond.species_id];
                    } else {
                        cond.species_id = -1;
                    }
                }
            }
        };
        convert_rules(sp.survival_rules);
        convert_rules(sp.birth_rules);
    }

    const int W = options.width;
    const int H = options.height;
    const int D = options.depth;
    if (W == 0 || H == 0 || D == 0) {
        std::cerr << "Invalid dimensions" << std::endl;
        return;
    }

    const int pW = W + 2, pH = H + 2, pD = D + 2;
    const size_t pN = (size_t)pW * pH * pD;

    std::vector<uint8_t> cur(S * pN, 0);
    std::vector<uint8_t> next(S * pN, 0);
    std::vector<uint8_t> neighbor_counts(S * pN, 0);

    {
        std::mt19937_64 rng(options.seed);
        std::bernoulli_distribution bd(options.initial_density);
        for (int s = 0; s < S; ++s) {
            for (int z = 1; z <= D; ++z) {
                for (int y = 1; y <= H; ++y) {
                    for (int x = 1; x <= W; ++x) {
                        if (bd(rng)) cur[s * pN + z * pH * pW + y * pW + x] = 1;
                    }
                }
            }
        }
    }

    std::vector<int> offsets;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (dx | dy | dz) offsets.push_back(dz * pH * pW + dy * pW + dx);

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    bool skip_output = (std::getenv("SKIP_OUTPUT") != nullptr);
    const char *outdir = "frames";
    mkdir(outdir, 0755);

    auto t0 = std::chrono::steady_clock::now();

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        #pragma omp parallel for
        for (int s = 0; s < S; ++s) {
            uint8_t* sp_grid = &cur[s * pN];
            uint8_t* sp_counts = &neighbor_counts[s * pN];

            update_ghost_cells_generic(sp_grid, W, H, D, pW, pH, pD);

            for (int z = 1; z <= D; ++z) {
                for (int y = 1; y <= H; ++y) {
                    for (int x = 1; x <= W; ++x) {
                        size_t idx = z * pH * pW + y * pW + x;
                        int live = 0;
                        for (int off : offsets) live += sp_grid[idx + off];
                        sp_counts[idx] = static_cast<uint8_t>(live);
                    }
                }
            }
        }

        #pragma omp parallel for collapse(2)
        for (int z = 1; z <= D; ++z) {
            for (int y = 1; y <= H; ++y) {
                size_t base_idx = z * pH * pW + y * pW + 1;
                for (int x = 0; x < W; ++x) {
                    size_t idx = base_idx + x;
                    int best_species = -1;
                    int max_strength = -1;
                    std::vector<uint8_t> local_n_counts_vec(S);
                    std::vector<uint8_t> current_alive_vec(S);
                    uint8_t* local_n_counts = local_n_counts_vec.data();
                    uint8_t* current_alive = current_alive_vec.data();
                    int current_occupant = -1;

                    for (int s = 0; s < S; ++s) {
                        local_n_counts[s] = neighbor_counts[s * pN + idx];
                        current_alive[s] = cur[s * pN + idx];
                        if (current_alive[s]) current_occupant = s;
                        next[s * pN + idx] = 0;
                    }

                    for (int s = 0; s < S; ++s) {
                        const auto& sp = species_list[s];
                        bool wants_to_live = false;
                        const std::vector<RuleGroup>* rules = (current_alive[s]) ? &sp.survival_rules : &sp.birth_rules;

                        for (const auto& group : *rules) {
                            bool group_sat = true;
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
                            if (sp.strength > max_strength) {
                                max_strength = sp.strength;
                                best_species = s;
                            } else if (sp.strength == max_strength) {
                                if (s == current_occupant) best_species = s;
                                else if (best_species == -1 || sp.id > species_list[best_species].id) {
                                    best_species = s;
                                }
                            }
                        }
                    }

                    if (best_species != -1) {
                        next[best_species * pN + idx] = 1;
                    }
                }
            }
        }

        bool equal_all = true;
        for (int s = 0; s < S; ++s) {
            uint8_t *cur_s = cur.data() + s * pN;
            uint8_t *next_s = next.data() + s * pN;
            bool eq = (std::memcmp(cur_s, next_s, pN * sizeof(uint8_t)) == 0);
            if (!eq) equal_all = false;
        }

        std::swap(cur, next);

        if (!skip_output && (output_every > 0 && (iter % output_every == 0 || equal_all || iter + 1 == max_iter))) {
            std::vector<uint8_t> proj(W * H, 0);
            for (int y = 1; y <= H; ++y) {
                for (int x = 1; x <= W; ++x) {
                    int best_id = 0;
                    int max_strength = -1;
                    for (int z = 1; z <= D; ++z) {
                        size_t pidx = z * pH * pW + y * pW + x;
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
                    }
                    proj[(y - 1) * W + (x - 1)] = best_id;
                }
            }
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            write_ppm_color(path, proj, W, H);

            std::string sdir = "frames_3d";
            mkdir(sdir.c_str(), 0755);
            for (int z = 1; z <= D; ++z) {
                std::vector<uint8_t> slice(W * H, 0);
                for (int y = 1; y <= H; ++y) {
                    for (int x = 1; x <= W; ++x) {
                        size_t pidx = z * pH * pW + y * pW + x;
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
                        slice[(y - 1) * W + (x - 1)] = best_id;
                    }
                }
                char spath[256];
                std::snprintf(spath, sizeof(spath), "%s/frame_%04zu_slice_%03zu.ppm", sdir.c_str(), iter, (std::size_t)z - 1);
                write_ppm_color(spath, slice, W, H);
            }
            std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
        }

        if (equal_all) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "3D OpenMP multi-species (CSV rules) run time: " << dt.count() << " s" << std::endl;
}
