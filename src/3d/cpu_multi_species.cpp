#include "dispatcher_3d.hpp"
#include "species.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#include <cstdio>
#include <chrono>
#include <map>
#include <string>

static inline std::size_t idx3(std::size_t x, std::size_t y, std::size_t z, std::size_t W, std::size_t H) {
    return z * (W * H) + y * W + x;
}
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
// Helper to check if a rule is satisfied given neighbor counts
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

void run_3d_cpu_multi_species(const SimulationOptions3D &options) {
    // Load species config
    SpeciesManager::instance().load_species("species/species.csv");
    const auto& all_species = SpeciesManager::instance().get_all_species();
    if (all_species.empty()) {
        std::cerr << "Error: No species loaded from CSV!" << std::endl;
        return;
    }
    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t D = options.depth;
    const std::size_t N = W * H * D;

    // Grid stores Species ID (0 = empty)
    std::vector<uint8_t> cur(N, 0);
    std::vector<uint8_t> next(N, 0);

    std::mt19937_64 rng(options.seed);
    // Random initialization with available species
    // Use initial_density to control how many cells start alive
    // If alive, pick random species from available species
    std::bernoulli_distribution bd(options.initial_density);
    std::vector<int> species_ids;
    for(auto& kv : all_species) species_ids.push_back(kv.first);
    
    if (species_ids.empty()) {
        std::cerr << "No species IDs available!" << std::endl;
        return;
    }
    
    std::uniform_int_distribution<int> sp_dist(0, species_ids.size() - 1);

    for (std::size_t i = 0; i < N; ++i) {
        if (bd(rng)) {
            cur[i] = species_ids[sp_dist(rng)];
        } else {
            cur[i] = 0;
        }
    }

    const std::size_t max_iter = options.iterations > 0 ? options.iterations : 1000;
    const std::size_t output_every = options.output_every > 0 ? options.output_every : 1;
    bool skip_output = (std::getenv("SKIP_OUTPUT") != nullptr);
    const char *outdir = "frames_multi";
    mkdir(outdir, 0755);
    
    std::string sdir = "frames_multi_3d";
    mkdir(sdir.c_str(), 0755);


    auto t0 = std::chrono::steady_clock::now();

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // Parallelize over Z slices
        #pragma omp parallel for
        for (long z = 0; z < (long)D; ++z) {
            for (long y = 0; y < (long)H; ++y) {
                for (long x = 0; x < (long)W; ++x) {
                    // 1. Count neighbors per species
                    std::map<int, int> neighbor_counts;
                    
                    for (int dz = -1; dz <= 1; ++dz) {
                        long zl = z + dz;
                        if (zl < 0) zl += D; else if (zl >= (long)D) zl -= D;
                        
                        for (int dy = -1; dy <= 1; ++dy) {
                            long yl = y + dy;
                            if (yl < 0) yl += H; else if (yl >= (long)H) yl -= H;
                            
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                
                                long xl = x + dx;
                                if (xl < 0) xl += W; else if (xl >= (long)W) xl -= W;
                                
                                uint8_t neighbor_id = cur[idx3(xl, yl, zl, W, H)];
                                if (neighbor_id != 0) {
                                    neighbor_counts[neighbor_id]++;
                                }
                            }
                        }
                    }

                    uint8_t current_id = cur[idx3(x, y, z, W, H)];
                    std::vector<int> candidates;

                    // 2. Check Survival (if occupied)
                    if (current_id != 0) {
                        const Species* sp = SpeciesManager::instance().get_species(current_id);
                        if (sp && check_rule_multi(sp->survival_rules, neighbor_counts)) {
                            candidates.push_back(current_id);
                        }
                    }

                    // 3. Check Birth (for all species)
                    // Optimization: Only check species that are present in neighbors? 
                    // Or just check all defined species? Checking all is safer for "spontaneous generation" rules if any (though usually requires neighbors).
                    // Given the rules "5 neighbors of A", we usually need neighbors.
                    // But let's check all species to be correct.
                    for (const auto& kv : all_species) {
                        int sp_id = kv.first;
                        // If it's the current species, we already checked survival. 
                        // Does birth apply to occupied cells? Usually birth is for empty cells.
                        // But if a stronger species can displace a weaker one?
                        // The prompt says: "Comparison of species strength to decide who stays/spawns".
                        // This implies multiple species might want to be here.
                        // If I am already here (and survived), do I compete with new births?
                        // Usually Game of Life is: Survival keeps you here. Birth puts you here if empty.
                        // But "Strength" implies conflict.
                        // Let's assume:
                        // Candidates = [Survivor (if any)] + [New Births (if any)]
                        // Winner = Max Strength among Candidates.
                        
                        if (sp_id == current_id) continue; // Already handled by survival

                        if (check_rule_multi(kv.second.birth_rules, neighbor_counts)) {
                            candidates.push_back(sp_id);
                        }
                    }

                    // 4. Resolve Conflict
                    if (candidates.empty()) {
                        next[idx3(x, y, z, W, H)] = 0;
                    } else if (candidates.size() == 1) {
                        next[idx3(x, y, z, W, H)] = candidates[0];
                    } else {
                        // Compare strength
                        int best_id = 0;
                        int max_str = -1;
                        for (int cand_id : candidates) {
                            const Species* sp = SpeciesManager::instance().get_species(cand_id);
                            if (sp) {
                                if (sp->strength > max_str) {
                                    max_str = sp->strength;
                                    best_id = cand_id;
                                } else if (sp->strength == max_str) {
                                    // Tie-break? ID? Random?
                                    // Let's use ID for determinism
                                    if (cand_id > best_id) best_id = cand_id;
                                }
                            }
                        }
                        next[idx3(x, y, z, W, H)] = best_id;
                    }
                }
            }
        }

        bool equal = (std::memcmp(cur.data(), next.data(), N * sizeof(uint8_t)) == 0);
        cur.swap(next);
        
        // Output logic
        if (!skip_output && (iter % output_every == 0 || equal || iter + 1 == max_iter)) {
            // Print statistics
            // std::map<int, int> counts;
            // for(auto v : cur) if(v!=0) counts[v]++;
            // std::cout << "Iter " << iter << ": ";
            // for(auto kv : counts) std::cout << "Sp" << kv.first << "=" << kv.second << " ";
            // std::cout << std::endl;

            // Produce a simple projection (max over depth) for visualization
            std::vector<int> proj(W * H, 0);
            for (std::size_t zz = 0; zz < D; ++zz) {
                for (std::size_t i = 0; i < W * H; ++i) {
                    int id = cur[zz * W * H + i];
                    if (id > 0) {
                        // Keep the highest ID (or could use max strength)
                        if (proj[i] == 0 || id > proj[i]) {
                            proj[i] = id;
                        }
                    }
                }
            }
            char path[256];
            std::snprintf(path, sizeof(path), "%s/frame_%04zu.ppm", outdir, iter);
            write_ppm_color(path, proj, W, H);

            // Also write per-slice images into frames_multi_3d for 3D viewer
            for (std::size_t zz = 0; zz < D; ++zz) {
                std::vector<int> slice(W * H);
                for (std::size_t i = 0; i < W * H; ++i) {
                    slice[i] = cur[zz * W * H + i];
                }
                char spath[256];
                std::snprintf(spath, sizeof(spath), "%s/frame_%04zu_slice_%03zu.ppm", sdir.c_str(), iter, zz);
                if (iter % 100 == 0 || zz == 0) {
                    std::printf("Writing slice: %s\n", spath);
                }
                write_ppm_color(spath, slice, W, H);
            }
            if (iter % 100 == 0 || equal || iter + 1 == max_iter) {
                std::cout << "Wrote " << path << " (iter=" << iter << ")" << std::endl;
            }
        }

        if (equal) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }
    
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "3D CPU Multi-Species run time: " << dt.count() << " s" << std::endl;
}
