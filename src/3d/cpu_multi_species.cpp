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

static inline std::size_t idx3(std::size_t x, std::size_t y, std::size_t z, std::size_t W, std::size_t H) {
    return z * (W * H) + y * W + x;
}

// Helper to check if a rule is satisfied given neighbor counts
static bool check_rule(const Rule& rule, const std::map<int, int>& neighbor_counts) {
    // Rule is OR of Conditions
    for (const auto& condition : rule) {
        // Condition is AND of (SpeciesID, Count)
        bool condition_met = true;
        for (const auto& req : condition) {
            int sp_id = req.first;
            int req_count = req.second;
            
            auto it = neighbor_counts.find(sp_id);
            int actual_count = (it != neighbor_counts.end()) ? it->second : 0;
            
            if (actual_count < req_count) {
                condition_met = false;
                break;
            }
        }
        if (condition_met) return true; // One condition set satisfied is enough
    }
    return false;
}

void run_3d_cpu_multi_species(const SimulationOptions3D &options) {
    // Load species config
    SpeciesManager::instance().load_species("species/species.csv");
    const auto& all_species = SpeciesManager::instance().get_all_species();

    const std::size_t W = options.width;
    const std::size_t H = options.height;
    const std::size_t D = options.depth;
    const std::size_t N = W * H * D;

    // Grid stores Species ID (0 = empty)
    std::vector<uint8_t> cur(N, 0);
    std::vector<uint8_t> next(N, 0);

    std::mt19937_64 rng(options.seed);
    // Random initialization with available species
    // We'll give each species equal probability for now, or use density
    // For simplicity: density is total density of life. If alive, pick random species.
    std::bernoulli_distribution bd(options.initial_density);
    std::uniform_int_distribution<int> sp_dist(0, all_species.size() - 1);
    std::vector<int> species_ids;
    for(auto& kv : all_species) species_ids.push_back(kv.first);

    if (species_ids.empty()) {
        std::cerr << "No species loaded!" << std::endl;
        return;
    }

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
                        if (sp && check_rule(sp->survival_rules, neighbor_counts)) {
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

                        if (check_rule(kv.second.birth_rules, neighbor_counts)) {
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

        cur.swap(next);
        
        // Output logic (simplified for multi-species, maybe just print stats)
        if (!skip_output && (iter % output_every == 0)) {
             std::map<int, int> counts;
             for(auto v : cur) if(v!=0) counts[v]++;
             std::cout << "Iter " << iter << ": ";
             for(auto kv : counts) std::cout << "Sp" << kv.first << "=" << kv.second << " ";
             std::cout << std::endl;
        }
    }
    
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "3D CPU Multi-Species run time: " << dt.count() << " s" << std::endl;
}
