#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <cstdint>

// A single condition: SpeciesID -> Required Count
// e.g. (1, 5) means 5 neighbors of species 1
using Condition = std::map<int, int>;

// A rule is a set of conditions (OR logic between sets, AND logic within a set)
// e.g. {(1,5)} {(2,2),(3,1)} means (5 of Sp1) OR (2 of Sp2 AND 1 of Sp3)
using Rule = std::vector<Condition>;

struct Species {
    int id;
    std::string name;
    Rule survival_rules;
    Rule birth_rules;
    int strength;
    std::string image_path;
    
    // Visual properties (placeholder for now)
    uint8_t r, g, b; 
};

class SpeciesManager {
public:
    static SpeciesManager& instance();

    void load_species(const std::string& csv_path);
    const Species* get_species(int id) const;
    const std::unordered_map<int, Species>& get_all_species() const;

    // Helper to parse rule string like "{(1,5)}{(2,7),(3,1)}"
    static Rule parse_rule(const std::string& rule_str);

private:
    SpeciesManager() = default;
    std::unordered_map<int, Species> species_map_;
};
