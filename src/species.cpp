#include "species.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <regex>

SpeciesManager& SpeciesManager::instance() {
    static SpeciesManager instance;
    return instance;
}

const Species* SpeciesManager::get_species(int id) const {
    auto it = species_map_.find(id);
    if (it != species_map_.end()) {
        return &it->second;
    }
    return nullptr;
}

const std::unordered_map<int, Species>& SpeciesManager::get_all_species() const {
    return species_map_;
}

// Helper to trim whitespace
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Parse string like "{(1,5)}{(2,7),(3,1)}"
Rule SpeciesManager::parse_rule(const std::string& rule_str) {
    Rule rule;
    std::string s = trim(rule_str);
    if (s.empty()) return rule;

    // Regex to find content inside {}
    // We will manually parse to handle nested structures if needed, but here it's simple {(...)}
    // Let's iterate through the string
    
    size_t pos = 0;
    while (pos < s.length()) {
        size_t open_brace = s.find('{', pos);
        if (open_brace == std::string::npos) break;
        size_t close_brace = s.find('}', open_brace);
        if (close_brace == std::string::npos) break; // Error or incomplete

        std::string content = s.substr(open_brace + 1, close_brace - open_brace - 1);
        
        // Content is like "(1,5),(2,7)"
        Condition condition;
        
        size_t inner_pos = 0;
        while (inner_pos < content.length()) {
            size_t open_paren = content.find('(', inner_pos);
            if (open_paren == std::string::npos) break;
            size_t close_paren = content.find(')', open_paren);
            if (close_paren == std::string::npos) break;

            std::string pair_str = content.substr(open_paren + 1, close_paren - open_paren - 1);
            size_t comma = pair_str.find(',');
            if (comma != std::string::npos) {
                int sp_id = std::stoi(pair_str.substr(0, comma));
                int count = std::stoi(pair_str.substr(comma + 1));
                condition[sp_id] = count;
            }
            inner_pos = close_paren + 1;
        }
        
        if (!condition.empty()) {
            rule.push_back(condition);
        }

        pos = close_brace + 1;
    }
    return rule;
}

void SpeciesManager::load_species(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open species CSV: " << csv_path << std::endl;
        return;
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> parts;

        // Simple CSV split (doesn't handle quoted commas well, but our format is simple)
        // Actually, our rules contain commas, so we need to be careful.
        // Format: ID,Name,Survival,Birth,Strength,Image
        // Survival/Birth are like "{(1,5)}" which has commas.
        // We can assume the "{(...)}" structure is quoted or we parse carefully.
        // Let's try to parse manually respecting quotes if present, or just finding the columns by context.
        // Given the example: 1,Bulbasaur,{(1,5)},{(1,12)},10,
        // The rules are NOT quoted in the example provided by user.
        // BUT the rules contain commas inside ().
        // Strategy: The top-level commas separate fields. The commas inside () are protected by ().
        
        int paren_depth = 0;
        int brace_depth = 0;
        std::string current_field;
        for (char c : line) {
            if (c == '{') brace_depth++;
            else if (c == '}') brace_depth--;
            else if (c == '(') paren_depth++;
            else if (c == ')') paren_depth--;
            
            if (c == ',' && brace_depth == 0 && paren_depth == 0) {
                parts.push_back(current_field);
                current_field.clear();
            } else {
                current_field += c;
            }
        }
        parts.push_back(current_field);

        if (parts.size() < 5) continue;

        Species sp;
        sp.id = std::stoi(trim(parts[0]));
        sp.name = trim(parts[1]);
        sp.survival_rules = parse_rule(parts[2]);
        sp.birth_rules = parse_rule(parts[3]);
        sp.strength = std::stoi(trim(parts[4]));
        if (parts.size() > 5) sp.image_path = trim(parts[5]);

        // Generate a color based on ID (hash)
        // Just some distinct colors for 1, 2, 3
        if (sp.id == 1) { sp.r = 0; sp.g = 255; sp.b = 0; } // Green (Bulbasaur)
        else if (sp.id == 2) { sp.r = 255; sp.g = 100; sp.b = 0; } // Orange (Ponyta)
        else if (sp.id == 3) { sp.r = 255; sp.g = 192; sp.b = 203; } // Pink (Sylveon)
        else {
            sp.r = (sp.id * 123) % 255;
            sp.g = (sp.id * 456) % 255;
            sp.b = (sp.id * 789) % 255;
        }

        species_map_[sp.id] = sp;
        std::cout << "Loaded Species: " << sp.name << " (ID=" << sp.id << ")" << std::endl;
    }
}
