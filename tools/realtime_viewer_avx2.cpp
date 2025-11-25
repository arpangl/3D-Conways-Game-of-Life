#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <thread>
#include <map>
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <sstream>
#include <fstream>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "species.hpp"

// ==========================================
// AVX2 計算相關結構和函數（從 avx2_multi_species.cpp）
// ==========================================

struct AVX2Condition {
    int species_id; // CSV 中的 ID
    int count;      // 所需數量
};

struct AVX2RuleGroup {
    std::vector<AVX2Condition> conditions;
};

struct SpeciesData {
    int id;
    int index;
    std::string name;
    int strength;
    std::string image_path;
    std::vector<AVX2RuleGroup> survival_rules;
    std::vector<AVX2RuleGroup> birth_rules;
};

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

std::vector<AVX2RuleGroup> parse_rule_string(const std::string& raw) {
    std::vector<AVX2RuleGroup> groups;
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
            AVX2RuleGroup rg;
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
                            AVX2Condition cond;
                            cond.species_id = sp_id;
                            cond.count = cnt;
                            rg.conditions.push_back(cond);
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

std::vector<SpeciesData> load_species_rules(const std::string& filename) {
    std::vector<SpeciesData> species_list;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return species_list;
    }

    std::string line;
    if(std::getline(file, line)) {} // Skip header

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

#if defined(__GNUC__)
__attribute__((target("avx2")))
#endif
void count_neighbors_avx2(const uint8_t* __restrict src, uint8_t* __restrict dst_counts, 
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

void update_ghost_cells_generic(uint8_t* grid, int W, int H, int D, int pW, int pH, int pD) {
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

// ==========================================
// 渲染和互動相關（從 realtime_viewer_v2.cpp）
// ==========================================

struct Camera {
    float x = 32.0f, y = 32.0f, z = 100.0f;
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 2.0f;
    float sensitivity = 0.2f;
};

int W = 128, H = 128, D = 128;
int INIT_W = 32, INIT_H = 32, INIT_D = 32;
int pW, pH, pD; // Padded dimensions
size_t pN; // Padded grid size

// AVX2 計算用的多層 grid
std::vector<uint8_t> cur_multi;  // S * pN
std::vector<uint8_t> next_multi; // S * pN
std::vector<uint8_t> neighbor_counts; // S * pN

// 渲染用的單一 ID grid
std::vector<uint8_t> grid; // W * H * D, stores species ID

std::vector<SpeciesData> species_list;
std::map<int, int> id_to_index; // CSV ID -> internal index
std::map<int, int> index_to_id; // internal index -> CSV ID

bool paused = false;
bool show_grid = true;
int target_fps = 60;
int selected_species_id = 1;
Camera cam;
bool mouse_captured = false;
std::map<int, GLuint> species_textures;

int min_x, min_y, min_z;
int max_x, max_y, max_z;

void reset_bounds() {
    min_x = W; min_y = H; min_z = D;
    max_x = -1; max_y = -1; max_z = -1;
}

void update_bounds(int x, int y, int z) {
    if (x < min_x) min_x = x;
    if (x > max_x) max_x = x;
    if (y < min_y) min_y = y;
    if (y > max_y) max_y = y;
    if (z < min_z) min_z = z;
    if (z > max_z) max_z = z;
}

void get_camera_vectors(float& fx, float& fy, float& fz, float& rx, float& ry, float& rz) {
    float radYaw = cam.yaw * M_PI / 180.0f;
    float radPitch = cam.pitch * M_PI / 180.0f;
    
    fx = cos(radYaw) * cos(radPitch);
    fy = sin(radPitch);
    fz = sin(radYaw) * cos(radPitch);
    
    rx = -fz; 
    ry = 0;
    rz = fx;
    
    float r_len = sqrt(rx*rx + rz*rz);
    if (r_len > 0.001f) {
        rx /= r_len;
        rz /= r_len;
    }
}

// 同步：從多層 grid 更新單一 ID grid
void sync_grids() {
    int S = species_list.size();
    std::fill(grid.begin(), grid.end(), 0);
    
    for (int z = 1; z <= D; ++z) {
        for (int y = 1; y <= H; ++y) {
            for (int x = 1; x <= W; ++x) {
                size_t pidx = z * pH * pW + y * pW + x;
                size_t idx = (z-1) * (W * H) + (y-1) * W + (x-1);
                
                // Find which species is alive at this position
                int best_id = 0;
                int max_strength = -1;
                
                for (int s = 0; s < S; ++s) {
                    if (cur_multi[s * pN + pidx]) {
                        const auto& sp = species_list[s];
                        if (sp.strength > max_strength) {
                            max_strength = sp.strength;
                            best_id = sp.id;
                        } else if (sp.strength == max_strength && sp.id > best_id) {
                            best_id = sp.id;
                        }
                    }
                }
                grid[idx] = best_id;
            }
        }
    }
}

// 同步：從單一 ID grid 更新多層 grid（用於互動編輯）
void sync_to_multi() {
    int S = species_list.size();
    std::fill(cur_multi.begin(), cur_multi.end(), 0);
    
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                size_t idx = z * (W * H) + y * W + x;
                uint8_t id = grid[idx];
                if (id != 0 && id_to_index.count(id)) {
                    int s = id_to_index[id];
                    size_t pidx = (z+1) * pH * pW + (y+1) * pW + (x+1);
                    cur_multi[s * pN + pidx] = 1;
                }
            }
        }
    }
    
    // Update ghost cells
    for (int s = 0; s < S; ++s) {
        update_ghost_cells_generic(&cur_multi[s * pN], W, H, D, pW, pH, pD);
    }
}

// AVX2 加速的模擬步驟
void simulation_step_avx2() {
    int S = species_list.size();
    
    int start_z = std::max(1, min_z);
    int end_z = std::min(D, max_z + 1);
    int start_y = std::max(1, min_y);
    int end_y = std::min(H, max_y + 1);
    int start_x = std::max(1, min_x);
    int end_x = std::min(W, max_x + 1);

    std::vector<int> offsets;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (dx|dy|dz) offsets.push_back(dz * pH * pW + dy * pW + dx);

    int next_min_x = W, next_min_y = H, next_min_z = D;
    int next_max_x = -1, next_max_y = -1, next_max_z = -1;

    // Step 1: Count neighbors (AVX2)
    #pragma omp parallel for
    for (int s = 0; s < S; ++s) {
        uint8_t* sp_grid = &cur_multi[s * pN];
        uint8_t* sp_counts = &neighbor_counts[s * pN];
        
        update_ghost_cells_generic(sp_grid, W, H, D, pW, pH, pD);
        
        for (int z = start_z; z <= end_z; ++z) {
            for (int y = start_y; y <= end_y; ++y) {
                size_t idx = z * pH * pW + y * pW + start_x;
                count_neighbors_avx2(sp_grid + idx, sp_counts + idx, end_x - start_x + 1, offsets);
            }
        }
    }

    // Step 2: Resolve rules & competition
    #pragma omp parallel
    {
        int t_min_x = W, t_min_y = H, t_min_z = D;
        int t_max_x = -1, t_max_y = -1, t_max_z = -1;

        #pragma omp for collapse(2) nowait
        for (int z = start_z; z <= end_z; ++z) {
            for (int y = start_y; y <= end_y; ++y) {
                for (int x = start_x; x <= end_x; ++x) {
                    size_t pidx = z * pH * pW + y * pW + x;
                    
                    int best_species = -1;
                    int max_strength = -1;
                    int current_occupant = -1;
                    
                    uint8_t local_n_counts[16];
                    bool current_alive[16];
                    
                    for(int s = 0; s < S && s < 16; ++s) {
                        local_n_counts[s] = neighbor_counts[s * pN + pidx];
                        current_alive[s] = cur_multi[s * pN + pidx];
                        if (current_alive[s]) current_occupant = s;
                        next_multi[s * pN + pidx] = 0;
                    }

                    for (int s = 0; s < S; ++s) {
                        const auto& sp = species_list[s];
                        bool wants_to_live = false;
                        
                        const std::vector<AVX2RuleGroup>* rules = (current_alive[s]) ? &sp.survival_rules : &sp.birth_rules;

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
                        next_multi[best_species * pN + pidx] = 1;
                        
                        int gx = x - 1, gy = y - 1, gz = z - 1;
                        if (gx < t_min_x) t_min_x = gx;
                        if (gx > t_max_x) t_max_x = gx;
                        if (gy < t_min_y) t_min_y = gy;
                        if (gy > t_max_y) t_max_y = gy;
                        if (gz < t_min_z) t_min_z = gz;
                        if (gz > t_max_z) t_max_z = gz;
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            if (t_min_x < next_min_x) next_min_x = t_min_x;
            if (t_max_x > next_max_x) next_max_x = t_max_x;
            if (t_min_y < next_min_y) next_min_y = t_min_y;
            if (t_max_y > next_max_y) next_max_y = t_max_y;
            if (t_min_z < next_min_z) next_min_z = t_min_z;
            if (t_max_z > next_max_z) next_max_z = t_max_z;
        }
    }
    
    min_x = next_min_x; max_x = next_max_x;
    min_y = next_min_y; max_y = next_max_y;
    min_z = next_min_z; max_z = next_max_z;

    if (min_x > max_x) {
        min_x = W/2; max_x = W/2;
        min_y = H/2; max_y = H/2;
        min_z = D/2; max_z = D/2;
    }

    std::swap(cur_multi, next_multi);
    sync_grids();
}

GLuint load_texture(const std::string& path) {
    int w, h, comp;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << path << std::endl;
        return 0;
    }
    
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    int mode = GL_RGB;
    if (comp == 4) mode = GL_RGBA;
    
    glTexImage2D(GL_TEXTURE_2D, 0, mode, w, h, 0, mode, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    stbi_image_free(data);
    return texture;
}

void init_gl_resources() {
    for (const auto& sp : species_list) {
        if (!sp.image_path.empty()) {
            GLuint tex = load_texture(sp.image_path);
            if (tex != 0) {
                species_textures[sp.id] = tex;
            }
        }
    }
}

void render_full_grid() {
    if (!show_grid) return;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glColor4f(0.5f, 0.5f, 0.5f, 0.3f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0); glVertex3f(0, H, 0);
    glVertex3f(W, 0, 0); glVertex3f(W, H, 0);
    glVertex3f(W, 0, D); glVertex3f(W, H, D);
    glVertex3f(0, 0, D); glVertex3f(0, H, D);
    glVertex3f(0, 0, 0); glVertex3f(W, 0, 0);
    glVertex3f(W, 0, 0); glVertex3f(W, 0, D);
    glVertex3f(W, 0, D); glVertex3f(0, 0, D);
    glVertex3f(0, 0, D); glVertex3f(0, 0, 0);
    glVertex3f(0, H, 0); glVertex3f(W, H, 0);
    glVertex3f(W, H, 0); glVertex3f(W, H, D);
    glVertex3f(W, H, D); glVertex3f(0, H, D);
    glVertex3f(0, H, D); glVertex3f(0, H, 0);
    glEnd();

    int margin = 2;
    int sx = std::max(0, min_x - margin);
    int ex = std::min(W, max_x + margin + 1);
    int sy = std::max(0, min_y - margin);
    int ey = std::min(H, max_y + margin + 1);
    int sz = std::max(0, min_z - margin);
    int ez = std::min(D, max_z + margin + 1);

    glColor4f(0.3f, 0.3f, 0.3f, 0.2f);
    glBegin(GL_LINES);
    for (int x = sx; x <= ex; ++x) { for (int z = sz; z <= ez; ++z) { glVertex3f(x, sy, z); glVertex3f(x, ey, z); } }
    for (int y = sy; y <= ey; ++y) { for (int z = sz; z <= ez; ++z) { glVertex3f(sx, y, z); glVertex3f(ex, y, z); } }
    for (int x = sx; x <= ex; ++x) { for (int y = sy; y <= ey; ++y) { glVertex3f(x, y, sz); glVertex3f(x, y, ez); } }
    glEnd();
    
    glColor4f(1.0f, 0.2f, 0.2f, 0.3f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glVertex3f(min_x, min_y, min_z); glVertex3f(max_x, min_y, min_z);
    glVertex3f(min_x, min_y, min_z); glVertex3f(min_x, max_y, min_z);
    glVertex3f(min_x, min_y, min_z); glVertex3f(min_x, min_y, max_z);
    glVertex3f(max_x, max_y, max_z); glVertex3f(min_x, max_y, max_z);
    glVertex3f(max_x, max_y, max_z); glVertex3f(max_x, min_y, max_z);
    glVertex3f(max_x, max_y, max_z); glVertex3f(max_x, max_y, min_z);
    glEnd();
    
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void render_billboard_circle_optimized(float x, float y, float z, float size, uint8_t r, uint8_t g, uint8_t b, const float* right, const float* up, GLuint texture_id) {
    float s = size / 2.0f;
    
    if (texture_id != 0) {
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER, 0.1f);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glColor3f(1, 1, 1);
        
        glBegin(GL_QUADS);
        glTexCoord2f(0, 1);
        glVertex3f(x - right[0]*s - up[0]*s, y - right[1]*s - up[1]*s, z - right[2]*s - up[2]*s);
        glTexCoord2f(1, 1);
        glVertex3f(x + right[0]*s - up[0]*s, y + right[1]*s - up[1]*s, z + right[2]*s - up[2]*s);
        glTexCoord2f(1, 0);
        glVertex3f(x + right[0]*s + up[0]*s, y + right[1]*s + up[1]*s, z + right[2]*s + up[2]*s);
        glTexCoord2f(0, 0);
        glVertex3f(x - right[0]*s + up[0]*s, y - right[1]*s + up[1]*s, z - right[2]*s + up[2]*s);
        glEnd();
        
        glDisable(GL_ALPHA_TEST);
        glDisable(GL_TEXTURE_2D);
    } else {
        glColor3ub(r, g, b);
        glBegin(GL_TRIANGLE_FAN);
        glVertex3f(x, y, z);
        int segments = 12;
        for (int i = 0; i <= segments; ++i) {
            float theta = 2.0f * M_PI * float(i) / float(segments);
            float dx = cos(theta) * s;
            float dy = sin(theta) * s;
            glVertex3f(x + right[0] * dx + up[0] * dy, y + right[1] * dx + up[1] * dy, z + right[2] * dx + up[2] * dy);
        }
        glEnd();
    }
}

void render_scene() {
    float modelview[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
    float right[3] = { modelview[0], modelview[4], modelview[8] };
    float up[3]    = { modelview[1], modelview[5], modelview[9] };

    int start_z = std::max(0, min_z);
    int end_z = std::min(D - 1, max_z);
    int start_y = std::max(0, min_y);
    int end_y = std::min(H - 1, max_y);
    int start_x = std::max(0, min_x);
    int end_x = std::min(W - 1, max_x);

    for (int z = start_z; z <= end_z; ++z) {
        for (int y = start_y; y <= end_y; ++y) {
            for (int x = start_x; x <= end_x; ++x) {
                uint8_t id = grid[z * (W * H) + y * W + x];
                if (id != 0) {
                    // Find species data
                    const SpeciesData* sp = nullptr;
                    for (const auto& s : species_list) {
                        if (s.id == id) {
                            sp = &s;
                            break;
                        }
                    }
                    if (sp) {
                        // Get color from SpeciesManager or use default
                        const Species* sm_sp = SpeciesManager::instance().get_species(id);
                        uint8_t r = sm_sp ? sm_sp->r : 255;
                        uint8_t g = sm_sp ? sm_sp->g : 255;
                        uint8_t b = sm_sp ? sm_sp->b : 255;
                        
                        GLuint tex = 0;
                        if (species_textures.count(id)) tex = species_textures[id];
                        render_billboard_circle_optimized(x + 0.5f, y + 0.5f, z + 0.5f, 0.8f, r, g, b, right, up, tex);
                    }
                }
            }
        }
    }
}

void render_hud(int width, int height) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    const Species* sp = SpeciesManager::instance().get_species(selected_species_id);
    if (sp) {
        GLuint tex = 0;
        if (species_textures.count(selected_species_id)) tex = species_textures[selected_species_id];
        
        if (tex != 0) {
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, tex);
            glColor3f(1, 1, 1);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(10, 10);
            glTexCoord2f(1, 0); glVertex2f(60, 10);
            glTexCoord2f(1, 1); glVertex2f(60, 60);
            glTexCoord2f(0, 1); glVertex2f(10, 60);
            glEnd();
            glDisable(GL_TEXTURE_2D);
        } else {
            glColor3ub(sp->r, sp->g, sp->b);
            glBegin(GL_QUADS);
            glVertex2f(10, 10);
            glVertex2f(60, 10);
            glVertex2f(60, 60);
            glVertex2f(10, 60);
            glEnd();
        }
    }
    
    glColor3f(1, 1, 1);
    glBegin(GL_LINE_LOOP);
    glVertex2f(10, 10);
    glVertex2f(60, 10);
    glVertex2f(60, 60);
    glVertex2f(10, 60);
    glEnd();

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

bool get_target_block(int /*mx*/, int /*my*/, int width, int height, bool is_place, int& out_x, int& out_y, int& out_z) {
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);

    float winX = width / 2.0f;
    float winY = height / 2.0f; 

    GLdouble nx = 0, ny = 0, nz = 0, fx = 0, fy = 0, fz = 0;
    if (gluUnProject(winX, winY, 0.0, model, proj, view, &nx, &ny, &nz) == GL_FALSE) return false;
    if (gluUnProject(winX, winY, 1.0, model, proj, view, &fx, &fy, &fz) == GL_FALSE) return false;

    double dx = fx - nx;
    double dy = fy - ny;
    double dz = fz - nz;
    double len = sqrt(dx*dx + dy*dy + dz*dz);
    if (len < 0.0001) return false;
    dx /= len; dy /= len; dz /= len;

    double cx = nx, cy = ny, cz = nz;
    int last_gx = -1, last_gy = -1, last_gz = -1;
    
    for (float t = 0; t < 20.0f; t += 0.1f) {
        cx = nx + dx * t;
        cy = ny + dy * t;
        cz = nz + dz * t;

        int gx = (int)floor(cx);
        int gy = (int)floor(cy);
        int gz = (int)floor(cz);

        if (gx >= 0 && gx < W && gy >= 0 && gy < H && gz >= 0 && gz < D) {
            if (grid[gz * (W * H) + gy * W + gx] != 0) {
                if (is_place) {
                    if (last_gx != -1) {
                        out_x = last_gx; out_y = last_gy; out_z = last_gz;
                        return true;
                    } else {
                        return false; 
                    }
                } else {
                    out_x = gx; out_y = gy; out_z = gz;
                    return true;
                }
            }
        }
        last_gx = gx; last_gy = gy; last_gz = gz;
    }
    
    float fixed_dist = 10.0f;
    cx = nx + dx * fixed_dist;
    cy = ny + dy * fixed_dist;
    cz = nz + dz * fixed_dist;
    
    if (std::isnan(cx) || std::isnan(cy) || std::isnan(cz)) {
        return false;
    }

    out_x = (int)floor(cx);
    out_y = (int)floor(cy);
    out_z = (int)floor(cz);
    
    if (out_x < 0) out_x = 0; if (out_x >= W) out_x = W-1;
    if (out_y < 0) out_y = 0; if (out_y >= H) out_y = H-1;
    if (out_z < 0) out_z = 0; if (out_z >= D) out_z = D-1;
    
    return false;
}

void render_crosshair(int width, int height) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    glColor3f(1, 1, 1);
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float size = 10.0f;

    glBegin(GL_LINES);
    glVertex2f(cx - size, cy); glVertex2f(cx + size, cy);
    glVertex2f(cx, cy - size); glVertex2f(cx, cy + size);
    glEnd();

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void render_preview(int x, int y, int z) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glColor4f(1, 1, 1, 0.5f);
    glLineWidth(2.0f);
    glPushMatrix();
    glTranslatef(x, y, z);
    glScalef(1.01f, 1.01f, 1.01f);
    
    glBegin(GL_LINE_LOOP); glVertex3f(0,0,0); glVertex3f(1,0,0); glVertex3f(1,1,0); glVertex3f(0,1,0); glEnd();
    glBegin(GL_LINE_LOOP); glVertex3f(0,0,1); glVertex3f(1,0,1); glVertex3f(1,1,1); glVertex3f(0,1,1); glEnd();
    glBegin(GL_LINES); 
    glVertex3f(0,0,0); glVertex3f(0,0,1);
    glVertex3f(1,0,0); glVertex3f(1,0,1);
    glVertex3f(1,1,0); glVertex3f(1,1,1);
    glVertex3f(0,1,0); glVertex3f(0,1,1);
    glEnd();
    
    glPopMatrix();
    glLineWidth(1.0f);
    glDisable(GL_BLEND);
}

void handle_click(int mx, int my, int width, int height, bool is_place) {
    int hit_x, hit_y, hit_z;
    bool hit = get_target_block(mx, my, width, height, is_place, hit_x, hit_y, hit_z);
    
    if (hit_x < 0 || hit_x >= W || hit_y < 0 || hit_y >= H || hit_z < 0 || hit_z >= D) {
        return;
    }

    if (is_place) {
        int idx = hit_z * (W * H) + hit_y * W + hit_x;
        if (idx < 0 || idx >= (int)grid.size()) {
            return;
        }
        grid[idx] = selected_species_id;
        update_bounds(hit_x, hit_y, hit_z);
        sync_to_multi();
    } else {
        if (hit) {
            int idx = hit_z * (W * H) + hit_y * W + hit_x;
            if (idx >= 0 && idx < (int)grid.size()) {
                grid[idx] = 0;
                sync_to_multi();
            }
        }
    }
}

int target_sim_fps = 10;

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fps" && i + 1 < argc) {
            target_sim_fps = std::stoi(argv[++i]);
        }
    }
    std::cout << "Target Simulation FPS: " << target_sim_fps << std::endl;

    // Load species from CSV
    std::string csv_path = "species/species.csv";
    species_list = load_species_rules(csv_path);
    
    if (species_list.empty()) {
        std::cerr << "No species rules found in " << csv_path << ". Exiting." << std::endl;
        return 1;
    }
    
    // Also load into SpeciesManager for rendering compatibility
    SpeciesManager::instance().load_species(csv_path);
    
    // Build ID mappings
    for (const auto& sp : species_list) {
        id_to_index[sp.id] = sp.index;
        index_to_id[sp.index] = sp.id;
    }
    
    // Convert rules to use internal indices
    for (auto& sp : species_list) {
        auto convert_rules = [&](std::vector<AVX2RuleGroup>& groups) {
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
    
    std::cout << "Loaded " << species_list.size() << " species from CSV." << std::endl;

    // Setup grids
    pW = W + 2; pH = H + 2; pD = D + 2;
    pN = (size_t)pW * pH * pD;
    int S = species_list.size();
    
    cur_multi.resize(S * pN, 0);
    next_multi.resize(S * pN, 0);
    neighbor_counts.resize(S * pN, 0);
    grid.resize(W * H * D, 0);
    
    std::cout << "Grid initialized. Size: " << grid.size() << std::endl;
    
    reset_bounds();

    // Initial spawn
    int start_x = (W - INIT_W) / 2;
    int start_y = (H - INIT_H) / 2;
    int start_z = (D - INIT_D) / 2;

    for (int z = start_z; z < start_z + INIT_D; ++z) {
        for (int y = start_y; y < start_y + INIT_H; ++y) {
            for (int x = start_x; x < start_x + INIT_W; ++x) {
                if (rand() % 10 == 0) {
                    int id = (rand() % S) + 1;
                    if (id <= (int)species_list.size()) {
                        grid[z * (W * H) + y * W + x] = species_list[id-1].id;
                        update_bounds(x, y, z);
                    }
                }
            }
        }
    }
    
    sync_to_multi();

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window* my_window = nullptr;
    my_window = SDL_CreateWindow("3D Game of Life - AVX2 Realtime", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    
    if (!my_window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_GLContext context = SDL_GL_CreateContext(my_window);
    (void)context; // Context is required for OpenGL but not directly used
    
    init_gl_resources();

    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    gluPerspective(45.0f, 1280.0f/720.0f, 0.1f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);

    cam.x = W / 2.0f; cam.y = H / 2.0f; cam.z = D * 1.5f;
    selected_species_id = species_list[0].id;

    bool running = true;
    SDL_Event event;
    
    auto last_time = std::chrono::high_resolution_clock::now();
    float frame_accum = 0.0f;
    int frames = 0;
    
    float sim_accum = 0.0f;
    float sim_step = 1.0f / target_sim_fps;

    SDL_SetRelativeMouseMode(SDL_TRUE);
    mouse_captured = true;

    while (running) {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta = current_time - last_time;
        last_time = current_time;
        float dt = delta.count();

        int mx, my;
        SDL_GetMouseState(&mx, &my);

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) running = false;
                if (event.key.keysym.sym == SDLK_t) paused = !paused;
                if (event.key.keysym.sym == SDLK_g) show_grid = !show_grid;
                if (event.key.keysym.sym == SDLK_r) {
                    // Cycle through species
                    int current_idx = -1;
                    for (size_t i = 0; i < species_list.size(); ++i) {
                        if (species_list[i].id == selected_species_id) {
                            current_idx = i;
                            break;
                        }
                    }
                    if (current_idx >= 0) {
                        selected_species_id = species_list[(current_idx + 1) % species_list.size()].id;
                    }
                }
            }
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    if (paused) {
                        int w, h;
                        SDL_GetWindowSize(my_window, &w, &h);
                        handle_click(mx, my, w, h, true); 
                    } else {
                        mouse_captured = true;
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                    }
                }
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    if (paused) {
                        int w, h;
                        SDL_GetWindowSize(my_window, &w, &h);
                        handle_click(mx, my, w, h, false); 
                    }
                }
            }
            if (event.type == SDL_MOUSEMOTION && mouse_captured) {
                int xrel = event.motion.xrel;
                int yrel = event.motion.yrel;
                if (xrel > 50) xrel = 50; if (xrel < -50) xrel = -50;
                if (yrel > 50) yrel = 50; if (yrel < -50) yrel = -50;

                cam.yaw -= xrel * cam.sensitivity;
                cam.pitch += yrel * cam.sensitivity;
                
                if (cam.pitch > 89.0f) cam.pitch = 89.0f;
                if (cam.pitch < -89.0f) cam.pitch = -89.0f;
            }
        }

        if (paused) {
            int w, h;
            SDL_GetWindowSize(my_window, &w, &h);
            int hit_x, hit_y, hit_z;
            Uint32 buttons = SDL_GetMouseState(NULL, NULL);
            
            if (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) {
                get_target_block(mx, my, w, h, true, hit_x, hit_y, hit_z);
                int idx = hit_z * (W * H) + hit_y * W + hit_x;
                if (idx >= 0 && idx < (int)grid.size()) {
                    grid[idx] = selected_species_id;
                    update_bounds(hit_x, hit_y, hit_z);
                    sync_to_multi();
                }
            }
            if (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
                if (get_target_block(mx, my, w, h, false, hit_x, hit_y, hit_z)) {
                     int idx = hit_z * (W * H) + hit_y * W + hit_x;
                     if (idx >= 0 && idx < (int)grid.size()) {
                         grid[idx] = 0;
                         sync_to_multi();
                     }
                }
            }
        }

        const Uint8* state = SDL_GetKeyboardState(NULL);
        float fx, fy, fz, rx, ry, rz;
        get_camera_vectors(fx, fy, fz, rx, ry, rz);
        
        if (state[SDL_SCANCODE_W]) { cam.x += fx * cam.speed; cam.y += fy * cam.speed; cam.z += fz * cam.speed; }
        if (state[SDL_SCANCODE_S]) { cam.x -= fx * cam.speed; cam.y -= fy * cam.speed; cam.z -= fz * cam.speed; }
        if (state[SDL_SCANCODE_A]) { cam.x -= rx * cam.speed; cam.y -= ry * cam.speed; cam.z -= rz * cam.speed; }
        if (state[SDL_SCANCODE_D]) { cam.x += rx * cam.speed; cam.y += ry * cam.speed; cam.z += rz * cam.speed; }
        if (state[SDL_SCANCODE_SPACE]) { cam.y += cam.speed; }
        if (state[SDL_SCANCODE_X]) { cam.y -= cam.speed; }

        if (!paused) {
            sim_accum += dt;
            if (sim_accum >= sim_step) {
                simulation_step_avx2();
                sim_accum = 0.0f;
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        
        float lx = cam.x + fx;
        float ly = cam.y + fy;
        float lz = cam.z + fz;
        gluLookAt(cam.x, cam.y, cam.z, lx, ly, lz, 0.0f, 1.0f, 0.0f);

        render_full_grid(); 
        render_scene(); 
        
        if (paused) {
            int w, h;
            SDL_GetWindowSize(my_window, &w, &h);
            int hit_x, hit_y, hit_z;
            get_target_block(mx, my, w, h, true, hit_x, hit_y, hit_z);
            render_preview(hit_x, hit_y, hit_z);
        }
        
        int w, h;
        SDL_GetWindowSize(my_window, &w, &h);
        render_crosshair(w, h); 
        render_hud(w, h);

        SDL_GL_SwapWindow(my_window);

        frames++;
        frame_accum += dt;
        if (frame_accum >= 1.0f) {
            char title[256];
            snprintf(title, sizeof(title), "FPS: %.2f | SimFPS: %d | Species: %d | Paused: %s | AVX2", 
                    frames / frame_accum, target_sim_fps, selected_species_id, paused ? "YES" : "NO");
            SDL_SetWindowTitle(my_window, title);
            frames = 0;
            frame_accum = 0;
        }
    }

    SDL_Quit();
    return 0;
}

