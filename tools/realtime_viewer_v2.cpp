#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <thread>
#include <map>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "species.hpp"

// Camera
struct Camera {
    float x = 32.0f, y = 32.0f, z = 100.0f;
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 2.0f; // Increased speed
    float sensitivity = 0.2f;
};

// Global State
int W = 128, H = 128, D = 128; // Larger Map
int INIT_W = 32, INIT_H = 32, INIT_D = 32; // Initial Spawn Area
std::vector<uint8_t> grid;
std::vector<uint8_t> next_grid;
bool paused = false;
bool show_grid = true;
int target_fps = 60;
int selected_species_id = 1;
Camera cam;
bool mouse_captured = false;
std::map<int, GLuint> species_textures;

// Bounding Box for Optimization
int min_x, min_y, min_z;
int max_x, max_y, max_z;

// Forward Declaration
void render_billboard_circle_optimized(float x, float y, float z, float size, uint8_t r, uint8_t g, uint8_t b, const float* right, const float* up, GLuint texture_id);

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

// Helper to get forward vector
void get_camera_vectors(float& fx, float& fy, float& fz, float& rx, float& ry, float& rz) {
    float radYaw = cam.yaw * M_PI / 180.0f;
    float radPitch = cam.pitch * M_PI / 180.0f;
    
    fx = cos(radYaw) * cos(radPitch);
    fy = sin(radPitch);
    fz = sin(radYaw) * cos(radPitch);
    
    // Right vector
    rx = -fz; 
    ry = 0;
    rz = fx;
    
    float r_len = sqrt(rx*rx + rz*rz);
    if (r_len > 0.001f) {
        rx /= r_len;
        rz /= r_len;
    }
}

// Simulation Step
void simulation_step() {
    const auto& all_species = SpeciesManager::instance().get_all_species();
    
    int start_z = std::max(0, min_z - 1);
    int end_z = std::min(D - 1, max_z + 1);
    int start_y = std::max(0, min_y - 1);
    int end_y = std::min(H - 1, max_y + 1);
    int start_x = std::max(0, min_x - 1);
    int end_x = std::min(W - 1, max_x + 1);

    int next_min_x = W, next_min_y = H, next_min_z = D;
    int next_max_x = -1, next_max_y = -1, next_max_z = -1;

    #pragma omp parallel
    {
        int t_min_x = W, t_min_y = H, t_min_z = D;
        int t_max_x = -1, t_max_y = -1, t_max_z = -1;

        #pragma omp for collapse(2) nowait
        for (long z = start_z; z <= end_z; ++z) {
            for (long y = start_y; y <= end_y; ++y) {
                for (long x = start_x; x <= end_x; ++x) {
                    long idx = z * (W * H) + y * W + x;
                    
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
                                
                                uint8_t nid = grid[zl * (W * H) + yl * W + xl];
                                if (nid != 0) neighbor_counts[nid]++;
                            }
                        }
                    }

                    uint8_t current_id = grid[idx];
                    std::vector<int> candidates;

                    if (current_id != 0) {
                        const Species* sp = SpeciesManager::instance().get_species(current_id);
                        bool survived = false;
                        if (sp) {
                            for (const auto& cond : sp->survival_rules) {
                                bool met = true;
                                for (const auto& req : cond) {
                                    if (neighbor_counts[req.first] != req.second) { met = false; break; }
                                }
                                if (met) { survived = true; break; }
                            }
                        }
                        if (survived) candidates.push_back(current_id);
                    }

                    for (const auto& kv : all_species) {
                        int sp_id = kv.first;
                        if (sp_id == current_id) continue;
                        bool born = false;
                        for (const auto& cond : kv.second.birth_rules) {
                            bool met = true;
                            for (const auto& req : cond) {
                                if (neighbor_counts[req.first] != req.second) { met = false; break; }
                            }
                            if (met) { born = true; break; }
                        }
                        if (born) candidates.push_back(sp_id);
                    }

                    uint8_t final_id = 0;
                    if (candidates.empty()) final_id = 0;
                    else if (candidates.size() == 1) final_id = candidates[0];
                    else {
                        int best = 0, max_s = -1;
                        for (int cid : candidates) {
                            const Species* sp = SpeciesManager::instance().get_species(cid);
                            if (sp && sp->strength > max_s) { max_s = sp->strength; best = cid; }
                            else if (sp && sp->strength == max_s && cid > best) best = cid;
                        }
                        final_id = best;
                    }
                    
                    next_grid[idx] = final_id;

                    if (final_id != 0) {
                        if (x < t_min_x) t_min_x = x;
                        if (x > t_max_x) t_max_x = x;
                        if (y < t_min_y) t_min_y = y;
                        if (y > t_max_y) t_max_y = y;
                        if (z < t_min_z) t_min_z = z;
                        if (z > t_max_z) t_max_z = z;
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

    grid.swap(next_grid);
    std::fill(next_grid.begin(), next_grid.end(), 0); 
}

// Texture Loading
GLuint load_texture(const std::string& path) {
    int w, h, comp;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << path << " Error: " << stbi_failure_reason() << std::endl;
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
    const auto& all_species = SpeciesManager::instance().get_all_species();
    for (const auto& kv : all_species) {
        if (!kv.second.image_path.empty()) {
            GLuint tex = load_texture(kv.second.image_path);
            if (tex != 0) {
                species_textures[kv.first] = tex;
            }
        }
    }
}

void render_full_grid() {
    if (!show_grid) return;
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    // Global Bounds
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

    // Local Grid
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
    
    // Active Box
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

// Optimized Billboard Renderer
void render_billboard_circle_optimized(float x, float y, float z, float size, uint8_t r, uint8_t g, uint8_t b, const float* right, const float* up, GLuint texture_id) {
    float s = size / 2.0f;
    
    if (texture_id != 0) {
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_ALPHA_TEST); // Enable Alpha Testing for transparency
        glAlphaFunc(GL_GREATER, 0.1f); // Discard transparent pixels

        glBindTexture(GL_TEXTURE_2D, texture_id);
        glColor3f(1, 1, 1); // White for texture
        
        glBegin(GL_QUADS);
        // Bottom Left
        glTexCoord2f(0, 1);
        glVertex3f(x - right[0]*s - up[0]*s, y - right[1]*s - up[1]*s, z - right[2]*s - up[2]*s);
        // Bottom Right
        glTexCoord2f(1, 1);
        glVertex3f(x + right[0]*s - up[0]*s, y + right[1]*s - up[1]*s, z + right[2]*s - up[2]*s);
        // Top Right
        glTexCoord2f(1, 0);
        glVertex3f(x + right[0]*s + up[0]*s, y + right[1]*s + up[1]*s, z + right[2]*s + up[2]*s);
        // Top Left
        glTexCoord2f(0, 0);
        glVertex3f(x - right[0]*s + up[0]*s, y - right[1]*s + up[1]*s, z - right[2]*s + up[2]*s);
        glEnd();
        
        glDisable(GL_ALPHA_TEST);
        glDisable(GL_TEXTURE_2D);
    } else {
        glColor3ub(r, g, b);
        glBegin(GL_TRIANGLE_FAN);
        glVertex3f(x, y, z); // Center
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
                    const Species* sp = SpeciesManager::instance().get_species(id);
                    if (sp) {
                        GLuint tex = 0;
                        if (species_textures.count(id)) tex = species_textures[id];
                        render_billboard_circle_optimized(x + 0.5f, y + 0.5f, z + 0.5f, 0.8f, sp->r, sp->g, sp->b, right, up, tex);
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
            glTexCoord2f(1, 0); glVertex2f(60, 10); // Larger icon (50x50)
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

// Raycast helper to find target block
// If hit, returns true and sets out_x,y,z to the block coordinate.
// If is_place is true, it returns the ADJACENT block (face normal).
// If no hit, returns false but sets out_x,y,z to a fixed distance point.
bool get_target_block(int mx, int my, int width, int height, bool is_place, int& out_x, int& out_y, int& out_z) {
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
    if (len < 0.0001) return false; // Degenerate ray
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
                // Hit!
                if (is_place) {
                    // Return previous empty block (face neighbor)
                    if (last_gx != -1) {
                        out_x = last_gx; out_y = last_gy; out_z = last_gz;
                        return true;
                    } else {
                        // Ray started inside a block?
                        return false; 
                    }
                } else {
                    // Return the block itself for deletion
                    out_x = gx; out_y = gy; out_z = gz;
                    return true;
                }
            }
        }
        last_gx = gx; last_gy = gy; last_gz = gz;
    }
    
    // No hit, return fixed distance
    float fixed_dist = 10.0f;
    cx = nx + dx * fixed_dist;
    cy = ny + dy * fixed_dist;
    cz = nz + dz * fixed_dist;
    
    if (std::isnan(cx) || std::isnan(cy) || std::isnan(cz)) {
        std::cerr << "NaN detected in raycast!" << std::endl;
        return false;
    }

    out_x = (int)floor(cx);
    out_y = (int)floor(cy);
    out_z = (int)floor(cz);
    
    // Clamp to bounds
    if (out_x < 0) out_x = 0; if (out_x >= W) out_x = W-1;
    if (out_y < 0) out_y = 0; if (out_y >= H) out_y = H-1;
    if (out_z < 0) out_z = 0; if (out_z >= D) out_z = D-1;
    
    return false; // Not a hit on existing, but valid coords
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
    
    // Wireframe box
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
    // We use get_target_block. If it returns true (hit existing) OR false (fixed distance), we use the coords.
    // BUT for deletion (is_place=false), we ONLY want to delete if we actually hit something.
    bool hit = get_target_block(mx, my, width, height, is_place, hit_x, hit_y, hit_z);
    
    // Safety check for bounds
    if (hit_x < 0 || hit_x >= W || hit_y < 0 || hit_y >= H || hit_z < 0 || hit_z >= D) {
        // std::cerr << "Click out of bounds: " << hit_x << "," << hit_y << "," << hit_z << std::endl;
        return;
    }

    if (is_place) {
        // Place allows both hit (adjacent) and no-hit (fixed distance)
        int idx = hit_z * (W * H) + hit_y * W + hit_x;
        if (idx < 0 || idx >= (int)grid.size()) {
             std::cerr << "Index out of bounds! " << idx << " size: " << grid.size() << std::endl;
             return;
        }
        grid[idx] = selected_species_id;
        update_bounds(hit_x, hit_y, hit_z);
        // std::cout << "Placed at " << hit_x << "," << hit_y << "," << hit_z << " Bounds: " << min_x << "," << max_x << std::endl;
    } else {
        // Delete only if we hit something
        if (hit) {
            int idx = hit_z * (W * H) + hit_y * W + hit_x;
            if (idx >= 0 && idx < (int)grid.size()) {
                grid[idx] = 0;
                std::cout << "Removed at " << hit_x << "," << hit_y << "," << hit_z << std::endl;
            }
        }
    }
}

int target_sim_fps = 10; // Default

int main(int argc, char* argv[]) {
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fps" && i + 1 < argc) {
            target_sim_fps = std::stoi(argv[++i]);
        }
    }
    std::cout << "Target Simulation FPS: " << target_sim_fps << std::endl;

    SpeciesManager::instance().load_species("species/species.csv");

    grid.resize(W * H * D, 0);
    next_grid.resize(W * H * D, 0);
    std::cout << "Grid initialized. Size: " << grid.size() << std::endl;
    
    reset_bounds();

    int start_x = (W - INIT_W) / 2;
    int start_y = (H - INIT_H) / 2;
    int start_z = (D - INIT_D) / 2;

    for (int z = start_z; z < start_z + INIT_D; ++z) {
        for (int y = start_y; y < start_y + INIT_H; ++y) {
            for (int x = start_x; x < start_x + INIT_W; ++x) {
                if (rand() % 10 == 0) {
                    grid[z * (W * H) + y * W + x] = (rand() % 3) + 1;
                    update_bounds(x, y, z);
                }
            }
        }
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window* my_window = nullptr;
    my_window = SDL_CreateWindow("3D Game of Life - Realtime", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    
    if (!my_window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_GLContext context = SDL_GL_CreateContext(my_window);
    
    init_gl_resources(); // Load textures

    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    gluPerspective(45.0f, 1280.0f/720.0f, 0.1f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);

    cam.x = W / 2.0f; cam.y = H / 2.0f; cam.z = D * 1.5f;

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
                    selected_species_id = (selected_species_id % 3) + 1; 
                }
            }
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) { // Left Click = Place
                    if (paused) {
                        int w, h;
                        SDL_GetWindowSize(my_window, &w, &h);
                        handle_click(mx, my, w, h, true); 
                    } else {
                        mouse_captured = true;
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                    }
                }
                if (event.button.button == SDL_BUTTON_RIGHT) { // Right Click = Remove
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
                // Clamp
                if (xrel > 50) xrel = 50; if (xrel < -50) xrel = -50;
                if (yrel > 50) yrel = 50; if (yrel < -50) yrel = -50;

                // INVERTED MOUSE CONTROLS
                cam.yaw -= xrel * cam.sensitivity;
                cam.pitch += yrel * cam.sensitivity;
                
                if (cam.pitch > 89.0f) cam.pitch = 89.0f;
                if (cam.pitch < -89.0f) cam.pitch = -89.0f;
            }
        }

        // Mass placement/deletion
        if (paused) {
            int w, h;
            SDL_GetWindowSize(my_window, &w, &h);
            int hit_x, hit_y, hit_z;
            Uint32 buttons = SDL_GetMouseState(NULL, NULL);
            
            if (buttons & SDL_BUTTON(SDL_BUTTON_LEFT)) { // Place
                get_target_block(mx, my, w, h, true, hit_x, hit_y, hit_z);
                int idx = hit_z * (W * H) + hit_y * W + hit_x;
                if (idx >= 0 && idx < (int)grid.size()) {
                    grid[idx] = selected_species_id;
                    update_bounds(hit_x, hit_y, hit_z);
                }
            }
            if (buttons & SDL_BUTTON(SDL_BUTTON_RIGHT)) { // Remove
                if (get_target_block(mx, my, w, h, false, hit_x, hit_y, hit_z)) {
                     int idx = hit_z * (W * H) + hit_y * W + hit_x;
                     if (idx >= 0 && idx < (int)grid.size()) grid[idx] = 0;
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
                simulation_step();
                sim_accum = 0.0f; // Reset to drop frames if we are too slow
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
            // Always show preview for placement (Right click logic)
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
            sprintf(title, "FPS: %.2f | SimFPS: %d | Species: %d | Paused: %s", frames / frame_accum, target_sim_fps, selected_species_id, paused ? "YES" : "NO");
            SDL_SetWindowTitle(my_window, title);
            frames = 0;
            frame_accum = 0;
        }
        
        // No delay, run as fast as possible for smooth camera
    }

    SDL_Quit();
    return 0;
}
