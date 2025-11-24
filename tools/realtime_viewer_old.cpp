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

#include "species.hpp"

// Camera
struct Camera {
    float x = 32.0f, y = 32.0f, z = 100.0f;
    float yaw = -90.0f;
    float pitch = 0.0f;
    float speed = 0.5f;
    float sensitivity = 0.2f;
};

// Global State
int W = 64, H = 64, D = 64;
std::vector<uint8_t> grid;
std::vector<uint8_t> next_grid;
bool paused = false;
bool show_grid = true;
int target_fps = 60;
int selected_species_id = 1;
Camera cam;
bool mouse_captured = false;

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
    
    #pragma omp parallel for
    for (long z = 0; z < (long)D; ++z) {
        for (long y = 0; y < (long)H; ++y) {
            for (long x = 0; x < (long)W; ++x) {
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
                                if (neighbor_counts[req.first] < req.second) { met = false; break; }
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
                            if (neighbor_counts[req.first] < req.second) { met = false; break; }
                        }
                        if (met) { born = true; break; }
                    }
                    if (born) candidates.push_back(sp_id);
                }

                if (candidates.empty()) next_grid[idx] = 0;
                else if (candidates.size() == 1) next_grid[idx] = candidates[0];
                else {
                    int best = 0, max_s = -1;
                    for (int cid : candidates) {
                        const Species* sp = SpeciesManager::instance().get_species(cid);
                        if (sp && sp->strength > max_s) { max_s = sp->strength; best = cid; }
                        else if (sp && sp->strength == max_s && cid > best) best = cid;
                    }
                    next_grid[idx] = best;
                }
            }
        }
    }
    grid.swap(next_grid);
}

void render_cube(float x, float y, float z, float size, uint8_t r, uint8_t g, uint8_t b) {
    glColor3ub(r, g, b);
    glPushMatrix();
    glTranslatef(x, y, z);
    float s = size / 2.0f;
    glBegin(GL_QUADS);
        glVertex3f(-s, -s, s); glVertex3f(s, -s, s); glVertex3f(s, s, s); glVertex3f(-s, s, s);
        glVertex3f(-s, -s, -s); glVertex3f(-s, s, -s); glVertex3f(s, s, -s); glVertex3f(s, -s, -s);
        glVertex3f(-s, s, -s); glVertex3f(-s, s, s); glVertex3f(s, s, s); glVertex3f(s, s, -s);
        glVertex3f(-s, -s, -s); glVertex3f(s, -s, -s); glVertex3f(s, -s, s); glVertex3f(-s, -s, s);
        glVertex3f(s, -s, -s); glVertex3f(s, s, -s); glVertex3f(s, s, s); glVertex3f(s, -s, s);
        glVertex3f(-s, -s, -s); glVertex3f(-s, -s, s); glVertex3f(-s, s, s); glVertex3f(-s, s, -s);
    glEnd();
    glPopMatrix();
}

void render_full_grid() {
    if (!show_grid) return;
    
    glColor3f(1.0f, 1.0f, 1.0f); // White grid
    glBegin(GL_LINES);
    
    for (int x = 0; x <= W; ++x) {
        for (int z = 0; z <= D; ++z) {
            glVertex3f(x, 0, z); glVertex3f(x, H, z);
        }
    }
    for (int y = 0; y <= H; ++y) {
        for (int z = 0; z <= D; ++z) {
            glVertex3f(0, y, z); glVertex3f(W, y, z);
        }
    }
    for (int x = 0; x <= W; ++x) {
        for (int y = 0; y <= H; ++y) {
            glVertex3f(x, y, 0); glVertex3f(x, y, D);
        }
    }
    glEnd();
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

    // Draw selection box
    const Species* sp = SpeciesManager::instance().get_species(selected_species_id);
    if (sp) {
        glColor3ub(sp->r, sp->g, sp->b);
        glBegin(GL_QUADS);
        glVertex2f(10, 10);
        glVertex2f(40, 10);
        glVertex2f(40, 40);
        glVertex2f(10, 40);
        glEnd();
    }
    
    // Draw border
    glColor3f(1, 1, 1);
    glBegin(GL_LINE_LOOP);
    glVertex2f(10, 10);
    glVertex2f(40, 10);
    glVertex2f(40, 40);
    glVertex2f(10, 40);
    glEnd();

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// Raycast and place/remove
void handle_click(int mx, int my, int width, int height, bool left_click) {
    GLdouble model[16], proj[16];
    GLint view[4];
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);
    glGetIntegerv(GL_VIEWPORT, view);

    float winX = (float)mx;
    float winY = (float)height - (float)my; // Invert Y

    GLdouble nx, ny, nz, fx, fy, fz;
    gluUnProject(winX, winY, 0.0, model, proj, view, &nx, &ny, &nz);
    gluUnProject(winX, winY, 1.0, model, proj, view, &fx, &fy, &fz);

    double dx = fx - nx;
    double dy = fy - ny;
    double dz = fz - nz;
    double len = sqrt(dx*dx + dy*dy + dz*dz);
    dx /= len; dy /= len; dz /= len;

    // Ray march
    double cx = nx, cy = ny, cz = nz;
    for (float t = 0; t < 200.0f; t += 0.5f) {
        cx = nx + dx * t;
        cy = ny + dy * t;
        cz = nz + dz * t;

        int gx = (int)floor(cx);
        int gy = (int)floor(cy);
        int gz = (int)floor(cz);

        if (gx >= 0 && gx < W && gy >= 0 && gy < H && gz >= 0 && gz < D) {
            if (left_click) {
                grid[gz * (W * H) + gy * W + gx] = selected_species_id;
            }
            return; 
        }
    }
}

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    SpeciesManager::instance().load_species("species/species.csv");

    grid.resize(W * H * D, 0);
    next_grid.resize(W * H * D, 0);
    
    for(auto& c : grid) {
        if (rand() % 10 == 0) c = (rand() % 3) + 1; 
    }

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window* window = SDL_CreateWindow("3D Game of Life - Realtime", 
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    
    SDL_GLContext context = SDL_GL_CreateContext(window);
    
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    gluPerspective(45.0f, 1280.0f/720.0f, 0.1f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);

    cam.x = W / 2.0f; cam.y = H / 2.0f; cam.z = D * 2.0f;

    bool running = true;
    SDL_Event event;
    
    auto last_time = std::chrono::high_resolution_clock::now();
    float frame_accum = 0.0f;
    int frames = 0;

    SDL_SetRelativeMouseMode(SDL_FALSE);

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
                if (event.button.button == SDL_BUTTON_LEFT) {
                    if (paused) {
                        int w, h;
                        SDL_GetWindowSize(window, &w, &h);
                        handle_click(mx, my, w, h, true);
                    } else {
                        mouse_captured = true;
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                    }
                }
            }
            if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    mouse_captured = false;
                    SDL_SetRelativeMouseMode(SDL_FALSE);
                }
            }
            if (event.type == SDL_MOUSEMOTION && mouse_captured) {
                cam.yaw += event.motion.xrel * cam.sensitivity;
                cam.pitch -= event.motion.yrel * cam.sensitivity;
                if (cam.pitch > 89.0f) cam.pitch = 89.0f;
                if (cam.pitch < -89.0f) cam.pitch = -89.0f;
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
            simulation_step();
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        
        float lx = cam.x + fx;
        float ly = cam.y + fy;
        float lz = cam.z + fz;
        gluLookAt(cam.x, cam.y, cam.z, lx, ly, lz, 0.0f, 1.0f, 0.0f);

        render_scene();
        
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        render_hud(w, h);

        SDL_GL_SwapWindow(window);

        frames++;
        frame_accum += dt;
        if (frame_accum >= 1.0f) {
            std::string title = "FPS: " + std::to_string(frames) + " | Species: " + std::to_string(selected_species_id) + " | Paused: " + (paused ? "YES" : "NO");
            SDL_SetWindowTitle(window, title.c_str());
            frames = 0;
            frame_accum = 0;
        }
        
        float frame_time = 1.0f / target_fps;
        if (dt < frame_time) {
            SDL_Delay((Uint32)((frame_time - dt) * 1000));
        }
    }

    SDL_Quit();
    return 0;
}
