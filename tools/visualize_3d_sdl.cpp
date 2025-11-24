// Simple SDL2 + OpenGL slice-stack viewer
// Usage: ./tools/visualize_3d_sdl --dir frames_3d --iter 0

#include <SDL2/SDL.h>
#include <GL/gl.h>

static void myLookAt(float eyeX, float eyeY, float eyeZ,
                     float centerX, float centerY, float centerZ,
                     float upX, float upY, float upZ) {
    // compute f = normalize(center - eye)
    float fx = centerX - eyeX; float fy = centerY - eyeY; float fz = centerZ - eyeZ;
    float fl = sqrtf(fx*fx + fy*fy + fz*fz);
    if (fl > 0.0f) { fx /= fl; fy /= fl; fz /= fl; }
    // compute s = normalize(cross(f, up))
    float sx = fy * upZ - fz * upY;
    float sy = fz * upX - fx * upZ;
    float sz = fx * upY - fy * upX;
    float sl = sqrtf(sx*sx + sy*sy + sz*sz);
    if (sl > 0.0f) { sx /= sl; sy /= sl; sz /= sl; }
    // compute u = cross(s, f)
    float ux = sy * fz - sz * fy;
    float uy = sz * fx - sx * fz;
    float uz = sx * fy - sy * fx;
    // build rotation matrix (column-major for OpenGL)
    float m[16];
    m[0] = sx; m[4] = sy; m[8]  = sz; m[12] = 0.0f;
    m[1] = ux; m[5] = uy; m[9]  = uz; m[13] = 0.0f;
    m[2] = -fx; m[6] = -fy; m[10] = -fz; m[14] = 0.0f;
    m[3] = 0.0f; m[7] = 0.0f; m[11] = 0.0f; m[15] = 1.0f;
    glMultMatrixf(m);
    // then translate by -eye
    glTranslatef(-eyeX, -eyeY, -eyeZ);
}

#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <map>
#include <array>

namespace fs = std::filesystem;

struct Slice {
    int width, height;
    std::vector<unsigned char> data; // RGB
};

static bool read_ppm(const std::string &path, Slice &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    f >> magic;
    if (magic != "P6") return false;
    while (f.peek() == '#') { std::string line; std::getline(f, line); }
    int w=0,h=0,maxv=0;
    f >> w >> h >> maxv;
    if (w<=0 || h<=0) return false;
    f.get();
    out.width = w; out.height = h;
    out.data.resize((size_t)w*h*3);
    f.read(reinterpret_cast<char*>(out.data.data()), out.data.size());
    return true;
}

int main(int argc, char **argv) {
    std::string dir = "frames_3d";
    int iter = 0;
    bool test_mode = false;
    for (int i=1;i<argc;++i) {
        std::string a = argv[i];
        if (a=="--dir" && i+1<argc) dir = argv[++i];
        else if (a=="--iter" && i+1<argc) iter = std::stoi(argv[++i]);
        else if (a=="--test") test_mode = true;
    }

    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "Directory not found: " << dir << std::endl;
        return 1;
    }

    // build stacks per iteration
    std::vector<std::vector<Slice>> stacks;
    std::vector<int> iterations;
    // vector for special test-mode positions (50 cubes)
    std::vector<std::array<float,3>> test_positions;
    float spacing = 3.0f;
    if (test_mode) {
        // create two synthetic iterations: 4x4x4 and 5x5x5 solid cubes
        // iteration 0: 4x4x4
        {
            int W = 4, H = 4, D = 4;
            std::vector<Slice> stack;
            for (int z=0; z<D; ++z) {
                Slice s; s.width = W; s.height = H; s.data.assign((size_t)W*H*3, 255); // white voxels
                stack.push_back(std::move(s));
            }
            stacks.push_back(std::move(stack));
            iterations.push_back(0);
        }
        // iteration 1: 5x5x5
        {
            int W = 5, H = 5, D = 5;
            std::vector<Slice> stack;
            for (int z=0; z<D; ++z) {
                Slice s; s.width = W; s.height = H; s.data.assign((size_t)W*H*3, 200); // slightly gray
                stack.push_back(std::move(s));
            }
            stacks.push_back(std::move(stack));
            iterations.push_back(1);
        }
        // prepare test positions: arrange up to 50 cubes in a 5x5x2 grid centered around origin
        int nx = 5, ny = 5, nz = 2;
        float spacing = 3.0f;
        for (int iz=0; iz<nz; ++iz) for (int iy=0; iy<ny; ++iy) for (int ix=0; ix<nx; ++ix) {
            if ((int)test_positions.size() >= 50) break;
            float x = (ix - (nx-1)/2.0f) * spacing;
            float y = (iy - (ny-1)/2.0f) * spacing;
            float z = (iz - (nz-1)/2.0f) * spacing;
            test_positions.push_back(std::array<float,3>{x,y,z});
        }
    } else {
        std::map<int, std::map<int, fs::path>> groups; // iter -> (slice_idx -> path)
        for (auto &p: fs::directory_iterator(dir)) {
            if (!p.is_regular_file()) continue;
            std::string name = p.path().filename().string();
            if (p.path().extension() != ".ppm") continue;
            // look for pattern frame_<iter>_slice_<slice>.ppm
            int iter_idx = -1, slice_idx = -1;
            if (sscanf(name.c_str(), "frame_%d_slice_%d.ppm", &iter_idx, &slice_idx) == 2) {
                groups[iter_idx][slice_idx] = p.path();
            }
        }
        if (groups.empty()) {
            std::cerr << "No slice files found in " << dir << std::endl;
            return 1;
        }
        for (auto &it: groups) iterations.push_back(it.first);
        std::sort(iterations.begin(), iterations.end());
        for (int itidx: iterations) {
            auto &mp = groups[itidx];
            std::vector<int> slice_keys;
            for (auto &s: mp) slice_keys.push_back(s.first);
            std::sort(slice_keys.begin(), slice_keys.end());
            std::vector<Slice> stack;
            for (int sk: slice_keys) {
                Slice s; if (!read_ppm(mp[sk].string(), s)) { std::cerr << "Failed to read " << mp[sk] << std::endl; return 1; }
                stack.push_back(std::move(s));
            }
            stacks.push_back(std::move(stack));
        }
    }

    // current iteration index into stacks
    size_t cur_iter = 0;
    // playback timing (100ms ~ 10 FPS)
    int delay_ms = 100; // default frame delay (~10 FPS)
    Uint32 last_frame_ms = SDL_GetTicks();
    std::string last_title;

    int W = stacks[0].front().width;
    int H = stacks[0].front().height;
    int D = (int)stacks[0].size();
    std::cout << "Loaded stack series: " << W << "x" << H << "x" << D << " (" << stacks.size() << " iterations)\n";

    if (SDL_Init(SDL_INIT_VIDEO) != 0) { std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl; return 1; }
    SDL_Window *win = SDL_CreateWindow("3D Viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1024, 768, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!win) { std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl; SDL_Quit(); return 1; }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) { std::cerr << "SDL_GL_CreateContext Error: " << SDL_GetError() << std::endl; SDL_DestroyWindow(win); SDL_Quit(); return 1; }
    SDL_GL_SetSwapInterval(1);

    // basic lighting setup
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat light_pos[] = { 0.5f, 1.0f, 0.5f, 0.0f };
    GLfloat light_diff[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    GLfloat light_amb[] = { 0.2f, 0.2f, 0.2f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diff);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_NORMALIZE);

    // we render voxels directly as cubes, so no texture upload needed

    bool running = true;
    bool mouse_locked = true;
    bool mouse_down = false;
    SDL_SetRelativeMouseMode(SDL_TRUE);

    // simple camera
        float yaw = -0.78539816339f; // -45 degrees
        float pitch = -0.52359877559f; // -30 degrees
    // place camera so the stack is visible by default (centered and pulled back)
    float cube_size_def = 1.0f;
    float spacing_def = 1.05f;
    // camera position is in stack-local coordinates (stack is centered at origin).
    float camx = 0.0f;
    float camy = 0.0f;
    float camz = std::max(W, std::max(H,D)) * 3.0f; // distance along +Z from origin
    float speed = 10.0f;

    Uint32 last = SDL_GetTicks();

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) running = false;
                else if (e.key.keysym.sym == SDLK_TAB) {
                    mouse_locked = !mouse_locked; SDL_SetRelativeMouseMode(mouse_locked ? SDL_TRUE : SDL_FALSE);
                }
            } else if (e.type == SDL_MOUSEBUTTONDOWN) {
                mouse_down = true;
            } else if (e.type == SDL_MOUSEBUTTONUP) {
                mouse_down = false;
            } else if (e.type == SDL_MOUSEMOTION) {
                if (mouse_locked && mouse_down) {
                    // adjust X motion so dragging left makes you look right (user expectation)
                    yaw += e.motion.xrel * 0.002f;
                    // vertical drag should move view up when dragging up: invert sign accordingly
                    pitch -= e.motion.yrel * 0.002f;
                    if (pitch > 1.5f) pitch = 1.5f;
                    if (pitch < -1.5f) pitch = -1.5f;
                }
            }
        }

        // delta time
        Uint32 now = SDL_GetTicks(); float dt = (now - last) / 1000.0f; last = now;
        const uint8_t *kbd = SDL_GetKeyboardState(NULL);
        // compute movement vectors from yaw/pitch so movement is camera-relative
        // forward vector in world/stack coords (OpenGL camera looks along -Z by default)
        float fx = sinf(yaw) * cosf(pitch);
        float fy = sinf(pitch);
        float fz = -cosf(yaw) * cosf(pitch);
        // normalize forward
        float flen = sqrtf(fx*fx + fy*fy + fz*fz);
        if (flen > 0.0f) { fx /= flen; fy /= flen; fz /= flen; }
        // right = cross(forward, up) where up=(0,1,0)
        float rx = fy*0.0f - fz*1.0f; // = -fz
        float ry = fz*0.0f - fx*0.0f; // = 0
        float rz = fx*1.0f - fy*0.0f; // = fx
        float rlen = sqrtf(rx*rx + ry*ry + rz*rz);
        if (rlen > 0.0f) { rx /= rlen; ry /= rlen; rz /= rlen; }

        if (kbd[SDL_SCANCODE_W]) { camx += fx * speed * dt; camy += fy * speed * dt; camz += fz * speed * dt; }
        if (kbd[SDL_SCANCODE_S]) { camx -= fx * speed * dt; camy -= fy * speed * dt; camz -= fz * speed * dt; }
        if (kbd[SDL_SCANCODE_D]) { camx += rx * speed * dt; camy += ry * speed * dt; camz += rz * speed * dt; }
        if (kbd[SDL_SCANCODE_A]) { camx -= rx * speed * dt; camy -= ry * speed * dt; camz -= rz * speed * dt; }
        // vertical control: Space = up, X = down
        if (kbd[SDL_SCANCODE_SPACE]) camy += speed * dt;
        if (kbd[SDL_SCANCODE_X]) camy -= speed * dt;

        // clamp camera height to avoid falling through the world
        if (camy < -20.0f) camy = -20.0f;
        if (camy >  20.0f) camy = 20.0f;

        // advance iteration automatically based on delay (only if multiple iterations)
        if (stacks.size() > 1 && (int)(now - last_frame_ms) >= delay_ms) {
            cur_iter = (cur_iter + 1) % stacks.size();
            last_frame_ms = now;
        }

        // current stack to render and its dimensions (recomputed per-iteration)
        auto &stack = stacks[cur_iter];
        int CW = stack.front().width;
        int CH = stack.front().height;
        int CD = (int)stack.size();
        int w,h; SDL_GetWindowSize(win, &w, &h);
        glViewport(0,0,w,h);
        glClearColor(0.1f,0.1f,0.12f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_2D);
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        // manually construct perspective matrix (avoid dependency on GLU)
        double fovy = 60.0 * 3.14159265358979323846 / 180.0;
        double nearv = 0.1;
        double top = tan(fovy/2.0) * nearv;
        double rightv = top * (double)w / (double)h;
        glFrustum(-rightv, rightv, -top, top, nearv, 10000.0);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();

        // camera transform: use gluLookAt so movement vectors and rotation stay consistent
        float cx = CW * 0.5f; float cy = CH * 0.5f; float cz = ((float)CD - 1.0f) * 0.5f;
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        // compute forward vector from yaw/pitch
        float ffx = sinf(yaw) * cosf(pitch);
        float ffy = sinf(pitch);
        float ffz = -cosf(yaw) * cosf(pitch);
        // eye position is camx/camy/camz in stack-local coords
        myLookAt(camx, camy, camz, camx + ffx, camy + ffy, camz + ffz, 0.0f, 1.0f, 0.0f);
        // now translate stack so its center is at origin in view space
        glTranslatef(-cx, -cy, -cz);

        // render grid under stack for reference (use current dims)
        glDisable(GL_LIGHTING);
        glColor3f(0.22f, 0.22f, 0.25f);
        float cube_size = 1.0f;
        float grid_z = -0.25f * cube_size; // slightly below bottom layer so it's visible
        glBegin(GL_LINES);
        for (int xg = 0; xg <= CW; ++xg) {
            float gx = xg * cube_size;
            glVertex3f(gx, 0.0f, grid_z);
            glVertex3f(gx, CH * cube_size, grid_z);
        }
        for (int yg = 0; yg <= CH; ++yg) {
            float gy = yg * cube_size;
            glVertex3f(0.0f, gy, grid_z);
            glVertex3f(CW * cube_size, gy, grid_z);
        }
        glEnd();
        glEnable(GL_LIGHTING);
        // render voxels as cubes (Minecraft-like)
        glDisable(GL_TEXTURE_2D);
        int alive_count = 0;
        if (test_mode && !test_positions.empty()) {
            // make the size difference more visible and smooth: pulse between small and large
            float unit_small = 0.6f; // visual unit for 4x
            float unit_large = 1.4f; // visual unit for 5x
            Uint32 now_ms = SDL_GetTicks();
            float period_ms = std::max(200.0f, (float)delay_ms * 2.0f);
            float phase = fmodf((float)now_ms, period_ms) / period_ms; // 0..1
            // smooth pulse (sinusoidal) between small and large
            const float TWO_PI = 6.28318530717958647692f;
            float pulse = 0.5f * (1.0f + sinf(phase * TWO_PI));
            float scale = unit_small + (unit_large - unit_small) * pulse;
            float half = 0.5f * cube_size * scale;
            // draw full 3D grid covering test positions
            float minx=1e9f,maxx=-1e9f,miny=1e9f,maxy=-1e9f,minz=1e9f,maxz=-1e9f;
            for (auto &p: test_positions) {
                minx = std::min(minx, p[0]); maxx = std::max(maxx, p[0]);
                miny = std::min(miny, p[1]); maxy = std::max(maxy, p[1]);
                minz = std::min(minz, p[2]); maxz = std::max(maxz, p[2]);
            }
            // expand bounds
            minx -= spacing; maxx += spacing;
            miny -= spacing; maxy += spacing;
            minz -= spacing; maxz += spacing;
            // draw grid lines along X/Y/Z at integer spacing = cube_size
            glDisable(GL_LIGHTING);
            glColor3f(0.18f,0.18f,0.2f);
            glBegin(GL_LINES);
            for (float x=minx; x<=maxx+0.001f; x += cube_size) {
                for (float y=miny; y<=maxy+0.001f; y += cube_size) {
                    glVertex3f(x,y,minz); glVertex3f(x,y,maxz);
                }
            }
            for (float x=minx; x<=maxx+0.001f; x += cube_size) {
                for (float z=minz; z<=maxz+0.001f; z += cube_size) {
                    glVertex3f(x,miny,z); glVertex3f(x,maxy,z);
                }
            }
            for (float y=miny; y<=maxy+0.001f; y += cube_size) {
                for (float z=minz; z<=maxz+0.001f; z += cube_size) {
                    glVertex3f(minx,y,z); glVertex3f(maxx,y,z);
                }
            }
            glEnd();
            glEnable(GL_LIGHTING);

            // draw cubes at positions
            for (auto &p: test_positions) {
                ++alive_count;
                float cxp = p[0]; float cyp = p[1]; float czp = p[2];
                float x0f = cxp - half; float x1f = cxp + half;
                float y0f = cyp - half; float y1f = cyp + half;
                float z0f = czp - half; float z1f = czp + half;
                glColor3f(0.7f,0.7f,0.7f);
                glBegin(GL_QUADS);
                // +Z
                glNormal3f(0,0,1);
                glVertex3f(x0f,y0f,z1f);
                glVertex3f(x1f,y0f,z1f);
                glVertex3f(x1f,y1f,z1f);
                glVertex3f(x0f,y1f,z1f);
                // -Z
                glNormal3f(0,0,-1);
                glVertex3f(x1f,y0f,z0f);
                glVertex3f(x0f,y0f,z0f);
                glVertex3f(x0f,y1f,z0f);
                glVertex3f(x1f,y1f,z0f);
                // +Y
                glNormal3f(0,1,0);
                glVertex3f(x0f,y1f,z0f);
                glVertex3f(x1f,y1f,z0f);
                glVertex3f(x1f,y1f,z1f);
                glVertex3f(x0f,y1f,z1f);
                // -Y
                glNormal3f(0,-1,0);
                glVertex3f(x0f,y0f,z1f);
                glVertex3f(x1f,y0f,z1f);
                glVertex3f(x1f,y0f,z0f);
                glVertex3f(x0f,y0f,z0f);
                // +X
                glNormal3f(1,0,0);
                glVertex3f(x1f,y0f,z1f);
                glVertex3f(x1f,y0f,z0f);
                glVertex3f(x1f,y1f,z0f);
                glVertex3f(x1f,y1f,z1f);
                // -X
                glNormal3f(-1,0,0);
                glVertex3f(x0f,y0f,z0f);
                glVertex3f(x0f,y0f,z1f);
                glVertex3f(x0f,y1f,z1f);
                glVertex3f(x0f,y1f,z0f);
                glEnd();
            }
        } else {
            for (int z=0; z<CD; ++z) {
                // sample image data to decide filled voxels
                const auto &s = stack[z];
                for (int y0=0; y0<CH; ++y0) {
                    for (int x0=0; x0<CW; ++x0) {
                        size_t idx = (y0 * W + x0) * 3;
                        unsigned char r = s.data[idx];
                        if (r == 0) continue; // empty
                        ++alive_count;
                        float xoff = (float)x0 * cube_size;
                        float yoff = (float)(CH - 1 - y0) * cube_size; // flip Y so image coords match world
                        float zoff = (float)z * (cube_size + 0.05f);
                        unsigned char g = s.data[idx+1];
                        unsigned char b = s.data[idx+2];
                        glColor3f(r/255.0f, g/255.0f, b/255.0f);
                        float x0f = xoff;
                        float x1f = xoff + cube_size;
                        float y0f = yoff;
                        float y1f = yoff + cube_size;
                        float z0f = zoff;
                        float z1f = zoff + cube_size;
                        glBegin(GL_QUADS);
                        // +Z
                        glNormal3f(0,0,1);
                        glVertex3f(x0f,y0f,z1f);
                        glVertex3f(x1f,y0f,z1f);
                        glVertex3f(x1f,y1f,z1f);
                        glVertex3f(x0f,y1f,z1f);
                        // -Z
                        glNormal3f(0,0,-1);
                        glVertex3f(x1f,y0f,z0f);
                        glVertex3f(x0f,y0f,z0f);
                        glVertex3f(x0f,y1f,z0f);
                        glVertex3f(x1f,y1f,z0f);
                        // +Y
                        glNormal3f(0,1,0);
                        glVertex3f(x0f,y1f,z0f);
                        glVertex3f(x1f,y1f,z0f);
                        glVertex3f(x1f,y1f,z1f);
                        glVertex3f(x0f,y1f,z1f);
                        // -Y
                        glNormal3f(0,-1,0);
                        glVertex3f(x0f,y0f,z1f);
                        glVertex3f(x1f,y0f,z1f);
                        glVertex3f(x1f,y0f,z0f);
                        glVertex3f(x0f,y0f,z0f);
                        // +X
                        glNormal3f(1,0,0);
                        glVertex3f(x1f,y0f,z1f);
                        glVertex3f(x1f,y0f,z0f);
                        glVertex3f(x1f,y1f,z0f);
                        glVertex3f(x1f,y1f,z1f);
                        // -X
                        glNormal3f(-1,0,0);
                        glVertex3f(x0f,y0f,z0f);
                        glVertex3f(x0f,y0f,z1f);
                        glVertex3f(x0f,y1f,z1f);
                        glVertex3f(x0f,y1f,z0f);
                        glEnd();
                    }
                }
            }
        }
        glEnable(GL_TEXTURE_2D);

        // update window title only when it changes (reduces flicker)
        std::string title = "3D Viewer - iter=" + std::to_string(iterations[cur_iter]) + " (idx=" + std::to_string(cur_iter) + ") alive=" + std::to_string(alive_count);
        if (title != last_title) {
            SDL_SetWindowTitle(win, title.c_str());
            last_title = title;
        }

        // if there are no alive voxels, draw a faint wireframe grid to show stack bounds
        if (alive_count == 0) {
            glDisable(GL_LIGHTING);
            glColor3f(0.3f, 0.3f, 0.35f);
            for (int z=0; z<D; ++z) {
                for (int y0=0; y0<H; ++y0) {
                    for (int x0=0; x0<W; ++x0) {
                        float xoff = (float)x0 * cube_size;
                        float yoff = (float)(H - 1 - y0) * cube_size;
                        float zoff = (float)z * (cube_size + 0.05f);
                        float x0f = xoff;
                        float x1f = xoff + cube_size;
                        float y0f = yoff;
                        float y1f = yoff + cube_size;
                        float z0f = zoff;
                        float z1f = zoff + cube_size;
                        glBegin(GL_LINE_LOOP);
                        // simple box outline (12 edges as lines)
                        glVertex3f(x0f,y0f,z0f); glVertex3f(x1f,y0f,z0f);
                        glVertex3f(x1f,y1f,z0f); glVertex3f(x0f,y1f,z0f);
                        glEnd();
                        glBegin(GL_LINE_LOOP);
                        glVertex3f(x0f,y0f,z1f); glVertex3f(x1f,y0f,z1f);
                        glVertex3f(x1f,y1f,z1f); glVertex3f(x0f,y1f,z1f);
                        glEnd();
                        glBegin(GL_LINES);
                        glVertex3f(x0f,y0f,z0f); glVertex3f(x0f,y0f,z1f);
                        glVertex3f(x1f,y0f,z0f); glVertex3f(x1f,y0f,z1f);
                        glVertex3f(x1f,y1f,z0f); glVertex3f(x1f,y1f,z1f);
                        glVertex3f(x0f,y1f,z0f); glVertex3f(x0f,y1f,z1f);
                        glEnd();
                    }
                }
            }
            glEnable(GL_LIGHTING);
        }

        SDL_GL_SwapWindow(win);
    }

    // textures were removed; nothing to delete
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
