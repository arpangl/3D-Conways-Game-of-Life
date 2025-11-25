// SDL-based PPM sequence viewer with simple interaction.
// Usage: ./ppm_viewer_sdl --dir frames --delay 100 --duration 0

#include <SDL2/SDL.h>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

struct Frame {
    std::string path;
    int width;
    int height;
    std::vector<unsigned char> data; // RGB
};

static bool read_ppm(const std::string &path, Frame &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    f >> magic;
    if (magic != "P6") return false;
    // skip comments
    while (f.peek() == '#') {
        std::string line;
        std::getline(f, line);
    }
    int w=0,h=0,maxv=0;
    f >> w >> h >> maxv;
    if (w <= 0 || h <= 0) return false;
    f.get(); // consume single whitespace
    out.width = w; out.height = h;
    out.data.resize(static_cast<std::size_t>(w) * h * 3);
    f.read(reinterpret_cast<char*>(out.data.data()), out.data.size());
    return true;
}

int main(int argc, char **argv) {
    std::string dir = "frames";
    int delay_ms = 100;
    int duration_s = 0; // 0 == forever until user quits
    for (int i=1;i<argc;++i) {
        std::string a = argv[i];
        if (a == "--dir" && i+1<argc) dir = argv[++i];
        else if (a == "--delay" && i+1<argc) delay_ms = std::stoi(argv[++i]);
        else if (a == "--duration" && i+1<argc) duration_s = std::stoi(argv[++i]);
    }

    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "Directory not found: " << dir << std::endl;
        return 1;
    }

    std::vector<fs::path> files;
    for (auto &p: fs::directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        std::string name = p.path().filename().string();
        if (name.rfind("frame_",0) == 0 && p.path().extension() == ".ppm") files.push_back(p.path());
    }
    if (files.empty()) {
        std::cerr << "No frames found in " << dir << std::endl;
        return 1;
    }
    std::sort(files.begin(), files.end());

    std::vector<Frame> frames;
    for (auto &p: files) {
        Frame f;
        if (read_ppm(p.string(), f)) {
            f.path = p.string();
            frames.push_back(std::move(f));
        }
    }
    if (frames.empty()) { std::cerr << "No valid frames." << std::endl; return 1; }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    const Frame &first = frames.front();
    SDL_Window *win = SDL_CreateWindow("Conway Viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, first.width, first.height, SDL_WINDOW_RESIZABLE);
    if (!win) { std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl; SDL_Quit(); return 1; }
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren) { std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl; SDL_DestroyWindow(win); SDL_Quit(); return 1; }

    size_t idx = 0;
    bool running = true;
    bool paused = false;
    auto start = std::chrono::steady_clock::now();

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_SPACE) paused = !paused;
                else if (e.key.keysym.sym == SDLK_RIGHT) idx = (idx + 1) % frames.size();
                else if (e.key.keysym.sym == SDLK_LEFT) idx = (idx + frames.size() - 1) % frames.size();
                else if (e.key.keysym.sym == SDLK_ESCAPE || e.key.keysym.sym == SDLK_q) running = false;
                else if (e.key.keysym.sym == SDLK_UP) delay_ms = std::max(1, delay_ms - 10);
                else if (e.key.keysym.sym == SDLK_DOWN) delay_ms += 10;
            }
        }

        if (!paused) {
            const Frame &fr = frames[idx];
            SDL_Surface *surf = SDL_CreateRGBSurfaceFrom((void*)fr.data.data(), fr.width, fr.height, 24, fr.width * 3,
                                                         0x0000FF, 0x00FF00, 0xFF0000, 0);
            if (!surf) { std::cerr << "SDL_CreateRGBSurfaceFrom: " << SDL_GetError() << std::endl; break; }
            SDL_Texture *tex = SDL_CreateTextureFromSurface(ren, surf);
            SDL_FreeSurface(surf);
            if (!tex) { std::cerr << "SDL_CreateTextureFromSurface: " << SDL_GetError() << std::endl; break; }
            SDL_RenderClear(ren);
            SDL_RenderCopy(ren, tex, nullptr, nullptr);
            SDL_DestroyTexture(tex);
            SDL_RenderPresent(ren);

            idx = (idx + 1) % frames.size();
        }

        if (duration_s > 0) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= duration_s) break;
        }

        SDL_Delay(delay_ms);
    }

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
