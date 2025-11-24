// Simple terminal PPM viewer for frame sequences.
// Usage: ./ppm_viewer --dir frames --delay 100 --duration 5

#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

struct Frame {
    std::string path;
    std::size_t width;
    std::size_t height;
    std::vector<unsigned char> data; // RGB
};

static bool read_ppm(const std::string &path, Frame &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    f >> magic;
    if (magic != "P6") return false;
    // skip whitespace/comments
    int w=0,h=0,maxv=0;
    while (f.peek() == '#') {
        std::string line;
        std::getline(f, line);
    }
    f >> w >> h >> maxv;
    f.get(); // single whitespace
    out.width = static_cast<std::size_t>(w);
    out.height = static_cast<std::size_t>(h);
    std::size_t n = out.width * out.height * 3;
    out.data.resize(n);
    f.read(reinterpret_cast<char*>(out.data.data()), n);
    return true;
}

static void clear_screen() {
    std::cout << "\x1b[H\x1b[2J";
}

int main(int argc, char **argv) {
    std::string dir = "frames";
    int delay_ms = 100;
    int duration_s = 5;
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
    if (frames.empty()) {
        std::cerr << "No valid PPM frames loaded." << std::endl;
        return 1;
    }

    auto start = std::chrono::steady_clock::now();
    std::size_t idx = 0;
    const bool loop = true;
    while (true) {
        const Frame &fr = frames[idx];
        // Render as ASCII by mapping each pixel block to a character
        // Downscale to terminal size roughly
        const int term_w = 80;
        const int term_h = 24;
        int sx = std::max<int>(1, fr.width / term_w);
        int sy = std::max<int>(1, fr.height / term_h);

        clear_screen();
        for (std::size_t y = 0; y < fr.height; y += sy) {
            for (std::size_t x = 0; x < fr.width; x += sx) {
                int rsum=0,gsum=0,bsum=0;
                int cnt=0;
                for (int yy=0; yy<sy && y+yy<fr.height; ++yy) for (int xx=0; xx<sx && x+xx<fr.width; ++xx) {
                    std::size_t pos = ((y+yy)*fr.width + (x+xx)) * 3;
                    rsum += fr.data[pos+0];
                    gsum += fr.data[pos+1];
                    bsum += fr.data[pos+2];
                    ++cnt;
                }
                int r = rsum / cnt;
                int g = gsum / cnt;
                int b = bsum / cnt;
                int lum = (r*299 + g*587 + b*114) / 1000;
                if (lum > 128) std::cout << "█";
                else std::cout << " ";
            }
            std::cout << "\n";
        }
        std::cout << "Frame: " << fr.path << " (" << (idx+1) << "/" << frames.size() << ")" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        idx++;
        if (idx >= frames.size()) {
            if (loop) idx = 0;
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= duration_s) break;
    }

    return 0;
}
