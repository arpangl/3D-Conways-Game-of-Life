#include "dispatcher_2d.hpp"
#include "dispatcher_3d.hpp"
#include "simulation_options.hpp"

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool parse_bool(const std::string &value) {
    const auto lowered = to_lower(value);
    return lowered == "1" || lowered == "true" || lowered == "yes";
}

Backend parse_backend(const std::string &value) {
    const auto lowered = to_lower(value);
    if (lowered == "cpu") {
        return Backend::CPU;
    }
    if (lowered == "avx2") {
        return Backend::AVX2;
    }
    if (lowered == "openmp") {
        return Backend::OpenMP;
    }
    if (lowered == "cuda") {
        return Backend::CUDA;
    }
    throw std::invalid_argument("Unsupported backend: " + value);
}

MultiSpeciesMode parse_mode(const std::string &value) {
    const auto lowered = to_lower(value);
    if (lowered == "food_chain") {
        return MultiSpeciesMode::FoodChain;
    }
    if (lowered == "symbiosis") {
        return MultiSpeciesMode::Symbiosis;
    }
    throw std::invalid_argument("Unsupported multi-species mode: " + value);
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  ./main 2d --width 800 --height 600 --iterations 10000 --backend <cpu|avx2|openmp|cuda> --parallel_core 4 --species <1-4> --mode <food_chain|symbiosis> --seed 42 [--cuda_device 0 --cuda_streams 2 --avx2_tile 64 --avx2_alignment 32 --visualize 1]\n"
              << "  ./main 3d --width 512 --height 512 --depth 512 --iterations 10000 --seed 42 ...\n";
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const std::string dimension = to_lower(argv[1]);

    std::unordered_map<std::string, std::string> args;
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            std::cerr << "Unexpected token: " << key << "\n";
            return 1;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for argument: " << key << "\n";
            return 1;
        }
        args[key.substr(2)] = argv[++i];
    }

    try {
        if (dimension == "2d") {
            SimulationOptions2D options;
            if (auto it = args.find("width"); it != args.end()) {
                options.width = std::stoull(it->second);
            }
            if (auto it = args.find("height"); it != args.end()) {
                options.height = std::stoull(it->second);
            }
            if (auto it = args.find("iterations"); it != args.end()) {
                options.iterations = std::stoull(it->second);
            }
            if (auto it = args.find("seed"); it != args.end()) {
                options.seed = static_cast<std::uint32_t>(std::stoul(it->second));
            }
            if (auto it = args.find("visualize"); it != args.end()) {
                options.visualize = parse_bool(it->second);
            }
            if (auto it = args.find("backend"); it != args.end()) {
                options.backend = parse_backend(it->second);
            }
            if (auto it = args.find("parallel_core"); it != args.end()) {
                options.parallel_core = std::stoi(it->second);
            }
            if (auto it = args.find("species"); it != args.end()) {
                options.species = std::stoi(it->second);
            }
            if (options.species > 1) {
                if (auto it = args.find("mode"); it != args.end()) {
                    options.multi_species_mode = parse_mode(it->second);
                } else {
                    throw std::invalid_argument("--mode is required when species > 1");
                }
            }
            if (auto it = args.find("cuda_device"); it != args.end()) {
                options.cuda_device = std::stoi(it->second);
            }
            if (auto it = args.find("cuda_streams"); it != args.end()) {
                options.cuda_streams = std::stoi(it->second);
            }
            if (auto it = args.find("avx2_tile"); it != args.end()) {
                options.avx2_tile = std::stoull(it->second);
            }
            if (auto it = args.find("avx2_alignment"); it != args.end()) {
                options.avx2_alignment = std::stoull(it->second);
            }

            if (options.backend != Backend::OpenMP && args.find("parallel_core") != args.end()) {
                std::cout << "Warning: --parallel_core is only used with the OpenMP backend." << std::endl;
            }

            if (options.species <= 1) {
                switch (options.backend) {
                case Backend::CPU:
                    run_2d_cpu_single_species(options);
                    break;
                case Backend::AVX2:
                    run_2d_avx2_single_species(options);
                    break;
                case Backend::OpenMP:
                    run_2d_openmp_single_species(options);
                    break;
                case Backend::CUDA:
                    run_2d_cuda_single_species(options);
                    break;
                }
            } else {
                switch (options.backend) {
                case Backend::CPU:
                    run_2d_cpu_multi_species(options);
                    break;
                case Backend::AVX2:
                    run_2d_avx2_multi_species(options);
                    break;
                case Backend::OpenMP:
                    run_2d_openmp_multi_species(options);
                    break;
                case Backend::CUDA:
                    run_2d_cuda_multi_species(options);
                    break;
                }
            }
        } else if (dimension == "3d") {
            SimulationOptions3D options;
            if (auto it = args.find("width"); it != args.end()) {
                options.width = std::stoull(it->second);
            }
            if (auto it = args.find("height"); it != args.end()) {
                options.height = std::stoull(it->second);
            }
            if (auto it = args.find("depth"); it != args.end()) {
                options.depth = std::stoull(it->second);
            }
            if (auto it = args.find("iterations"); it != args.end()) {
                options.iterations = std::stoull(it->second);
            }
            if (auto it = args.find("seed"); it != args.end()) {
                options.seed = static_cast<std::uint32_t>(std::stoul(it->second));
            }
            if (auto it = args.find("visualize"); it != args.end()) {
                options.visualize = parse_bool(it->second);
            }
            if (auto it = args.find("backend"); it != args.end()) {
                options.backend = parse_backend(it->second);
            }
            if (auto it = args.find("parallel_core"); it != args.end()) {
                options.parallel_core = std::stoi(it->second);
            }
            if (auto it = args.find("species"); it != args.end()) {
                options.species = std::stoi(it->second);
            }
            if (options.species > 1) {
                if (auto it = args.find("mode"); it != args.end()) {
                    options.multi_species_mode = parse_mode(it->second);
                } else {
                    throw std::invalid_argument("--mode is required when species > 1");
                }
            }
            if (auto it = args.find("cuda_device"); it != args.end()) {
                options.cuda_device = std::stoi(it->second);
            }
            if (auto it = args.find("cuda_streams"); it != args.end()) {
                options.cuda_streams = std::stoi(it->second);
            }
            if (auto it = args.find("avx2_tile"); it != args.end()) {
                options.avx2_tile = std::stoull(it->second);
            }
            if (auto it = args.find("avx2_alignment"); it != args.end()) {
                options.avx2_alignment = std::stoull(it->second);
            }

            if (options.backend != Backend::OpenMP && args.find("parallel_core") != args.end()) {
                std::cout << "Warning: --parallel_core is only used with the OpenMP backend." << std::endl;
            }

            if (options.species <= 1) {
                switch (options.backend) {
                case Backend::CPU:
                    run_3d_cpu_single_species(options);
                    break;
                case Backend::AVX2:
                    run_3d_avx2_single_species(options);
                    break;
                case Backend::OpenMP:
                    run_3d_openmp_single_species(options);
                    break;
                case Backend::CUDA:
                    run_3d_cuda_single_species(options);
                    break;
                }
            } else {
                switch (options.backend) {
                case Backend::CPU:
                    run_3d_cpu_multi_species(options);
                    break;
                case Backend::AVX2:
                    run_3d_avx2_multi_species(options);
                    break;
                case Backend::OpenMP:
                    run_3d_openmp_multi_species(options);
                    break;
                case Backend::CUDA:
                    run_3d_cuda_multi_species(options);
                    break;
                }
            }
        } else {
            print_usage();
            return 1;
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
