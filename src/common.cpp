#include "simulation_options.hpp"

std::string backend_to_string(Backend backend) {
    switch (backend) {
    case Backend::CPU:
        return "CPU";
    case Backend::AVX2:
        return "AVX2";
    case Backend::OpenMP:
        return "OpenMP";
    case Backend::CUDA:
        return "CUDA";
    }
    return "Unknown";
}
