#include "life_common.h"

#include <iostream>

namespace gol3d {

#ifndef USE_CUDA

void step_cuda(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg) {
    static bool warned = false;
    if (!warned) {
        std::cerr << "[CUDA] Backend compiled without CUDA support; falling back to AVX implementation.\n";
        warned = true;
    }
    step_avx(current, next, cfg);
}

#endif  // USE_CUDA

}  // namespace gol3d

