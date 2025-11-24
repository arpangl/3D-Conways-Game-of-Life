#include "dispatcher_3d.hpp"
#include <iostream>

// Wrapper: call the CUDA implementation if compiled with nvcc, otherwise fall back to CPU.
#ifdef CUDA_STUB
extern void run_3d_cpu_single_species(const SimulationOptions3D &options);
void run_3d_cuda_single_species(const SimulationOptions3D &options) {
    std::cerr << "CUDA backend not compiled with nvcc; falling back to CPU." << std::endl;
    run_3d_cpu_single_species(options);
}
#else
extern void run_3d_cuda_single_species_impl(const SimulationOptions3D &options);
void run_3d_cuda_single_species(const SimulationOptions3D &options) {
    run_3d_cuda_single_species_impl(options);
}
#endif

