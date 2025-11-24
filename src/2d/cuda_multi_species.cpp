#include "dispatcher_2d.hpp"
#include <iostream>

// Wrapper: call the CUDA implementation if compiled with nvcc, otherwise fall back to CPU.
#ifdef CUDA_STUB
extern void run_2d_cpu_multi_species(const SimulationOptions2D &options);
void run_2d_cuda_multi_species(const SimulationOptions2D &options) {
    std::cerr << "CUDA backend not compiled with nvcc; falling back to CPU." << std::endl;
    run_2d_cpu_multi_species(options);
}
#else
extern void run_2d_cuda_multi_species_impl(const SimulationOptions2D &options);
void run_2d_cuda_multi_species(const SimulationOptions2D &options) {
    run_2d_cuda_multi_species_impl(options);
}
#endif

