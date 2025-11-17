#pragma once

#include "simulation_options.hpp"

void run_3d_cpu_single_species(const SimulationOptions3D &options);
void run_3d_cpu_multi_species(const SimulationOptions3D &options);
void run_3d_avx2_single_species(const SimulationOptions3D &options);
void run_3d_avx2_multi_species(const SimulationOptions3D &options);
void run_3d_openmp_single_species(const SimulationOptions3D &options);
void run_3d_openmp_multi_species(const SimulationOptions3D &options);
void run_3d_cuda_single_species(const SimulationOptions3D &options);
void run_3d_cuda_multi_species(const SimulationOptions3D &options);

void visualize_3d_state(const VisualizationOptions3D &options);
