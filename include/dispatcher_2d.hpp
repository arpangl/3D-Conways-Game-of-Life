#pragma once

#include "simulation_options.hpp"

void run_2d_cpu_single_species(const SimulationOptions2D &options);
void run_2d_cpu_multi_species(const SimulationOptions2D &options);
void run_2d_avx2_single_species(const SimulationOptions2D &options);
void run_2d_avx2_multi_species(const SimulationOptions2D &options);
void run_2d_openmp_single_species(const SimulationOptions2D &options);
void run_2d_openmp_multi_species(const SimulationOptions2D &options);
void run_2d_cuda_single_species(const SimulationOptions2D &options);
void run_2d_cuda_multi_species(const SimulationOptions2D &options);

void visualize_2d_state(const VisualizationOptions2D &options);
