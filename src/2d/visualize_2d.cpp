#include "dispatcher_2d.hpp"
#include <iostream>
#include <thread>

void visualize_2d_state(const VisualizationOptions2D &options) {
    // Minimal visualization helper. The simulation currently writes PPM
    // frames to ./frames/. This function is left as a small helper that
    // informs the user how to view frames. A full real-time renderer
    // would be added later if desired.
    (void)options;
    std::cout << "Visualization: frames written to ./frames/ (if enabled by backend)\n";
    std::cout << "You can view frames with an image viewer or combine into a GIF." << std::endl;
}
