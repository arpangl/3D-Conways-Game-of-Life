#include "life_common.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

namespace gol3d {

void run_visualization(const LifeConfig3D& cfg,
                       StepFunction3D stepper,
                       Backend backend,
                       std::size_t totalSteps,
                       double delaySeconds) {
    if (!stepper) {
        stepper = &step_single;
        backend = Backend::Single;
    }

    Grid3D current(cfg.width * cfg.height * cfg.depth, 0);
    Grid3D next(current.size(), 0);

    randomize(current, cfg);

    const auto sleepDuration = std::chrono::duration<double>(delaySeconds);
    const std::size_t middlePlane = cfg.depth / 2;

    for (std::size_t step = 0; step < totalSteps; ++step) {
        std::cout << "\033[2J\033[H";  // Clear screen and move cursor home.
        std::cout << "3D Game of Life [" << backend_to_string(backend) << "] "
                  << "(slice z=" << middlePlane << ") — step " << step << '\n';
        print_grid_slice(current, cfg, middlePlane, std::cout);
        std::cout.flush();

        stepper(current, next, cfg);
        current.swap(next);

        if (delaySeconds > 0.0) {
            std::this_thread::sleep_for(sleepDuration);
        }
    }
}

}  // namespace gol3d
