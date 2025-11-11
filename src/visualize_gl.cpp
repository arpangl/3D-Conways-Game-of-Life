#include "life_common.h"

#include <stdexcept>

#ifdef USE_GLFW

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace gol3d {

namespace {

struct CameraSettings {
    float distance = 2.5f;
    float pitchDeg = 30.0f;
    float yawDeg = 45.0f;
};

void setup_matrices(int width, int height, const LifeConfig3D& cfg, const CameraSettings& cam) {
    const float aspect = (height == 0) ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
    constexpr float kPi = 3.14159265358979323846f;
    const float fov = 45.0f;
    const float nearPlane = 0.1f;
    const float farPlane = 100.0f;

    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    const float top = std::tan(fov * kPi / 360.0f) * nearPlane;
    const float bottom = -top;
    const float right = top * aspect;
    const float left = -right;
    const float A = (farPlane + nearPlane) / (nearPlane - farPlane);
    const float B = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
    const float proj[16] = {
        (2.0f * nearPlane) / (right - left), 0.0f, 0.0f, 0.0f,
        0.0f, (2.0f * nearPlane) / (top - bottom), 0.0f, 0.0f,
        (right + left) / (right - left), (top + bottom) / (top - bottom), A, -1.0f,
        0.0f, 0.0f, B, 0.0f};
    glLoadMatrixf(proj);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(0.0f, 0.0f, -cam.distance);
    glRotatef(cam.pitchDeg, 1.0f, 0.0f, 0.0f);
    glRotatef(cam.yawDeg, 0.0f, 1.0f, 0.0f);
    glTranslatef(-static_cast<float>(cfg.width) / 2.0f,
                 -static_cast<float>(cfg.height) / 2.0f,
                 -static_cast<float>(cfg.depth) / 2.0f);
}

void draw_grid(const Grid3D& grid, const LifeConfig3D& cfg) {
    glPointSize(4.0f);
    glBegin(GL_POINTS);
    for (std::size_t z = 0; z < cfg.depth; ++z) {
        for (std::size_t y = 0; y < cfg.height; ++y) {
            for (std::size_t x = 0; x < cfg.width; ++x) {
                const auto idx = index_3d(cfg, x, y, z);
                if (!grid[idx]) {
                    continue;
                }
                const float intensity = 0.3f + 0.7f * static_cast<float>(z) / static_cast<float>(cfg.depth);
                glColor3f(intensity, 0.8f * intensity, 1.0f);
                glVertex3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
            }
        }
    }
    glEnd();
}

}  // namespace

void run_visualization_gl(const LifeConfig3D& cfg,
                          StepFunction3D stepper,
                          Backend backend,
                          std::size_t totalSteps,
                          double secondsPerStep) {
    if (!stepper) {
        stepper = &step_single;
        backend = Backend::Single;
    }
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    std::string title = "3D Game of Life (" + backend_to_string(backend) + ")";
    GLFWwindow* window = glfwCreateWindow(960, 720, title.c_str(), nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

    Grid3D current(cfg.width * cfg.height * cfg.depth, 0);
    Grid3D next(current.size(), 0);
    randomize(current, cfg);

    CameraSettings cam;
    cam.distance = static_cast<float>(std::max({cfg.width, cfg.height, cfg.depth})) * 1.8f;

    const double stepInterval = secondsPerStep > 0.0 ? secondsPerStep : 0.0;
    double lastStepTime = glfwGetTime();
    std::size_t stepsSimulated = 0;

    while (!glfwWindowShouldClose(window)) {
        int frameWidth = 0;
        int frameHeight = 0;
        glfwGetFramebufferSize(window, &frameWidth, &frameHeight);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        setup_matrices(std::max(frameWidth, 1), std::max(frameHeight, 1), cfg, cam);

        draw_grid(current, cfg);

        glfwSwapBuffers(window);
        glfwPollEvents();

        const double now = glfwGetTime();
        if (stepInterval == 0.0 || now - lastStepTime >= stepInterval) {
            stepper(current, next, cfg);
            current.swap(next);
            lastStepTime = now;
            ++stepsSimulated;

            if (totalSteps > 0 && stepsSimulated >= totalSteps) {
                break;
            }
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}

}  // namespace gol3d

#else

namespace gol3d {

void run_visualization_gl(const LifeConfig3D&, StepFunction3D, Backend, std::size_t, double) {
    throw std::runtime_error("OpenGL visualization requested but built without USE_GLFW=1");
}

}  // namespace gol3d

#endif  // USE_GLFW
