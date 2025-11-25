# 0.1.3.0更新
更新./tools/realtime_viewer_terrain.cpp
- 3D即時互動版
- 鄰居與Rules計算平行化
- 加入模擬地形功能(特殊初始化)
- 已測試模擬進化鍊

# 0.0.2.5進度條


## 1、2D單物種：
ok但沒互動純計時

## 2、2D多物種 （物種間無互動） ：
ok （加速過avx2  加速過cuda）但沒互動純計時

## 3、3D 單物種：
ok（加速過avx2 加速過cuda）但沒互動純計時

## 4、3D自定義多物種：
只有cpu互動版
各種加速版: 待補
離線計時版: 待補

## 5、重力+地形版：

未做


# 3D Conway's Game of Life

This project explores high-performance implementations of Conway's Game of Life extended to three dimensions. Multiple acceleration strategies – OpenMP, AVX/AVX2, AVX-512, and CUDA – are provided alongside a single-threaded baseline, ASCII visualizer, an optional OpenGL point-cloud viewer, benchmarking harness, and a reference 2D implementation.

## Rules: 2D vs 3D

**Classic 2D Life (B3/S23)**: Each cell observes eight neighbors on a 2D grid.

- A live cell survives with two or three neighbors.
- A dead cell becomes alive with exactly three neighbors.

**3D Life variant used here (B5/S45)**: Cells inhabit a 3D lattice and observe all 26 neighbors in the surrounding cube.

- A live cell survives when it has four or five neighbors.
- A dead cell is born when it has exactly five neighbors.

The 3D rule encourages richer volumetric structures and slower decay than the 2D counterpart because of the larger neighborhood.

## Repository Layout

```
.
├── AGENTS.md              — Task brief
├── README.md              — This document
├── Makefile               — Build script (g++)
├── include/
│   └── life_common.h      — Shared data structures and declarations
├── main.cpp               — Entry point and CLI
└── src/
    ├── 2d.cpp             — 2D Conway implementation
    ├── avx.cpp            — AVX/AVX2 accelerated stepper
    ├── avx512.cpp         — AVX-512 accelerated stepper
    ├── benchmark.cpp      — Benchmark harness
    ├── common.cpp         — Utilities (randomization, indexing, etc.)
    ├── cuda_stub.cpp      — CPU fallback used when CUDA is unavailable
    ├── cuda.cu            — CUDA kernel wrapper (built with NVCC when enabled)
    ├── openmp.cpp         — OpenMP accelerated stepper
    ├── single.cpp         — Scalar 3D baseline
    ├── visiualize.cpp     — ASCII visualization loop
    └── visualize_gl.cpp   — OpenGL/GLFW visualization (optional)
```

## Building

The project builds with `make` using `g++` and C++17:

```bash
make
```

Optional toggles (pass as `VAR=1 make`):

- `USE_OPENMP=0` – disable `-fopenmp` if your toolchain lacks OpenMP.
- `USE_AVX2=1` – compile with `-mavx2` to enable the AVX2 path.
- `USE_AVX512=1` – compile with `-mavx512f -mavx512bw` to activate AVX-512.
- `USE_CUDA=1` – compile CUDA support (requires NVCC; see below).
- `USE_GLFW=1` – build the OpenGL 3D viewer (needs GLFW, OpenGL headers/libs).

### CUDA builds

The provided `cuda.cu` contains a CUDA kernel, while `cuda_stub.cpp` keeps the binary functional when CUDA is disabled. Enable CUDA by compiling the project with NVCC for that compilation unit:

```bash
make clean
make USE_CUDA=1 NVCC=nvcc
```

You can override `NVCC` if the compiler lives elsewhere (e.g., `NVCC=/opt/cuda/bin/nvcc`). Linking might require `-lcudart` depending on your CUDA setup.

### GLFW/OpenGL builds

To enable the 3D viewport, install GLFW and OpenGL development packages, then:

```bash
make USE_GLFW=1 GLFW_LIBS="-lglfw -lGL -ldl -lpthread"
```

Adjust `GLFW_LIBS`/`GLFW_CFLAGS` if your installation lives outside default system paths.

## Running

The executable is named `life`. It supports several modes:

```bash
./life <mode> [options]
```

Common options:

- `--width/--height/--depth <n>` – grid dimensions (depth unused in 2D mode).
- `--steps <n>` – number of simulation steps.
- `--density <p>` – initial probability of a live cell.
- `--seed <n>` – RNG seed.
- `--wrap` / `--no-wrap` – enable or disable toroidal wrapping.
- `--backend <name>` – in `visualize`/`visualize3d` modes, pick the stepping backend (`single`, `openmp`, `avx`, `avx512`, `cuda`).

### Single run examples

```bash
# Scalar baseline
./life single --width 64 --height 64 --depth 64 --steps 50

# OpenMP with toroidal boundaries
./life openmp --width 96 --height 96 --depth 64 --wrap

# AVX baseline (requires -mavx2 at build time for SIMD acceleration)
./life avx --width 64 --height 64 --depth 64

# AVX-512 path (requires -mavx512f -mavx512bw)
./life avx512 --width 128 --height 128 --depth 64

# CUDA build (compiled with USE_CUDA=1)
./life cuda --width 128 --height 128 --depth 64

# Classic 2D Conway's Life
./life 2d --width 256 --height 256 --steps 200
```

### Visualization

Displays a middle Z slice in the terminal with ANSI escape codes:

```bash
./life visualize --width 48 --height 48 --depth 48 --backend openmp --visual-delay 0.1 --visual-steps 200
```

### 3D OpenGL visualization

Fixed-angle point cloud rendering of the full volume (requires `USE_GLFW=1` build):

```bash
./life visualize3d --width 48 --height 48 --depth 48 --backend avx512 --visual-delay 0.05 --visual-steps 0
```

Use `--visual-delay` to control seconds per simulation step (`0` for fastest). Close the window to exit.

### Benchmarking

Benchmark one or more backends:

```bash
# Run all CPU backends with defaults
./life benchmark --width 64 --height 64 --depth 64

# Specific backends and custom parameters
./life benchmark --backend single --backend avx --warmup 8 --measure 128 --reps 10
```

The benchmark reports average/min/max milliseconds per run along with cell counts.

## Extending

- Adjust the 3D rule in `src/single.cpp` and matching SIMD/CUDA implementations to explore alternate automata.
- Swap out the ASCII visualizer with an OpenGL/GLFW renderer by extending `run_visualization`.
- Integrate persistent data capture or volumetric outputs by re-using the grid helpers in `include/life_common.h`.

## License

Educational sample code – adapt freely for coursework or experimentation.
