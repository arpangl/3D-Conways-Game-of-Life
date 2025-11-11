## 3D Conway's Game of Life

### What's this project
This project is related to building an 3D Conway's Game of Life, we're going to make use of OpenMP, CUDA, AVX512, AVX/AVX2 to build a paralled 3D Conway's Game of Life. 
Besides, we'll also conduct a benchmark for each component of the speed-up techniques individually.

### Files supposed to be compelete
1. README.md
In `README.md`, you should list the structure of this project and explain that what's the rule of 3D Conway's Game of Life, what's the difference between 2D's Conway's Game of Life. Additionally, you should write down how to run this project under different conditions.
2. main.cpp
This is the entrypoint of the entire project.
3. src/single.cpp
This is the implementation of single threaded version.
4. src/avx.cpp
This file is the implementation of AVX/AVX2 version.
5. src/avx512.cpp
This file is the implementation of AVX512 version.
6. src/openmp.cpp
This file is the implementation of OpenMP version.
7. src/cuda.cu
This file is the implementation of CUDA version.
8. src/visiualize.cpp
This file is to visiulize the process of the game, i think it will use OpenGL to render the frame simultaneously.
9. src/benchmark.cpp
This file runs the benchmark of different acceleration techniques, even on screen or off screen.
10. src/2d.cpp
This file is the implementation of 2D Conway's Game of Life.
11. Makefile
This file is to build up the whole project
12. Other files as needed

### Building System
only `make` can be used

### Commit?
No need.
