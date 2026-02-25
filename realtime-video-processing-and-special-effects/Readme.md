# CS 5330 Project 1: Real-Time Video Filtering

**Student:** Harsh Vijay Mamania  
**Date:** February 1, 2026  
**Course:** CS 5330 Pattern Recognition & Computer Vision  
**Instructor:** Professor Bruce Maxwell  
**Institution:** Northeastern University

---

## Project Overview

This project implements a real-time video processing interface with multiple image filters and effects, including classical computer vision techniques and modern deep learning-based depth estimation. 

---

## File Descriptions

### Core Implementation Files

**`imgDisplay.cpp`**  
Basic image loading and display program demonstrating fundamental OpenCV operations.

**`filter.h`**  
Header file with function declarations for all filters and effects.

**`filter.cpp`**  
Complete implementation of all filters including blur, edge detection, face effects, and creative extensions.

**`vidDisplay.cpp`**  
Main real-time video processing application with interactive keyboard-controlled filter selection.

### Extension 2: FFT Performance Study Files

**`experiment_filter.cpp`**  
Generalized blur implementations for performance comparison across naive, separable, FFT, and OpenCV methods.

**`experiment_benchmark.cpp`**  
Automated benchmarking program that measures and compares blur performance across multiple kernel sizes.

---

## Key Features

- Real-time video filtering with 15+ effects
- Face detection using Haar cascades
- Depth estimation with Depth Anything V2 neural network
- Custom creative extensions:
  - **The Rockstar Aura:** Adaptive sunglasses with three modes and animated particle effects
  - **FFT Performance Study:** Rigorous benchmarking of spatial vs. frequency domain blur approaches

---

## Dependencies

- OpenCV 4.x
- Visual Studio 2022
- ONNX Runtime (for depth estimation)
- C++17 or later

---

## Build Instructions

1. Open the solution file in Visual Studio 2022
2. Ensure OpenCV and ONNX Runtime are properly configured in project properties
3. Set the desired `.cpp` file as the startup file:
   - For image display: `imgDisplay.cpp`
   - For video filtering: `vidDisplay.cpp`
   - For FFT benchmarking: `experiment_benchmark.cpp`
4. Build and run (F5)

---

## Usage

### Video Display Mode (`vidDisplay.cpp`)
- **'g'** - Grayscale
- **'h'** - Custom grayscale
- **'e'** - Sepia tone
- **'b'** - Fast blur (separable)
- **'x'/'y'** - Sobel X/Y edge detection
- **'m'** - Gradient magnitude
- **'f'** - Face detection
- **'d'** - Depth estimation
- **'p'** - Portrait mode
- **'7'/'8'/'9'** - Sunglasses modes (opaque/adaptive/reflective)
- **'1'/'2'** - Adjust brightness
- **'3'/'4'** - Adjust contrast
- **'s'** - Save screenshot
- **'q'** - Quit

### FFT Benchmark Mode (`experiment_benchmark.cpp`)
Automatically runs performance tests and outputs results to console and `benchmark_results.csv`.

---
