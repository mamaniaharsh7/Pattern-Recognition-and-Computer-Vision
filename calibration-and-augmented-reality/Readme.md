# Project 4: Calibration and Augmented Reality

**Name:** Harsh Vijay Mamania  

---

## Project Overview

This project implements a complete camera calibration and augmented reality pipeline using C++ and OpenCV. It covers chessboard corner detection, camera intrinsic calibration, real-time pose estimation using solvePnP, and rendering of virtual 3D objects above a physical checkerboard target. Two extensions explore Harris vs ORB feature detection comparison and simultaneous multi-target AR.

---

## Code Files

### Calibration Program (Tasks 1 to 3)

**`calibration.h`**  
Header file for the calibration program. Declares constants for board dimensions (9x6 internal corners), square size, and function signatures for corner detection, calibration frame saving, camera calibration, and intrinsics file writing.

**`calibration.cpp`**  
Implementation of all calibration functions. Handles grayscale conversion, chessboard corner detection with sub-pixel refinement, world point construction, camera matrix initialization, calibration via cv::calibrateCamera, and saving intrinsics to a yml file.

**`main.cpp`**  
Entry point for the calibration program. Opens the webcam, runs the real-time corner detection loop, and handles keypresses: 's' to save a calibration frame, 'c' to run calibration (minimum 5 frames required), and 'q' to quit.

---

### AR Program (Tasks 4 to 6 + Extension: Multiple Targets)

**`ar.h`**  
Header file for the AR program. Declares board constants mirroring the calibration program, and function signatures for loading intrinsics, corner detection, world point construction, pose estimation, axis projection, and virtual object rendering.

**`ar.cpp`**  
Implementation of all AR functions. Loads camera intrinsics from file, detects corners with sub-pixel refinement, builds world point sets, estimates board pose using solvePnP, projects 3D coordinate axes onto the image, and renders a virtual rocket ship constructed from 24 3D points across four components (body, nose cone, fins, exhaust flame). Supports two color schemes for multi-target rendering.

**`ar_main.cpp`**  
Entry point for the AR program. Loads intrinsics, opens the webcam, and runs the real-time AR loop. Implements a two-pass masking strategy to detect and track two independent checkerboards simultaneously, estimating independent poses and rendering a distinctly colored rocket above each board. Press 's' to save a screenshot, 'q' to quit.

---

### Feature Detection (Task 7 + Extension: Harris vs ORB)

**`features.cpp`**  
Standalone program for Task 7. Detects and visualizes Harris corners on a live webcam feed in real time. Implements temporal smoothing and non-maximum suppression for stable detection. Press '+' and '-' to adjust the detection threshold, 'q' to quit.

**`features_compare.cpp`**  
Extension program comparing Harris corner detection and ORB feature detection side by side on a live webcam feed. Displays three simultaneous windows: Harris detection, ORB detection, and ORB feature matching between consecutive frames. Match quality is evaluated using a dynamic distance threshold. Press '+'/'-' to adjust Harris threshold, 'u'/'d' to adjust ORB feature count, 's' to save screenshots of all three windows simultaneously, 'q' to quit.

---

## Notes

- Run the calibration program first and press 'c' to generate `intrinsics.yml` before running the AR program.
- `intrinsics.yml` must be present in the same directory as the AR program executable.
- To switch between programs in Visual Studio, set the desired entry point file to included in build and exclude all other main files via Solution Explorer properties.
