# CS 5330 Project 2: Content-Based Image Retrieval
 
**Contributors:** Harsh Mamania
**Date:** February 9th, 2026
 
---
 
## Tasks Summary
 
- **Task 1:** Baseline Matching - using pixel-level features (ssd_match.cpp)
- **Task 2:** Histogram Matching - RG Chromaticity & RGB (matching.cpp, features.cpp)
- **Task 3:** Multi-Histogram Matching (matching.cpp, features.cpp)
- **Task 4:** Texture (Sobel gradient magnitude histograms) and Color (HSV histogram) matching (image_match.cpp, filter.cpp)
- **Task 5:** Deep Network Embeddings - using pre-trained ResNet18 (matching.cpp, features.cpp)
- **Task 6:** DNN Embeddings & Classic Features Comparison (analysis across all methods)
- **Task 7:** Hybrid Deep and Classical Feature Fusion (custom_match.cpp)
- **Extension1:**  PCA experimentation study: to look closely at the intrinsic dimensionality and subspace structure of 512-dimensional ResNet embeddings under varying compression regimes (matching.cpp, features.cpp)
- **Extension2:** Segmentation-Based Deep Features for Image Retrieval

---
 
## File Descriptions
 
### Core Implementation Files
 
**features.h**  
Header file declaring all feature extraction and distance metric functions for Tasks 2, 3, 5, and Extension (PCA). Contains function prototypes for histogram computation, deep network embeddings, PCA dimensionality reduction, and match overlap calculation.
 
**features.cpp**  
Implementation of all feature extraction and distance computation functions for Tasks 2, 3, 5, and Extension (PCA). Includes RG/RGB histograms, multi-histogram matching, ResNet18 embedding processing, PCA-based dimensionality reduction, and similarity metrics.
 
**matching.cpp**  
Main retrieval program for Tasks 2, 3, 5, and Extension (PCA). Performs histogram-based matching (RG chromaticity, RGB, multi-histogram) and deep network embedding retrieval with comprehensive PCA dimensionality reduction analysis from 512D down to 8D.
 
**ssd_match.cpp**  
**Task 1:** Baseline matching using 7×7 center pixel patch with manual Sum of Squared Differences computation. Validates implementation correctness and establishes pixel-level matching performance baseline.
 
**image_match.cpp**  
**Task 4:** Texture and color matching combining HSV color histograms (16×16×16 bins) with Sobel gradient texture histograms (32 bins). Extracts features from center 160×160 patch with 30% color and 70% texture weighting.
 
**custom_match.cpp**  
**Task 7:** Custom hybrid design fusing ResNet18 deep embeddings (60%), HSV color histograms (20%), and Sobel texture histograms (20%). Displays both most similar and least similar images for comprehensive retrieval analysis.
 
**filter.h**  
Header file for image filtering and gradient computation utilities used in Task 4. Declares grayscale conversion, Sobel X/Y operators, and gradient magnitude calculation functions.
 
**filter.cpp**  
Implementation of Sobel gradient operators and grayscale conversion functions for Task 4 texture feature extraction. Provides manual 3×3 Sobel kernels for horizontal and vertical gradient computation with magnitude calculation.
 
---
 
## Compilation & Usage
 
Compile each executable separately based on task requirements. All programs follow standard usage:
```
./<executable> <target_image> <image_directory> <topN>
```
 
For `matching.cpp`, specify feature type:
```
./matching <target_image> <directory> <feature_type> [N]
```
Where `feature_type` is: `rg`, `rgb`, `multi`, or `dnn`
 
---
