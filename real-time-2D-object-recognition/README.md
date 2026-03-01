# Project 3: Real-time 2D Object Recognition

**Name:** Harsh Vijay Mamania
**Date:** February 17, 2026
**Course:** CS 5330 Pattern Recognition & Computer Vision
**Instructor:** Professor Bruce Maxwell
**Institution:** Northeastern University
**OS, IDE:** Windows; Visual Studio 2022

---

## Project Overview

This project implements a real-time 2D object recognition system capable of identifying objects placed on a uniform white background using a downward-facing webcam. The system processes live video through a multi-stage pipeline: dynamic thresholding, morphological filtering, connected components analysis, moment-based feature extraction, and nearest-neighbor classification. It supports recognition of up to 10 distinct objects and can detect up to 3 objects simultaneously. A CNN embedding-based one-shot classification mode using a pre-trained ResNet18 network is also implemented alongside the hand-crafted feature pipeline.

**Link to Demo Video:** https://drive.google.com/file/d/1dA2q8kXkPgLNkyLNd-A5uUciM8vvJaxO/view?usp=sharing

---

## Tasks and Related Files

| Task | Description |
|------|-------------|
| Task 1 | Thresholding |
| Task 2 | Morphological Filtering |
| Task 3 | Connected Components Analysis |
| Task 4 | Feature Computation |
| Task 5 | Training Data Collection |
| Task 6 | Classification |
| Task 7 | Confusion Matrix |
| Task 8 | CNN Embedding Classification |
| Extension | Multi-Object Detection |
| Extension | PCA Embedding Visualization |
| Main Loop | Camera capture, display, keyboard input |

All task implementations are located in **objectRecognition.h** and **objectRecognition.cpp**, except Task 8 which also uses **embeddingUtils.h** and **embeddingUtils.cpp**. The main loop is in **main.cpp**.

---

## What Each File Does

- **objectRecognition.h** - Header file declaring all functions across the full recognition pipeline, organized by task.
- **objectRecognition.cpp** - Implementation of all pipeline functions including thresholding, morphological filtering, connected components, feature computation, training, classification, confusion matrix, and embedding utilities.
- **embeddingUtils.h** - Header for professor-provided embedding utility functions. Original code by Bruce A. Maxwell.
- **embeddingUtils.cpp** - Implements `getEmbedding` (runs ResNet18 and extracts 512D embedding) and `prepEmbeddingImage` (rotates frame and extracts axis-aligned ROI). Original code by Bruce A. Maxwell.
- **main.cpp** - Main entry point. Handles camera capture, runs the full pipeline per frame, manages display windows, and handles all keyboard input.
- **object_db.csv** - Training database containing labeled feature vectors for all 10 objects collected during training mode.
- **resnet18.onnx** - Pre-trained ResNet18 network used for CNN embedding classification.

## Training Database

The file `object_db.csv` contains all pre-collected training examples for all 10 objects and is included in the submission. It is loaded automatically at startup. If you wish to retrain from scratch, delete or rename the existing `object_db.csv` and use the `n` key to collect new training examples during runtime.

---

## Key Features

- Dynamic ISODATA thresholding adapts automatically to lighting conditions without manual tuning
- Morphological closing implemented from scratch using custom dilate and erode functions
- Multi-object detection and classification of up to 3 objects simultaneously in real time
- Nearest-neighbor classification with scaled Euclidean distance normalization
- CNN one-shot classification using ResNet18 512D embeddings
- PCA-based 2D embedding scatter plot for visual inspection of embedding separability
- Supports 10 distinct object categories
- Screenshot capture saving all 5 pipeline stages simultaneously

---

## Dependencies

- OpenCV 4.x
- OpenCV DNN module (for ResNet18 inference)
- Visual Studio 2022
- resnet18.onnx (pre-trained network file, must be placed in the project directory)

---

## Build Instructions

1. Clone or download the project directory.
2. Open the solution in Visual Studio 2022.
3. Ensure OpenCV is correctly linked in the project properties (include directories, library directories, and additional dependencies).
4. Place resnet18.onnx in the project working directory.
5. Build the solution in Release or Debug mode.
6. Run the executable. The webcam will open automatically.

---

## Usage

| Key | Action |
|-----|--------|
| n | Training mode: prompts for a label and saves the current feature vector to object_db.csv |
| t | Embedding training mode: prompts for a label and saves the current CNN embedding to memory |
| x | Toggle between hand-crafted feature classification and CNN embedding classification |
| v | Display PCA 2D scatter plot of all stored embeddings |
| e | Evaluation mode: prompts for the true label and logs the true/predicted pair for confusion matrix |
| m | Print the confusion matrix to the console from all logged evaluation pairs |
| s | Save screenshots of all 5 pipeline windows as numbered image files |
| q | Quit the application |
