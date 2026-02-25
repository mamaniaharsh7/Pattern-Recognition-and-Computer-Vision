/*
 * Harsh Vijay Mamania
 * February 1, 2026
 * CS 5330 Project 1
 *
 * Purpose: Header file containing filter function declarations and structures
 */

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>
#include <vector>

 // Particle structure for animated sparkle effects
struct Sparkle {
    float x, y;           // Current position
    float vx, vy;         // Velocity components
    int lifetime;         // Frames until respawn
    bool visible;         // Twinkling state
};

// Custom grayscale conversion
int my_new_grayscale(cv::Mat& src, cv::Mat& dst);

// Sepia tone filter
int my_sepia_filter(cv::Mat& src, cv::Mat& dst);

// Naive 5x5 Gaussian blur using at() accessor
int blur5x5_1(cv::Mat& src, cv::Mat& dst);

// Optimized 5x5 Gaussian blur using separable filters
int blur5x5_2(cv::Mat& src, cv::Mat& dst);

// 3x3 Sobel filter for horizontal edges
int sobelX3x3(cv::Mat& src, cv::Mat& dst);

// 3x3 Sobel filter for vertical edges
int sobelY3x3(cv::Mat& src, cv::Mat& dst);

// Gradient magnitude from Sobel X and Y
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

// Blur and quantize image to specified levels
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);

// Adjust image brightness by offset
int modify_brightness(cv::Mat& src, cv::Mat& dst, int brightness_offset);

// Adjust image contrast by factor
int modify_contrast(cv::Mat& src, cv::Mat& dst, float contrast_factor);

// Embossing effect using directional gradients
int emboss(cv::Mat& src, cv::Mat& dst);

// Portrait mode with face-focused blur
int portraitMode(cv::Mat& src, cv::Mat& dst, std::vector<cv::Rect>& faces);

// Render sunglasses on detected face with mode selection
int addSunglasses(cv::Mat& frame, cv::Rect face, int mode);

// Update and render animated sparkle particles around face
void updateAndDrawSparkles(cv::Mat& frame, cv::Rect face, std::vector<Sparkle>& sparkles);

// FFT Performance Study Functions
// Naive blur with generalized kernel size
int experiment_blurNaive(cv::Mat& src, cv::Mat& dst, int kernel_size);

// Separable blur with generalized kernel size
int experiment_blurSeparable(cv::Mat& src, cv::Mat& dst, int kernel_size);

// FFT-based blur with optional visualization
int experiment_blurFFT(cv::Mat& src, cv::Mat& dst, int kernel_size, bool show_steps = false);

// OpenCV GaussianBlur wrapper for benchmarking
int experiment_blurOpenCV(cv::Mat& src, cv::Mat& dst, int kernel_size);

// Generate Gaussian kernel of specified size
cv::Mat experiment_createGaussianKernel(int kernel_size, double sigma);

#endif