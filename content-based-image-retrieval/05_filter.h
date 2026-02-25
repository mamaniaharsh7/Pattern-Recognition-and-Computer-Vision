/*
 * Harsh Mamania
 * 10th February 2026
 *
 * Header file for image filtering and gradient computation functions.
 * Used in Task 4 for texture feature extraction.
 */

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

 // Converts image to grayscale using OpenCV function
cv::Mat toGrayscale(const cv::Mat& input);

// Custom grayscale conversion with weighted RGB channels
int GreyScale(const cv::Mat& src, cv::Mat& dst);

// Computes horizontal Sobel gradient (3x3 kernel)
int sobelX3x3(cv::Mat& src, cv::Mat& dst);

// Computes vertical Sobel gradient (3x3 kernel)
int sobelY3x3(cv::Mat& src, cv::Mat& dst);

// Computes gradient magnitude from X and Y Sobel outputs
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

#endif