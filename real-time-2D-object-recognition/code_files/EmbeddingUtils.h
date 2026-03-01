/*
 * Name: Harsh Vijay Mamania
 * Date: 17th Feb 2026
 *
 * embeddingUtils.h
 *
 * Header for professor-provided embedding utility functions.
 * Original code by Bruce A. Maxwell.
 * Modified for Project 3: removed features.h and vision.h dependencies.
 */

#ifndef EMBEDDING_UTILS_H
#define EMBEDDING_UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


 // Runs the input image through a ResNet18 network and returns a 512D embedding vector.
int getEmbedding(cv::Mat& src, cv::Mat& embedding, cv::dnn::Net& net, int debug);

// Rotates the original frame to align the primary axis with the X-axis and extracts the ROI.
void prepEmbeddingImage(cv::Mat& frame, cv::Mat& embimage, int cx, int cy, float theta,
    float minE1, float maxE1, float minE2, float maxE2, int debug);


#endif // EMBEDDING_UTILS_H