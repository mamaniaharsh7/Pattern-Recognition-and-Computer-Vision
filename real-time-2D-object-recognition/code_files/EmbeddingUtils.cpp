/*
 * Name: Harsh Vijay Mamania
 * Date: 17th Feb 2026
 * 
 * embeddingUtils.cpp
 *
 * Utility functions for computing embeddings using ResNet18.
 * Original code by Bruce A. Maxwell.
 * Modified for Project 3: removed features.h and vision.h dependencies.
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "embeddingUtils.h"


 /*
  * Runs the input image through a ResNet18 network and returns a 512D embedding vector.
  *
  * src       - pre-processed ROI image (output of prepEmbeddingImage)
  * embedding - output 512D feature vector from the second-to-last layer
  * net       - pre-loaded ResNet18 DNN network
  * debug     - 1 to print embedding values and show intermediate images, 0 for silent
  *
  * Returns 0 on success
  */
int getEmbedding(cv::Mat& src, cv::Mat& embedding, cv::dnn::Net& net, int debug) {
    const int ORNet_size = 224;
    cv::Mat blob;
    cv::Mat resized;

    cv::resize(src, resized, cv::Size(ORNet_size, ORNet_size));

    cv::dnn::blobFromImage(resized,
        blob,
        (1.0 / 255.0) * (1 / 0.226),
        cv::Size(ORNet_size, ORNet_size),
        cv::Scalar(124, 116, 104),
        true,
        false,
        CV_32F);

    net.setInput(blob);
    embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0");

    if (debug) {
        std::cout << embedding << std::endl;
    }

    return 0;
}


/*
 * Rotates the original frame to align the primary axis with the X-axis and extracts the ROI.
 *
 * frame    - original BGR camera frame
 * embimage - output cropped and rotated ROI image
 * cx       - x coordinate of region centroid
 * cy       - y coordinate of region centroid
 * theta    - orientation angle of the primary axis in radians
 * minE1    - minimum projection along primary axis (negative value)
 * maxE1    - maximum projection along primary axis (positive value)
 * minE2    - minimum projection along secondary axis (negative value)
 * maxE2    - maximum projection along secondary axis (positive value)
 * debug    - 1 to show intermediate images, 0 for silent
 */
void prepEmbeddingImage(cv::Mat& frame, cv::Mat& embimage, int cx, int cy, float theta,
    float minE1, float maxE1, float minE2, float maxE2, int debug) {
    // Rotate the image to align the primary region axis with the x-axis
    cv::Mat rotatedImage;
    cv::Mat M;

    M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), -theta * 180 / M_PI, 1.0);
    int largest = frame.cols > frame.rows ? frame.cols : frame.rows;
    largest = (int)(1.414 * largest);
    cv::warpAffine(frame, rotatedImage, M, cv::Size(largest, largest));

    if (debug) {
        cv::imshow("rotated", rotatedImage);
    }

    int left = cx + (int)minE1;
    int top = cy - (int)maxE2;
    int width = (int)maxE1 - (int)minE1;
    int height = (int)maxE2 - (int)minE2;

    // Bounds check the ROI
    if (left < 0) {
        width += left;
        left = 0;
    }
    if (top < 0) {
        height += top;
        top = 0;
    }
    if (left + width >= rotatedImage.cols) {
        width = (rotatedImage.cols - 1) - left;
    }
    if (top + height >= rotatedImage.rows) {
        height = (rotatedImage.rows - 1) - top;
    }

    if (debug) {
        printf("ROI box: %d %d %d %d\n", left, top, width, height);
    }

    // Crop the image to the bounding box of the object
    cv::Rect objroi(left, top, width, height);
    cv::rectangle(rotatedImage,
        cv::Point2d(objroi.x, objroi.y),
        cv::Point2d(objroi.x + objroi.width, objroi.y + objroi.height),
        200, 4);

    cv::Mat extractedImage(rotatedImage, objroi);

    if (debug) {
        cv::imshow("extracted", extractedImage);
    }

    extractedImage.copyTo(embimage);
}