/*
 * Harsh Mamania
 * 10th February 2026
 *
 * Header file for Content-Based Image Retrieval (CBIR) feature extraction and distance metrics.
 * Contains function declarations for Tasks 2, 3, 5, and Extension (PCA dimensionality reduction).
 */

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <string>

 // Structure to hold image filename and distance for sorting matches
struct ImageMatch {
    std::string filename;
    float distance;

    // Comparison operator for sorting (smallest distance first = most similar)
    bool operator<(const ImageMatch& other) const {
        return distance < other.distance;
    }
};

// ========== TASK 2: HISTOGRAM MATCHING ==========

// Computes RG chromaticity histogram with brightness normalization
cv::Mat computeRGHistogram(const cv::Mat& src, int histsize);

// Computes RGB color histogram with specified bins per channel
cv::Mat computeRGBHistogram(const cv::Mat& src, int bins_per_channel);

// Computes histogram intersection distance between two histograms
float histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2);



// ========== TASK 3: MULTI-HISTOGRAM MATCHING ==========

// Computes RGB histogram for center 50% region of image
cv::Mat computeCenterHistogram(const cv::Mat& src, int bins_per_channel);

// Placeholder function (not used in final implementation)
cv::Mat computeOuterHistogram(const cv::Mat& src, int bins_per_channel);

// Computes weighted distance combining whole-image and center-region histograms
float multiHistogramDistance(const cv::Mat& center_hist1, const cv::Mat& outer_hist1,
    const cv::Mat& center_hist2, const cv::Mat& outer_hist2);



// ========== TASK 5: DEEP NETWORK EMBEDDINGS ==========

// Reads 512D ResNet18 embeddings from CSV file into map
std::map<std::string, std::vector<float>> readEmbeddings(const std::string& csv_file);

// Computes sum of squared differences between two feature vectors
float computeSSD(const std::vector<float>& v1, const std::vector<float>& v2);

// Computes cosine distance (1 - cosine similarity) between two vectors
float cosineDistance(const std::vector<float>& v1, const std::vector<float>& v2);

// ========== EXTENSION: PCA DIMENSIONALITY REDUCTION ==========

// Applies PCA to reduce embedding dimensionality, returns reduced embeddings
std::map<std::string, std::vector<float>> applyPCA(
    const std::map<std::string, std::vector<float>>& embeddings,
    int n_components,
    std::vector<float>& mean_vec,
    cv::Mat& eigenvectors
);

// Projects a single embedding to PCA space using pre-computed mean and eigenvectors
std::vector<float> projectToPCA(
    const std::vector<float>& embedding,
    const std::vector<float>& mean_vec,
    const cv::Mat& eigenvectors
);

// Computes percentage overlap between two sets of top-N matches
float computeMatchOverlap(const std::vector<ImageMatch>& matches1,
    const std::vector<ImageMatch>& matches2, int N);

#endif