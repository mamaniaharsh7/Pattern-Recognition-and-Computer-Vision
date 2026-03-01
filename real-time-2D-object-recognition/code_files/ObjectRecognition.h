/*
 * Name: Harsh Vijay Mamania
 * Date: 17th Feb 2026
 *
 * objectRecognition.h
 *
 * Header file for Project 3: Real-time 2D Object Recognition
 * Declares all functions used across the recognition pipeline.
 */

#ifndef OBJECT_RECOGNITION_H
#define OBJECT_RECOGNITION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


 // TASK 1 - Thresholding

 // Blurs the input frame, converts to HSV, and computes saturation-weighted brightness into a single channel.
int preprocessFrame(const cv::Mat& src, cv::Mat& dst);

// Estimates a binary threshold value using the ISODATA algorithm (k-means K=2) on a pixel sample.
int computeDynamicThreshold(const cv::Mat& modifiedV);

// Applies a binary threshold to the modified V channel, producing a foreground/background image.
int applyThreshold(const cv::Mat& modifiedV, cv::Mat& dst, int threshold_value);

// Full Task 1 pipeline wrapper: preprocessFrame -> computeDynamicThreshold -> applyThreshold.
int thresholdFrame(const cv::Mat& src, cv::Mat& dst);


// TASK 2 - Morphological Filtering

// Applies a single dilation pass to the binary image using a 3x3 kernel.
int dilate(const cv::Mat& src, cv::Mat& dst);

// Applies a single erosion pass to the binary image using a 3x3 kernel.
int erode(const cv::Mat& src, cv::Mat& dst);

// Applies morphological closing (dilation then erosion) to fill holes in foreground regions.
int applyMorphology(const cv::Mat& src, cv::Mat& dst);


// TASK 3 - Connected Components Analysis

// Runs connected components on the cleaned binary image and returns labeled region map with stats.
int computeConnectedComponents(const cv::Mat& cleaned, cv::Mat& labels, cv::Mat& stats, cv::Mat& centroids);

// Finds the single largest valid region, ignoring regions that are too small or touch the image boundary.
int findLargestRegion(const cv::Mat& stats, int num_labels, int img_rows, int img_cols, int min_area);

// Finds the top N largest valid regions, ignoring regions that are too small or touch the image boundary.
std::vector<int> findTopRegions(const cv::Mat& stats, int num_labels, int img_rows, int img_cols, int min_area, int max_regions);

// Draws all valid regions in distinct random colors on a black image for visualization.
int drawRegionMap(const cv::Mat& labels, const cv::Mat& stats, int num_labels, int img_rows, int img_cols, int min_area, cv::Mat& dst);

// Draws the selected region in a distinct color on a black output image.
int drawRegion(const cv::Mat& labels, int region_id, cv::Mat& dst);


// TASK 4 - Feature Computation

// Computes orientation angle of the region using second order central moments.
double computeOrientation(const cv::Moments& moments);

// Computes the oriented bounding box of the region using minAreaRect.
cv::RotatedRect computeOrientedBoundingBox(const cv::Mat& labels, int region_id);

// Computes the feature vector: percent filled, aspect ratio, Hu1, Hu2.
int computeFeatures(const cv::Mat& labels, int region_id, const cv::Moments& moments, const cv::RotatedRect& obb, std::vector<double>& features);

// Draws the primary axis and oriented bounding box overlaid on the region display image.
int drawFeaturesOverlay(cv::Mat& display, const cv::Moments& moments, const cv::RotatedRect& obb);


// TASK 5 - Collecting Training Data

// Appends a labeled feature vector to the training database CSV file.
int saveTrainingExample(const std::string& db_path, const std::string& label, const std::vector<double>& features);


// TASK 6 - Classification

// Struct to hold a single training example
struct TrainingExample {
    std::string label;
    std::vector<double> features;
};

// Loads all training examples from the CSV database file into memory.
int loadDatabase(const std::string& db_path, std::vector<TrainingExample>& db);

// Computes the scaled Euclidean distance between two feature vectors using per-feature standard deviations.
double computeScaledDistance(const std::vector<double>& f1, const std::vector<double>& f2,
    const std::vector<double>& stdevs);

// Computes the per-feature standard deviations across all training examples.
int computeFeatureStdevs(const std::vector<TrainingExample>& db, std::vector<double>& stdevs);

// Classifies an unknown feature vector using nearest neighbor and returns the predicted label.
std::string classifyObject(const std::vector<double>& features, const std::vector<TrainingExample>& db,
    const std::vector<double>& stdevs);


// TASK 7 - Confusion Matrix

// Adds a true/predicted label pair to the evaluation log.
int logEvaluationResult(std::vector<std::pair<std::string, std::string>>& eval_log,
    const std::string& true_label, const std::string& predicted_label);

// Builds and prints the confusion matrix from the evaluation log to the console.
int printConfusionMatrix(const std::vector<std::pair<std::string, std::string>>& eval_log);


// TASK 8 - CNN Embedding Classification

// Struct to hold a single embedding training example
struct EmbeddingExample {
    std::string label;
    cv::Mat     embedding;
};

// Computes the min and max projections of region pixels along the primary and secondary axes.
int computeAxisExtents(const cv::Mat& labels, int region_id, double cx, double cy, double theta,
    float& minE1, float& maxE1, float& minE2, float& maxE2);

// Saves a labeled embedding to the embedding database in memory.
int saveEmbeddingExample(std::vector<EmbeddingExample>& emb_db,
    const std::string& label, const cv::Mat& embedding);

// Classifies an unknown embedding using sum-squared difference against the embedding database.
std::string classifyEmbedding(const cv::Mat& embedding, const std::vector<EmbeddingExample>& emb_db);


// EXTENSION 4 - 2D Embedding Plot

// Projects all stored embeddings to 2D using PCA and displays them as a labeled scatter plot.
int plotEmbeddings2D(const std::vector<EmbeddingExample>& emb_db);


#endif // OBJECT_RECOGNITION_H