/*
 * Name: Harsh Vijay Mamania
 * Date: 17th Feb 2026
 *
 * objectRecognition.cpp
 *
 * Implementation file for Project 3: Real-time 2D Object Recognition
 * Contains all function implementations for the recognition pipeline.
 */

#include "objectRecognition.h"
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <algorithm>


 // TASK 1 - Thresholding

 /*
  * Blurs the input frame, converts to HSV, and computes saturation-weighted
  * brightness into a single channel.
  *
  * src - input BGR frame captured from the camera
  * dst - output single-channel 8-bit image containing modified V values
  *
  * Returns 0 on success, -1 if the input image is empty
  */
int preprocessFrame(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) {
        return -1;
    }

    // Step 1: Blur the image to reduce noise and smooth region boundaries
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(5, 5), 0);

    // Step 2: Convert from BGR to HSV color space
    // HSV separates color (H), vividness (S), and brightness (V) cleanly
    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    // Step 3: Compute saturation-weighted brightness for each pixel
    // Formula: modified_V = V * (1 - S / 255)
    // Effect: vivid/colorful pixels (high S) get darkened toward 0
    //         white/gray pixels (low S) retain their brightness
    // This ensures both dark objects AND colorful objects threshold as foreground
    dst.create(src.rows, src.cols, CV_8UC1);

    for (int row = 0; row < hsv.rows; row++) {
        for (int col = 0; col < hsv.cols; col++) {
            // Each HSV pixel is a 3-channel uchar: [H, S, V]
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(row, col);
            float S = pixel[1];
            float V = pixel[2];

            // Darken vivid pixels so they appear as foreground after thresholding
            float modified_V = V * (1.0f - S / 255.0f);

            // Clamp to valid 8-bit range and store
            dst.at<uchar>(row, col) = static_cast<uchar>(std::min(modified_V, 255.0f));
        }
    }

    return 0;
}


/*
 * Estimates a binary threshold value using the ISODATA algorithm (k-means K=2)
 * on a random sample of pixel values from the modified V channel.
 *
 * modifiedV - single-channel 8-bit image output from preprocessFrame
 *
 * Returns the computed threshold value as an integer between 0 and 255
 */
int computeDynamicThreshold(const cv::Mat& modifiedV) {
    // Collect a random sample of 1/16 of all pixels
    // Sampling avoids processing every pixel, which speeds up the algorithm
    std::vector<float> samples;
    int total_pixels = modifiedV.rows * modifiedV.cols;
    int sample_size = total_pixels / 16;
    samples.reserve(sample_size);

    for (int i = 0; i < sample_size; i++) {
        int row = rand() % modifiedV.rows;
        int col = rand() % modifiedV.cols;
        samples.push_back(static_cast<float>(modifiedV.at<uchar>(row, col)));
    }

    // Initialize two means: one in the dark range, one in the bright range
    // These are just starting guesses -- the algorithm will correct them
    float mean_dark = 64.0f;
    float mean_bright = 192.0f;

    // Iteratively refine the two means until they converge
    for (int iteration = 0; iteration < 100; iteration++) {
        float sum_dark = 0.0f;
        float sum_bright = 0.0f;
        int   count_dark = 0;
        int   count_bright = 0;

        // Assign each sample to whichever mean it is closest to
        for (float val : samples) {
            float dist_to_dark = std::abs(val - mean_dark);
            float dist_to_bright = std::abs(val - mean_bright);

            if (dist_to_dark < dist_to_bright) {
                sum_dark += val;
                count_dark++;
            }
            else {
                sum_bright += val;
                count_bright++;
            }
        }

        // Recompute means from assigned samples
        // Guard against division by zero if all samples fall into one cluster
        float new_mean_dark = (count_dark > 0) ? sum_dark / count_dark : mean_dark;
        float new_mean_bright = (count_bright > 0) ? sum_bright / count_bright : mean_bright;

        // Check for convergence -- stop if means barely moved
        float change_dark = std::abs(new_mean_dark - mean_dark);
        float change_bright = std::abs(new_mean_bright - mean_bright);

        mean_dark = new_mean_dark;
        mean_bright = new_mean_bright;

        if (change_dark < 0.5f && change_bright < 0.5f) {
            break;
        }
    }

    // Threshold is the midpoint between the two converged means
    int threshold_value = static_cast<int>((mean_dark + mean_bright) / 2.0f);
    return threshold_value;
}


/*
 * Applies a binary threshold to the modified V channel, producing a foreground/background image.
 *
 * modifiedV       - single-channel 8-bit input image from preprocessFrame
 * dst             - output binary image, same size as input, values are 0 or 255
 * threshold_value - cutoff intensity; pixels below this become foreground (255)
 *
 * Returns 0 on success, -1 if the input image is empty
 */
int applyThreshold(const cv::Mat& modifiedV, cv::Mat& dst, int threshold_value) {
    if (modifiedV.empty()) {
        return -1;
    }

    // Allocate output image: single channel, same size as input
    dst.create(modifiedV.rows, modifiedV.cols, CV_8UC1);

    // Loop over every pixel and apply the threshold decision
    for (int row = 0; row < modifiedV.rows; row++) {
        for (int col = 0; col < modifiedV.cols; col++) {
            uchar pixel_value = modifiedV.at<uchar>(row, col);

            // Dark or colorful pixels are foreground (object)
            // Bright unsaturated pixels are background
            if (pixel_value < threshold_value) {
                dst.at<uchar>(row, col) = 255; // foreground
            }
            else {
                dst.at<uchar>(row, col) = 0;   // background
            }
        }
    }

    return 0;
}


/*
 * Full Task 1 pipeline wrapper: preprocessFrame -> computeDynamicThreshold -> applyThreshold.
 *
 * src - input BGR frame from the camera
 * dst - output binary image with foreground pixels set to 255
 *
 * Returns 0 on success, -1 if any stage of the pipeline fails
 */
int thresholdFrame(const cv::Mat& src, cv::Mat& dst) {
    // Stage 1: blur, convert to HSV, compute saturation-weighted brightness
    cv::Mat modifiedV;
    if (preprocessFrame(src, modifiedV) != 0) {
        return -1;
    }

    // Stage 2: compute a lighting-adaptive threshold from the image itself
    int threshold_value = computeDynamicThreshold(modifiedV);

    // Stage 3: apply threshold to produce the binary output
    if (applyThreshold(modifiedV, dst, threshold_value) != 0) {
        return -1;
    }

    return 0;
}


// TASK 2 - Morphological Filtering

/*
 * Applies a single dilation pass to the binary image using a 3x3 kernel.
 *
 * src - input binary image (CV_8UC1, values 0 or 255)
 * dst - output binary image after dilation, same size as input
 *
 * Returns 0 on success, -1 if the input image is empty
 */
int dilate(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.rows, src.cols, CV_8UC1);

    // For each pixel, check all neighbors in a 3x3 window
    // If ANY neighbor is foreground (255), output pixel becomes foreground
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            uchar max_val = 0;

            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int neighbor_row = row + kr;
                    int neighbor_col = col + kc;

                    // Skip neighbors that fall outside image boundaries
                    if (neighbor_row < 0 || neighbor_row >= src.rows ||
                        neighbor_col < 0 || neighbor_col >= src.cols) {
                        continue;
                    }

                    uchar neighbor_val = src.at<uchar>(neighbor_row, neighbor_col);
                    if (neighbor_val > max_val) {
                        max_val = neighbor_val;
                    }
                }
            }

            dst.at<uchar>(row, col) = max_val;
        }
    }

    return 0;
}


/*
 * Applies a single erosion pass to the binary image using a 3x3 kernel.
 *
 * src - input binary image (CV_8UC1, values 0 or 255)
 * dst - output binary image after erosion, same size as input
 *
 * Returns 0 on success, -1 if the input image is empty
 */
int erode(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) {
        return -1;
    }

    dst.create(src.rows, src.cols, CV_8UC1);

    // For each pixel, check all neighbors in a 3x3 window
    // If ANY neighbor is background (0), output pixel becomes background
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            uchar min_val = 255;

            for (int kr = -1; kr <= 1; kr++) {
                for (int kc = -1; kc <= 1; kc++) {
                    int neighbor_row = row + kr;
                    int neighbor_col = col + kc;

                    // Skip neighbors that fall outside image boundaries
                    if (neighbor_row < 0 || neighbor_row >= src.rows ||
                        neighbor_col < 0 || neighbor_col >= src.cols) {
                        continue;
                    }

                    uchar neighbor_val = src.at<uchar>(neighbor_row, neighbor_col);
                    if (neighbor_val < min_val) {
                        min_val = neighbor_val;
                    }
                }
            }

            dst.at<uchar>(row, col) = min_val;
        }
    }

    return 0;
}


/*
 * Applies morphological closing (dilation then erosion) to fill holes in foreground regions.
 *
 * src - input binary image from applyThreshold
 * dst - output cleaned binary image with holes filled
 *
 * Returns 0 on success, -1 if any stage fails
 */
int applyMorphology(const cv::Mat& src, cv::Mat& dst) {
    // Apply 5 passes of dilation then 5 passes of erosion for stronger hole filling
    // More passes fill larger holes without increasing kernel size
    cv::Mat temp = src.clone();

    for (int i = 0; i < 5; i++) {
        cv::Mat dilated;
        if (dilate(temp, dilated) != 0) {
            return -1;
        }
        temp = dilated;
    }

    for (int i = 0; i < 5; i++) {
        cv::Mat eroded;
        if (erode(temp, eroded) != 0) {
            return -1;
        }
        temp = eroded;
    }

    dst = temp;
    return 0;
}


// TASK 3 - Connected Components Analysis

/*
 * Runs connected components on the cleaned binary image and returns labeled region map with stats.
 *
 * cleaned   - input binary image from applyMorphology
 * labels    - output integer map where each pixel contains its region ID
 * stats     - output matrix with one row per region: x, y, width, height, area
 * centroids - output matrix with one row per region: cx, cy
 *
 * Returns number of labels (including background) on success, -1 if input is empty
 */
int computeConnectedComponents(const cv::Mat& cleaned, cv::Mat& labels, cv::Mat& stats, cv::Mat& centroids) {
    if (cleaned.empty()) {
        return -1;
    }

    // cv::CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA
    // are the five stats columns returned per region
    int num_labels = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8, CV_32S);
    return num_labels;
}


/*
 * Finds the single largest valid region, ignoring regions that are too small or touch the image boundary.
 *
 * stats     - stats matrix from computeConnectedComponents
 * num_labels - total number of labels including background (label 0)
 * img_rows  - height of the image in pixels
 * img_cols  - width of the image in pixels
 * min_area  - minimum pixel area for a region to be considered valid
 *
 * Returns the region ID of the largest valid region, or -1 if none found
 */
int findLargestRegion(const cv::Mat& stats, int num_labels, int img_rows, int img_cols, int min_area) {
    int best_region = -1;
    int best_area = 0;

    // Start from label 1 -- label 0 is always the background
    for (int i = 1; i < num_labels; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Filter out regions that are too small
        if (area < min_area) {
            continue;
        }

        // Filter out regions whose bounding box touches any image boundary
        if (x == 0 || y == 0 || x + width >= img_cols || y + height >= img_rows) {
            continue;
        }

        // Keep track of the largest passing region
        if (area > best_area) {
            best_area = area;
            best_region = i;
        }
    }

    return best_region;
}


/*
 * Draws the selected region in a distinct color on a black output image.
 *
 * labels    - label map from computeConnectedComponents
 * region_id - ID of the region to draw, as returned by findLargestRegion
 * dst       - output BGR image with the region colored and background black
 *
 * Returns 0 on success, -1 if inputs are invalid
 */
int drawRegion(const cv::Mat& labels, int region_id, cv::Mat& dst) {
    if (labels.empty() || region_id < 0) {
        return -1;
    }

    // Create a black BGR output image
    dst = cv::Mat::zeros(labels.rows, labels.cols, CV_8UC3);

    // Color for the selected region -- a distinct cyan
    cv::Vec3b region_color(255, 255, 0);

    for (int row = 0; row < labels.rows; row++) {
        for (int col = 0; col < labels.cols; col++) {
            if (labels.at<int>(row, col) == region_id) {
                dst.at<cv::Vec3b>(row, col) = region_color;
            }
        }
    }

    return 0;
}


// TASK 4 - Feature Computation

/*
 * Computes orientation angle of the region using second order central moments.
 *
 * moments - OpenCV moments struct computed from the region mask
 *
 * Returns orientation angle in radians
 */
double computeOrientation(const cv::Moments& moments) {
    // mu11, mu20, mu02 are the second order central moments
    // atan2 returns angle in [-pi/2, pi/2] range
    double theta = 0.5 * std::atan2(2.0 * moments.mu11, moments.mu20 - moments.mu02);
    return theta;
}


/*
 * Computes the oriented bounding box of the region using minAreaRect.
 *
 * labels    - label map from computeConnectedComponents
 * region_id - ID of the region to compute the bounding box for
 *
 * Returns a RotatedRect representing the oriented bounding box
 */
cv::RotatedRect computeOrientedBoundingBox(const cv::Mat& labels, int region_id) {
    // Collect all foreground pixel coordinates for this region
    std::vector<cv::Point> points;
    for (int row = 0; row < labels.rows; row++) {
        for (int col = 0; col < labels.cols; col++) {
            if (labels.at<int>(row, col) == region_id) {
                points.push_back(cv::Point(col, row));
            }
        }
    }

    // minAreaRect finds the tightest rotated rectangle around the point set
    return cv::minAreaRect(points);
}


/*
 * Computes the feature vector: percent filled, aspect ratio, Hu1, Hu2.
 *
 * labels    - label map from computeConnectedComponents
 * region_id - ID of the region to compute features for
 * moments   - precomputed moments of the region
 * obb       - oriented bounding box from computeOrientedBoundingBox
 * features  - output vector: [percent_filled, aspect_ratio, Hu1, Hu2]
 *
 * Returns 0 on success, -1 if inputs are invalid
 */
int computeFeatures(const cv::Mat& labels, int region_id,
    const cv::Moments& moments, const cv::RotatedRect& obb,
    std::vector<double>& features) {
    if (labels.empty() || region_id < 0) {
        return -1;
    }

    // Region area from M00
    double area = moments.m00;

    // Oriented bounding box dimensions
    float box_w = obb.size.width;
    float box_h = obb.size.height;
    double box_area = static_cast<double>(box_w * box_h);

    // Percent filled: how much of the bounding box is covered by the region
    double percent_filled = (box_area > 0.0) ? area / box_area : 0.0;

    // Aspect ratio: always >= 1 regardless of orientation
    double aspect_ratio = (std::min(box_w, box_h) > 0.0f)
        ? static_cast<double>(std::max(box_w, box_h)) / std::min(box_w, box_h)
        : 1.0;

    // Hu moments -- invariant to translation, scale, and rotation
    double hu[7];
    cv::HuMoments(moments, hu);

    features.clear();
    features.push_back(percent_filled);
    features.push_back(aspect_ratio);
    features.push_back(hu[0]);
    features.push_back(hu[1]);

    return 0;
}


/*
 * Draws the primary axis and oriented bounding box overlaid on the region display image.
 *
 * display - BGR image to draw on (modified in place), typically the region display from drawRegion
 * moments - precomputed moments of the region
 * obb     - oriented bounding box from computeOrientedBoundingBox
 *
 * Returns 0 on success, -1 if display image is empty
 */
int drawFeaturesOverlay(cv::Mat& display, const cv::Moments& moments, const cv::RotatedRect& obb) {
    if (display.empty()) {
        return -1;
    }

    // Centroid from moments
    double cx = moments.m10 / moments.m00;
    double cy = moments.m01 / moments.m00;

    // Orientation angle
    double theta = computeOrientation(moments);

    // Draw the primary axis as a line through the centroid
    int axis_length = 80;
    cv::Point axis_p1(static_cast<int>(cx + axis_length * std::cos(theta)),
        static_cast<int>(cy + axis_length * std::sin(theta)));
    cv::Point axis_p2(static_cast<int>(cx - axis_length * std::cos(theta)),
        static_cast<int>(cy - axis_length * std::sin(theta)));
    cv::line(display, axis_p1, axis_p2, cv::Scalar(0, 0, 255), 2);

    // Draw the centroid as a small circle
    cv::circle(display, cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
        5, cv::Scalar(0, 255, 0), -1);

    // Draw the oriented bounding box as a rotated rectangle
    cv::Point2f box_corners[4];
    obb.points(box_corners);
    for (int i = 0; i < 4; i++) {
        cv::line(display, box_corners[i], box_corners[(i + 1) % 4],
            cv::Scalar(255, 0, 255), 2);
    }

    return 0;
}


// TASK 5 - Collecting Training Data

/*
 * Appends a labeled feature vector to the training database CSV file.
 *
 * db_path  - path to the CSV file to append to (created if it does not exist)
 * label    - object name entered by the user (e.g. "banana")
 * features - feature vector: [percent_filled, aspect_ratio, Hu1, Hu2]
 *
 * Returns 0 on success, -1 if the file could not be opened
 */
int saveTrainingExample(const std::string& db_path, const std::string& label,
    const std::vector<double>& features) {
    // Open in append mode so existing entries are not overwritten
    std::ofstream file(db_path, std::ios::app);
    if (!file.is_open()) {
        return -1;
    }

    // Write label followed by each feature value, comma separated
    file << label;
    for (double val : features) {
        file << "," << std::fixed << std::setprecision(9) << val;
    }
    file << "\n";

    file.close();
    return 0;
}


// TASK 6 - Classification

/*
 * Loads all training examples from the CSV database file into memory.
 *
 * db_path - path to the CSV file written by saveTrainingExample
 * db      - output vector of TrainingExample structs populated from the file
 *
 * Returns 0 on success, -1 if the file could not be opened
 */
int loadDatabase(const std::string& db_path, std::vector<TrainingExample>& db) {
    std::ifstream file(db_path);
    if (!file.is_open()) {
        return -1;
    }

    db.clear();
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        TrainingExample example;

        // First token is the label
        std::getline(ss, token, ',');
        example.label = token;

        // Remaining tokens are feature values
        while (std::getline(ss, token, ',')) {
            example.features.push_back(std::stod(token));
        }

        if (!example.label.empty() && !example.features.empty()) {
            db.push_back(example);
        }
    }

    file.close();
    return 0;
}


/*
 * Computes the per-feature standard deviations across all training examples.
 *
 * db     - vector of training examples loaded by loadDatabase
 * stdevs - output vector of standard deviations, one per feature
 *
 * Returns 0 on success, -1 if the database is empty
 */
int computeFeatureStdevs(const std::vector<TrainingExample>& db, std::vector<double>& stdevs) {
    if (db.empty()) {
        return -1;
    }

    int num_features = static_cast<int>(db[0].features.size());
    int num_examples = static_cast<int>(db.size());

    // Compute mean for each feature
    std::vector<double> means(num_features, 0.0);
    for (const TrainingExample& ex : db) {
        for (int i = 0; i < num_features; i++) {
            means[i] += ex.features[i];
        }
    }
    for (int i = 0; i < num_features; i++) {
        means[i] /= num_examples;
    }

    // Compute standard deviation for each feature
    stdevs.assign(num_features, 0.0);
    for (const TrainingExample& ex : db) {
        for (int i = 0; i < num_features; i++) {
            double diff = ex.features[i] - means[i];
            stdevs[i] += diff * diff;
        }
    }
    for (int i = 0; i < num_features; i++) {
        stdevs[i] = std::sqrt(stdevs[i] / num_examples);
        // Avoid division by zero if a feature has zero variance
        if (stdevs[i] < 1e-6) {
            stdevs[i] = 1e-6;
        }
    }

    return 0;
}


/*
 * Computes the scaled Euclidean distance between two feature vectors.
 *
 * f1     - first feature vector
 * f2     - second feature vector
 * stdevs - per-feature standard deviations from computeFeatureStdevs
 *
 * Returns the scaled Euclidean distance as a double
 */
double computeScaledDistance(const std::vector<double>& f1, const std::vector<double>& f2,
    const std::vector<double>& stdevs) {
    double sum = 0.0;
    for (int i = 0; i < static_cast<int>(f1.size()); i++) {
        double scaled_diff = (f1[i] - f2[i]) / stdevs[i];
        sum += scaled_diff * scaled_diff;
    }
    return std::sqrt(sum);
}


/*
 * Classifies an unknown feature vector using nearest neighbor and returns the predicted label.
 *
 * features - feature vector of the unknown object
 * db       - training database loaded by loadDatabase
 * stdevs   - per-feature standard deviations from computeFeatureStdevs
 *
 * Returns the label of the nearest training example, or "unknown" if db is empty
 */
std::string classifyObject(const std::vector<double>& features,
    const std::vector<TrainingExample>& db,
    const std::vector<double>& stdevs) {
    if (db.empty()) {
        return "unknown";
    }

    std::string best_label = "unknown";
    double best_distance = std::numeric_limits<double>::max();

    for (const TrainingExample& ex : db) {
        double dist = computeScaledDistance(features, ex.features, stdevs);
        if (dist < best_distance) {
            best_distance = dist;
            best_label = ex.label;
        }
    }

    return best_label;
}


// TASK 7 - Confusion Matrix

/*
 * Adds a true/predicted label pair to the evaluation log.
 *
 * eval_log        - in-memory log of (true_label, predicted_label) pairs
 * true_label      - the actual object label entered by the user
 * predicted_label - the label assigned by classifyObject
 *
 * Returns 0 on success
 */
int logEvaluationResult(std::vector<std::pair<std::string, std::string>>& eval_log,
    const std::string& true_label, const std::string& predicted_label) {
    eval_log.push_back(std::make_pair(true_label, predicted_label));
    return 0;
}


/*
 * Builds and prints the confusion matrix from the evaluation log to the console.
 *
 * eval_log - vector of (true_label, predicted_label) pairs from logEvaluationResult
 *
 * Returns 0 on success, -1 if the log is empty
 */
int printConfusionMatrix(const std::vector<std::pair<std::string, std::string>>& eval_log) {
    if (eval_log.empty()) {
        return -1;
    }

    // Collect all unique labels in the order they first appear
    std::vector<std::string> labels;
    for (const auto& entry : eval_log) {
        const std::string& true_label = entry.first;
        const std::string& pred_label = entry.second;

        if (std::find(labels.begin(), labels.end(), true_label) == labels.end()) {
            labels.push_back(true_label);
        }
        if (std::find(labels.begin(), labels.end(), pred_label) == labels.end()) {
            labels.push_back(pred_label);
        }
    }

    int n = static_cast<int>(labels.size());

    // Build the matrix as a 2D vector initialized to zero
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));

    for (const auto& entry : eval_log) {
        int true_idx = static_cast<int>(std::find(labels.begin(), labels.end(), entry.first) - labels.begin());
        int pred_idx = static_cast<int>(std::find(labels.begin(), labels.end(), entry.second) - labels.begin());
        matrix[true_idx][pred_idx]++;
    }

    // Print the matrix with labels
    // Column width for alignment
    const int col_width = 14;

    // Print header row with predicted labels
    printf("\nConfusion Matrix (rows = true label, cols = predicted label)\n\n");
    printf("%*s", col_width, "");
    for (const std::string& label : labels) {
        printf("%*s", col_width, label.c_str());
    }
    printf("\n");

    // Print each row
    for (int i = 0; i < n; i++) {
        printf("%*s", col_width, labels[i].c_str());
        for (int j = 0; j < n; j++) {
            printf("%*d", col_width, matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}


// TASK 8 - CNN Embedding Classification

/*
 * Computes the min and max projections of region pixels along the primary and secondary axes.
 *
 * labels    - label map from computeConnectedComponents
 * region_id - ID of the region to analyze
 * cx        - x coordinate of the region centroid
 * cy        - y coordinate of the region centroid
 * theta     - orientation angle of the primary axis in radians
 * minE1     - output minimum projection along primary axis
 * maxE1     - output maximum projection along primary axis
 * minE2     - output minimum projection along secondary axis
 * maxE2     - output maximum projection along secondary axis
 *
 * Returns 0 on success, -1 if the label map is empty
 */
int computeAxisExtents(const cv::Mat& labels, int region_id, double cx, double cy, double theta,
    float& minE1, float& maxE1, float& minE2, float& maxE2) {
    if (labels.empty()) {
        return -1;
    }

    minE1 = std::numeric_limits<float>::max();
    maxE1 = -std::numeric_limits<float>::max();
    minE2 = std::numeric_limits<float>::max();
    maxE2 = -std::numeric_limits<float>::max();

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    for (int row = 0; row < labels.rows; row++) {
        for (int col = 0; col < labels.cols; col++) {
            if (labels.at<int>(row, col) != region_id) {
                continue;
            }

            // Translate pixel to centroid-relative coordinates
            double dx = col - cx;
            double dy = row - cy;

            // Project onto primary axis (direction of theta)
            float e1 = static_cast<float>(dx * cos_theta + dy * sin_theta);

            // Project onto secondary axis (perpendicular to theta)
            float e2 = static_cast<float>(-dx * sin_theta + dy * cos_theta);

            if (e1 < minE1) minE1 = e1;
            if (e1 > maxE1) maxE1 = e1;
            if (e2 < minE2) minE2 = e2;
            if (e2 > maxE2) maxE2 = e2;
        }
    }

    return 0;
}


/*
 * Saves a labeled embedding to the embedding database in memory.
 *
 * emb_db    - in-memory vector of EmbeddingExample structs
 * label     - object label for this embedding
 * embedding - 512D embedding vector from getEmbedding
 *
 * Returns 0 on success
 */
int saveEmbeddingExample(std::vector<EmbeddingExample>& emb_db,
    const std::string& label, const cv::Mat& embedding) {
    EmbeddingExample ex;
    ex.label = label;
    embedding.copyTo(ex.embedding);
    emb_db.push_back(ex);
    return 0;
}


/*
 * Classifies an unknown embedding using sum-squared difference against the embedding database.
 *
 * embedding - 512D embedding vector of the unknown object
 * emb_db    - embedding database built during training mode
 *
 * Returns the label of the nearest embedding, or "unknown" if the database is empty
 */
std::string classifyEmbedding(const cv::Mat& embedding,
    const std::vector<EmbeddingExample>& emb_db) {
    if (emb_db.empty()) {
        return "unknown";
    }

    std::string best_label = "unknown";
    double      best_dist = std::numeric_limits<double>::max();

    for (const EmbeddingExample& ex : emb_db) {
        // Sum-squared difference between two embedding vectors
        cv::Mat diff = embedding - ex.embedding;
        double dist = diff.dot(diff);

        if (dist < best_dist) {
            best_dist = dist;
            best_label = ex.label;
        }
    }

    return best_label;
}


// EXTENSION 3 - Multi-Object Detection

/*
 * Finds the top N largest valid regions, ignoring regions too small or touching the boundary.
 *
 * stats       - stats matrix from computeConnectedComponents
 * num_labels  - total number of labels including background
 * img_rows    - height of the image in pixels
 * img_cols    - width of the image in pixels
 * min_area    - minimum pixel area for a region to be considered valid
 * max_regions - maximum number of regions to return
 *
 * Returns a vector of region IDs sorted by area descending, up to max_regions
 */
std::vector<int> findTopRegions(const cv::Mat& stats, int num_labels, int img_rows, int img_cols,
    int min_area, int max_regions) {
    // Collect all valid regions as (area, id) pairs
    std::vector<std::pair<int, int>> valid_regions;

    for (int i = 1; i < num_labels; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area < min_area) {
            continue;
        }

        if (x == 0 || y == 0 || x + width >= img_cols || y + height >= img_rows) {
            continue;
        }

        valid_regions.push_back(std::make_pair(area, i));
    }

    // Sort by area descending
    std::sort(valid_regions.begin(), valid_regions.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.first > b.first;
        });

    // Return up to max_regions region IDs
    std::vector<int> result;
    int count = std::min(max_regions, static_cast<int>(valid_regions.size()));
    for (int i = 0; i < count; i++) {
        result.push_back(valid_regions[i].second);
    }

    return result;
}


// EXTENSION 4 - 2D Embedding Plot

/*
 * Projects all stored embeddings to 2D using PCA and displays them as a labeled scatter plot.
 *
 * emb_db - in-memory embedding database built using saveEmbeddingExample
 *
 * Returns 0 on success, -1 if fewer than 2 embeddings are stored
 */
int plotEmbeddings2D(const std::vector<EmbeddingExample>& emb_db) {
    if (emb_db.size() < 2) {
        printf("Need at least 2 embeddings to plot -- collect more with t key\n");
        return -1;
    }

    // Stack all embeddings into a single matrix -- one row per embedding
    int num_embeddings = static_cast<int>(emb_db.size());
    int embedding_size = emb_db[0].embedding.cols;
    cv::Mat data(num_embeddings, embedding_size, CV_32F);

    for (int i = 0; i < num_embeddings; i++) {
        emb_db[i].embedding.copyTo(data.row(i));
    }

    // Run PCA keeping only 2 principal components
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 2);

    // Project all embeddings to 2D
    cv::Mat projected = pca.project(data);

    // Find min and max of projected points for normalization
    double min_x, max_x, min_y, max_y;
    cv::minMaxLoc(projected.col(0), &min_x, &max_x);
    cv::minMaxLoc(projected.col(1), &min_y, &max_y);

    // Add small margin to avoid points at the very edge
    double range_x = max_x - min_x + 1e-6;
    double range_y = max_y - min_y + 1e-6;

    // Create a blank white canvas for the plot
    const int canvas_size = 600;
    const int margin = 60;
    cv::Mat canvas(canvas_size, canvas_size, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw axis lines
    cv::line(canvas, cv::Point(margin, canvas_size - margin),
        cv::Point(canvas_size - margin, canvas_size - margin),
        cv::Scalar(180, 180, 180), 1);
    cv::line(canvas, cv::Point(margin, margin),
        cv::Point(margin, canvas_size - margin),
        cv::Scalar(180, 180, 180), 1);

    // Assign a distinct color to each unique label
    std::vector<cv::Scalar> palette = {
        cv::Scalar(220,  20,  60),  // crimson
        cv::Scalar(30, 144, 255),  // dodger blue
        cv::Scalar(34, 139,  34),  // forest green
        cv::Scalar(255, 165,   0),  // orange
        cv::Scalar(148,   0, 211),  // purple
        cv::Scalar(0, 206, 209),  // dark turquoise
        cv::Scalar(255,  20, 147),  // deep pink
        cv::Scalar(139,  69,  19),  // saddle brown
        cv::Scalar(128, 128,   0),  // olive
        cv::Scalar(0, 128, 128)   // teal
    };

    // Collect unique labels to assign colors consistently
    std::vector<std::string> unique_labels;
    for (const EmbeddingExample& ex : emb_db) {
        if (std::find(unique_labels.begin(), unique_labels.end(), ex.label) == unique_labels.end()) {
            unique_labels.push_back(ex.label);
        }
    }

    // Plot each embedding as a filled circle with its label
    for (int i = 0; i < num_embeddings; i++) {
        // Normalize projected coordinates to canvas space
        float px = projected.at<float>(i, 0);
        float py = projected.at<float>(i, 1);

        int canvas_x = margin + static_cast<int>((px - min_x) / range_x * (canvas_size - 2 * margin));
        int canvas_y = (canvas_size - margin) - static_cast<int>((py - min_y) / range_y * (canvas_size - 2 * margin));

        // Find color for this label
        int label_idx = static_cast<int>(
            std::find(unique_labels.begin(), unique_labels.end(), emb_db[i].label) - unique_labels.begin()
            );
        cv::Scalar color = palette[label_idx % palette.size()];

        // Draw filled circle for this point
        cv::circle(canvas, cv::Point(canvas_x, canvas_y), 8, color, -1);
        cv::circle(canvas, cv::Point(canvas_x, canvas_y), 8, cv::Scalar(0, 0, 0), 1);

        // Draw label text slightly offset from the point
        cv::putText(canvas, emb_db[i].label,
            cv::Point(canvas_x + 10, canvas_y + 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.45,
            color, 1);
    }

    // Draw title
    cv::putText(canvas, "CNN Embeddings -- PCA Projection to 2D",
        cv::Point(margin, margin - 15),
        cv::FONT_HERSHEY_SIMPLEX, 0.55,
        cv::Scalar(50, 50, 50), 1);

    cv::imshow("Embedding Plot", canvas);
    return 0;
}


/*
 * Draws all valid regions in distinct random colors on a black image for visualization.
 *
 * labels    - label map from computeConnectedComponents
 * stats     - stats matrix from computeConnectedComponents
 * num_labels - total number of labels including background
 * img_rows  - height of the image in pixels
 * img_cols  - width of the image in pixels
 * min_area  - minimum pixel area for a region to be shown
 * dst       - output BGR image with each region in a distinct color
 *
 * Returns 0 on success, -1 if label map is empty
 */
int drawRegionMap(const cv::Mat& labels, const cv::Mat& stats, int num_labels,
    int img_rows, int img_cols, int min_area, cv::Mat& dst) {
    if (labels.empty()) {
        return -1;
    }

    // Start with a black canvas
    dst = cv::Mat::zeros(labels.rows, labels.cols, CV_8UC3);

    // Fixed color palette for up to 10 regions -- visually distinct colors
    std::vector<cv::Vec3b> palette = {
        cv::Vec3b(255,   0,   0),  // blue
        cv::Vec3b(0, 255,   0),  // green
        cv::Vec3b(0,   0, 255),  // red
        cv::Vec3b(255, 255,   0),  // cyan
        cv::Vec3b(255,   0, 255),  // magenta
        cv::Vec3b(0, 255, 255),  // yellow
        cv::Vec3b(128, 255,   0),  // lime
        cv::Vec3b(255, 128,   0),  // sky blue
        cv::Vec3b(0, 128, 255),  // orange
        cv::Vec3b(128,   0, 255)   // pink
    };

    int color_idx = 0;

    // Assign a color to each valid region
    std::vector<cv::Vec3b> region_color_map(num_labels, cv::Vec3b(0, 0, 0));

    for (int i = 1; i < num_labels; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Skip regions that are too small or touch the boundary
        if (area < min_area) continue;
        if (x == 0 || y == 0 || x + width >= img_cols || y + height >= img_rows) continue;

        region_color_map[i] = palette[color_idx % palette.size()];
        color_idx++;
    }

    // Paint each pixel with its region color
    for (int row = 0; row < labels.rows; row++) {
        for (int col = 0; col < labels.cols; col++) {
            int label = labels.at<int>(row, col);
            if (label > 0) {
                dst.at<cv::Vec3b>(row, col) = region_color_map[label];
            }
        }
    }

    return 0;
}