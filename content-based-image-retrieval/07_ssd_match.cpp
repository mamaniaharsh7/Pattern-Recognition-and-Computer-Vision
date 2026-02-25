/*
 * Harsh Mamania
 * 10th February 2026
 *
 * Task 1: Baseline matching using 7x7 center pixel patch and Sum of Squared Differences.
 * Implements pixel-level feature extraction and manual SSD computation without OpenCV functions.
 *
 * Usage: ./ssd_match <target_image> <directory> <topN>
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

// Structure to hold filename and distance for each match
struct MatchResult {
    std::string filename;
    double distance;

    // Comparison operator for sorting (smallest distance first)
    bool operator<(const MatchResult& other) const {
        return distance < other.distance;
    }
};

/*
 * Extracts 7x7 pixel patch from center of image as feature vector.
 * Returns flattened vector of 147 values (7*7*3 BGR channels).
 */
std::vector<int> extractFeature(const cv::Mat& img) {
    const int patchSize = 7;

    // Find center coordinates
    int cx = img.cols / 2;
    int cy = img.rows / 2;
    int half = patchSize / 2;  // 3 pixels on each side of center

    // Reserve space for efficiency (49 pixels * 3 channels = 147 values)
    std::vector<int> feature;
    feature.reserve(patchSize * patchSize * 3);

    // Extract 7x7 patch centered at (cx, cy)
    for (int y = cy - half; y <= cy + half; ++y) {
        for (int x = cx - half; x <= cx + half; ++x) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

            // Store all three BGR channels
            feature.push_back(pixel[0]);  // Blue
            feature.push_back(pixel[1]);  // Green
            feature.push_back(pixel[2]);  // Red
        }
    }

    return feature;
}

/*
 * Manually computes Sum of Squared Differences between two feature vectors.
 * Does not use OpenCV functions as per assignment requirements.
 */
double computeSSD(const std::vector<int>& f1, const std::vector<int>& f2) {
    double ssd = 0.0;

    // Sum squared differences across all 147 feature values
    for (size_t i = 0; i < f1.size(); ++i) {
        double diff = static_cast<double>(f1[i]) - f2[i];
        ssd += diff * diff;
    }

    return ssd;
}

int main(int argc, char** argv) {

    // ========== PARSE COMMAND LINE ARGUMENTS ==========

    if (argc < 4) {
        std::cout << "Usage: ./ssd_match <target_image> <directory> <topN>\n";
        return -1;
    }

    std::string targetPath = argv[1];   // Target image path
    std::string directory = argv[2];     // Database directory
    int topN = std::stoi(argv[3]);       // Number of matches to return

    // ========== LOAD AND VALIDATE TARGET IMAGE ==========

    cv::Mat target = cv::imread(targetPath);
    if (target.empty()) {
        std::cerr << "Error: Could not read target image\n";
        return -1;
    }

    // Display target image
    cv::Mat small;
    cv::resize(target, small, cv::Size(400, 400));
    cv::imshow("Target Image", small);

    // Validate dimensions (requirement: 640x512)
    if (target.cols != 640 || target.rows != 512) {
        std::cerr << "Warning: Target image must be 640x512 for consistent matching\n";
        std::cerr << "Current size: " << target.cols << "x" << target.rows << "\n";
    }

    // Extract 7x7 center patch from target
    auto targetFeature = extractFeature(target);

    // ========== PROCESS ALL IMAGES IN DATABASE ==========

    std::vector<MatchResult> matches;

    std::cout << "Processing images in directory: " << directory << "\n";

    // Loop through all files in directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string path = entry.path().string();

        // Load database image
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;  // Skip if cannot load

        // Skip images with different dimensions
        if (img.cols != 640 || img.rows != 512) continue;

        // Extract feature from database image
        auto feature = extractFeature(img);

        // Compute SSD distance
        double dist = computeSSD(targetFeature, feature);

        // Verify self-comparison returns zero distance
        if (path == targetPath) {
            if (dist != 0) {
                std::cerr << "Warning: Self-comparison distance is not zero! Distance: " << dist << "\n";
            }
            continue;  // Skip target image itself
        }

        // Store match result
        matches.push_back({ path, dist });
    }

    // ========== SORT AND DISPLAY RESULTS ==========

    // Sort by distance (ascending: smallest = most similar)
    std::sort(matches.begin(), matches.end());

    // Print top N matches to console
    std::cout << "\nTop " << topN << " Matches:\n";
    std::cout << "========================================\n";

    for (int i = 0; i < std::min(topN, (int)matches.size()); ++i) {
        std::cout << i + 1 << ". " << matches[i].filename
            << " | SSD: " << std::fixed << std::setprecision(2)
            << matches[i].distance << "\n";
    }

    // ========== SAVE RESULTS TO FILE ==========

    // Append results to text file
    std::ofstream out("match_results.txt", std::ios::app);
    out << "========================================\n";
    out << "Target: " << fs::path(targetPath).filename().string() << "\n";
    out << "========================================\n";

    for (int i = 0; i < std::min(topN, (int)matches.size()); ++i) {
        out << std::setw(3) << (i + 1) << ". "
            << std::setw(20) << fs::path(matches[i].filename).filename().string()
            << " | SSD: " << std::fixed << std::setprecision(2)
            << matches[i].distance << "\n";
    }
    out << "\n";
    out.close();

    // ========== DISPLAY MATCHED IMAGES ==========

    // Show top N matching images
    for (int i = 0; i < std::min(topN, (int)matches.size()); ++i) {
        cv::Mat img = cv::imread(matches[i].filename);
        if (!img.empty()) {
            cv::resize(img, small, cv::Size(400, 400));
            cv::imshow("Match " + std::to_string(i + 1), small);
        }
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}