/*
 * Harsh Mamania
 * 10th February 2026
 *
 * Task 4: Texture and Color matching using HSV color histograms and Sobel texture histograms.
 * Extracts features from center 160x160 patch and combines color and texture with equal weighting.
 *
 * Usage: ./image_match <target_image> <directory> <topN>
 */

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

/*
 * Extracts center region of specified size from image.
 * Default patch size is 160x160 pixels.
 */
cv::Mat extractCenter(const cv::Mat& img, int patchSize = 160) {
    // Find center coordinates
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    int half = patchSize / 2;

    // Calculate patch boundaries, ensuring they stay within image bounds
    int x = std::max(0, cx - half);
    int y = std::max(0, cy - half);

    int w = std::min(patchSize, img.cols - x);
    int h = std::min(patchSize, img.rows - y);

    // Return center region as sub-image
    return img(cv::Rect(x, y, w, h));
}

/*
 * Computes HSV color histogram with 16 bins per channel.
 * HSV color space provides better lighting invariance than RGB.
 */
cv::Mat computeColorHist(const cv::Mat& img) {
    // Convert BGR to HSV color space
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Define histogram parameters
    int h_bins = 16, s_bins = 16, v_bins = 16;
    int histSize[] = { h_bins, s_bins, v_bins };

    // Define value ranges for each HSV channel
    float h_range[] = { 0, 180 };    // Hue range in OpenCV
    float s_range[] = { 0, 256 };    // Saturation range
    float v_range[] = { 0, 256 };    // Value range
    const float* ranges[] = { h_range, s_range, v_range };

    // Use all three HSV channels
    int channels[] = { 0, 1, 2 };

    // Compute 3D histogram
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);

    // Normalize histogram to sum to 1
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

    // Flatten 3D histogram to 1D vector for storage
    return hist.reshape(1, 1);
}

/*
 * Computes texture histogram using Sobel gradient magnitude.
 * Uses 32 bins to capture edge strength distribution.
 */
cv::Mat computeTextureHist(const cv::Mat& img) {
    // Convert to grayscale for gradient computation
    cv::Mat gray, grad_x, grad_y, grad_mag;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute horizontal and vertical gradients using Sobel operator
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);  // X gradient
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);  // Y gradient

    // Compute gradient magnitude
    cv::magnitude(grad_x, grad_y, grad_mag);

    // Compute histogram of gradient magnitudes
    int histSize = 32;
    float range[] = { 0, 255 };
    const float* histRange = { range };

    cv::Mat hist;
    cv::calcHist(&grad_mag, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // Normalize histogram
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

    // Flatten to 1D
    return hist.reshape(1, 1);
}

/*
 * Computes weighted distance combining color and texture histograms.
 * Uses chi-square distance for histogram comparison.
 * Weighting: 30% color, 70% texture.
 */
double computeDistance(const cv::Mat& hist1_color, const cv::Mat& hist1_tex,
    const cv::Mat& hist2_color, const cv::Mat& hist2_tex) {
    // Compute chi-square distance for color histograms
    double dist_color = cv::compareHist(hist1_color, hist2_color, cv::HISTCMP_CHISQR);

    // Compute chi-square distance for texture histograms
    double dist_texture = cv::compareHist(hist1_tex, hist2_tex, cv::HISTCMP_CHISQR);

    // Weighted combination (70% texture, 30% color)
    return 0.3 * dist_color + 0.7 * dist_texture;
}

int main(int argc, char** argv) {

    // ========== PARSE COMMAND LINE ARGUMENTS ==========

    if (argc < 4) {
        std::cout << "Usage: ./image_match <target_image> <directory> <topN>\n";
        return -1;
    }

    std::string target_path = argv[1];
    std::string directory = argv[2];
    int topN = std::stoi(argv[3]);

    // ========== LOAD TARGET IMAGE ==========

    cv::Mat target = cv::imread(target_path);
    if (target.empty()) {
        std::cerr << "Error: Cannot read target image.\n";
        return -1;
    }

    // Display target image
    cv::Mat small;
    cv::resize(target, small, cv::Size(400, 400));
    cv::imshow("Target Image", small);

    // ========== EXTRACT TARGET FEATURES ==========

    // Extract center 160x160 patch
    cv::Mat targetFeature = extractCenter(target);

    // Compute color and texture histograms from center patch
    cv::Mat target_color_hist = computeColorHist(targetFeature);
    cv::Mat target_texture_hist = computeTextureHist(targetFeature);

    // ========== PROCESS ALL DATABASE IMAGES ==========

    std::vector<std::pair<double, std::string>> distances;

    std::cout << "Processing images in directory: " << directory << "\n";

    // Loop through all files in directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string file_path = entry.path().string();

        // Skip target image itself
        if (file_path == target_path) continue;

        // Load database image
        cv::Mat img = cv::imread(file_path);
        if (img.empty()) continue;

        // Extract center patch from database image
        cv::Mat imgFeature = extractCenter(img);

        // Compute color and texture histograms
        cv::Mat color_hist = computeColorHist(imgFeature);
        cv::Mat texture_hist = computeTextureHist(imgFeature);

        // Compute combined distance
        double dist = computeDistance(target_color_hist, target_texture_hist,
            color_hist, texture_hist);

        // Store result
        distances.push_back({ dist, file_path });
    }

    // ========== SORT AND DISPLAY RESULTS ==========

    // Sort by distance (ascending: smallest = most similar)
    std::sort(distances.begin(), distances.end());

    // Print top N matches
    std::cout << "\nTop " << topN << " matches for " << target_path << ":\n";
    std::cout << "========================================\n";

    for (int i = 0; i < std::min(topN, (int)distances.size()); ++i) {
        std::cout << i + 1 << ". " << fs::path(distances[i].second).filename().string()
            << " | Distance: " << std::fixed << std::setprecision(4)
            << distances[i].first << "\n";
    }

    // ========== DISPLAY MATCHED IMAGES ==========

    // Show top N matching images
    for (int i = 0; i < std::min(topN, (int)distances.size()); ++i) {
        cv::Mat img = cv::imread(distances[i].second);
        if (!img.empty()) {
            cv::resize(img, small, cv::Size(400, 400));
            cv::imshow("Match " + std::to_string(i + 1), small);
        }
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}