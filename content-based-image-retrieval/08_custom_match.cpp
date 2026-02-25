/*
 * Harsh Mamania
 * 10th February 2026
 *
 * Task 7: Custom design combining deep embeddings, color histograms, and texture histograms.
 * Hybrid approach: 60% ResNet18 embeddings + 20% HSV color + 20% Sobel texture.
 * Displays both most similar and least similar images for comprehensive analysis.
 *
 * Usage: ./custom_match <target_image> <directory> <topN>
 */

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

// Structure to hold all three feature types for an image
struct ImageFeatures {
    cv::Mat deep;      // 512D ResNet18 embedding
    cv::Mat color;     // HSV color histogram
    cv::Mat texture;   // Sobel texture histogram
};

/*
 * Computes HSV color histogram with 8 bins per channel.
 * Returns normalized histogram as 1D row vector.
 */
cv::Mat computeColorHist(const cv::Mat& img) {
    // Convert to HSV color space
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Define histogram parameters (8x8x8 = 512 bins)
    int h_bins = 8, s_bins = 8, v_bins = 8;
    int histSize[] = { h_bins, s_bins, v_bins };

    // Define value ranges for HSV channels
    float h_range[] = { 0, 180 };
    float s_range[] = { 0, 256 };
    float v_range[] = { 0, 256 };
    const float* ranges[] = { h_range, s_range, v_range };

    int channels[] = { 0, 1, 2 };

    // Compute 3D histogram
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);

    // Normalize using L2 norm for better stability
    cv::normalize(hist, hist, 1, 0, cv::NORM_L2);

    // Flatten to 1D row vector
    return hist.reshape(1, 1);
}

/*
 * Computes texture histogram using Sobel gradient magnitude.
 * Uses 32 bins to capture edge strength distribution.
 */
cv::Mat computeTextureHist(const cv::Mat& img) {
    // Convert to grayscale
    cv::Mat gray, grad_x, grad_y, mag;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute gradients using Sobel operator
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    // Compute gradient magnitude
    cv::magnitude(grad_x, grad_y, mag);

    // Compute histogram of magnitudes
    int bins = 32;
    float range[] = { 0, 255 };
    const float* histRange = { range };

    cv::Mat hist;
    cv::calcHist(&mag, 1, 0, cv::Mat(), hist, 1, &bins, &histRange, true, false);

    // Normalize using L2 norm
    cv::normalize(hist, hist, 1, 0, cv::NORM_L2);

    // Flatten to 1D row vector
    return hist.reshape(1, 1);
}

/*
 * Loads ResNet18 deep network embeddings from CSV file.
 * CSV format: filename, feature1, feature2, ..., feature512
 */
std::unordered_map<std::string, cv::Mat> loadDeepFeatures(const std::string& csv_file) {
    std::unordered_map<std::string, cv::Mat> embeddings;

    std::ifstream file(csv_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open CSV file " << csv_file << "\n";
        return embeddings;
    }

    std::string line;
    int line_count = 0;

    // Read each line from CSV
    while (std::getline(file, line)) {
        line_count++;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string filename;

        // Extract filename (first column)
        if (!std::getline(ss, filename, ',')) continue;

        // Remove path from filename (keep only filename)
        size_t last_slash = filename.find_last_of("/\\");
        if (last_slash != std::string::npos)
            filename = filename.substr(last_slash + 1);

        // Extract 512 feature values
        std::vector<float> vec;
        std::string value_str;
        while (std::getline(ss, value_str, ',')) {
            try {
                vec.push_back(std::stof(value_str));
            }
            catch (...) {
                break;
            }
        }

        // Verify we got exactly 512 features
        if (vec.size() != 512) {
            std::cerr << "Warning: Line " << line_count << " has "
                << vec.size() << " features (expected 512), skipping.\n";
            continue;
        }

        // Store as row vector
        embeddings[filename] = cv::Mat(vec).clone().reshape(1, 1);
    }

    std::cout << "Loaded " << embeddings.size() << " deep features from " << csv_file << "\n";
    return embeddings;
}

/*
 * Computes cosine distance between two vectors.
 * Distance = 1 - cosine_similarity, where similarity is normalized dot product.
 */
double cosineDistance(const cv::Mat& a, const cv::Mat& b) {
    // Ensure vectors are 1D row vectors
    cv::Mat a1 = a.reshape(1, 1);
    cv::Mat b1 = b.reshape(1, 1);

    // Compute dot product and norms
    double dot = a1.dot(b1);
    double normA = cv::norm(a1);
    double normB = cv::norm(b1);

    // Handle zero vectors
    if (normA < 1e-10 || normB < 1e-10) return 1.0;

    // Cosine similarity = dot / (normA * normB)
    // Distance = 1 - similarity
    return 1.0 - (dot / (normA * normB));
}

/*
 * Computes weighted distance combining deep, color, and texture features.
 * Weighting: 60% deep + 20% color + 20% texture.
 */
double computeDistance(const ImageFeatures& a, const ImageFeatures& b) {
    // Deep feature distance using cosine distance
    double dDeep = cosineDistance(a.deep, b.deep);

    // Color distance using Bhattacharyya distance (range 0 to 1)
    double dColor = cv::compareHist(a.color, b.color, cv::HISTCMP_BHATTACHARYYA);

    // Texture distance using Bhattacharyya distance
    double dTexture = cv::compareHist(a.texture, b.texture, cv::HISTCMP_BHATTACHARYYA);

    // Weighted fusion (all components normalized to [0, 1] range)
    return (0.6 * dDeep) + (0.2 * dColor) + (0.2 * dTexture);
}

int main(int argc, char** argv) {

    // ========== PARSE COMMAND LINE ARGUMENTS ==========

    if (argc < 4) {
        std::cout << "Usage: ./custom_match <target_image> <directory> <topN>\n";
        return -1;
    }

    std::string targetPath = argv[1];
    std::string directory = argv[2];
    int topN = std::stoi(argv[3]);

    // Path to ResNet18 embeddings CSV (update this path as needed)
    std::string csvPath = "ResNet18_olym.csv";

    // ========== LOAD DEEP FEATURES ==========

    std::cout << "Loading deep features from CSV...\n";
    auto deepMap = loadDeepFeatures(csvPath);

    // ========== EXTRACT TARGET FEATURES ==========

    // Load target image
    cv::Mat targetImg = cv::imread(targetPath);
    if (targetImg.empty()) {
        std::cerr << "Error: Failed to load target image\n";
        return -1;
    }

    // Display target image
    cv::Mat small;
    cv::resize(targetImg, small, cv::Size(400, 400));
    cv::imshow("Target Image", small);

    // Get target filename
    std::string targetName = fs::path(targetPath).filename().string();

    // Verify target has deep features
    if (deepMap.find(targetName) == deepMap.end()) {
        std::cerr << "Error: Target deep feature missing in CSV!\n";
        return -1;
    }

    // Gather all three feature types for target
    ImageFeatures targetFeat;
    targetFeat.deep = deepMap[targetName];
    targetFeat.color = computeColorHist(targetImg);
    targetFeat.texture = computeTextureHist(targetImg);

    // ========== PROCESS DATABASE IMAGES ==========

    std::vector<std::pair<double, std::string>> results;

    std::cout << "Processing database images...\n";

    // Loop through all images in database directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string filePath = entry.path().string();
        std::string name = entry.path().filename().string();

        // Skip target image itself
        if (name == targetName) continue;

        // Skip if deep features not available
        if (deepMap.find(name) == deepMap.end()) continue;

        // Load database image
        cv::Mat img = cv::imread(filePath);
        if (img.empty()) continue;

        // Gather all three feature types
        ImageFeatures feat;
        feat.deep = deepMap[name];
        feat.color = computeColorHist(img);
        feat.texture = computeTextureHist(img);

        // Compute combined distance
        double dist = computeDistance(targetFeat, feat);

        // Store result
        results.push_back({ dist, name });
    }

    // ========== SORT RESULTS ==========

    // Sort by distance (ascending: smallest = most similar)
    std::sort(results.begin(), results.end(),
        [](auto& a, auto& b) { return a.first < b.first; });

    // ========== DISPLAY TOP MATCHES ==========

    std::cout << "\n====== TOP " << topN << " MOST SIMILAR MATCHES ======\n";

    for (int i = 0; i < std::min(topN, (int)results.size()); ++i) {
        std::cout << i + 1 << ". " << results[i].second
            << " | Distance: " << std::fixed << std::setprecision(4)
            << results[i].first << "\n";

        // Display matched image
        std::string imgPath = directory + "/" + results[i].second;
        cv::Mat img = cv::imread(imgPath);

        if (!img.empty()) {
            cv::resize(img, small, cv::Size(400, 400));
            cv::imshow("Most Similar " + std::to_string(i + 1), small);
        }
    }

    // ========== DISPLAY LEAST SIMILAR MATCHES ==========

    std::cout << "\n====== TOP " << topN << " LEAST SIMILAR MATCHES ======\n";

    // Show least similar images (from end of sorted list)
    for (int i = results.size() - 1; i >= std::max(0, (int)results.size() - topN); --i) {
        int rank = results.size() - i;

        std::cout << results[i].second
            << " | Distance: " << std::fixed << std::setprecision(4)
            << results[i].first << "\n";

        // Display least similar image
        std::string imgPath = directory + "/" + results[i].second;
        cv::Mat img = cv::imread(imgPath);

        if (!img.empty()) {
            cv::resize(img, small, cv::Size(400, 400));
            cv::imshow("Least Similar " + std::to_string(rank), small);
        }
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}