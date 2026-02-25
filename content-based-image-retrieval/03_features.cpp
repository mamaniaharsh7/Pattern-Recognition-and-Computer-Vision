/*
 * Harsh Mamania
 * 10th February 2026
 *
 * Implementation of feature extraction and distance metric functions for CBIR.
 * Implements Tasks 2, 3, 5, and Extension (PCA dimensionality reduction).
 */

#include "features.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>

 // ========== TASK 2: HISTOGRAM MATCHING IMPLEMENTATION ==========

 /*
  * Computes RG chromaticity histogram with brightness normalization.
  * Uses r = R/(R+G+B) and g = G/(R+G+B) for lighting invariance.
  */
cv::Mat computeRGHistogram(const cv::Mat& src, int histsize) {
    // Initialize 2D histogram matrix with zeros
    cv::Mat hist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32FC1);

    // Loop over all pixels in the image
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);  // Get row pointer for efficiency

        for (int j = 0; j < src.cols; j++) {
            // Extract BGR values (OpenCV stores images as BGR, not RGB)
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            // Compute normalization divisor (R+G+B), avoid division by zero
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0;

            // Calculate chromaticity values (normalized to [0,1])
            float r = R / divisor;
            float g = G / divisor;

            // Convert continuous values to discrete bin indexes
            // Adding 0.5 performs rounding to nearest bin
            int rindex = (int)(r * (histsize - 1) + 0.5);
            int gindex = (int)(g * (histsize - 1) + 0.5);

            // Increment the count in the corresponding bin
            hist.at<float>(rindex, gindex)++;
        }
    }

    // Normalize histogram by total pixel count to get proportions
    hist /= (src.rows * src.cols);

    return hist;
}

/*
 * Computes RGB color histogram with specified bins per channel.
 * Flattens 3D histogram into 1D array for storage.
 */
cv::Mat computeRGBHistogram(const cv::Mat& src, int bins_per_channel) {
    // Total bins = bins_per_channel^3 (e.g., 8x8x8 = 512 bins)
    int total_bins = bins_per_channel * bins_per_channel * bins_per_channel;

    // Create 1D histogram (flattened 3D structure)
    cv::Mat hist = cv::Mat::zeros(1, total_bins, CV_32FC1);

    // Loop over all pixels
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            // Get BGR values
            int B = ptr[j][0];
            int G = ptr[j][1];
            int R = ptr[j][2];

            // Map each channel value [0,255] to bin index [0, bins_per_channel-1]
            // Use floor and min to ensure we stay within valid bin range
            int rindex = std::min((int)floor((R / 256.0) * bins_per_channel), bins_per_channel - 1);
            int gindex = std::min((int)floor((G / 256.0) * bins_per_channel), bins_per_channel - 1);
            int bindex = std::min((int)floor((B / 256.0) * bins_per_channel), bins_per_channel - 1);

            // Flatten 3D index (r,g,b) into 1D index
            // Formula: index = r*(bins^2) + g*bins + b
            int bin_index = rindex * (bins_per_channel * bins_per_channel) +
                gindex * bins_per_channel +
                bindex;

            // Increment count in this bin
            hist.at<float>(0, bin_index)++;
        }
    }

    // Normalize by total pixel count
    hist /= (src.rows * src.cols);

    return hist;
}

/*
 * Computes histogram intersection distance between two histograms.
 * Returns 1 - intersection (0 = identical, 1 = completely different).
 */
float histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2) {
    float intersection = 0.0;

    // Check if we have 2D histogram (rows > 1 and cols > 1)
    if (hist1.rows > 1 && hist1.cols > 1) {
        // Loop through 2D histogram
        for (int i = 0; i < hist1.rows; i++) {
            const float* ptr1 = hist1.ptr<float>(i);
            const float* ptr2 = hist2.ptr<float>(i);

            for (int j = 0; j < hist1.cols; j++) {
                // Accumulate minimum values (histogram intersection)
                intersection += std::min(ptr1[j], ptr2[j]);
            }
        }
    }
    else {
        // Handle 1D histogram (flattened RGB)
        for (int i = 0; i < hist1.cols; i++) {
            intersection += std::min(hist1.at<float>(0, i), hist2.at<float>(0, i));
        }
    }

    // Convert intersection to distance: 0 = identical, 1 = different
    return 1.0 - intersection;
}

// ========== TASK 3: MULTI-HISTOGRAM MATCHING IMPLEMENTATION ==========

/*
 * Computes RGB histogram for center 50% region of image.
 * Focuses on middle area (25% margin on all sides).
 */
cv::Mat computeCenterHistogram(const cv::Mat& src, int bins_per_channel) {
    // Calculate center region boundaries (middle 50% of image)
    int center_start_row = src.rows / 4;       // 25% from top
    int center_end_row = src.rows * 3 / 4;     // 75% from top
    int center_start_col = src.cols / 4;       // 25% from left
    int center_end_col = src.cols * 3 / 4;     // 75% from left

    // Create histogram for center region
    int total_bins = bins_per_channel * bins_per_channel * bins_per_channel;
    cv::Mat hist = cv::Mat::zeros(1, total_bins, CV_32FC1);

    int pixel_count = 0;  // Track pixels in center region

    // Loop only over center region
    for (int i = center_start_row; i < center_end_row; i++) {
        const cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);

        for (int j = center_start_col; j < center_end_col; j++) {
            // Get BGR values
            int B = ptr[j][0];
            int G = ptr[j][1];
            int R = ptr[j][2];

            // Map to bin indexes (same as RGB histogram)
            int rindex = std::min((int)floor((R / 256.0) * bins_per_channel), bins_per_channel - 1);
            int gindex = std::min((int)floor((G / 256.0) * bins_per_channel), bins_per_channel - 1);
            int bindex = std::min((int)floor((B / 256.0) * bins_per_channel), bins_per_channel - 1);

            // Flatten to 1D index
            int bin_index = rindex * (bins_per_channel * bins_per_channel) +
                gindex * bins_per_channel +
                bindex;

            // Increment bin and pixel count
            hist.at<float>(0, bin_index)++;
            pixel_count++;
        }
    }

    // Normalize by center region pixel count
    if (pixel_count > 0) {
        hist /= pixel_count;
    }

    return hist;
}

/*
 * Computes weighted distance combining whole-image and center-region histograms.
 * Uses 0.6 weight for whole image, 0.4 weight for center region.
 */
float multiHistogramDistance(const cv::Mat& center_hist1, const cv::Mat& outer_hist1,
    const cv::Mat& center_hist2, const cv::Mat& outer_hist2) {

    // Compute histogram intersection distance for whole image histograms
    // (outer_hist actually contains whole image histogram)
    float whole_distance = histogramIntersection(outer_hist1, outer_hist2);

    // Compute histogram intersection distance for center region
    float center_distance = histogramIntersection(center_hist1, center_hist2);

    // Weighted combination: whole image (60%), center region (40%)
    float combined_distance = 0.6 * whole_distance + 0.4 * center_distance;

    return combined_distance;
}

// ========== TASK 5: DEEP NETWORK EMBEDDINGS IMPLEMENTATION ==========

/*
 * Reads 512D ResNet18 embeddings from CSV file.
 * CSV format: filename, feature1, feature2, ..., feature512
 */
std::map<std::string, std::vector<float>> readEmbeddings(const std::string& csv_file) {
    std::map<std::string, std::vector<float>> embeddings;

    // Open CSV file
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        printf("Error: Cannot open CSV file %s\n", csv_file.c_str());
        return embeddings;
    }

    std::string line;
    int line_count = 0;

    // Read each line from CSV
    while (std::getline(file, line)) {
        line_count++;

        // Skip empty lines
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string filename;
        std::vector<float> features;

        // First column is filename
        if (!std::getline(ss, filename, ',')) {
            printf("Warning: Line %d has no filename, skipping\n", line_count);
            continue;
        }

        // Extract just the filename (remove any path)
        size_t last_slash = filename.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            filename = filename.substr(last_slash + 1);
        }

        // Read remaining 512 feature values
        std::string value_str;
        while (std::getline(ss, value_str, ',')) {
            try {
                float value = std::stof(value_str);
                features.push_back(value);
            }
            catch (...) {
                printf("Warning: Invalid value in line %d, skipping rest of line\n", line_count);
                break;
            }
        }

        // Verify we got exactly 512 features
        if (features.size() != 512) {
            printf("Warning: Line %d has %lu features (expected 512), skipping\n",
                line_count, features.size());
            continue;
        }

        // Store in map: filename -> feature vector
        embeddings[filename] = features;
    }

    file.close();
    printf("Loaded %lu embeddings from %s\n", embeddings.size(), csv_file.c_str());

    return embeddings;
}

/*
 * Computes sum of squared differences between two feature vectors.
 * Lower SSD = more similar vectors.
 */
float computeSSD(const std::vector<float>& v1, const std::vector<float>& v2) {
    // Verify vectors have same size
    if (v1.size() != v2.size()) {
        printf("Error: Vector size mismatch in SSD computation\n");
        return -1.0f;
    }

    float ssd = 0.0f;

    // Sum of squared differences
    for (size_t i = 0; i < v1.size(); i++) {
        float diff = v1[i] - v2[i];
        ssd += diff * diff;
    }

    return ssd;
}

/*
 * Computes cosine distance (1 - cosine similarity) between two vectors.
 * Measures angular difference in high-dimensional space.
 */
float cosineDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    // Verify vectors have same size
    if (v1.size() != v2.size()) {
        printf("Error: Vector size mismatch in cosine distance computation\n");
        return -1.0f;
    }

    // Compute dot product and magnitudes
    float dot_product = 0.0f;
    float magnitude_v1 = 0.0f;
    float magnitude_v2 = 0.0f;

    for (size_t i = 0; i < v1.size(); i++) {
        dot_product += v1[i] * v2[i];
        magnitude_v1 += v1[i] * v1[i];
        magnitude_v2 += v2[i] * v2[i];
    }

    // Compute L2-norms (Euclidean lengths)
    magnitude_v1 = sqrt(magnitude_v1);
    magnitude_v2 = sqrt(magnitude_v2);

    // Avoid division by zero
    if (magnitude_v1 == 0.0f || magnitude_v2 == 0.0f) {
        return 1.0f;  // Maximum distance if either vector is zero
    }

    // Cosine similarity = dot product of normalized vectors
    float cosine_similarity = dot_product / (magnitude_v1 * magnitude_v2);

    // Cosine distance = 1 - similarity
    // Clamp to [0, 2] range to handle floating point errors
    float distance = 1.0f - cosine_similarity;
    distance = std::max(0.0f, std::min(2.0f, distance));

    return distance;
}

// ========== EXTENSION: PCA DIMENSIONALITY REDUCTION IMPLEMENTATION ==========

/*
 * Applies PCA to reduce embedding dimensionality using SVD.
 * Returns reduced embeddings and outputs mean vector and eigenvectors.
 */
std::map<std::string, std::vector<float>> applyPCA(
    const std::map<std::string, std::vector<float>>& embeddings,
    int n_components,
    std::vector<float>& mean_vec,
    cv::Mat& eigenvectors) {

    // Validate input
    if (embeddings.empty()) {
        printf("Error: No embeddings to apply PCA\n");
        return std::map<std::string, std::vector<float>>();
    }

    int n_images = embeddings.size();
    int n_features = embeddings.begin()->second.size();  // Should be 512

    printf("\nApplying PCA:\n");
    printf("  Images: %d\n", n_images);
    printf("  Original dimensions: %d\n", n_features);
    printf("  Target dimensions: %d\n", n_components);

    // Step 1: Build data matrix (each row is an embedding)
    cv::Mat data_matrix(n_images, n_features, CV_32F);
    std::vector<std::string> filenames;

    int row_idx = 0;
    for (const auto& entry : embeddings) {
        filenames.push_back(entry.first);
        const std::vector<float>& embedding = entry.second;

        // Copy embedding to data matrix row
        for (int col = 0; col < n_features; col++) {
            data_matrix.at<float>(row_idx, col) = embedding[col];
        }
        row_idx++;
    }

    // Step 2: Compute mean vector across all embeddings
    mean_vec.resize(n_features, 0.0f);
    for (int col = 0; col < n_features; col++) {
        float sum = 0.0f;
        for (int row = 0; row < n_images; row++) {
            sum += data_matrix.at<float>(row, col);
        }
        mean_vec[col] = sum / n_images;
    }

    // Step 3: Center the data (subtract mean from each embedding)
    cv::Mat centered_data(n_images, n_features, CV_32F);
    for (int row = 0; row < n_images; row++) {
        for (int col = 0; col < n_features; col++) {
            centered_data.at<float>(row, col) =
                data_matrix.at<float>(row, col) - mean_vec[col];
        }
    }

    // Step 4: Transpose centered data for SVD
    cv::Mat centered_data_T;
    cv::transpose(centered_data, centered_data_T);

    // Step 5: Apply Singular Value Decomposition
    printf("  Computing SVD...\n");
    cv::SVD svd(centered_data_T, cv::SVD::FULL_UV);

    // Extract first n_components eigenvectors (principal components)
    eigenvectors = svd.u(cv::Rect(0, 0, n_components, n_features)).clone();

    printf("  Eigenvectors matrix: %d x %d\n", eigenvectors.rows, eigenvectors.cols);

    // Display top 10 eigenvalues (variance explained by each component)
    printf("  Top 10 eigenvalues: ");
    for (int i = 0; i < std::min(10, (int)svd.w.rows); i++) {
        float eigenval = svd.w.at<float>(i, 0) * svd.w.at<float>(i, 0);
        printf("%.1f ", eigenval);
    }
    printf("\n");

    // Step 6: Project all embeddings to reduced PCA space
    std::map<std::string, std::vector<float>> reduced_embeddings;

    for (int i = 0; i < n_images; i++) {
        std::string filename = filenames[i];
        std::vector<float> reduced(n_components);

        // Project centered embedding onto each principal component
        for (int j = 0; j < n_components; j++) {
            float projection = 0.0f;

            // Dot product: centered_embedding · eigenvector[j]
            for (int k = 0; k < n_features; k++) {
                projection += centered_data.at<float>(i, k) *
                    eigenvectors.at<float>(k, j);
            }

            reduced[j] = projection;
        }

        reduced_embeddings[filename] = reduced;
    }

    printf("  PCA complete! Reduced embeddings: %lu\n", reduced_embeddings.size());

    return reduced_embeddings;
}

/*
 * Projects a single embedding to PCA space using pre-computed mean and eigenvectors.
 * Used for projecting query images at runtime.
 */
std::vector<float> projectToPCA(
    const std::vector<float>& embedding,
    const std::vector<float>& mean_vec,
    const cv::Mat& eigenvectors) {

    int n_features = embedding.size();
    int n_components = eigenvectors.cols;

    // Step 1: Center the embedding by subtracting mean
    std::vector<float> centered(n_features);
    for (int i = 0; i < n_features; i++) {
        centered[i] = embedding[i] - mean_vec[i];
    }

    // Step 2: Project onto eigenvectors (principal components)
    std::vector<float> reduced(n_components);

    for (int j = 0; j < n_components; j++) {
        float projection = 0.0f;

        // Dot product with j-th eigenvector
        for (int k = 0; k < n_features; k++) {
            projection += centered[k] * eigenvectors.at<float>(k, j);
        }

        reduced[j] = projection;
    }

    return reduced;
}

/*
 * Computes percentage overlap between two sets of top-N matches.
 * Used to measure retrieval accuracy preservation after PCA.
 */
float computeMatchOverlap(const std::vector<ImageMatch>& matches1,
    const std::vector<ImageMatch>& matches2, int N) {
    int common = 0;

    // Count how many filenames appear in both top-N lists
    for (int i = 0; i < N && i < matches1.size(); i++) {
        for (int j = 0; j < N && j < matches2.size(); j++) {
            if (matches1[i].filename == matches2[j].filename) {
                common++;
                break;
            }
        }
    }

    // Return percentage overlap
    return (float)common / N * 100.0f;
}