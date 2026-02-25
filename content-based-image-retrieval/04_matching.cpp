/*
 * Harsh Mamania
 * 10th February 2026
 *
 * Main program for Content-Based Image Retrieval using histogram matching and deep embeddings.
 * Implements Tasks 2 (RG/RGB histograms), Task 3 (Multi-histogram), Task 5 (DNN embeddings),
 * and Extension (PCA dimensionality reduction).
 *
 * Usage: ./matching <target_image> <image_directory> <feature_type> [N]
 *        feature_type: 'rg', 'rgb', 'multi', or 'dnn'
 *        N: number of top matches to return (default: 3)
 */

#include "features.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <chrono>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {

    // ========== PARSE COMMAND LINE ARGUMENTS ==========

    if (argc < 4) {
        printf("Usage: %s <target_image> <image_directory> <feature_type> [N]\n", argv[0]);
        printf("  feature_type: 'rg', 'rgb', 'multi', or 'dnn'\n");
        printf("  N: number of top matches (default: 3)\n");
        return -1;
    }

    std::string target_filename = argv[1];  // Target image path
    std::string directory = argv[2];         // Directory containing database images
    std::string feature_type = argv[3];      // Feature type to use
    int N = (argc >= 5) ? atoi(argv[4]) : 3; // Number of matches to return

    // Validate feature type
    if (feature_type != "rg" && feature_type != "rgb" && feature_type != "multi" && feature_type != "dnn") {
        printf("Error: feature_type must be 'rg', 'rgb', 'multi', or 'dnn'\n");
        return -1;
    }

    // ========== TASK 5 & EXTENSION: DNN EMBEDDINGS WITH PCA ==========

    if (feature_type == "dnn") {
        printf("Loading DNN embeddings from CSV...\n");
        std::string csv_file = "ResNet18_olym.csv";

        // Load all 512D embeddings from CSV
        std::map<std::string, std::vector<float>> embeddings = readEmbeddings(csv_file);

        if (embeddings.empty()) {
            printf("Error: No embeddings loaded\n");
            return -1;
        }

        // Extract target filename
        std::string target_fname_pca = fs::path(target_filename).filename().string();

        // Verify target exists in embeddings
        if (embeddings.find(target_fname_pca) == embeddings.end()) {
            printf("Error: Target image %s not found in embeddings\n", target_fname_pca.c_str());
            return -1;
        }

        // Define dimensionality levels to test for PCA study
        std::vector<int> dimensions;
        dimensions.push_back(256);
        dimensions.push_back(128);
        dimensions.push_back(64);
        dimensions.push_back(50);
        dimensions.push_back(32);
        dimensions.push_back(16);
        dimensions.push_back(8);

        // ========== COMPUTE 512D BASELINE MATCHES ==========

        std::vector<ImageMatch> matches_512d;
        std::vector<float> target_512d = embeddings[target_fname_pca];

        printf("\nComputing 512D baseline matches...\n");
        auto start_512 = std::chrono::high_resolution_clock::now();

        // Compute SSD distance from target to all database images
        for (const auto& entry : embeddings) {
            // Skip target image itself
            if (entry.first == target_fname_pca) continue;

            float dist = computeSSD(target_512d, entry.second);

            // Store result
            ImageMatch match;
            match.filename = entry.first;
            match.distance = dist;
            matches_512d.push_back(match);
        }

        auto end_512 = std::chrono::high_resolution_clock::now();
        long duration_512 = std::chrono::duration_cast<std::chrono::milliseconds>(end_512 - start_512).count();

        // Normalize SSD distances to [0, 1] range for fair comparison
        float max_ssd = 0;
        for (size_t i = 0; i < matches_512d.size(); i++) {
            if (matches_512d[i].distance > max_ssd) max_ssd = matches_512d[i].distance;
        }
        if (max_ssd > 0) {
            for (size_t i = 0; i < matches_512d.size(); i++) {
                matches_512d[i].distance /= max_ssd;
            }
        }

        // Sort by distance (ascending: smallest distance = most similar)
        std::sort(matches_512d.begin(), matches_512d.end());

        // Display 512D baseline results
        printf("\n========================================\n");
        printf("PCA DIMENSIONALITY REDUCTION STUDY\n");
        printf("========================================\n\n");

        printf("Baseline 512D:\n");
        printf("  Time: %ld ms\n", duration_512);
        printf("  Top 3: ");
        for (int i = 0; i < 3 && i < matches_512d.size(); i++) {
            printf("%s (%.4f)  ", matches_512d[i].filename.c_str(), matches_512d[i].distance);
        }
        printf("\n\n");

        // ========== TEST EACH DIMENSIONALITY LEVEL ==========

        for (size_t dim_idx = 0; dim_idx < dimensions.size(); dim_idx++) {
            int n_comp = dimensions[dim_idx];
            std::vector<float> mean_vec;
            cv::Mat eigenvectors;

            printf("Testing %dD...\n", n_comp);

            // Apply PCA to reduce all embeddings to n_comp dimensions
            auto reduced = applyPCA(embeddings, n_comp, mean_vec, eigenvectors);
            auto target_reduced = reduced[target_fname_pca];

            // Compute distances in reduced space
            std::vector<ImageMatch> matches_reduced;
            auto start = std::chrono::high_resolution_clock::now();

            for (const auto& entry : reduced) {
                // Skip target image
                if (entry.first == target_fname_pca) continue;

                float dist = computeSSD(target_reduced, entry.second);

                ImageMatch match;
                match.filename = entry.first;
                match.distance = dist;
                matches_reduced.push_back(match);
            }

            auto end = std::chrono::high_resolution_clock::now();
            long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // Normalize reduced-dimension distances
            max_ssd = 0;
            for (size_t i = 0; i < matches_reduced.size(); i++) {
                if (matches_reduced[i].distance > max_ssd) max_ssd = matches_reduced[i].distance;
            }
            if (max_ssd > 0) {
                for (size_t i = 0; i < matches_reduced.size(); i++) {
                    matches_reduced[i].distance /= max_ssd;
                }
            }

            // Sort by distance
            std::sort(matches_reduced.begin(), matches_reduced.end());

            // Compute overlap with 512D baseline
            float overlap = computeMatchOverlap(matches_512d, matches_reduced, N);
            float speedup = (float)duration_512 / duration;

            // Display images for current dimensionality (only for 8D to reduce clutter)
            if (n_comp == 8) {
                for (int i = 0; i < N && i < matches_reduced.size(); i++) {
                    std::string match_path = directory + "/" + matches_reduced[i].filename;
                    cv::Mat match_img = cv::imread(match_path);

                    if (match_img.data != NULL) {
                        cv::Mat match_display;
                        cv::resize(match_img, match_display, cv::Size(), 0.5, 0.5);
                        std::string window_name = "8D Match " + std::to_string(i + 1);
                        cv::imshow(window_name, match_display);
                    }
                }
            }

            // Display results for this dimensionality
            printf("%dD Results:\n", n_comp);
            printf("  Time: %ld ms (%.2fx speedup)\n", duration, speedup);
            printf("  Top-%d overlap with 512D: %.1f%%\n", N, overlap);
            printf("  Top 3: ");
            for (int i = 0; i < 3 && i < matches_reduced.size(); i++) {
                printf("%s (%.4f)  ", matches_reduced[i].filename.c_str(), matches_reduced[i].distance);
            }
            printf("\n\n");
        }

        // ========== DISPLAY TARGET AND 512D BASELINE MATCHES ==========

        // Display target image
        cv::Mat target_img = cv::imread(target_filename);
        if (target_img.data != NULL) {
            cv::Mat target_display;
            cv::resize(target_img, target_display, cv::Size(), 0.5, 0.5);
            cv::imshow("Target Image", target_display);
        }

        // Display top N matches from 512D baseline
        for (int i = 0; i < N && i < matches_512d.size(); i++) {
            std::string match_path = directory + "/" + matches_512d[i].filename;
            cv::Mat match_img = cv::imread(match_path);

            if (match_img.data != NULL) {
                cv::Mat match_display;
                cv::resize(match_img, match_display, cv::Size(), 0.5, 0.5);
                std::string window_name = "512D Match " + std::to_string(i + 1);
                cv::imshow(window_name, match_display);
            }
        }

        cv::waitKey(0);
        cv::destroyAllWindows();

        return 0;  // Exit after DNN processing
    }

    // ========== TASKS 2 & 3: CLASSICAL HISTOGRAM MATCHING ==========

    // Set histogram parameters
    int histsize_rg = 16;   // 16x16 bins for RG chromaticity
    int bins_rgb = 8;       // 8x8x8 bins for RGB (512 total)

    printf("Loading target image: %s\n", target_filename.c_str());

    // Load target image
    cv::Mat target_img = cv::imread(target_filename);
    if (target_img.data == NULL) {
        printf("Error: Cannot read target image %s\n", target_filename.c_str());
        return -1;
    }

    // Compute target features based on selected feature type
    cv::Mat target_hist;
    cv::Mat target_center_hist, target_outer_hist;

    if (feature_type == "rg") {
        // TASK 2: RG chromaticity histogram
        printf("Computing RG chromaticity histogram (16x16 bins)...\n");
        target_hist = computeRGHistogram(target_img, histsize_rg);
    }
    else if (feature_type == "rgb") {
        // TASK 2: RGB histogram
        printf("Computing RGB histogram (8x8x8 bins)...\n");
        target_hist = computeRGBHistogram(target_img, bins_rgb);
    }
    else if (feature_type == "multi") {
        // TASK 3: Multi-histogram (whole + center)
        printf("Computing multi-histogram (whole + center, 8x8x8 bins each)...\n");
        target_outer_hist = computeRGBHistogram(target_img, bins_rgb);       // Whole image
        target_center_hist = computeCenterHistogram(target_img, bins_rgb);   // Center 50%
    }

    // Validate directory
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        printf("Error: Cannot open directory %s\n", directory.c_str());
        return -1;
    }

    // ========== PROCESS ALL IMAGES IN DATABASE DIRECTORY ==========

    std::vector<ImageMatch> matches;

    printf("\nProcessing images in directory: %s\n", directory.c_str());
    printf("========================================\n");

    // Loop through all files in directory
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            fs::path filepath = entry.path();
            std::string extension = filepath.extension().string();

            // Check if file is an image
            if (extension == ".jpg" || extension == ".jpeg" ||
                extension == ".png" || extension == ".ppm" || extension == ".tif") {

                std::string filename = filepath.filename().string();

                // Skip target image itself
                if (filename == fs::path(target_filename).filename().string()) {
                    continue;
                }

                // Load database image
                cv::Mat img = cv::imread(filepath.string());
                if (img.data == NULL) {
                    printf("Warning: Cannot read %s, skipping...\n", filename.c_str());
                    continue;
                }

                // Compute features and distance based on feature type
                cv::Mat img_hist;
                cv::Mat img_center_hist, img_outer_hist;
                float distance;

                if (feature_type == "rg") {
                    // RG chromaticity matching
                    img_hist = computeRGHistogram(img, histsize_rg);
                    distance = histogramIntersection(target_hist, img_hist);
                }
                else if (feature_type == "rgb") {
                    // RGB histogram matching
                    img_hist = computeRGBHistogram(img, bins_rgb);
                    distance = histogramIntersection(target_hist, img_hist);
                }
                else {  // multi
                    // Multi-histogram matching
                    img_outer_hist = computeRGBHistogram(img, bins_rgb);
                    img_center_hist = computeCenterHistogram(img, bins_rgb);
                    distance = multiHistogramDistance(target_center_hist, target_outer_hist,
                        img_center_hist, img_outer_hist);
                }

                // Store match result
                ImageMatch match;
                match.filename = filename;
                match.distance = distance;
                matches.push_back(match);

                // Print progress
                printf("Processed: %-20s (distance: %.4f)\n", filename.c_str(), distance);
            }
        }
    }

    // ========== SORT AND DISPLAY RESULTS ==========

    if (matches.empty()) {
        printf("\nNo images found in directory!\n");
        return -1;
    }

    // Sort by distance (ascending: smallest distance = most similar)
    std::sort(matches.begin(), matches.end());

    // Print top N matches
    printf("\n========================================\n");
    printf("Top %d matches for %s:\n", N, target_filename.c_str());
    printf("========================================\n");

    for (int i = 0; i < N && i < matches.size(); i++) {
        printf("%d. %-20s (distance: %.4f)\n",
            i + 1, matches[i].filename.c_str(), matches[i].distance);
    }

    printf("========================================\n");
    printf("Total images processed: %lu\n", matches.size());

    // ========== DISPLAY IMAGES ==========

    // Display target image
    cv::Mat target_display;
    cv::resize(target_img, target_display, cv::Size(), 0.5, 0.5);
    cv::imshow("Target Image", target_display);

    // Display top N matching images
    for (int i = 0; i < N && i < matches.size(); i++) {
        std::string match_path = directory + "/" + matches[i].filename;
        cv::Mat match_img = cv::imread(match_path);

        if (match_img.data != NULL) {
            cv::Mat match_display;
            cv::resize(match_img, match_display, cv::Size(), 0.5, 0.5);
            std::string window_name = "Match " + std::to_string(i + 1) + ": " + matches[i].filename;
            cv::imshow(window_name, match_display);
        }
    }

    // Wait for key press and cleanup
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}