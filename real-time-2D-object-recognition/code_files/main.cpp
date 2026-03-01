/*
 * Name: Harsh Vijay Mamania
 * Date: 17th Feb 2026
 *
 * main.cpp
 *
 * Main entry point for Project 3: Real-time 2D Object Recognition
 * Handles camera capture, pipeline execution, display, and user input.
 */

#include "objectRecognition.h"
#include "embeddingUtils.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <iostream>


int main() {
    // Seed the random number generator used in computeDynamicThreshold
    srand(static_cast<unsigned int>(time(nullptr)));

    // Open the default webcam
    cv::VideoCapture capdev(0, cv::CAP_DSHOW);
    if (!capdev.isOpened()) {
        printf("Error: unable to open video device\n");
        return -1;
    }

    // Print camera resolution for reference
    cv::Size frame_size(
        static_cast<int>(capdev.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(capdev.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    printf("Camera resolution: %d x %d\n", frame_size.width, frame_size.height);

    // Create display windows
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Cleaned", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Region Map", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Regions", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    cv::Mat thresholded;
    cv::Mat cleaned;
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::Mat region_display;
    cv::Mat region_map;
    int screenshot_counter = 1;

    // Evaluation log for confusion matrix -- stores (true_label, predicted_label) pairs
    std::vector<std::pair<std::string, std::string>> eval_log;

    // Path to the training database CSV file
    const std::string DB_PATH = "object_db.csv";

    // Load ResNet18 network for embedding-based classification
    const std::string NET_PATH = "resnet18.onnx";
    cv::dnn::Net net = cv::dnn::readNet(NET_PATH);
    if (net.empty()) {
        printf("Warning: could not load ResNet18 from %s -- embedding classification disabled\n", NET_PATH.c_str());
    }
    else {
        printf("Loaded ResNet18 network\n");
    }

    // In-memory embedding database for one-shot classification
    std::vector<EmbeddingExample> emb_db;

    // Flag to toggle between hand-crafted and embedding classification display
    bool use_embedding = false;
    std::vector<TrainingExample> db;
    std::vector<double> stdevs;

    if (loadDatabase(DB_PATH, db) != 0) {
        printf("Warning: could not load database from %s -- classification disabled\n", DB_PATH.c_str());
    }
    else {
        printf("Loaded %d training examples\n", static_cast<int>(db.size()));
        computeFeatureStdevs(db, stdevs);
    }

    // Minimum region area in pixels -- regions smaller than this are ignored
    const int MIN_REGION_AREA = 500;

    // Maximum number of objects to detect simultaneously
    const int MAX_REGIONS = 3;

    printf("Controls: n = train features | t = train embedding | x = toggle mode | v = plot embeddings | e = evaluate | m = confusion matrix | s = screenshot | q = quit\n");

    for (;;) {
        // Capture a new frame from the camera
        capdev >> frame;
        if (frame.empty()) {
            printf("Error: empty frame received\n");
            break;
        }

        // Task 1: threshold the frame
        if (thresholdFrame(frame, thresholded) != 0) {
            printf("Error: thresholding pipeline failed\n");
            break;
        }

        // Task 2: morphological cleanup
        if (applyMorphology(thresholded, cleaned) != 0) {
            printf("Error: morphology pipeline failed\n");
            break;
        }

        // Task 3: connected components
        int num_labels = computeConnectedComponents(cleaned, labels, stats, centroids);
        if (num_labels < 0) {
            printf("Error: connected components failed\n");
            break;
        }

        // Find top N valid regions
        std::vector<int> top_regions = findTopRegions(stats, num_labels, frame.rows, frame.cols,
            MIN_REGION_AREA, MAX_REGIONS);

        // Draw region map -- all valid regions in distinct colors
        drawRegionMap(labels, stats, num_labels, frame.rows, frame.cols, MIN_REGION_AREA, region_map);

        // Start with a black display image
        region_display = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);

        // Colors for each region -- cyan, yellow, green
        std::vector<cv::Vec3b> region_colors = {
            cv::Vec3b(255, 255, 0),
            cv::Vec3b(0, 255, 255),
            cv::Vec3b(0, 255, 0)
        };

        // Draw the region if one was found
        std::vector<double> features;
        cv::Mat embedding;

        // Track best region for training and evaluation (largest one)
        int best_region = top_regions.empty() ? -1 : top_regions[0];

        if (!top_regions.empty()) {
            // Process each detected region
            for (int r = 0; r < static_cast<int>(top_regions.size()); r++) {
                int region_id = top_regions[r];
                cv::Vec3b color = region_colors[r % region_colors.size()];

                // Draw this region in its assigned color
                for (int row = 0; row < labels.rows; row++) {
                    for (int col = 0; col < labels.cols; col++) {
                        if (labels.at<int>(row, col) == region_id) {
                            region_display.at<cv::Vec3b>(row, col) = color;
                        }
                    }
                }

                // Compute moments for this region
                cv::Mat region_mask = (labels == region_id);
                cv::Moments region_moments = cv::moments(region_mask, true);

                // Compute oriented bounding box
                cv::RotatedRect obb = computeOrientedBoundingBox(labels, region_id);

                // Compute feature vector (used for first region for training/eval)
                std::vector<double> region_features;
                computeFeatures(labels, region_id, region_moments, obb, region_features);
                if (r == 0) {
                    features = region_features;
                }

                // Draw axis and bounding box overlay
                drawFeaturesOverlay(region_display, region_moments, obb);

                // Classify this region
                std::string predicted_label = "unknown";
                std::string embedding_label = "unknown";
                cv::Mat     region_embedding;

                if (!db.empty() && !region_features.empty()) {
                    predicted_label = classifyObject(region_features, db, stdevs);
                }

                // Compute embedding for this region if network is loaded
                if (!net.empty()) {
                    float minE1, maxE1, minE2, maxE2;
                    double cx = region_moments.m10 / region_moments.m00;
                    double cy = region_moments.m01 / region_moments.m00;
                    double theta = computeOrientation(region_moments);

                    if (computeAxisExtents(labels, region_id, cx, cy, theta,
                        minE1, maxE1, minE2, maxE2) == 0) {
                        cv::Mat embimage;
                        prepEmbeddingImage(frame, embimage,
                            static_cast<int>(cx), static_cast<int>(cy),
                            static_cast<float>(theta),
                            minE1, maxE1, minE2, maxE2, 0);

                        if (!embimage.empty()) {
                            getEmbedding(embimage, region_embedding, net, 0);
                            if (r == 0) embedding = region_embedding;

                            if (!emb_db.empty() && !region_embedding.empty()) {
                                embedding_label = classifyEmbedding(region_embedding, emb_db);
                            }
                        }
                    }
                }

                // Display label for this region
                std::string display_label = use_embedding ? embedding_label : predicted_label;
                double cx = region_moments.m10 / region_moments.m00;
                double cy = region_moments.m01 / region_moments.m00;
                cv::putText(region_display, display_label,
                    cv::Point(static_cast<int>(cx) - 40, static_cast<int>(cy) - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);

                printf("Region %d -- Filled: %.3f | Aspect: %.3f | Label: %s | Emb: %s\n",
                    r + 1, region_features[0], region_features[1],
                    predicted_label.c_str(), embedding_label.c_str());
            }

            // Show current mode indicator
            std::string mode_text = use_embedding ? "Mode: Embedding" : "Mode: Hand-crafted";
            cv::putText(region_display, mode_text,
                cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 2);

        }
        else {
            // No valid regions found
            region_display = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
        }

        // Display all four stages
        cv::imshow("Original", frame);
        cv::imshow("Thresholded", thresholded);
        cv::imshow("Cleaned", cleaned);
        cv::imshow("Region Map", region_map);
        cv::imshow("Regions", region_display);

        // Handle keyboard input
        char key = static_cast<char>(cv::waitKey(10));

        if (key == 'q') {
            printf("Quitting\n");
            break;
        }
        else if (key == 't') {
            // Embedding training mode -- save current embedding with a label
            if (!embedding.empty()) {
                cv::destroyAllWindows();

                printf("Enter label for embedding training example: ");
                std::string label;
                std::cin >> label;

                saveEmbeddingExample(emb_db, label, embedding);
                printf("Saved embedding for: %s (total: %d)\n",
                    label.c_str(), static_cast<int>(emb_db.size()));

                cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Cleaned", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Region Map", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Regions", cv::WINDOW_AUTOSIZE);
            }
            else {
                printf("No embedding available -- place an object in frame first\n");
            }
        }
        else if (key == 'x') {
            // Toggle between hand-crafted features and embedding classification display
            use_embedding = !use_embedding;
            printf("Classification mode: %s\n", use_embedding ? "embedding" : "hand-crafted features");
        }
        else if (key == 'v') {
            // Visualize embeddings as a 2D PCA scatter plot
            if (emb_db.empty()) {
                printf("No embeddings stored yet -- collect some with t key first\n");
            }
            else {
                plotEmbeddings2D(emb_db);
                printf("Embedding plot generated for %d examples\n", static_cast<int>(emb_db.size()));
            }
        }
        else if (key == 'e') {
            // Evaluation mode -- log true vs predicted label pair
            if (best_region > 0 && !features.empty()) {
                // Destroy windows to release keyboard focus so std::cin works
                cv::destroyAllWindows();

                printf("Enter true label for current object: ");
                std::string true_label;
                std::cin >> true_label;

                // Use whichever classifier is currently active
                std::string predicted_label;
                if (use_embedding && !embedding.empty() && !emb_db.empty()) {
                    predicted_label = classifyEmbedding(embedding, emb_db);
                }
                else {
                    predicted_label = classifyObject(features, db, stdevs);
                }

                logEvaluationResult(eval_log, true_label, predicted_label);
                printf("Logged -- True: %s | Predicted: %s [%s]\n",
                    true_label.c_str(), predicted_label.c_str(),
                    use_embedding ? "embedding" : "hand-crafted");

                // Recreate all display windows
                cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Cleaned", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Region Map", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Regions", cv::WINDOW_AUTOSIZE);
            }
            else {
                printf("No valid region detected -- place an object in frame first\n");
            }
        }
        else if (key == 'm') {
            // Print confusion matrix from all logged evaluations
            if (eval_log.empty()) {
                printf("No evaluation data yet -- press e to log true/predicted pairs first\n");
            }
            else {
                printConfusionMatrix(eval_log);
            }
        }
        else if (key == 'n') {
            // Training mode -- only save if a valid region is currently detected
            if (best_region > 0 && !features.empty()) {
                // Destroy windows to release keyboard focus so std::cin works
                cv::destroyAllWindows();

                printf("Enter label for current object: ");
                std::string label;
                std::cin >> label;

                if (saveTrainingExample(DB_PATH, label, features) == 0) {
                    printf("Saved training example for: %s\n", label.c_str());
                }
                else {
                    printf("Error: could not save training example\n");
                }

                // Recreate all display windows
                cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Cleaned", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Region Map", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("Regions", cv::WINDOW_AUTOSIZE);
            }
            else {
                printf("No valid region detected -- place an object in frame first\n");
            }
        }
        else if (key == 's') {
            char original_name[256];
            char threshold_name[256];
            char cleaned_name[256];
            char region_map_name[256];
            char region_name[256];

            sprintf(original_name, "screenshot_original_%d.jpg", screenshot_counter);
            sprintf(threshold_name, "screenshot_thresholded_%d.jpg", screenshot_counter);
            sprintf(cleaned_name, "screenshot_cleaned_%d.jpg", screenshot_counter);
            sprintf(region_map_name, "screenshot_regionmap_%d.jpg", screenshot_counter);
            sprintf(region_name, "screenshot_regions_%d.jpg", screenshot_counter);

            cv::imwrite(original_name, frame);
            cv::imwrite(threshold_name, thresholded);
            cv::imwrite(cleaned_name, cleaned);
            cv::imwrite(region_map_name, region_map);
            cv::imwrite(region_name, region_display);

            printf("Saved screenshots %d\n", screenshot_counter);
            screenshot_counter++;
        }
    }

    capdev.release();
    cv::destroyAllWindows();
    return 0;
}