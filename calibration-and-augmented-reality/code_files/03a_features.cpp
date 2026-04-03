// features.cpp
// Harsh Vijay Mamania
// 22 March 2026
// Standalone program for Task 7 of Project 4.
// Detects and visualizes Harris corners in a live webcam feed.
// Experiments with different thresholds to understand feature detection.
// Press '+' to increase threshold, '-' to decrease, 'q' to quit.

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the default webcam
    cv::VideoCapture capdev(0, cv::CAP_DSHOW);
    if (!capdev.isOpened()) {
        printf("Error: unable to open video device\n");
        return -1;
    }

    // Harris detection parameters
    // blockSize    : size of neighborhood considered for corner detection
    // apertureSize : aperture parameter for Sobel gradient computation
    // k            : Harris detector free parameter (typically 0.04-0.06)
    int    blockSize = 2;
    int    apertureSize = 3;
    double k = 0.04;

    // Threshold as a fraction of the max normalized response (0.0 to 1.0)
    // Higher = fewer but stronger corners
    // Lower  = more corners including weaker ones
    double threshold = 0.5; // was 0.1

    printf("Controls:\n");
    printf("  +  - increase threshold (fewer corners)\n");
    printf("  -  - decrease threshold (more corners)\n");
    printf("  q  - quit\n\n");

    cv::Mat frame, gray, response, response_norm;
    //
    while (true) {
        capdev >> frame;
        if (frame.empty()) {
            printf("Error: blank frame grabbed\n");
            break;
        }

        // Handle keypresses FIRST so threshold updates before display
        int key = cv::waitKey(10);

        if (key == 'q' || key == 'Q') {
            printf("Quitting.\n");
            break;
        }
        if (key == '+' || key == '=') {
            // threshold = std::min(threshold + 0.01, 0.5);
            threshold = std::min(threshold + 0.05, 0.99);  // was 0.01
            printf("Threshold increased to: %.4f\n", threshold);
        }
        if (key == '-' || key == '_') {
            //threshold = std::max(threshold - 0.01, 0.001);
            threshold = std::max(threshold - 0.05, 0.01);  // was 0.01
            printf("Threshold decreased to: %.4f\n", threshold);
        }

        // Step 1: convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        //// Step 2: compute Harris corner response map
        //response = cv::Mat::zeros(gray.size(), CV_32FC1);
        //cv::cornerHarris(gray, response, blockSize, apertureSize, k);

        //// Step 3: normalize response to 0-255
        //cv::normalize(response, response_norm,
        //    0, 255, cv::NORM_MINMAX, CV_32FC1);

        cv::Mat raw_response = cv::Mat::zeros(gray.size(), CV_32FC1);
        cv::cornerHarris(gray, raw_response, blockSize, apertureSize, k);

        // Accumulate response over time using weighted average
        // 0.8 = weight of previous frames, 0.2 = weight of current frame
        // Higher first value = more smoothing but slower to react
        if (response.empty()) {
            response = raw_response.clone();
        }
        else {
            cv::addWeighted(response, 0.8, raw_response, 0.2, 0, response);
        }

        cv::normalize(response, response_norm,
            0, 255, cv::NORM_MINMAX, CV_32FC1);

        // Step 4: find global max for absolute thresholding
        double global_max;
        cv::minMaxLoc(response_norm, nullptr, &global_max);

        // Step 5: threshold with non-maximum suppression
        int corner_count = 0;
        for (int r = 0; r < response_norm.rows; r++) {
            for (int c = 0; c < response_norm.cols; c++) {
                float val = response_norm.at<float>(r, c);

                // Absolute threshold against global max
                if (val > threshold * global_max) {

                    // Non-maximum suppression in 5x5 neighborhood
                    cv::Mat neighborhood = response_norm(
                        /*cv::Range(std::max(0, r - 5),
                            std::min(response_norm.rows, r + 5)),
                        cv::Range(std::max(0, c - 5),
                            std::min(response_norm.cols, c + 5))*/
                        cv::Range(std::max(0, r - 10),
                            std::min(response_norm.rows, r + 10)),
                        cv::Range(std::max(0, c - 10),
                            std::min(response_norm.cols, c + 10))
                    );
                    double local_max;
                    cv::minMaxLoc(neighborhood, nullptr, &local_max);

                    if (val == local_max) {
                        cv::circle(frame, cv::Point(c, r),
                            4, cv::Scalar(0, 0, 255), 2);
                        corner_count++;
                    }
                }
            }
        }

        // Step 6: display UPDATED threshold and corner count
        char info[100];
        sprintf(info, "Threshold: %.4f | Corners: %d",
            threshold, corner_count);
        cv::putText(frame, info,
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Harris Corner Detection", frame);
    }
    //
    capdev.release();
    cv::destroyAllWindows();
    return 0;
}
