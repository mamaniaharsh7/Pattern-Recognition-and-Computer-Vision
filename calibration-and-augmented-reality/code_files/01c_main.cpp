// main.cpp
// Harsh Vijay Mamania
// 22 March 2026
// Main entry point for Project 4 calibration program.
// Opens webcam, detects chessboard corners in real time,
// saves calibration frames on 's', calibrates on 'c', quits on 'q'.

#include "calibration.h"
#include <iostream>

int main() {
    // Open the default webcam
    cv::VideoCapture capdev(0, cv::CAP_DSHOW);
    if (!capdev.isOpened()) {
        printf("Error: unable to open video device\n");
        return -1;
    }

    printf("Controls:\n");
    printf("  s - save current frame for calibration\n");
    printf("  c - run calibration (requires at least 5 frames)\n");
    printf("  q - quit\n\n");

    // Lists that accumulate across all saved frames
    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Vec3f>>   point_list;

    // Build the world point set once — it never changes
    std::vector<cv::Vec3f> point_set;
    buildWorldPoints(point_set);

    // Camera intrinsics — filled by runCalibration()
    cv::Mat camera_matrix, dist_coeffs;

    cv::Mat frame;
    std::vector<cv::Point2f> corner_set;
    bool last_found = false;

    while (true) {
        // Grab a frame from the webcam
        capdev >> frame;
        if (frame.empty()) {
            printf("Error: blank frame grabbed\n");
            break;
        }

        // Try to detect chessboard corners
        bool found = detectChessboardCorners(frame, corner_set);
        last_found = found;

        // Draw corners onto the frame
        if (!corner_set.empty()) {
            drawDetectedCorners(frame, corner_set, found);
        }

        // Print corner info when found
        if (found) {
            printf("Corners found: %zu | First corner: (%.2f, %.2f)\n",
                corner_set.size(),
                corner_set[0].x,
                corner_set[0].y);
        }

        cv::imshow("Calibration", frame);

        int key = cv::waitKey(10);

        if (key == 'q' || key == 'Q') {
            printf("Quitting.\n");
            break;
        }

        // Save frame only if board was fully detected
        if (key == 's' || key == 'S') {
            if (last_found) {
                saveCalibrationFrame(corner_set, point_set,
                    corner_list, point_list);
            }
            else {
                printf("No board detected — frame not saved.\n");
            }
        }

        // Run calibration only if at least 5 frames are saved
        if (key == 'c' || key == 'C') {
            if (corner_list.size() < 5) {
                printf("Need at least 5 frames. Currently have: %zu\n",
                    corner_list.size());
            }
            else {
                runCalibration(corner_list, point_list,
                    frame.size(), camera_matrix, dist_coeffs);
                saveIntrinsics("intrinsics.yml",
                    camera_matrix, dist_coeffs);
            }
        }
    }

    capdev.release();
    cv::destroyAllWindows();
    return 0;
}