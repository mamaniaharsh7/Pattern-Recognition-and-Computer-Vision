// ar_main.cpp
// Harsh Vijay Mamania
// 22 March 2026
// Main entry point for Project 4 AR program.
// Loads camera intrinsics, detects two chessboards simultaneously,
// estimates independent poses, and renders a virtual rocket above each.
// Press 'q' to quit.

#include "ar.h"
#include <iostream>
#include <ctime>

int main() {
    // Load intrinsics saved by the calibration program
    cv::Mat camera_matrix, dist_coeffs;
    if (!loadIntrinsics("intrinsics.yml", camera_matrix, dist_coeffs)) {
        return -1;
    }

    printf("Camera matrix:\n");
    for (int r = 0; r < 3; r++) {
        printf("  [%.4f, %.4f, %.4f]\n",
            camera_matrix.at<double>(r, 0),
            camera_matrix.at<double>(r, 1),
            camera_matrix.at<double>(r, 2));
    }

    // Open webcam
    cv::VideoCapture capdev(0, cv::CAP_DSHOW);
    if (!capdev.isOpened()) {
        printf("Error: unable to open video device\n");
        return -1;
    }

    // Build world points once — same for both boards
    std::vector<cv::Vec3f> point_set;
    buildWorldPointsAR(point_set);

    printf("\nControls:\n");
    printf("  q - quit\n\n");

    cv::Mat frame;
    cv::Mat rvec1, tvec1, rvec2, tvec2;

    while (true) {
        capdev >> frame;
        if (frame.empty()) {
            printf("Error: blank frame grabbed\n");
            break;
        }

        // --- Detect corners for both boards ---
        // We need to find TWO separate sets of checkerboard corners
        // Strategy: find all corners in the full frame for board 1,
        // mask that region out, then find board 2 in the remaining area

        std::vector<cv::Point2f> corners1, corners2;
        bool found1 = detectCornersAR(frame, corners1);

        // For board 2: mask out board 1's region before searching
        cv::Mat frame2 = frame.clone();
        if (found1 && !corners1.empty()) {
            // Compute bounding box of board 1's corners
            cv::Rect board1_rect = cv::boundingRect(corners1);

            // Expand bounding box slightly to cover full board area
            int margin = 20;
            board1_rect.x = std::max(0, board1_rect.x - margin);
            board1_rect.y = std::max(0, board1_rect.y - margin);
            board1_rect.width = std::min(frame.cols - board1_rect.x,
                board1_rect.width + 2 * margin);
            board1_rect.height = std::min(frame.rows - board1_rect.y,
                board1_rect.height + 2 * margin);

            // Fill board 1's region with gray so detector ignores it
            cv::rectangle(frame2, board1_rect,
                cv::Scalar(128, 128, 128), cv::FILLED);
        }

        bool found2 = detectCornersAR(frame2, corners2);

        // --- Pose estimation and rendering ---
        if (found1) {
            estimatePose(corners1, point_set,
                camera_matrix, dist_coeffs,
                rvec1, tvec1);
            projectAndDrawAxes(frame, rvec1, tvec1,
                camera_matrix, dist_coeffs);
            // Board 1: original color scheme (0)
            drawVirtualObject(frame, rvec1, tvec1,
                camera_matrix, dist_coeffs, 0);

            printf("Board 1 | tvec: [%.2f, %.2f, %.2f]\n",
                tvec1.at<double>(0),
                tvec1.at<double>(1),
                tvec1.at<double>(2));
        }

        if (found2) {
            estimatePose(corners2, point_set,
                camera_matrix, dist_coeffs,
                rvec2, tvec2);
            projectAndDrawAxes(frame, rvec2, tvec2,
                camera_matrix, dist_coeffs);
            // Board 2: alternate color scheme (1)
            drawVirtualObject(frame, rvec2, tvec2,
                camera_matrix, dist_coeffs, 1);

            printf("Board 2 | tvec: [%.2f, %.2f, %.2f]\n",
                tvec2.at<double>(0),
                tvec2.at<double>(1),
                tvec2.at<double>(2));
        }

        // Status line
        printf("Boards detected: %d/2\n\n",
            (int)found1 + (int)found2);

        cv::imshow("AR - Multiple Targets", frame);

        int key = cv::waitKey(10);
        if (key == 'q' || key == 'Q') {
            printf("Quitting.\n");
            break;
        }
        if (key == 's' || key == 'S') {
            time_t now = time(0);
            char filename[100];
            sprintf(filename, "multi_target_%ld.png", now);
            cv::imwrite(filename, frame);
            printf("Screenshot saved: %s\n", filename);
        }
    }

    capdev.release();
    cv::destroyAllWindows();
    return 0;
}