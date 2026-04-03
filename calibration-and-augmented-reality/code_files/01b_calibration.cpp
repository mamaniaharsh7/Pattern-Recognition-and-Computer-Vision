// calibration.cpp
// Harsh Vijay Mamania
// 22 March 2026
// Implementation of chessboard corner detection and calibration functions.

#include "calibration.h"

// Detects chessboard corners in a frame and refines to sub-pixel accuracy.
// frame      : input color image from the webcam
// corners    : output vector filled with (x,y) pixel locations of each corner
// Returns true if all corners were found, false otherwise.
bool detectChessboardCorners(const cv::Mat& frame,
    std::vector<cv::Point2f>& corners) {
    // Step 1: convert to grayscale
    // findChessboardCorners works on single-channel (grayscale) images
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Step 2: find the corners
    // CALIB_CB_ADAPTIVE_THRESH helps handle uneven lighting
    // CALIB_CB_NORMALIZE_IMAGE normalizes brightness before detection
    bool found = cv::findChessboardCorners(
        gray,
        BOARD_SIZE,
        corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
    );

    // Step 3: if found, refine to sub-pixel accuracy
    if (found) {
        // Size(11,11) is the search window around each corner
        // Size(-1,-1) means no dead zone in the center
        // The TermCriteria tells OpenCV when to stop refining:
        //   stop after 30 iterations OR when movement < 0.001 pixels
        cv::cornerSubPix(
            gray,
            corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                30, 0.001
            )
        );
    }

    return found;
}

// Draws detected corners onto the frame for visualization.
// frame      : the color image to draw on (modified in place)
// corners    : the corners found by detectChessboardCorners
// found      : whether detection succeeded (changes drawing color)
void drawDetectedCorners(cv::Mat& frame,
    const std::vector<cv::Point2f>& corners,
    bool found) {
    cv::drawChessboardCorners(frame, BOARD_SIZE, corners, found);
}

// Builds the fixed 3D world coordinates for one full chessboard.
// These are always the same regardless of camera angle.
// point_set  : output vector filled with (x, y, 0) world coords per corner
void buildWorldPoints(std::vector<cv::Vec3f>& point_set) {
    point_set.clear();

    // Walk row by row, column by column
    // Y is negative because the Y axis points downward in board space
    for (int row = 0; row < BOARD_HEIGHT; row++) {
        for (int col = 0; col < BOARD_WIDTH; col++) {
            point_set.push_back(cv::Vec3f(
                col * SQUARE_SIZE,   // X: moves right across the board
                -row * SQUARE_SIZE,  // Y: moves downward (hence negative)
                0.0f                 // Z: always 0, all corners are flat
            ));
        }
    }
}

// Saves one calibration frame by appending corner and world point data.
// corners     : the 2D image corners from the current frame
// point_set   : the 3D world points for one board
// corner_list : growing list of all saved 2D corner sets
// point_list  : growing list of all saved 3D world point sets
void saveCalibrationFrame(
    const std::vector<cv::Point2f>& corners,
    const std::vector<cv::Vec3f>& point_set,
    std::vector<std::vector<cv::Point2f>>& corner_list,
    std::vector<std::vector<cv::Vec3f>>& point_list) {

    corner_list.push_back(corners);
    point_list.push_back(point_set);

    printf("Calibration frame saved. Total frames: %zu\n",
        corner_list.size());

    // Print all saved frames so far
    for (int i = 0; i < corner_list.size(); i++) {
        printf("  Frame %d | First 2D corner: (%.2f, %.2f) | "
            "First 3D point: (%.1f, %.1f, %.1f)\n",
            i + 1,
            corner_list[i][0].x, corner_list[i][0].y,
            point_list[i][0][0], point_list[i][0][1], point_list[i][0][2]);
    }
}

// Calibrates the camera using saved corner and world point data.
// corner_list  : all saved 2D corner sets
// point_list   : all saved 3D world point sets
// frame_size   : size of the calibration images
// camera_matrix: output 3x3 intrinsic matrix (modified in place)
// dist_coeffs  : output distortion coefficients (modified in place)
// Returns the reprojection error (lower is better, < 1.0 is good).
double runCalibration(
    std::vector<std::vector<cv::Point2f>>& corner_list,
    std::vector<std::vector<cv::Vec3f>>& point_list,
    cv::Size frame_size,
    cv::Mat& camera_matrix,
    cv::Mat& dist_coeffs) {

    // Initialize camera matrix with a reasonable starting guess
    // fx=1, fy=1 (will be solved), cx/cy at image center
    camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 2) = frame_size.width / 2.0;
    camera_matrix.at<double>(1, 2) = frame_size.height / 2.0;

    // Start with 5 distortion coefficients, all zero
    dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

    // Rotation and translation vectors — one per calibration frame
    // calibrateCamera fills these but we don't need them after this step
    std::vector<cv::Mat> rvecs, tvecs;

    printf("\n--- Before Calibration ---\n");
    printf("Camera matrix:\n");
    for (int r = 0; r < 3; r++) {
        printf("  [%.4f, %.4f, %.4f]\n",
            camera_matrix.at<double>(r, 0),
            camera_matrix.at<double>(r, 1),
            camera_matrix.at<double>(r, 2));
    }
    printf("Distortion coefficients:\n");
    printf("  [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
        dist_coeffs.at<double>(0), dist_coeffs.at<double>(1),
        dist_coeffs.at<double>(2), dist_coeffs.at<double>(3),
        dist_coeffs.at<double>(4));

    // Run calibration
    // CV_CALIB_FIX_ASPECT_RATIO assumes fx == fy (square pixels)
    double error = cv::calibrateCamera(
        point_list,
        corner_list,
        frame_size,
        camera_matrix,
        dist_coeffs,
        rvecs, tvecs,
        cv::CALIB_FIX_ASPECT_RATIO
    );

    printf("\n--- After Calibration ---\n");
    printf("Camera matrix:\n");
    for (int r = 0; r < 3; r++) {
        printf("  [%.4f, %.4f, %.4f]\n",
            camera_matrix.at<double>(r, 0),
            camera_matrix.at<double>(r, 1),
            camera_matrix.at<double>(r, 2));
    }
    printf("Distortion coefficients:\n");
    printf("  [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
        dist_coeffs.at<double>(0), dist_coeffs.at<double>(1),
        dist_coeffs.at<double>(2), dist_coeffs.at<double>(3),
        dist_coeffs.at<double>(4));

    printf("Reprojection error: %.4f pixels\n\n", error);

    return error;
}

// Saves camera matrix and distortion coefficients to a file.
// filename      : path to output .yml file
// camera_matrix : the solved 3x3 intrinsic matrix
// dist_coeffs   : the solved distortion coefficients
void saveIntrinsics(const std::string& filename,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) {

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        printf("Error: could not open file %s for writing\n",
            filename.c_str());
        return;
    }

    fs << "camera_matrix" << camera_matrix;
    fs << "dist_coeffs" << dist_coeffs;
    fs.release();

    printf("Intrinsics saved to %s\n", filename.c_str());
}