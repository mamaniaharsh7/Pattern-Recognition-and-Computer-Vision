// ar.h
// Harsh Vijay Mamania
// 22 March 2026
// Header file for Project 4 AR program.
// Declares functions for pose estimation and augmented reality rendering.

#ifndef AR_H
#define AR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Board constants — must match calibration program
const int AR_BOARD_WIDTH = 9;
const int AR_BOARD_HEIGHT = 6;
const cv::Size AR_BOARD_SIZE(AR_BOARD_WIDTH, AR_BOARD_HEIGHT);
const float AR_SQUARE_SIZE = 1.0f;

// Loads camera intrinsics from a yml file written by the calibration program.
// filename      : path to the .yml file
// camera_matrix : output 3x3 intrinsic matrix
// dist_coeffs   : output distortion coefficients
// Returns true if file was loaded successfully, false otherwise.
bool loadIntrinsics(const std::string& filename,
    cv::Mat& camera_matrix,
    cv::Mat& dist_coeffs);

// Detects chessboard corners in a frame and refines to sub-pixel accuracy.
// Identical in logic to the calibration program's version.
// frame   : input color image
// corners : output vector of detected corner pixel locations
// Returns true if all corners were found, false otherwise.
bool detectCornersAR(const cv::Mat& frame,
    std::vector<cv::Point2f>& corners);

// Builds the fixed 3D world coordinates for one full chessboard.
// point_set : output vector of (x, y, 0) world coords per corner
void buildWorldPointsAR(std::vector<cv::Vec3f>& point_set);

// Estimates the pose of the board using solvePnP.
// corners       : detected 2D corner locations in the image
// point_set     : corresponding 3D world coordinates
// camera_matrix : loaded camera intrinsics
// dist_coeffs   : loaded distortion coefficients
// rvec          : output rotation vector
// tvec          : output translation vector
void estimatePose(const std::vector<cv::Point2f>& corners,
    const std::vector<cv::Vec3f>& point_set,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    cv::Mat& rvec,
    cv::Mat& tvec);

// Projects 3D coordinate axes onto the image and draws them.
// Visualizes X (red), Y (green), Z (blue) axes from the board origin.
// frame         : color image to draw on (modified in place)
// rvec          : rotation vector from solvePnP
// tvec          : translation vector from solvePnP
// camera_matrix : loaded camera intrinsics
// dist_coeffs   : loaded distortion coefficients
void projectAndDrawAxes(cv::Mat& frame,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs);

// Constructs and draws a 3D virtual rocket above the board.
// Projects all object points using the current pose and draws
// the object as colored line segments on the frame.
// frame         : color image to draw on (modified in place)
// rvec          : rotation vector from solvePnP
// tvec          : translation vector from solvePnP
// camera_matrix : loaded camera intrinsics
// dist_coeffs   : loaded distortion coefficients
// color_scheme  : 0 = original (white/red/blue/yellow)
//                 1 = alternate (white/cyan/orange/green)
void drawVirtualObject(cv::Mat& frame,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    int color_scheme = 0);

#endif // AR_H