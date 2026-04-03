// calibration.h
// Harsh Vijay Mamania
// 22 March 2026
// Header file for Project 4: camera calibration functions.
// Declares constants and functions for chessboard corner detection
// and camera calibration.

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// ---- Board constants ----
// Number of internal corners, NOT number of squares
const int BOARD_WIDTH = 9;  // corners along the horizontal
const int BOARD_HEIGHT = 6;  // corners along the vertical
const cv::Size BOARD_SIZE(BOARD_WIDTH, BOARD_HEIGHT);

// We treat each square as 1 unit in world space
const float SQUARE_SIZE = 1.0f;

// ---- Function declarations ----

// Detects chessboard corners in a frame and refines to sub-pixel accuracy.
// frame      : input color image from the webcam
// corners    : output vector filled with (x,y) pixel locations of each corner
// Returns true if all corners were found, false otherwise.
bool detectChessboardCorners(const cv::Mat& frame,
    std::vector<cv::Point2f>& corners);

// Draws detected corners onto the frame for visualization.
// frame      : the color image to draw on (modified in place)
// corners    : the corners found by detectChessboardCorners
// found      : whether detection succeeded (changes drawing color)
void drawDetectedCorners(cv::Mat& frame,
    const std::vector<cv::Point2f>& corners,
    bool found);

// Builds the fixed 3D world coordinates for one full chessboard.
// These are always the same regardless of camera angle.
// point_set  : output vector filled with (x, y, 0) world coords per corner
void buildWorldPoints(std::vector<cv::Vec3f>& point_set);

// Saves one calibration frame by appending corner and world point data.
// corners     : the 2D image corners from the current frame
// point_set   : the 3D world points for one board
// corner_list : growing list of all saved 2D corner sets
// point_list  : growing list of all saved 3D world point sets
void saveCalibrationFrame(
    const std::vector<cv::Point2f>& corners,
    const std::vector<cv::Vec3f>& point_set,
    std::vector<std::vector<cv::Point2f>>& corner_list,
    std::vector<std::vector<cv::Vec3f>>& point_list);

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
    cv::Mat& dist_coeffs);

// Saves camera matrix and distortion coefficients to a file.
// filename      : path to output .yml file
// camera_matrix : the solved 3x3 intrinsic matrix
// dist_coeffs   : the solved distortion coefficients
void saveIntrinsics(const std::string& filename,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs);

#endif // CALIBRATION_H