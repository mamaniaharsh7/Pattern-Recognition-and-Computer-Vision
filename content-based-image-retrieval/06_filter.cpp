/*
 * Harsh Mamania
 * 10th February 2026
 *
 * Implementation of image filtering and gradient computation functions.
 * Provides Sobel operators for texture feature extraction in Task 4.
 */

#include "filter.h"
#include <algorithm>
#include <cmath>

 /*
  * Converts BGR image to grayscale using OpenCV's built-in function.
  * Helper function for quick grayscale conversion.
  */
cv::Mat toGrayscale(const cv::Mat& input) {
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

/*
 * Custom grayscale conversion using weighted RGB channels.
 * Applies inverted red emphasis: 0.6*(255-R) + 0.2*G + 0.2*B
 */
int GreyScale(const cv::Mat& src, cv::Mat& dst) {
    // Validate input: must be 3-channel 8-bit image
    if (src.empty() || src.type() != CV_8UC3) {
        return -1;
    }

    // Create output matrix with same size and type as source
    dst.create(src.rows, src.cols, src.type());

    // Process each pixel
    for (int y = 0; y < src.rows; y++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(y);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < src.cols; x++) {
            const cv::Vec3b& p = srcRow[x];

            // Weighted grayscale formula (inverted red channel)
            float gray = 0.6 * (255.0f - p[2]) + 0.2 * p[1] + 0.2 * p[0];

            // Clamp to valid range [0, 255]
            uint8_t g = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, gray)));

            // Set all channels to gray value
            dstRow[x] = cv::Vec3b(g, g, g);
        }
    }
    return 0;
}

/*
 * Computes horizontal Sobel gradient using 3x3 kernel.
 * Kernel: [-1 0 1; -2 0 2; -1 0 1]
 * Output is signed 16-bit to handle negative gradients.
 */
int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    // Validate input
    if (src.empty() || src.type() != CV_8UC3)
        return -1;

    // Create output: signed short to handle negative values
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    int rows = src.rows;
    int cols = src.cols;

    // Process interior pixels (skip 1-pixel border)
    for (int y = 1; y < rows - 1; y++) {
        // Get pointers to three adjacent rows
        const cv::Vec3b* row_m1 = src.ptr<cv::Vec3b>(y - 1);  // Row above
        const cv::Vec3b* row_0 = src.ptr<cv::Vec3b>(y);       // Current row
        const cv::Vec3b* row_p1 = src.ptr<cv::Vec3b>(y + 1);  // Row below

        cv::Vec3s* dstRow = dst.ptr<cv::Vec3s>(y);

        for (int x = 1; x < cols - 1; x++) {
            // Apply kernel to each color channel
            for (int c = 0; c < 3; c++) {
                // Horizontal Sobel kernel application
                int val =
                    row_m1[x - 1][c] * -1 + row_m1[x + 1][c] * 1 +
                    row_0[x - 1][c] * -2 + row_0[x + 1][c] * 2 +
                    row_p1[x - 1][c] * -1 + row_p1[x + 1][c] * 1;

                dstRow[x][c] = static_cast<short>(val);
            }
        }
    }
    return 0;
}

/*
 * Computes vertical Sobel gradient using 3x3 kernel.
 * Kernel: [-1 -2 -1; 0 0 0; 1 2 1]
 * Output is signed 16-bit to handle negative gradients.
 */
int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    // Validate input
    if (src.empty() || src.type() != CV_8UC3)
        return -1;

    // Create output: signed short
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    int rows = src.rows;
    int cols = src.cols;

    // Process interior pixels (skip 1-pixel border)
    for (int y = 1; y < rows - 1; y++) {
        // Get pointers to three adjacent rows
        const cv::Vec3b* row_m1 = src.ptr<cv::Vec3b>(y - 1);
        const cv::Vec3b* row_0 = src.ptr<cv::Vec3b>(y);
        const cv::Vec3b* row_p1 = src.ptr<cv::Vec3b>(y + 1);

        cv::Vec3s* dstRow = dst.ptr<cv::Vec3s>(y);

        for (int x = 1; x < cols - 1; x++) {
            // Apply kernel to each color channel
            for (int c = 0; c < 3; c++) {
                // Vertical Sobel kernel application
                int val =
                    row_m1[x - 1][c] * -1 + row_m1[x][c] * -2 + row_m1[x + 1][c] * -1 +
                    row_p1[x - 1][c] * 1 + row_p1[x][c] * 2 + row_p1[x + 1][c] * 1;

                dstRow[x][c] = static_cast<short>(val);
            }
        }
    }

    return 0;
}

/*
 * Computes gradient magnitude from horizontal and vertical Sobel outputs.
 * Magnitude = sqrt(sx^2 + sy^2), clamped to [0, 255] range.
 */
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
    // Validate inputs: must be signed 16-bit, same size
    if (sx.empty() || sy.empty() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3)
        return -1;
    if (sx.size() != sy.size())
        return -1;

    // Create output: 8-bit unsigned
    dst = cv::Mat::zeros(sx.size(), CV_8UC3);
    int rows = sx.rows;
    int cols = sx.cols;

    // Compute magnitude for each pixel
    for (int y = 0; y < rows; y++) {
        const cv::Vec3s* sxRow = sx.ptr<cv::Vec3s>(y);
        const cv::Vec3s* syRow = sy.ptr<cv::Vec3s>(y);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);

        for (int x = 0; x < cols; x++) {
            for (int c = 0; c < 3; c++) {
                // Euclidean magnitude: sqrt(sx^2 + sy^2)
                int val = static_cast<int>(std::sqrt(sxRow[x][c] * sxRow[x][c] +
                    syRow[x][c] * syRow[x][c]));

                // Clamp to valid pixel range [0, 255]
                dstRow[x][c] = static_cast<uchar>(std::min(255, val));
            }
        }
    }
    return 0;
}