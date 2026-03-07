// experiment_filter.cpp
// Harsh Vijay Mamania
// Purpose: Generalized blur implementations for FFT performance study

#include "filter.h"
#include <cmath>

// ===== Helper: Create Gaussian Kernel =====
cv::Mat experiment_createGaussianKernel(int kernel_size, double sigma) {
    cv::Mat kernel(kernel_size, kernel_size, CV_32F);
    int center = kernel_size / 2;
    double sum = 0.0;

    // If sigma not specified, use standard formula
    if (sigma <= 0) {
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8;
    }

    // Generate Gaussian kernel
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int x = i - center;
            int y = j - center;
            double value = exp(-(x * x + y * y) / (2.0 * sigma * sigma));
            kernel.at<float>(i, j) = value;
            sum += value;
        }
    }

    // Normalize
    kernel /= sum;

    return kernel;
}

// ===== Method 1: Naive Blur (at method) =====
int experiment_blurNaive(cv::Mat& src, cv::Mat& dst, int kernel_size) {
    src.copyTo(dst);

    // Create Gaussian kernel
    cv::Mat kernel = experiment_createGaussianKernel(kernel_size, 0);

    int offset = kernel_size / 2;

    // Apply convolution
    for (int i = offset; i < src.rows - offset; i++) {
        for (int j = offset; j < src.cols - offset; j++) {
            for (int c = 0; c < 3; c++) {
                float sum = 0.0;

                // Convolve with kernel
                for (int ki = 0; ki < kernel_size; ki++) {
                    for (int kj = 0; kj < kernel_size; kj++) {
                        int row = i + ki - offset;
                        int col = j + kj - offset;
                        sum += src.at<cv::Vec3b>(row, col)[c] * kernel.at<float>(ki, kj);
                    }
                }

                dst.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(sum);
            }
        }
    }

    return 0;
}

// ===== Method 2: Separable Blur (pointer method) =====
int experiment_blurSeparable(cv::Mat& src, cv::Mat& dst, int kernel_size) {
    // Create 1D Gaussian kernel
    cv::Mat kernel_1d(1, kernel_size, CV_32F);
    int center = kernel_size / 2;
    double sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8;
    double sum = 0.0;

    for (int i = 0; i < kernel_size; i++) {
        int x = i - center;
        double value = exp(-(x * x) / (2.0 * sigma * sigma));
        kernel_1d.at<float>(0, i) = value;
        sum += value;
    }
    kernel_1d /= sum;

    cv::Mat tmp;
    src.copyTo(tmp);

    int offset = kernel_size / 2;

    // Horizontal pass
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);

        for (int j = offset; j < src.cols - offset; j++) {
            for (int c = 0; c < 3; c++) {
                float sum = 0.0;
                for (int k = 0; k < kernel_size; k++) {
                    sum += sptr[j + k - offset][c] * kernel_1d.at<float>(0, k);
                }
                tptr[j][c] = cv::saturate_cast<uchar>(sum);
            }
        }
    }

    tmp.copyTo(dst);

    // Vertical pass
    for (int i = offset; i < tmp.rows - offset; i++) {
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < tmp.cols; j++) {
            for (int c = 0; c < 3; c++) {
                float sum = 0.0;
                for (int k = 0; k < kernel_size; k++) {
                    cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i + k - offset);
                    sum += tptr[j][c] * kernel_1d.at<float>(0, k);
                }
                dptr[j][c] = cv::saturate_cast<uchar>(sum);
            }
        }
    }

    return 0;
}

// ===== Method 3: FFT Blur (with visualization steps) =====
int experiment_blurFFT(cv::Mat& src, cv::Mat& dst, int kernel_size, bool show_steps) {
    // Split into channels
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> result_channels;

    // Process each channel (only show steps for first channel to avoid clutter)
    for (int c = 0; c < 3; c++) {
        bool show_this_channel = show_steps && (c == 0);  // Only show Blue channel steps

        if (show_this_channel) {
            std::cout << "\n=== Processing Blue Channel ===" << std::endl;
        }

        cv::Mat channel_float;
        channels[c].convertTo(channel_float, CV_32F);

        // STAGE 0: Original channel
        if (show_this_channel) {
            cv::Mat original_display;
            cv::normalize(channel_float, original_display, 0, 1, cv::NORM_MINMAX);
            cv::imshow("Stage 0: Original Channel (Blue)", original_display);
            std::cout << "Stage 0: Original grayscale channel" << std::endl;
            cv::waitKey(0);
        }

        // Get optimal DFT size
        int m = cv::getOptimalDFTSize(channel_float.rows);
        int n = cv::getOptimalDFTSize(channel_float.cols);

        // STAGE 1: Pad image
        cv::Mat padded;
        cv::copyMakeBorder(channel_float, padded, 0, m - channel_float.rows,
            0, n - channel_float.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        if (show_this_channel) {
            cv::Mat padded_display;
            cv::normalize(padded, padded_display, 0, 1, cv::NORM_MINMAX);
            cv::imshow("Stage 1: Padded Image", padded_display);
            std::cout << "Stage 1: Padded to optimal DFT size (" << m << "x" << n << ")" << std::endl;
            cv::waitKey(0);
        }

        // Create complex image for DFT
        cv::Mat planes[] = { padded, cv::Mat::zeros(padded.size(), CV_32F) };
        cv::Mat complex_img;
        cv::merge(planes, 2, complex_img);

        // STAGE 2: Forward DFT
        cv::dft(complex_img, complex_img);

        if (show_this_channel) {
            // Compute magnitude spectrum for visualization
            cv::Mat mag_planes[2];
            cv::split(complex_img, mag_planes);
            cv::magnitude(mag_planes[0], mag_planes[1], mag_planes[0]);
            cv::Mat magnitude = mag_planes[0];

            // Shift zero frequency to center for better visualization
            int cx = magnitude.cols / 2;
            int cy = magnitude.rows / 2;
            cv::Mat q0(magnitude, cv::Rect(0, 0, cx, cy));
            cv::Mat q1(magnitude, cv::Rect(cx, 0, cx, cy));
            cv::Mat q2(magnitude, cv::Rect(0, cy, cx, cy));
            cv::Mat q3(magnitude, cv::Rect(cx, cy, cx, cy));
            cv::Mat tmp;
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);
            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);

            // Log scale for better visualization
            magnitude += cv::Scalar::all(1);
            cv::log(magnitude, magnitude);
            cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

            cv::imshow("Stage 2: Frequency Domain (Magnitude Spectrum)", magnitude);
            std::cout << "Stage 2: After Forward DFT - converted to frequency domain" << std::endl;
            std::cout << "         (Bright = high magnitude, Center = low frequencies)" << std::endl;
            cv::waitKey(0);
        }

        // STAGE 3: Create Gaussian kernel
        cv::Mat kernel = experiment_createGaussianKernel(kernel_size, 0);
        cv::Mat kernel_float;
        kernel.convertTo(kernel_float, CV_32F);

        if (show_this_channel) {
            cv::Mat kernel_display;
            // Resize for better visibility
            cv::resize(kernel_float, kernel_display, cv::Size(200, 200), 0, 0, cv::INTER_NEAREST);
            cv::normalize(kernel_display, kernel_display, 0, 1, cv::NORM_MINMAX);
            cv::imshow("Stage 3: Gaussian Kernel (Spatial Domain)", kernel_display);
            std::cout << "Stage 3: " << kernel_size << "x" << kernel_size << " Gaussian kernel (spatial)" << std::endl;
            cv::waitKey(0);
        }

        // Pad kernel to same size as image
        cv::Mat kernel_padded = cv::Mat::zeros(padded.size(), CV_32F);
        int offset = kernel_size / 2;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int ki = (i - offset + padded.rows) % padded.rows;
                int kj = (j - offset + padded.cols) % padded.cols;
                kernel_padded.at<float>(ki, kj) = kernel_float.at<float>(i, j);
            }
        }

        // STAGE 4: DFT of kernel
        cv::Mat kernel_planes[] = { kernel_padded, cv::Mat::zeros(padded.size(), CV_32F) };
        cv::Mat kernel_complex;
        cv::merge(kernel_planes, 2, kernel_complex);
        cv::dft(kernel_complex, kernel_complex);

        if (show_this_channel) {
            cv::Mat kernel_mag_planes[2];
            cv::split(kernel_complex, kernel_mag_planes);
            cv::magnitude(kernel_mag_planes[0], kernel_mag_planes[1], kernel_mag_planes[0]);
            cv::Mat kernel_magnitude = kernel_mag_planes[0];

            // Shift for visualization
            int cx = kernel_magnitude.cols / 2;
            int cy = kernel_magnitude.rows / 2;
            cv::Mat q0(kernel_magnitude, cv::Rect(0, 0, cx, cy));
            cv::Mat q1(kernel_magnitude, cv::Rect(cx, 0, cx, cy));
            cv::Mat q2(kernel_magnitude, cv::Rect(0, cy, cx, cy));
            cv::Mat q3(kernel_magnitude, cv::Rect(cx, cy, cx, cy));
            cv::Mat tmp;
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);
            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);

            kernel_magnitude += cv::Scalar::all(1);
            cv::log(kernel_magnitude, kernel_magnitude);
            cv::normalize(kernel_magnitude, kernel_magnitude, 0, 1, cv::NORM_MINMAX);

            cv::imshow("Stage 4: Kernel in Frequency Domain", kernel_magnitude);
            std::cout << "Stage 4: Kernel after DFT - in frequency domain" << std::endl;
            std::cout << "         (Notice Gaussian shape preserved)" << std::endl;
            cv::waitKey(0);
        }

        // STAGE 5: Multiply in frequency domain (convolution becomes multiplication!)
        cv::mulSpectrums(complex_img, kernel_complex, complex_img, 0);

        if (show_this_channel) {
            cv::Mat result_mag_planes[2];
            cv::split(complex_img, result_mag_planes);
            cv::magnitude(result_mag_planes[0], result_mag_planes[1], result_mag_planes[0]);
            cv::Mat result_magnitude = result_mag_planes[0];

            // Shift for visualization
            int cx = result_magnitude.cols / 2;
            int cy = result_magnitude.rows / 2;
            cv::Mat q0(result_magnitude, cv::Rect(0, 0, cx, cy));
            cv::Mat q1(result_magnitude, cv::Rect(cx, 0, cx, cy));
            cv::Mat q2(result_magnitude, cv::Rect(0, cy, cx, cy));
            cv::Mat q3(result_magnitude, cv::Rect(cx, cy, cx, cy));
            cv::Mat tmp;
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);
            q1.copyTo(tmp);
            q2.copyTo(q1);
            tmp.copyTo(q2);

            result_magnitude += cv::Scalar::all(1);
            cv::log(result_magnitude, result_magnitude);
            cv::normalize(result_magnitude, result_magnitude, 0, 1, cv::NORM_MINMAX);

            cv::imshow("Stage 5: After Multiplication (Filtered Spectrum)", result_magnitude);
            std::cout << "Stage 5: Image spectrum * Kernel spectrum" << std::endl;
            std::cout << "         (High frequencies attenuated by Gaussian)" << std::endl;
            cv::waitKey(0);
        }

        // STAGE 6: Inverse DFT
        cv::idft(complex_img, complex_img, cv::DFT_SCALE);

        // Extract real part
        cv::split(complex_img, planes);
        cv::Mat result = planes[0];

        if (show_this_channel) {
            cv::Mat result_display;
            cv::normalize(result, result_display, 0, 1, cv::NORM_MINMAX);
            cv::imshow("Stage 6: After Inverse DFT (with padding)", result_display);
            std::cout << "Stage 6: After Inverse DFT - back to spatial domain" << std::endl;
            cv::waitKey(0);
        }

        // STAGE 7: Remove padding
        result = result(cv::Rect(0, 0, src.cols, src.rows));

        // Convert back to uchar
        cv::Mat result_uchar;
        result.convertTo(result_uchar, CV_8U);

        if (show_this_channel) {
            cv::imshow("Stage 7: Final Result (padding removed)", result_uchar);
            std::cout << "Stage 7: Final blurred channel (cropped to original size)" << std::endl;
            std::cout << "\nPress any key to continue..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        result_channels.push_back(result_uchar);
    }

    // Merge channels back
    cv::merge(result_channels, dst);

    return 0;
}

// ===== Method 4: OpenCV GaussianBlur (wrapper) =====
int experiment_blurOpenCV(cv::Mat& src, cv::Mat& dst, int kernel_size) {
    double sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8;
    cv::GaussianBlur(src, dst, cv::Size(kernel_size, kernel_size), sigma);
    return 0;

}
