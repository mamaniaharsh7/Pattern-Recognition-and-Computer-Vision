/*
 * Harsh Vijay Mamania
 * February 1, 2026
 * CS 5330 Project 1
 *
 * Purpose: Benchmark program for comparing blur implementation performance across kernel sizes
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include "filter.h"

 /*
  * Generic timing function for blur operations with warmup and averaging
  * blurFunc: blur function to benchmark, src: source image, kernel_size: kernel dimensions
  * num_runs: number of iterations to average
  * Returns average execution time in milliseconds
  */
template<typename BlurFunc>
double timeBlurFunction(BlurFunc blurFunc, cv::Mat& src, int kernel_size, int num_runs) {
    cv::Mat dst;

    // Warmup runs to populate cache and stabilize timing
    for (int i = 0; i < 3; i++) {
        blurFunc(src, dst, kernel_size);
    }

    // Timed runs for performance measurement
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        blurFunc(src, dst, kernel_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Return average time per run
    return duration.count() / (double)num_runs;
}

int main(int argc, char* argv[]) {
    std::cout << "===== FFT Blur Performance Benchmark =====" << std::endl;
    std::cout << std::endl;

    // Load high and low frequency test images
    std::cout << "Loading test images..." << std::endl;
    cv::Mat high_freq = cv::imread("high_frequency.jpg");
    cv::Mat low_freq = cv::imread("low_frequency.jpg");

    // Validate image loading
    if (high_freq.empty() || low_freq.empty()) {
        std::cout << "ERROR: Could not load test images!" << std::endl;
        std::cout << "Please place 'high_frequency.jpg' and 'low_frequency.jpg' in the project directory." << std::endl;
        return -1;
    }

    // Resize both images to consistent dimensions for fair comparison
    cv::resize(high_freq, high_freq, cv::Size(640, 480));
    cv::resize(low_freq, low_freq, cv::Size(640, 480));

    std::cout << "Images loaded successfully." << std::endl;
    std::cout << "Image size: " << high_freq.cols << "x" << high_freq.rows << std::endl;
    std::cout << std::endl;

    // Visualize FFT transformation stages for educational purposes
    std::cout << "\n========== FFT VISUALIZATION (5x5 kernel) ==========" << std::endl;
    std::cout << "Press any key in each window to see next stage..." << std::endl;
    cv::Mat visualization_dummy;

    // Show FFT stages for high frequency image
    experiment_blurFFT(high_freq, visualization_dummy, 5, true);

    // Show FFT stages for low frequency image
    experiment_blurFFT(low_freq, visualization_dummy, 5, true);

    std::cout << "\nVisualization complete. Starting benchmarks...\n" << std::endl;

    // Define test parameters
    std::vector<int> kernel_sizes = { 3, 5, 9, 15, 21, 31 };
    int num_runs = 10;

    // Structure to store benchmark results
    struct Result {
        std::string method;
        int kernel_size;
        double time_high_freq;
        double time_low_freq;
    };
    std::vector<Result> results;

    // Run comprehensive performance benchmarks
    std::cout << "Running benchmarks (" << num_runs << " runs per test)..." << std::endl;
    std::cout << std::endl;

    // Iterate through each kernel size
    for (int kernel_size : kernel_sizes) {
        std::cout << "===== Kernel Size: " << kernel_size << "x" << kernel_size << " =====" << std::endl;

        // Benchmark Method 1: Naive 2D convolution
        std::cout << "  Testing Naive..." << std::flush;
        double time_naive_hf = timeBlurFunction(experiment_blurNaive, high_freq, kernel_size, num_runs);
        double time_naive_lf = timeBlurFunction(experiment_blurNaive, low_freq, kernel_size, num_runs);
        std::cout << " Done. (High: " << time_naive_hf << "ms, Low: " << time_naive_lf << "ms)" << std::endl;
        results.push_back({ "Naive", kernel_size, time_naive_hf, time_naive_lf });

        // Benchmark Method 2: Separable filter optimization
        std::cout << "  Testing Separable..." << std::flush;
        double time_sep_hf = timeBlurFunction(experiment_blurSeparable, high_freq, kernel_size, num_runs);
        double time_sep_lf = timeBlurFunction(experiment_blurSeparable, low_freq, kernel_size, num_runs);
        std::cout << " Done. (High: " << time_sep_hf << "ms, Low: " << time_sep_lf << "ms)" << std::endl;
        results.push_back({ "Separable", kernel_size, time_sep_hf, time_sep_lf });

        // Benchmark Method 3: FFT-based frequency domain convolution (wrapped to disable visualization)
        std::cout << "  Testing FFT..." << std::flush;
        double time_fft_hf = timeBlurFunction(
            [](cv::Mat& s, cv::Mat& d, int k) { return experiment_blurFFT(s, d, k, false); },
            high_freq, kernel_size, num_runs
        );
        double time_fft_lf = timeBlurFunction(
            [](cv::Mat& s, cv::Mat& d, int k) { return experiment_blurFFT(s, d, k, false); },
            low_freq, kernel_size, num_runs
        );
        std::cout << " Done. (High: " << time_fft_hf << "ms, Low: " << time_fft_lf << "ms)" << std::endl;
        results.push_back({ "FFT", kernel_size, time_fft_hf, time_fft_lf });

        // Benchmark Method 4: OpenCV optimized implementation (baseline)
        std::cout << "  Testing OpenCV..." << std::flush;
        double time_cv_hf = timeBlurFunction(experiment_blurOpenCV, high_freq, kernel_size, num_runs);
        double time_cv_lf = timeBlurFunction(experiment_blurOpenCV, low_freq, kernel_size, num_runs);
        std::cout << " Done. (High: " << time_cv_hf << "ms, Low: " << time_cv_lf << "ms)" << std::endl;
        results.push_back({ "OpenCV", kernel_size, time_cv_hf, time_cv_lf });

        std::cout << std::endl;
    }

    // Display results summary for high frequency image
    std::cout << "===== RESULTS SUMMARY =====" << std::endl;
    std::cout << std::endl;
    std::cout << "High Frequency Image:" << std::endl;
    std::cout << "Kernel | Naive (ms) | Separable (ms) | FFT (ms) | OpenCV (ms)" << std::endl;
    std::cout << "-------|------------|----------------|----------|------------" << std::endl;

    for (int kernel_size : kernel_sizes) {
        std::cout << kernel_size << "x" << kernel_size << "   | ";
        for (const auto& r : results) {
            if (r.kernel_size == kernel_size) {
                printf("%10.2f | ", r.time_high_freq);
            }
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // Display results summary for low frequency image
    std::cout << "Low Frequency Image:" << std::endl;
    std::cout << "Kernel | Naive (ms) | Separable (ms) | FFT (ms) | OpenCV (ms)" << std::endl;
    std::cout << "-------|------------|----------------|----------|------------" << std::endl;

    for (int kernel_size : kernel_sizes) {
        std::cout << kernel_size << "x" << kernel_size << "   | ";
        for (const auto& r : results) {
            if (r.kernel_size == kernel_size) {
                printf("%10.2f | ", r.time_low_freq);
            }
        }
        std::cout << std::endl;
    }

    // Export results to CSV file for further analysis
    std::ofstream csv_file("benchmark_results.csv");
    csv_file << "Method,Kernel_Size,High_Freq_ms,Low_Freq_ms\n";
    for (const auto& r : results) {
        csv_file << r.method << "," << r.kernel_size << ","
            << r.time_high_freq << "," << r.time_low_freq << "\n";
    }
    csv_file.close();

    std::cout << std::endl;
    std::cout << "Results saved to 'benchmark_results.csv'" << std::endl;

    return 0;
}