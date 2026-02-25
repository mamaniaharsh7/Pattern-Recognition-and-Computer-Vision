/*
 * Harsh Vijay Mamania
 * February 1, 2026
 * CS 5330 Project 1
 *
 * Purpose: Read and display an image file with basic color channel manipulation
 */

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"

int main(int argc, char* argv[]) {
    cv::Mat src;
    char filename[256];

    // Validate command line arguments
    if (argc < 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        exit(-1);
    }

    strcpy(filename, argv[1]);

    // Load image (BGR format, 8-bit)
    src = cv::imread(filename);
    if (src.data == NULL) {
        printf("Unable to load image %s\n", filename);
        exit(-2);
    }

    // Display image properties
    printf("Size: %d rows x %d cols\n", src.rows, src.cols);
    printf("Channels: %d\n", src.channels());
    printf("Bytes per pixel: %d\n", (int)src.elemSize());

    // Display original image at half size
    cv::Mat display_halfsize_src;
    cv::resize(src, display_halfsize_src, cv::Size(), 0.5, 0.5);
    cv::imshow(filename, display_halfsize_src);

    // Create modified image with swapped red and blue channels
    cv::Mat mod;
    src.copyTo(mod);

    for (int i = 0; i < mod.rows; i++) {
        cv::Vec3b* ptr = mod.ptr<cv::Vec3b>(i);
        for (int j = 0; j < mod.cols; j++) {
            uchar temp = ptr[j][0];
            ptr[j][0] = ptr[j][2];
            ptr[j][2] = temp;
        }
    }

    // Display modified image
    cv::Mat display_halfsize_mod;
    cv::resize(mod, display_halfsize_mod, cv::Size(), 0.5, 0.5);
    cv::imshow("swapped", display_halfsize_mod);

    // Wait for quit command
    printf("Press 'q' or ENTER to quit\n");
    while (true) {
        char key_pressed = cv::waitKey(0);
        if (key_pressed == 'q' || key_pressed == 13) {
            cv::destroyAllWindows();
            break;
        }
        printf("Invalid key. Press 'q' or ENTER to quit\n");
    }

    return 0;
}