/*
 * Harsh Vijay Mamania
 * February 1, 2026
 * CS 5330 Project 1
 *
 * Purpose: Real-time video processing interface with multiple filters and effects
 */

#include <cstdio>
#include <cstring>
#include <string>
#include "opencv2/opencv.hpp"
#include "filter.h"
#include "faceDetect.h"
#include "DA2Network.hpp"

int main(int argc, char* argv[]) {
    cv::VideoCapture* capdev;

    // Open video device (webcam)
    capdev = new cv::VideoCapture(0, cv::CAP_DSHOW);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    // Get video properties
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1);

    // Initialize depth estimation network
    DA2Network da_net("model_fp16.onnx");
    float scale_factor = 256.0 / 480.0;

    // State variables
    cv::Mat frame;
    int screenshot_counter = 1;
    std::string filter_mode = "color";
    cv::Rect last(0, 0, 0, 0);
    int brightness_offset = 0;
    float contrast_factor = 1.0;
    int sunglasses_mode = 0;
    std::vector<Sparkle> sparkles;

    // Main video processing loop
    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("Frame is empty\n");
            break;
        }

        // Apply selected filter mode
        if (filter_mode == "grayscale") {
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        }

        if (filter_mode == "my_new_grayscale") {
            my_new_grayscale(frame, frame);
        }

        if (filter_mode == "sepia") {
            my_sepia_filter(frame, frame);
        }

        if (filter_mode == "naive_blur") {
            blur5x5_1(frame, frame);
        }

        if (filter_mode == "fast_blur") {
            blur5x5_2(frame, frame);
        }

        if (filter_mode == "sobel_x") {
            cv::Mat sobel_result_16b;
            sobelX3x3(frame, sobel_result_16b);
            cv::convertScaleAbs(sobel_result_16b, frame);
        }

        if (filter_mode == "sobel_y") {
            cv::Mat sobel_result_16b;
            sobelY3x3(frame, sobel_result_16b);
            cv::convertScaleAbs(sobel_result_16b, frame);
        }

        if (filter_mode == "magnitude") {
            cv::Mat sobelX_result_16b, sobelY_result_16b;
            sobelX3x3(frame, sobelX_result_16b);
            sobelY3x3(frame, sobelY_result_16b);
            magnitude(sobelX_result_16b, sobelY_result_16b, frame);
        }

        if (filter_mode == "blur_quantize") {
            blurQuantize(frame, frame, 10);
        }

        if (filter_mode == "face_detect") {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> faces;
            detectFaces(gray, faces);
            drawBoxes(frame, faces);

            // Temporal smoothing to reduce jitter
            if (faces.size() > 0) {
                last.x = (faces[0].x + last.x) / 2;
                last.y = (faces[0].y + last.y) / 2;
                last.width = (faces[0].width + last.width) / 2;
                last.height = (faces[0].height + last.height) / 2;
            }
        }

        if (filter_mode == "depth") {
            // Downsample for performance
            cv::Mat small_frame;
            cv::resize(frame, small_frame, cv::Size(), 0.5, 0.5);

            // Run depth estimation
            cv::Mat depth;
            da_net.set_input(small_frame, scale_factor);
            da_net.run_network(depth, small_frame.size());

            // Apply colormap and resize to original dimensions
            cv::Mat depth_colored;
            cv::applyColorMap(depth, depth_colored, cv::COLORMAP_INFERNO);
            cv::resize(depth_colored, frame, frame.size());
        }

        if (filter_mode == "face_distance") {
            // Get depth map
            cv::Mat small_frame;
            cv::resize(frame, small_frame, cv::Size(), 0.5, 0.5);

            cv::Mat depth;
            da_net.set_input(small_frame, scale_factor);
            da_net.run_network(depth, small_frame.size());

            cv::Mat depth_full;
            cv::resize(depth, depth_full, frame.size());

            // Detect faces
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            detectFaces(gray, faces);

            // Display depth at face center
            if (faces.size() > 0) {
                cv::Rect face = faces[0];
                int center_x = face.x + face.width / 2;
                int center_y = face.y + face.height / 2;
                int depth_value = depth_full.at<uchar>(center_y, center_x);

                cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);

                char text[100];
                sprintf(text, "Depth: %d", depth_value);
                cv::putText(frame, text, cv::Point(face.x, face.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
        }

        // Apply brightness adjustment if active
        if (brightness_offset != 0) {
            modify_brightness(frame, frame, brightness_offset);
        }

        // Apply contrast adjustment if active
        if (contrast_factor != 1.0) {
            modify_contrast(frame, frame, contrast_factor);
        }

        if (filter_mode == "emboss") {
            emboss(frame, frame);
        }

        if (filter_mode == "portrait") {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            detectFaces(gray, faces);
            portraitMode(frame, frame, faces);
        }

        if (filter_mode == "sunglasses") {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> faces;
            detectFaces(gray, faces);

            for (size_t i = 0; i < faces.size(); i++) {
                addSunglasses(frame, faces[i], sunglasses_mode);
                updateAndDrawSparkles(frame, faces[i], sparkles);
            }
        }

        // Display processed frame
        cv::imshow("Video", frame);

        // Handle keyboard input
        char key = cv::waitKey(10);

        if (key == 's') {
            char screenshot_name[256];
            sprintf(screenshot_name, "screenshot%d.jpg", screenshot_counter);
            cv::imwrite(screenshot_name, frame);
            screenshot_counter++;
        }

        // Filter selection keys
        if (key == 'g') filter_mode = "grayscale";
        if (key == 'c') filter_mode = "color";
        if (key == 'h') filter_mode = "my_new_grayscale";
        if (key == 'e') filter_mode = "sepia";
        if (key == 'n') filter_mode = "naive_blur";
        if (key == 'b') filter_mode = "fast_blur";
        if (key == 'x') filter_mode = "sobel_x";
        if (key == 'y') filter_mode = "sobel_y";
        if (key == 'm') filter_mode = "magnitude";
        if (key == 'l') filter_mode = "blur_quantize";
        if (key == 'f') filter_mode = "face_detect";
        if (key == 'd') filter_mode = "depth";
        if (key == 'z') filter_mode = "face_distance";
        if (key == 'i') filter_mode = "emboss";
        if (key == 'p') filter_mode = "portrait";

        // Brightness and contrast controls
        if (key == '1') brightness_offset += 5;
        if (key == '2') brightness_offset -= 5;
        if (key == '3') contrast_factor += 0.1;
        if (key == '4') contrast_factor -= 0.1;
        if (key == 'r') {
            brightness_offset = 0;
            contrast_factor = 1.0;
        }

        // Sunglasses mode selection
        if (key == '7') {
            filter_mode = "sunglasses";
            sunglasses_mode = 0;
        }
        if (key == '8') {
            filter_mode = "sunglasses";
            sunglasses_mode = 1;
        }
        if (key == '9') {
            filter_mode = "sunglasses";
            sunglasses_mode = 2;
        }

        // Quit
        if (key == 'q') {
            break;
        }
    }

    delete capdev;
    return 0;
}