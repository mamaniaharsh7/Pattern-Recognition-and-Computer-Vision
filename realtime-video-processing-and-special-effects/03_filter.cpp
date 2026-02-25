/*
 * Harsh Vijay Mamania
 * February 1, 2026
 * CS 5330 Project 1
 *
 * Purpose: Image filter implementations including blur, edge detection, and effects
 */

#include "filter.h"
#include <cmath>

 /*
  * Custom grayscale conversion using luminosity method
  * src: source color image, dst: destination grayscale image
  * Returns 0 on success
  */
int my_new_grayscale(cv::Mat& src, cv::Mat& dst) {
    src.copyTo(dst);

    for (int i = 0; i < dst.rows; i++) {
        cv::Vec3b* ptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            uchar b = ptr[j][0];
            uchar g = ptr[j][1];
            uchar r = ptr[j][2];
            uchar gray = (uchar)(0.114 * r + 0.299 * g + 0.587 * b);
            ptr[j][0] = ptr[j][1] = ptr[j][2] = gray;
        }
    }
    return 0;
}

/*
 * Apply sepia tone filter for vintage appearance
 * src: source image, dst: destination image
 * Returns 0 on success
 */
int my_sepia_filter(cv::Mat& src, cv::Mat& dst) {
    src.copyTo(dst);

    for (int i = 0; i < dst.rows; i++) {
        cv::Vec3b* ptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            uchar og_b = ptr[j][0];
            uchar og_g = ptr[j][1];
            uchar og_r = ptr[j][2];

            ptr[j][0] = std::min(255, (int)(0.272 * og_r + 0.534 * og_g + 0.131 * og_b));
            ptr[j][1] = std::min(255, (int)(0.349 * og_r + 0.686 * og_g + 0.168 * og_b));
            ptr[j][2] = std::min(255, (int)(0.393 * og_r + 0.769 * og_g + 0.189 * og_b));
        }
    }

    return 0;
}

/*
 * Naive 5x5 Gaussian blur using at() accessor
 * src: source image, dst: destination blurred image
 * Returns 0 on success
 */
int blur5x5_1(cv::Mat& src, cv::Mat& dst) {
    src.copyTo(dst);

    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            for (int c = 0; c < 3; c++) {
                int value =
                    (src.at<cv::Vec3b>(i - 2, j - 2)[c] * 1) +
                    (src.at<cv::Vec3b>(i - 2, j - 1)[c] * 2) +
                    (src.at<cv::Vec3b>(i - 2, j)[c] * 4) +
                    (src.at<cv::Vec3b>(i - 2, j + 1)[c] * 2) +
                    (src.at<cv::Vec3b>(i - 2, j + 2)[c] * 1) +
                    (src.at<cv::Vec3b>(i - 1, j - 2)[c] * 2) +
                    (src.at<cv::Vec3b>(i - 1, j - 1)[c] * 4) +
                    (src.at<cv::Vec3b>(i - 1, j)[c] * 8) +
                    (src.at<cv::Vec3b>(i - 1, j + 1)[c] * 4) +
                    (src.at<cv::Vec3b>(i - 1, j + 2)[c] * 2) +
                    (src.at<cv::Vec3b>(i, j - 2)[c] * 4) +
                    (src.at<cv::Vec3b>(i, j - 1)[c] * 8) +
                    (src.at<cv::Vec3b>(i, j)[c] * 16) +
                    (src.at<cv::Vec3b>(i, j + 1)[c] * 8) +
                    (src.at<cv::Vec3b>(i, j + 2)[c] * 4) +
                    (src.at<cv::Vec3b>(i + 1, j - 2)[c] * 2) +
                    (src.at<cv::Vec3b>(i + 1, j - 1)[c] * 4) +
                    (src.at<cv::Vec3b>(i + 1, j)[c] * 8) +
                    (src.at<cv::Vec3b>(i + 1, j + 1)[c] * 4) +
                    (src.at<cv::Vec3b>(i + 1, j + 2)[c] * 2) +
                    (src.at<cv::Vec3b>(i + 2, j - 2)[c] * 1) +
                    (src.at<cv::Vec3b>(i + 2, j - 1)[c] * 2) +
                    (src.at<cv::Vec3b>(i + 2, j)[c] * 4) +
                    (src.at<cv::Vec3b>(i + 2, j + 1)[c] * 2) +
                    (src.at<cv::Vec3b>(i + 2, j + 2)[c] * 1);

                dst.at<cv::Vec3b>(i, j)[c] = value / 100;
            }
        }
    }
    return 0;
}

/*
 * Optimized 5x5 Gaussian blur using separable 1D filters
 * src: source image, dst: destination blurred image
 * Returns 0 on success
 */
int blur5x5_2(cv::Mat& src, cv::Mat& dst) {
    cv::Mat tmp;
    src.copyTo(tmp);

    // Horizontal pass with [1 2 4 2 1] kernel
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        for (int j = 2; j < src.cols - 2; j++) {
            for (int c = 0; c < 3; c++) {
                int value =
                    (sptr[j - 2][c] * 1) +
                    (sptr[j - 1][c] * 2) +
                    (sptr[j][c] * 4) +
                    (sptr[j + 1][c] * 2) +
                    (sptr[j + 2][c] * 1);
                tptr[j][c] = value / 10;
            }
        }
    }

    tmp.copyTo(dst);

    // Vertical pass with [1 2 4 2 1] kernel
    for (int i = 2; i < tmp.rows - 2; i++) {
        cv::Vec3b* tptr_up2 = tmp.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b* tptr_up1 = tmp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        cv::Vec3b* tptr_dwn1 = tmp.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* tptr_dwn2 = tmp.ptr<cv::Vec3b>(i + 2);
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < tmp.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int value =
                    (tptr_up2[j][c] * 1) +
                    (tptr_up1[j][c] * 2) +
                    (tptr[j][c] * 4) +
                    (tptr_dwn1[j][c] * 2) +
                    (tptr_dwn2[j][c] * 1);
                dptr[j][c] = value / 10;
            }
        }
    }

    return 0;
}

/*
 * 3x3 Sobel X filter for detecting vertical edges
 * src: source image, dst: destination gradient image (16SC3 format)
 * Returns 0 on success
 */
int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    cv::Mat tmp;
    src.copyTo(tmp);

    // Vertical smoothing pass [1 2 1]
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b* sptr_up1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* sptr_dwn1 = src.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int value =
                    (sptr_up1[j][c] * 1) +
                    (sptr[j][c] * 2) +
                    (sptr_dwn1[j][c] * 1);
                tptr[j][c] = value / 4;
            }
        }
    }

    dst.create(src.size(), CV_16SC3);

    // Horizontal derivative pass [-1 0 1]
    for (int i = 0; i < tmp.rows; i++) {
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);

        for (int j = 1; j < tmp.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                int value =
                    (tptr[j - 1][c] * -1) +
                    (tptr[j][c] * 0) +
                    (tptr[j + 1][c] * 1);
                dptr[j][c] = value;
            }
        }
    }
    return 0;
}

/*
 * 3x3 Sobel Y filter for detecting horizontal edges
 * src: source image, dst: destination gradient image (16SC3 format)
 * Returns 0 on success
 */
int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    cv::Mat tmp;
    src.copyTo(tmp);

    // Horizontal smoothing pass [1 2 1]
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                int value =
                    (sptr[j - 1][c] * 1) +
                    (sptr[j][c] * 2) +
                    (sptr[j + 1][c] * 1);
                tptr[j][c] = value / 4;
            }
        }
    }

    dst.create(src.size(), CV_16SC3);

    // Vertical derivative pass [1 0 -1]
    for (int i = 1; i < tmp.rows - 1; i++) {
        cv::Vec3b* tptr_up1 = tmp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        cv::Vec3b* tptr_dwn1 = tmp.ptr<cv::Vec3b>(i + 1);
        cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);

        for (int j = 0; j < tmp.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int value =
                    (tptr_up1[j][c] * 1) +
                    (tptr[j][c] * 0) +
                    (tptr_dwn1[j][c] * -1);
                dptr[j][c] = value;
            }
        }
    }
    return 0;
}

/*
 * Compute gradient magnitude from Sobel X and Y components
 * sx: Sobel X gradient (16SC3), sy: Sobel Y gradient (16SC3), dst: magnitude image (8UC3)
 * Returns 0 on success
 */
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i < sx.rows; i++) {
        cv::Vec3s* sxptr = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s* syptr = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int sx_val = sxptr[j][c];
                int sy_val = syptr[j][c];
                int mag = sqrt(sx_val * sx_val + sy_val * sy_val);

                if (mag > 255) mag = 255;
                dptr[j][c] = mag;
            }
        }
    }
    return 0;
}

/*
 * Apply blur and quantize colors to specified number of levels
 * src: source image, dst: destination image, levels: number of color levels per channel
 * Returns 0 on success
 */
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
    cv::Mat blurred_img;
    blur5x5_2(src, blurred_img);
    blurred_img.copyTo(dst);

    int b = 255 / levels;

    for (int i = 0; i < dst.rows; i++) {
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int xt = dptr[j][c] / b;
                int xf = xt * b;
                dptr[j][c] = xf;
            }
        }
    }

    return 0;
}

/*
 * Adjust image brightness by adding offset to all pixels
 * src: source image, dst: destination image, brightness_offset: value to add (-255 to 255)
 * Returns 0 on success
 */
int modify_brightness(cv::Mat& src, cv::Mat& dst, int brightness_offset) {
    src.copyTo(dst);

    for (int i = 0; i < dst.rows; i++) {
        cv::Vec3b* ptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            for (int c = 0; c < 3; c++) {
                ptr[j][c] = cv::saturate_cast<uchar>(ptr[j][c] + brightness_offset);
            }
        }
    }
    return 0;
}

/*
 * Adjust image contrast by scaling around midpoint (128)
 * src: source image, dst: destination image, contrast_factor: scaling factor (0.5 to 3.0 typical)
 * Returns 0 on success
 */
int modify_contrast(cv::Mat& src, cv::Mat& dst, float contrast_factor) {
    src.copyTo(dst);

    for (int i = 0; i < dst.rows; i++) {
        cv::Vec3b* ptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            for (int c = 0; c < 3; c++) {
                float new_val = (ptr[j][c] - 128) * contrast_factor + 128;
                ptr[j][c] = cv::saturate_cast<uchar>(new_val);
            }
        }
    }
    return 0;
}

/*
 * Create embossing effect using directional gradient (45 degree)
 * src: source image, dst: destination embossed image
 * Returns 0 on success
 */
int emboss(cv::Mat& src, cv::Mat& dst) {
    // Compute Sobel gradients
    cv::Mat sobelX, sobelY;
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);

    dst.create(src.size(), CV_8UC3);

    // Apply directional derivative at 45 degrees
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3s* sx_ptr = sobelX.ptr<cv::Vec3s>(i);
        cv::Vec3s* sy_ptr = sobelY.ptr<cv::Vec3s>(i);
        cv::Vec3b* dst_ptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int sx = sx_ptr[j][c];
                int sy = sy_ptr[j][c];
                float emboss_val = sx * 0.7071 + sy * 0.7071 + 128;
                dst_ptr[j][c] = cv::saturate_cast<uchar>(emboss_val);
            }
        }
    }

    return 0;
}

/*
 * Create portrait mode effect by blurring background while keeping faces sharp
 * src: source image, dst: destination image, faces: vector of detected face rectangles
 * Returns 0 on success
 */
int portraitMode(cv::Mat& src, cv::Mat& dst, std::vector<cv::Rect>& faces) {
    cv::Mat original;
    src.copyTo(original);

    // Create heavily blurred background
    cv::Mat blurred;
    src.copyTo(blurred);
    for (int i = 0; i < 5; i++) {
        blur5x5_2(blurred, blurred);
    }

    blurred.copyTo(dst);

    // Restore sharp face regions
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Rect face = faces[i];
        if (face.x >= 0 && face.y >= 0 &&
            face.x + face.width <= original.cols &&
            face.y + face.height <= original.rows) {
            original(face).copyTo(dst(face));
        }
    }

    return 0;
}

/*
 * Render sunglasses overlay on detected face with multiple modes
 * frame: video frame to draw on, face: detected face rectangle, mode: style (0=opaque, 1=adaptive, 2=reflective)
 * Returns 0 on success
 */
int addSunglasses(cv::Mat& frame, cv::Rect face, int mode) {
    // Calculate lens geometry based on face dimensions
    int lens_width = face.width / 3;
    int lens_height = face.height / 6;
    int y_pos = face.y + face.height / 3;  // Eye level position

    // Calculate center positions for left and right lenses
    int left_center_x = face.x + face.width / 4;
    int left_center_y = y_pos + lens_height / 2;
    int right_center_x = face.x + 3 * face.width / 4;
    int right_center_y = y_pos + lens_height / 2;

    // Mode 0: Solid black opaque lenses
    if (mode == 0) {
        cv::ellipse(frame, cv::Point(left_center_x, left_center_y),
            cv::Size(lens_width / 2, lens_height / 2),
            0, 0, 360, cv::Scalar(15, 15, 15), -1);
        cv::ellipse(frame, cv::Point(right_center_x, right_center_y),
            cv::Size(lens_width / 2, lens_height / 2),
            0, 0, 360, cv::Scalar(15, 15, 15), -1);
    }
    // Mode 1: Brightness-adaptive transparency (auto-tinting)
    else if (mode == 1) {
        // Calculate scene brightness
        cv::Scalar mean_brightness = cv::mean(frame);
        float avg = (mean_brightness[0] + mean_brightness[1] + mean_brightness[2]) / 3.0;

        // Map brightness to opacity (brighter scene = more opaque lenses)
        float opacity = 0.1 + (avg / 255.0) * 0.9;

        // Draw lenses on overlay copy
        cv::Mat overlay = frame.clone();
        cv::ellipse(overlay, cv::Point(left_center_x, left_center_y),
            cv::Size(lens_width / 2, lens_height / 2),
            0, 0, 360, cv::Scalar(15, 15, 15), -1);
        cv::ellipse(overlay, cv::Point(right_center_x, right_center_y),
            cv::Size(lens_width / 2, lens_height / 2),
            0, 0, 360, cv::Scalar(15, 15, 15), -1);

        // Blend overlay with original using calculated opacity
        cv::addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame);
    }
    // Mode 2: Scene-reflecting mirror lenses
    else if (mode == 2) {
        // Create blurred and horizontally flipped version for reflection effect
        cv::Mat blurred, flipped;
        cv::blur(frame, blurred, cv::Size(15, 15));
        cv::flip(blurred, flipped, 1);

        // Create elliptical mask for lens regions
        cv::Mat lens_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::ellipse(lens_mask, cv::Point(left_center_x, left_center_y),
            cv::Size(lens_width / 2, lens_height / 2),
            0, 0, 360, cv::Scalar(255), -1);
        cv::ellipse(lens_mask, cv::Point(right_center_x, right_center_y),
            cv::Size(lens_width / 2, lens_height / 2),
            0, 0, 360, cv::Scalar(255), -1);

        // Copy flipped scene to lens regions only
        flipped.copyTo(frame, lens_mask);
    }

    // Draw frames and structural elements (same for all modes)

    // Lens outlines
    cv::ellipse(frame, cv::Point(left_center_x, left_center_y),
        cv::Size(lens_width / 2, lens_height / 2),
        0, 0, 360, cv::Scalar(0, 0, 0), 3);
    cv::ellipse(frame, cv::Point(right_center_x, right_center_y),
        cv::Size(lens_width / 2, lens_height / 2),
        0, 0, 360, cv::Scalar(0, 0, 0), 3);

    // Bridge connecting lenses
    cv::line(frame, cv::Point(left_center_x + lens_width / 2, left_center_y),
        cv::Point(right_center_x - lens_width / 2, right_center_y),
        cv::Scalar(0, 0, 0), 4);

    // Temple arms extending to face edges
    cv::line(frame, cv::Point(left_center_x - lens_width / 2, left_center_y),
        cv::Point(face.x, left_center_y), cv::Scalar(0, 0, 0), 3);
    cv::line(frame, cv::Point(right_center_x + lens_width / 2, right_center_y),
        cv::Point(face.x + face.width, right_center_y), cv::Scalar(0, 0, 0), 3);

    // Reflective highlights on lenses (simulates light reflection)
    cv::ellipse(frame, cv::Point(left_center_x - lens_width / 6, left_center_y - lens_height / 6),
        cv::Size(lens_width / 8, lens_height / 8),
        0, 0, 360, cv::Scalar(200, 200, 200), -1);
    cv::ellipse(frame, cv::Point(right_center_x - lens_width / 6, right_center_y - lens_height / 6),
        cv::Size(lens_width / 8, lens_height / 8),
        0, 0, 360, cv::Scalar(200, 200, 200), -1);

    return 0;
}

/*
 * Update and render animated sparkle particles radiating from face
 * frame: video frame to draw on, face: detected face rectangle, sparkles: particle state vector
 * Returns void
 */
void updateAndDrawSparkles(cv::Mat& frame, cv::Rect face, std::vector<Sparkle>& sparkles) {
    // Initialize particles on first call with random positions on face perimeter
    if (sparkles.empty()) {
        for (int i = 0; i < 40; i++) {
            Sparkle s;
            float angle = (rand() % 360) * 3.14159 / 180.0;
            float face_cx = face.x + face.width / 2.0;
            float face_cy = face.y + face.height / 2.0;
            float radius = face.width / 2.0;

            // Position particle on circle around face using polar coordinates
            s.x = face_cx + radius * cos(angle);
            s.y = face_cy + radius * sin(angle);

            // Set velocity to move radially outward
            s.vx = cos(angle) * 2.0;
            s.vy = sin(angle) * 2.0;

            s.lifetime = rand() % 60 + 30;
            s.visible = (rand() % 2 == 0);
            sparkles.push_back(s);
        }
    }

    // Calculate current face center for distance checks
    float face_cx = face.x + face.width / 2.0;
    float face_cy = face.y + face.height / 2.0;

    // Update each particle's position and state
    for (size_t i = 0; i < sparkles.size(); i++) {
        // Move particle according to velocity
        sparkles[i].x += sparkles[i].vx;
        sparkles[i].y += sparkles[i].vy;
        sparkles[i].lifetime--;

        // Random twinkling effect (10% chance per frame)
        if (rand() % 10 == 0) {
            sparkles[i].visible = !sparkles[i].visible;
        }

        // Check if particle needs respawning
        float dx = sparkles[i].x - face_cx;
        float dy = sparkles[i].y - face_cy;
        float dist = sqrt(dx * dx + dy * dy);

        // Respawn if particle traveled too far or lifetime expired
        if (dist > face.width * 1.5 || sparkles[i].lifetime <= 0) {
            float angle = (rand() % 360) * 3.14159 / 180.0;
            float radius = face.width / 2.0;

            // Reset position to face perimeter
            sparkles[i].x = face_cx + radius * cos(angle);
            sparkles[i].y = face_cy + radius * sin(angle);

            // Reset velocity pointing outward
            sparkles[i].vx = cos(angle) * 2.0;
            sparkles[i].vy = sin(angle) * 2.0;

            sparkles[i].lifetime = rand() % 60 + 30;
            sparkles[i].visible = (rand() % 2 == 0);
        }

        // Render particle as golden circle if visible
        if (sparkles[i].visible) {
            cv::circle(frame, cv::Point((int)sparkles[i].x, (int)sparkles[i].y),
                3, cv::Scalar(0, 215, 255), -1);
        }
    }
}