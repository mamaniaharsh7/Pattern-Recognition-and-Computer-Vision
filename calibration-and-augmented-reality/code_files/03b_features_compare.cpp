// features_compare.cpp
// Harsh Vijay Mamania
// 22 March 2026
// Extension: side-by-side comparison of Harris corner detection
// and ORB feature detection on a live webcam feed.
// Shows Harris on the left window, ORB on the right window,
// and ORB feature matching between consecutive frames.
// Press '+'/'-' to adjust Harris threshold.
// Press 'u'/'d' to adjust ORB feature count.
// Press 's' to save screenshots of all windows.
// Press 'q' to quit.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>

int main() {
    // Open the default webcam
    cv::VideoCapture capdev(0, cv::CAP_DSHOW);
    if (!capdev.isOpened()) {
        printf("Error: unable to open video device\n");
        return -1;
    }

    // --- Harris parameters ---
    int    blockSize = 2;
    int    apertureSize = 3;
    double k = 0.04;
    double threshold = 0.5;

    // --- ORB parameters ---
    int nfeatures = 100;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures);

    // --- Matcher for descriptor comparison ---
    // NORM_HAMMING is the correct distance metric for ORB binary descriptors
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    printf("Controls:\n");
    printf("  +/- : increase/decrease Harris threshold\n");
    printf("  u/d : increase/decrease ORB feature count\n");
    printf("  s   : save screenshots of all windows\n");
    printf("  q   : quit\n\n");

    cv::Mat frame, gray;
    cv::Mat harris_response, harris_response_norm;
    cv::Mat harris_frame, orb_frame;

    // Previous frame data for matching
    cv::Mat prev_gray, prev_frame;
    cv::Mat prev_descriptors;
    std::vector<cv::KeyPoint> prev_keypoints;
    bool has_prev = false;

    while (true) {
        capdev >> frame;
        if (frame.empty()) {
            printf("Error: blank frame grabbed\n");
            break;
        }

        // Handle keypresses first
        int key = cv::waitKey(10);
        if (key == 'q' || key == 'Q') {
            printf("Quitting.\n");
            break;
        }
        if (key == '+' || key == '=') {
            threshold = std::min(threshold + 0.05, 0.99);
            printf("Harris threshold: %.2f\n", threshold);
        }
        if (key == '-' || key == '_') {
            threshold = std::max(threshold - 0.05, 0.01);
            printf("Harris threshold: %.2f\n", threshold);
        }
        if (key == 'u' || key == 'U') {
            nfeatures += 50;
            orb = cv::ORB::create(nfeatures);
            printf("ORB features: %d\n", nfeatures);
        }
        if (key == 'd' || key == 'D') {
            nfeatures = std::max(50, nfeatures - 50);
            orb = cv::ORB::create(nfeatures);
            printf("ORB features: %d\n", nfeatures);
        }

        // Convert to grayscale once
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // -------------------------------------------------------
        // LEFT WINDOW: Harris corner detection
        // -------------------------------------------------------
        harris_frame = frame.clone();

        harris_response = cv::Mat::zeros(gray.size(), CV_32FC1);
        cv::cornerHarris(gray, harris_response,
            blockSize, apertureSize, k);

        // Temporal smoothing
        static cv::Mat accumulated;
        if (accumulated.empty()) {
            accumulated = harris_response.clone();
        }
        else {
            cv::addWeighted(accumulated, 0.8,
                harris_response, 0.2,
                0, accumulated);
        }

        cv::normalize(accumulated, harris_response_norm,
            0, 255, cv::NORM_MINMAX, CV_32FC1);

        double global_max;
        cv::minMaxLoc(harris_response_norm, nullptr, &global_max);

        int harris_count = 0;
        for (int r = 0; r < harris_response_norm.rows; r++) {
            for (int c = 0; c < harris_response_norm.cols; c++) {
                float val = harris_response_norm.at<float>(r, c);
                if (val > threshold * global_max) {
                    cv::Mat neighborhood = harris_response_norm(
                        cv::Range(std::max(0, r - 10),
                            std::min(harris_response_norm.rows, r + 10)),
                        cv::Range(std::max(0, c - 10),
                            std::min(harris_response_norm.cols, c + 10))
                    );
                    double local_max;
                    cv::minMaxLoc(neighborhood, nullptr, &local_max);
                    if (val == local_max) {
                        cv::circle(harris_frame,
                            cv::Point(c, r),
                            4, cv::Scalar(0, 0, 255), 2);
                        harris_count++;
                    }
                }
            }
        }

        char harris_info[100];
        sprintf(harris_info, "Harris | Threshold: %.2f | Corners: %d",
            threshold, harris_count);
        cv::putText(harris_frame, harris_info,
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6, cv::Scalar(0, 255, 0), 2);

        // -------------------------------------------------------
        // MIDDLE WINDOW: ORB feature detection (current frame)
        // -------------------------------------------------------
        orb_frame = frame.clone();

        // detectAndCompute gets keypoints AND descriptors in one call
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(gray, cv::noArray(),
            keypoints, descriptors);

        cv::drawKeypoints(orb_frame, keypoints, orb_frame,
            cv::Scalar(0, 255, 0),
            cv::DrawMatchesFlags::DEFAULT);

        char orb_info[100];
        sprintf(orb_info, "ORB | Max features: %d | Detected: %zu",
            nfeatures, keypoints.size());
        cv::putText(orb_frame, orb_info,
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6, cv::Scalar(0, 255, 0), 2);

        // -------------------------------------------------------
        // RIGHT WINDOW: ORB matching between consecutive frames
        // -------------------------------------------------------
        cv::Mat match_frame;

        if (has_prev && !descriptors.empty() &&
            !prev_descriptors.empty()) {

            // Match current descriptors to previous frame's descriptors
            std::vector<cv::DMatch> matches;
            matcher.match(descriptors, prev_descriptors, matches);

            // Sort matches by distance — lower = better match
            std::sort(matches.begin(), matches.end(),
                [](const cv::DMatch& a, const cv::DMatch& b) {
                    return a.distance < b.distance;
                });

            // Keep only the top 30 best matches for clean visualization
            /*int num_good = std::min((int)matches.size(), 30);
            std::vector<cv::DMatch> good_matches(
                matches.begin(),
                matches.begin() + num_good
            );*/
            // Find the best (minimum) match distance
            double min_dist = matches[0].distance;
            for (const auto& m : matches) {
                if (m.distance < min_dist) min_dist = m.distance;
            }

            // Keep only matches within 3x the best match distance
            // This filters out genuinely bad matches rather than
            // always keeping a fixed count
            std::vector<cv::DMatch> good_matches;
            for (const auto& m : matches) {
                if (m.distance <= 3.0 * min_dist) {
                    good_matches.push_back(m);
                }
            }

            // Draw matches — lines connect same feature across frames
            cv::drawMatches(frame, keypoints,
                prev_frame, prev_keypoints,
                good_matches, match_frame,
                cv::Scalar(0, 255, 0),  // match line color
                cv::Scalar(0, 0, 255),  // single point color
                std::vector<char>(),
                cv::DrawMatchesFlags::DEFAULT);

            char match_info[100];
            /*sprintf(match_info,
                "ORB Matching | Good matches: %d / %zu",
                num_good, matches.size());*/
            sprintf(match_info,
                "ORB Matching | Good matches: %zu / %zu",
                good_matches.size(), matches.size());
            cv::putText(match_frame, match_info,
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 255, 0), 2);

        }
        else {
            // First frame — no previous to match against yet
            match_frame = frame.clone();
            cv::putText(match_frame,
                "ORB Matching | Waiting for second frame...",
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 255, 255), 2);
        }

        // Save current frame as previous for next iteration
        prev_gray = gray.clone();
        prev_frame = frame.clone();
        prev_keypoints = keypoints;
        prev_descriptors = descriptors.clone();
        has_prev = true;

        // Screenshot all three windows
        if (key == 's' || key == 'S') {
            time_t now = time(0);
            char harris_fn[100], orb_fn[100], match_fn[100];
            sprintf(harris_fn, "harris_%ld.png", now);
            sprintf(orb_fn, "orb_%ld.png", now);
            sprintf(match_fn, "matches_%ld.png", now);

            cv::imwrite(harris_fn, harris_frame);
            cv::imwrite(orb_fn, orb_frame);
            cv::imwrite(match_fn, match_frame);

            printf("Saved: %s, %s, %s\n",
                harris_fn, orb_fn, match_fn);
        }

        // Show all three windows
        cv::imshow("Harris Corner Detection", harris_frame);
        cv::imshow("ORB Feature Detection", orb_frame);
        cv::imshow("ORB Feature Matching", match_frame);
    }

    capdev.release();
    cv::destroyAllWindows();
    return 0;
}