// ar.cpp
// Harsh Vijay Mamania
// 22 March 2026
// Implementation of pose estimation and AR rendering functions.

#include "ar.h"

// Loads camera intrinsics from a yml file written by the calibration program.
// filename      : path to the .yml file
// camera_matrix : output 3x3 intrinsic matrix
// dist_coeffs   : output distortion coefficients
// Returns true if file was loaded successfully, false otherwise.
bool loadIntrinsics(const std::string& filename,
    cv::Mat& camera_matrix,
    cv::Mat& dist_coeffs) {

    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        printf("Error: could not open intrinsics file: %s\n",
            filename.c_str());
        return false;
    }

    fs["camera_matrix"] >> camera_matrix;
    fs["dist_coeffs"] >> dist_coeffs;
    fs.release();

    printf("Intrinsics loaded from %s\n", filename.c_str());
    return true;
}

// Detects chessboard corners in a frame and refines to sub-pixel accuracy.
// frame   : input color image
// corners : output vector of detected corner pixel locations
// Returns true if all corners were found, false otherwise.
bool detectCornersAR(const cv::Mat& frame,
    std::vector<cv::Point2f>& corners) {

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    bool found = cv::findChessboardCorners(
        gray,
        AR_BOARD_SIZE,
        corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
    );

    if (found) {
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

// Builds the fixed 3D world coordinates for one full chessboard.
// point_set : output vector of (x, y, 0) world coords per corner
void buildWorldPointsAR(std::vector<cv::Vec3f>& point_set) {
    point_set.clear();
    for (int row = 0; row < AR_BOARD_HEIGHT; row++) {
        for (int col = 0; col < AR_BOARD_WIDTH; col++) {
            point_set.push_back(cv::Vec3f(
                col * AR_SQUARE_SIZE,
                -row * AR_SQUARE_SIZE,
                0.0f
            ));
        }
    }
}

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
    cv::Mat& tvec) {

    // Convert Vec3f to Point3f — solvePnP expects Point3f
    std::vector<cv::Point3f> object_points;
    for (const auto& p : point_set) {
        object_points.push_back(cv::Point3f(p[0], p[1], p[2]));
    }

    cv::solvePnP(object_points, corners,
        camera_matrix, dist_coeffs,
        rvec, tvec);
}

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
    const cv::Mat& dist_coeffs) {

    // Define 4 points in 3D world space:
    // origin and one point along each axis (3 units long for visibility)
    std::vector<cv::Point3f> axis_points = {
        cv::Point3f(0,  0,  0),   // origin
        cv::Point3f(3,  0,  0),   // X axis tip
        cv::Point3f(0, -3,  0),   // Y axis tip (negative = down in board space)
        cv::Point3f(0,  0, -3)    // Z axis tip (negative = toward viewer)
    };

    // Project all 4 points onto the image plane
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(axis_points, rvec, tvec,
        camera_matrix, dist_coeffs,
        image_points);

    // image_points[0] = projected origin
    // image_points[1] = projected X tip
    // image_points[2] = projected Y tip
    // image_points[3] = projected Z tip

    // Convert to integer pixel coordinates for drawing
    cv::Point origin = image_points[0];
    cv::Point x_tip = image_points[1];
    cv::Point y_tip = image_points[2];
    cv::Point z_tip = image_points[3];

    // Draw the three axes as colored lines
    // thickness = 3 for visibility
    cv::line(frame, origin, x_tip, cv::Scalar(0, 0, 255), 3);   // X = red
    cv::line(frame, origin, y_tip, cv::Scalar(0, 255, 0), 3);   // Y = green
    cv::line(frame, origin, z_tip, cv::Scalar(255, 0, 0), 3);   // Z = blue

    // Label each axis tip
    cv::putText(frame, "X", x_tip, cv::FONT_HERSHEY_SIMPLEX,
        0.6, cv::Scalar(0, 0, 255), 2);
    cv::putText(frame, "Y", y_tip, cv::FONT_HERSHEY_SIMPLEX,
        0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, "Z", z_tip, cv::FONT_HERSHEY_SIMPLEX,
        0.6, cv::Scalar(255, 0, 0), 2);
}

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
    int color_scheme) {

    float cx = 4.0f;
    float cy = -2.5f;
    float r = 0.8f;

    std::vector<cv::Point3f> pts;

    // Body rings — indices 0-7 (bottom), 8-15 (top)
    for (int ring = 0; ring < 2; ring++) {
        float z = (ring == 0) ? 1.0f : 4.0f;
        for (int i = 0; i < 8; i++) {
            float angle = i * CV_PI / 4.0f;
            pts.push_back(cv::Point3f(
                cx + r * cos(angle),
                cy + r * sin(angle),
                z
            ));
        }
    }

    // Nose cone tip (index 16)
    pts.push_back(cv::Point3f(cx, cy, 6.5f));

    // Fin 1 (indices 17-18)
    pts.push_back(cv::Point3f(cx, cy - 1.6f, 1.0f));
    pts.push_back(cv::Point3f(cx, cy - 1.6f, 2.5f));

    // Fin 2 (indices 19-20)
    pts.push_back(cv::Point3f(cx + 1.6f, cy, 1.0f));
    pts.push_back(cv::Point3f(cx + 1.6f, cy, 2.5f));

    // Fin 3 (indices 21-22)
    pts.push_back(cv::Point3f(cx - 1.6f, cy, 1.0f));
    pts.push_back(cv::Point3f(cx - 1.6f, cy, 2.5f));

    // Exhaust flame tip (index 23)
    pts.push_back(cv::Point3f(cx, cy, -0.5f));

    // --- Project all points at once ---
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(pts, rvec, tvec,
        camera_matrix, dist_coeffs, img_pts);

    // Convert to integer pixel coords
    std::vector<cv::Point> p;
    for (const auto& pt : img_pts) {
        p.push_back(cv::Point((int)pt.x, (int)pt.y));
    }

    // --- Select colors based on color_scheme ---
    cv::Scalar body_color, nose_color, fin_color, flame_color;

    if (color_scheme == 0) {
        // Original: white body, red nose, blue fins, yellow flame
        body_color = cv::Scalar(220, 220, 220);
        nose_color = cv::Scalar(0, 0, 220);
        fin_color = cv::Scalar(220, 100, 0);
        flame_color = cv::Scalar(0, 200, 255);
    }
    else {
        // Alternate: white body, yellow nose, orange fins, green flame
        body_color = cv::Scalar(220, 220, 220);
        nose_color = cv::Scalar(0, 220, 220);
        fin_color = cv::Scalar(0, 165, 255);
        flame_color = cv::Scalar(0, 255, 0);
    }

    // --- Draw body rings ---
    for (int i = 0; i < 8; i++) {
        cv::line(frame, p[i], p[(i + 1) % 8], body_color, 2);
    }
    for (int i = 0; i < 8; i++) {
        cv::line(frame, p[8 + i], p[8 + (i + 1) % 8], body_color, 2);
    }
    for (int i = 0; i < 8; i++) {
        cv::line(frame, p[i], p[8 + i], body_color, 2);
    }

    // --- Draw nose cone ---
    for (int i = 0; i < 8; i++) {
        cv::line(frame, p[8 + i], p[16], nose_color, 2);
    }

    // --- Draw fins ---
    // Fin 1
    cv::line(frame, p[3], p[17], fin_color, 2);
    cv::line(frame, p[17], p[18], fin_color, 2);
    cv::line(frame, p[18], p[11], fin_color, 2);

    // Fin 2
    cv::line(frame, p[2], p[19], fin_color, 2);
    cv::line(frame, p[19], p[20], fin_color, 2);
    cv::line(frame, p[20], p[10], fin_color, 2);

    // Fin 3
    cv::line(frame, p[4], p[21], fin_color, 2);
    cv::line(frame, p[21], p[22], fin_color, 2);
    cv::line(frame, p[22], p[12], fin_color, 2);

    // --- Draw exhaust flame ---
    for (int i = 0; i < 8; i++) {
        cv::line(frame, p[i], p[23], flame_color, 1);
    }
}