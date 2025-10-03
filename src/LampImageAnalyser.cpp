#include "LampImageAnalyser.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/fast_math.hpp>
#include <QDebug>
#include <QFileDialog>

cv::Mat imgBlur;
int thresh = 50;
std::vector<std::vector<cv::Point> > contours;

void onMouse(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN && userdata)
    {
        auto* imgPtr = static_cast<cv::Mat*>(userdata);

        QString filename = QFileDialog::getSaveFileName(
            nullptr,
            "Save Image As",
            "plasma.png",
            "Images (*.png *.jpg *.bmp);;All Files (*)"
        );

        if (!filename.isEmpty()) {
            if (!cv::imwrite(filename.toStdString(), *imgPtr)) {
                qWarning() << "Failed to save image.";
            } else {
                qDebug() << "Image saved to:" << filename;
            }
        }
    }
}

void LampImageAnalyser::RunEdgeDetection(QImage& inputImage)
{
    uint8_t * buffer = (uint8_t*)inputImage.bits();
    size_t width = inputImage.width();
    size_t height = inputImage.height();
    size_t stride = inputImage.bytesPerLine();

    try
    {
        cv::Mat img = cv::Mat(height, width, CV_8U, buffer, stride);
        cv::Mat imgContrasted, imgBlur;

        cv::convertScaleAbs(img, imgContrasted, 1);
        cv::blur(imgContrasted, imgBlur, cv::Size(10,10)); 

        cv::Mat sobelx, sobely, gradient;
 
        // Apply Sobel operator
        cv::Sobel(imgBlur, sobelx, CV_64F, 1, 0, 3);
        cv::Sobel(imgBlur, sobely, CV_64F, 0, 1, 3);
    
        // Compute gradient magnitude
        cv::magnitude(sobelx, sobely, gradient);
    
        // Convert to 8-bit image
        cv::Mat gradient_abs;
        cv::convertScaleAbs(gradient, gradient_abs);
        
        cv::Mat edges;
        cv::threshold(gradient_abs, edges, 10, 255, cv::THRESH_BINARY);

        this->edgeDetectedImage = edges;
        // Display result
        cv::imshow("Sobel Edge Detection", edges);

        //emit ImageProcessed(edgedImage);
    }
    catch(const cv::Exception e)
    {
        qDebug() << e.what() << '\n';
    }    
    catch(const std::exception e)
    {
        qDebug() << e.what() << '\n';
    }
}

void LampImageAnalyser::FindContours(cv::Mat inputImage)
{
    const char* source_window = "Source";
    cv::namedWindow(source_window);
    cv::imshow(source_window, inputImage);

    const int max_thresh = 255;
    cv::createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, &LampImageAnalyser::thresh_callback, this);
}

void LampImageAnalyser::thresh_callback(int, void* userdata)
{
    LampImageAnalyser* analyser = static_cast<LampImageAnalyser*>(userdata);
    cv::RNG rng(12345);
    cv::Mat canny_output;
    cv::Canny(analyser->edgeDetectedImage, canny_output, thresh, thresh*2);
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
    for(size_t i = 0; i< contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
        drawContours(drawing, contours, (int)i, color, 1, cv::LINE_8, hierarchy, 0);
    }
    imshow("Contours", drawing);
}

static double angleCos(const cv::Point2f& p0, const cv::Point2f& p1, const cv::Point2f& p2) {
    // cos of angle at p1 (p0—p1—p2)
    cv::Point2f v1 = p0 - p1;
    cv::Point2f v2 = p2 - p1;
    double denom = std::sqrt(v1.dot(v1)) * std::sqrt(v2.dot(v2)) + 1e-9;
    return (v1.dot(v2)) / denom;
}

std::vector<cv::Point> extractVerticesFromContour(const std::vector<cv::Point>& contour,
                                         double epsilonFrac = 0.01,
                                         double sharpCosThresh = 0.9,
                                         double minVertexSpacing = 5.0)
{
    // 1) simplify the contour
    double eps = epsilonFrac * cv::arcLength(contour, /*closed=*/true);
    std::vector<cv::Point> poly;
    approxPolyDP(contour, poly, eps, /*closed=*/true);

    // 2) (optional) pick only “sharp” turns by angle
    std::vector<cv::Point> vertices;
    const int n = (int)poly.size();
    if (n < 3) return poly; // nothing to filter

    auto tooClose = [&](const cv::Point& a, const cv::Point& b){
        return cv::norm(a - b) < minVertexSpacing;
    };

    for (int i = 0; i < n; ++i) {
        const cv::Point2f& prev = poly[(i - 1 + n) % n];
        const cv::Point2f& curr = poly[i];
        const cv::Point2f& next = poly[(i + 1) % n];

        double c = std::abs(angleCos(prev, curr, next)); // 1 ≈ straight, 0 ≈ 90°
        if (c < sharpCosThresh) { // e.g. < 0.9 keeps noticeably “bendy” points
            if (vertices.empty() || !tooClose(vertices.back(), curr))
                vertices.push_back(curr);
        }
    }
    return vertices;
}

void LampImageAnalyser::DrawVertices(QImage& inputImage)
{
    this->RunEdgeDetection(inputImage);
    this->FindContours(this->edgeDetectedImage);

    uint8_t * buffer = (uint8_t*)inputImage.bits();
    size_t width = inputImage.width();
    size_t height = inputImage.height();
    size_t stride = inputImage.bytesPerLine();

    cv::Mat img = cv::Mat(height, width, CV_8U, buffer, stride);

    for (auto& c : contours) {
        auto verts = extractVerticesFromContour(c,
            /*epsilonFrac=*/0.01,    // increase to simplify more
            /*sharpCosThresh=*/0.97, // lower -> more corners kept, higher -> only sharp
            /*minVertexSpacing=*/6.0 // suppress near-duplicates
        );

        // draw
        for (auto& p : verts) {
           cv::circle(img, p, 3, cv::Scalar(255,255,255), cv::FILLED);
        }
    }
    cv::imshow("Vertices", img);
}
