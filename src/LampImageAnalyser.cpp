#include "LampImageAnalyser.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/fast_math.hpp>
#include <QDebug>
#include <QFileDialog>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/ximgproc.hpp>

cv::Mat imgBlur;
int thresh = 5;
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

LampImageAnalyser::LampImageAnalyser()
{
    this->currentImageCv = std::make_shared<cv::Mat>();
};

void LampImageAnalyser::SetCurrentImage(const QImage& image)
{
    uint8_t * buffer = (uint8_t*)image.bits();
    size_t width = image.width();
    size_t height = image.height();
    size_t stride = image.bytesPerLine();

    *this->currentImageCv = cv::Mat(height, width, CV_8U, buffer, stride);
}

cv::Mat pruneSpurs(const cv::Mat& thin255, int iterations = 6) {
    CV_Assert(thin255.type() == CV_8U);
    cv::Mat skel; cv::threshold(thin255, skel, 0, 1, cv::THRESH_BINARY); // 0/1

    // 8-neighborhood count (ignore center)
    cv::Mat K = cv::Mat::ones(3,3,CV_8U); K.at<uchar>(1,1) = 0;

    for (int i = 0; i < iterations; ++i) {
        cv::Mat nbrs;
        cv::filter2D(skel, nbrs, CV_8U, K, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
        cv::Mat endpoints = (skel == 1) & (nbrs == 1);  // degree==1
        if (cv::countNonZero(endpoints) == 0) break;
        skel.setTo(0, endpoints);
    }
    skel *= 255;
    return skel;
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
        cv::Mat imgContrasted, imgBlur, gray;

        cv::convertScaleAbs(img, imgContrasted, 1.2);
        cv::blur(imgContrasted, imgBlur, cv::Size(10,10));

        // SOBEL
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
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
        //cv::imshow("SOBEL", edges);

        cv::Mat labels, stats, centroids;
        int n = cv::connectedComponentsWithStats(edges, labels, stats, centroids, 8, CV_32S);
        int min_area = 50; // tune
        for (int i = 1; i < n; ++i) {
            if (stats.at<int>(i, cv::CC_STAT_AREA) < min_area)
            {
                edges.setTo(0, labels == i);
            }
        }
        //cv::imshow("SOBEL shaved", edges);

        // THINNING
        cv::Mat thinned_edges;
        cv::ximgproc::thinning(edges, thinned_edges, cv::ximgproc::THINNING_ZHANGSUEN);
        //cv::imshow("thinned", thinned_edges);

        cv::Mat contouredImg = pruneSpurs(thinned_edges, 10);

        // TRACING
        cv::Mat toTrace = contouredImg.clone();
        this->contours.clear();
        cv::findContours(toTrace, this->contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        std::sort(this->contours.begin(), this->contours.end(),
              [](auto &a, auto &b){ return a.size() > b.size(); });
        

        cv::Mat coloredEdges; 
        cv::cvtColor(img, coloredEdges, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < this->contours.size(); ++i) 
        {
            cv::Scalar col(0, 255*(i<3), 255*(i==0)); // highlight top curves
            cv::polylines(coloredEdges, this->contours[i], false, col, 1, cv::LINE_AA);
        }
        //cv::imshow("Curves", coloredEdges);
        
        this->edgeDetectedImage = contouredImg;
        *this->currentImageCv = coloredEdges;
        emit ImageProcessed(this->currentImageCv);
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

void LampImageAnalyser::PaintVertice(const QPointF& imgPt)
{
    cv::circle(*this->currentImageCv, cv::Point(imgPt.x(), imgPt.y()), 3, cv::Scalar(255,255,255), cv::FILLED);
    cv::String coordinates = cv::String(std::format("({},{})", imgPt.x(), imgPt.y()));
    cv::putText(*this->currentImageCv, 
                coordinates, 
                cv::Point(imgPt.x() + 5, imgPt.y() + 5),
                cv::FONT_HERSHEY_PLAIN,
                1,
                cv::Scalar(255, 0, 0)
    );
    emit ShowVerticeOnImage(this->currentImageCv);
}

void LampImageAnalyser::TraceCurves()
{

}
