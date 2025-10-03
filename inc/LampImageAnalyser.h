#pragma once

#include <QObject>
#include <opencv2/core.hpp>

class LampImageAnalyser : public QObject
{
    Q_OBJECT
public:
    LampImageAnalyser(){};
    
    ~LampImageAnalyser(){};

    void RunEdgeDetection(QImage& inputImage);
    void FindContours(cv::Mat inputImage);
    void DrawVertices(QImage& inputImage);

signals:
    void ImageProcessed(const cv::Mat &processedImage);

private:
    cv::Mat edgeDetectedImage;

    static void thresh_callback(int, void* );
};