#pragma once

#include <QObject>
#include <opencv2/core.hpp>

class LampImageAnalyser : public QObject
{
    Q_OBJECT
public:
    LampImageAnalyser(){};
    
    ~LampImageAnalyser(){};

    void RunEdgeDetection(size_t height, size_t width, uint8_t* buffer, size_t stride);
signals:
    void ImageProcessed(const cv::Mat &processedImage);

};