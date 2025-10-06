#pragma once

#include <QObject>
#include <opencv2/core.hpp>
#include <memory>

class LampImageAnalyser : public QObject
{
    Q_OBJECT
public:
    LampImageAnalyser();
    
    ~LampImageAnalyser(){};

    void RunEdgeDetection(QImage& inputImage);
    void FindContours(cv::Mat inputImage);
    void DrawVertices(QImage& inputImage);

    void PaintVertice(const QPointF& imgPt);

    void SetCurrentImage(const QImage& image);

signals:
    void ImageProcessed(const cv::Mat &processedImage);
    void ShowVerticeOnImage(std::shared_ptr<cv::Mat> img);

private:
    cv::Mat edgeDetectedImage;
    std::shared_ptr<cv::Mat> currentImageCv;

    static void thresh_callback(int, void* );
};