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

    void PaintVertice(const QPointF& imgPt);

    void SetCurrentImage(const QImage& image);

signals:
    void ImageProcessed(std::shared_ptr<cv::Mat> img);
    void ShowVerticeOnImage(std::shared_ptr<cv::Mat> img);
    void RadiiCalculated(QString radii);

private:
    cv::Mat edgeDetectedImage;
    std::shared_ptr<cv::Mat> currentImageCv;
    std::vector<std::vector<cv::Point>> contours;

    static void thresh_callback(int, void* );
    void CalculateRadii();
    void PaintCircleCenter(cv::Point point, int curveNumber, cv::Scalar col);
};