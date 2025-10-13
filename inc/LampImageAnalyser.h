#pragma once

#include <QObject>
#include <opencv2/core.hpp>
#include <memory>


typedef enum 
{
    APEX_OK,
    APEX_ERR_CONTOUR_TOO_THIN,
    APEX_ERR_APEX_OUTSIDE,
    APEX_ERR_APEX_FAR,
    APEX_ERR_EMPTY_VECTOR,
    APEX_ERR_FAULTY_CONTOUR
} ApexSanityResult;

class LampImageAnalyser : public QObject
{
    Q_OBJECT
public:
    LampImageAnalyser();
    
    ~LampImageAnalyser(){};

    void SetCurrentImage(const QImage& image);

    void ProcessLampWalls(QImage& inputImage);
    void PaintVertice(const QPointF& imgPt);
    void GetCurvatures(std::vector<float> * curvatures);
    void ProcessElectrodes(QImage& inputImage);
    
    QPair<float, float> CalculateDistanceAngle(cv::Point start, cv::Point end);
    QPair<float, float> CalculateDistanceAngle(QPointF start, QPointF end);

signals:
    void ImageProcessed(std::shared_ptr<cv::Mat> img);
    void ShowVerticeOnImage(std::shared_ptr<cv::Mat> img);
    void RadiiCalculated(QString radii);

private:
    cv::Mat edgeDetectedImage;
    std::shared_ptr<cv::Mat> currentImageCv;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> electrodes;
    std::vector<cv::Point> curveCenters;
    std::vector<float> curvatures;

    static void thresh_callback(int, void* );
    void CalculateRadii();
    void PaintCircleCenter(cv::Point point, int curveNumber, cv::Scalar col);
};