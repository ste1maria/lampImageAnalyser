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

    void SetOriginalImage(const QImage& image);
    void SetCurrentImage(const QImage& image);
    void ResetImage();

    void ProcessLampWalls(QImage& inputImage);
    void PaintVertice(const QPointF& imgPt);
    void PaintVertice(const cv::Point& imgPt);
    void GetCurvatures(std::vector<float> * curvatures);
    void ProcessElectrodes(QImage& inputImage);
    
    QPair<float, float> CalculateDistanceAngle(cv::Point start, cv::Point end);
    QPair<float, float> CalculateDistanceAngle(QPointF start, QPointF end);

signals:
    void ImageProcessed(std::shared_ptr<cv::Mat> img);
    void ShowVerticeOnImage(std::shared_ptr<cv::Mat> img);
    void RadiiCalculated(QString radii);
    void ElectrodesAnalysisFailed();
    void WallsAnalysisFailed();

private:

    class Lamp
    {
    public:
        Lamp(){};
        ~Lamp(){};

    public:
        std::vector<std::vector<cv::Point>> contours;
        std::vector<std::vector<cv::Point>> electrodes;
        QPair<cv::Point, cv::Point> electrodesApexes;
        cv::Point2f lampCenter;
    };

    Lamp lamp;
    cv::Mat edgeDetectedImage;
    std::shared_ptr<cv::Mat> originalImageCv;
    std::shared_ptr<cv::Mat> currentImageCv;

    static void thresh_callback(int, void* );
    void CalculateRadii();
    void PaintCircleCenter(cv::Point point, int curveNumber, cv::Scalar col);
};