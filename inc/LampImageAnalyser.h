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
    void ElectrodesProcessed(QPoint coordFirst, QPoint coordSecond, size_t distance);

private:

    class Lamp
    {
    public:
        Lamp(){};
        ~Lamp(){};
        void ResetLampModel()
        {
            lampContours.clear();
            electrodes.clear();
            electrodesApexes = QPair<cv::Point, cv::Point>(undeterminedPoint, undeterminedPoint);
            lampCenter = undeterminedPoint;
        }

    public:
        std::vector<std::vector<cv::Point>> lampContours;
        std::vector<std::vector<cv::Point>> electrodes;
        std::vector<cv::Point> lampWallsApexes;
        QPair<cv::Point, cv::Point> electrodesApexes;
        const cv::Point2f undeterminedPoint= cv::Point2f(0,0);
        cv::Point2f lampCenter = undeterminedPoint;
        size_t electrodesDistance = 0;
        double pixelMmRatio = 0;
    };

    const double standardLampWidth_mm = 7.00;

    std::vector<std::vector<cv::Point>> contours;
    Lamp lamp;
    cv::Mat edgeDetectedImage;
    std::shared_ptr<cv::Mat> originalImageCv;
    std::shared_ptr<cv::Mat> currentImageCv;

    static void thresh_callback(int, void* );
    void CalculateRadii();
    void PaintCircleCenter(cv::Point point, int curveNumber, cv::Scalar col);
    bool CalculateElectrodes(QImage& inputImage);
    void SimplifyLampContours();
    void FindPixelMmRatio(int lampWidthPx);
};