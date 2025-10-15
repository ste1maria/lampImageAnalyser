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
    void ProcessElectrodes(QImage& inputImage);
    void RunFullAnalysis(QImage& imgToProcess);
    
    QPair<float, float> CalculateDistanceAngle(cv::Point start, cv::Point end);
    QPair<float, float> CalculateDistanceAngle(QPointF start, QPointF end);

signals:
    void ImageProcessed(std::shared_ptr<cv::Mat> img);
    void ShowVerticeOnImage(std::shared_ptr<cv::Mat> img);
    //void RadiiCalculated(QString radii);
    //void WallsCalculated(size_t leftWall, size_t rightWall);
    void ElectrodesAnalysisFailed();
    void WallsAnalysisFailed();

    void ElectrodesProcessed
    (
        QPoint coordFirst, 
        QPoint coordSecond, 
        double distance, 
        double angle
    );
    void LampWallsProcessed
    (
        std::vector<QPoint> centers, 
        std::vector<float> radii, 
        QPair<double, double> thickness
    );

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
        std::vector<cv::Point> wallsApexes;
        QPair<double, double>  wallsThickness;
        std::vector<float> wallsRadii;
        std::vector<QPoint> wallsCenters;
        QPair<cv::Point, cv::Point> electrodesApexes;
        const cv::Point2f undeterminedPoint = cv::Point2f(0,0);

        cv::Point2f lampCenter = undeterminedPoint;
        double electrodesDistancePx = 0;
        double electrodesAngle = 0.0f;
        double pixelMmRatio = 0;
    };

    const double standardLampWidth_mm = 7.00;

    std::vector<std::vector<cv::Point>> contours;
    Lamp lamp;
    cv::Mat edgeDetectedImage;
    std::shared_ptr<cv::Mat> originalImageCv;
    std::shared_ptr<cv::Mat> currentImageCv;

    static void thresh_callback(int, void* );
    void CalculateRadii(std::vector<float>* radii, std::vector<QPoint>* centers);
    void PaintCircleCenter(cv::Point point, int curveNumber, cv::Scalar col);
    bool CalculateElectrodes(QImage& inputImage);
    void SimplifyLampContours();
    void FindPixelMmRatio(int lampWidthPx);
    QPair<double, double> CalculateWallThickness();
};