#pragma once
#include <QObject>
#include "GLVisualisation.h"
#include <memory>

#include <opencv2/core.hpp>

class MainWindow;
class GLVisualisation;
class LampImageAnalyser;

typedef struct 
{
    std::vector<QString> radii;
    std::vector<QString> thickness;
    QString electrodesDist;
} ResultsType;


typedef enum
{
    FAILURE_IMG_TOO_SMALL,
    FAILURE_ELECTRODES,
    FAILURE_LAMP_CONTOURS
} AnalysisFailure;

class ImageProcessor : public QObject
{
    Q_OBJECT

public:
    ImageProcessor(class MainWindow *mainWindow);
    ~ImageProcessor();

    void SetVisualizationWidget(GLVisualisation* visWidget);
    bool UploadImage(QString fileName);
    void ResetImage();
    bool IsImagePresent();

signals:
    void DistanceCalculated(float distance, float angle);
    void PrintRadii(QString radii);
    void PrintFullResults(ResultsType results);
    void AnalysisFailed(AnalysisFailure failure);

public slots:
    void SaveCalculations();
    void SaveImage();

    void RunEdgeDetectionSlot();
    void RunFullAnalysis();
    void ProcessElectrodesSlot();

    void StartManualSelection();
    void ResetManualSelection();

    void PaintVerticeSlot(const QPointF& imgPt);
    void ShowVerticeOnImage(std::shared_ptr<cv::Mat> img);
    void ShowEdgedImage(std::shared_ptr<cv::Mat> processedImage);

    void ElectrodesProcessingFail();
    void WallsAnalysisFailSlot();

private:
    MainWindow *mainWindow;
    GLVisualisation* visualization;
    QImage currentImage;
    QImage originalImage;
    std::unique_ptr<LampImageAnalyser> analyser;

    bool imageUploaded = false;
    QPair<QPointF, QPointF> vertices;
    int vertex_count = 0;

    void ShowImage(const QImage& image);
};
