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

class ImageProcessor : public QObject
{
    Q_OBJECT

public:
    ImageProcessor(class MainWindow *mainWindow);
    ~ImageProcessor();

    void SetVisualizationWidget(std::shared_ptr<GLVisualisation> visWidget);
    bool UploadImage(QString fileName);
    void ResetImage();

signals:
    void DistanceCalculated(float distance, float angle);
    void PrintRadii(QString radii);
    void PrintFullResults(ResultsType results);

public slots:
    void SaveCalculations();
    void SaveImage();
    void RunEdgeDetectionSlot();
    void PaintVerticeSlot(const QPointF& imgPt);
    void ShowVerticeOnImage(std::shared_ptr<cv::Mat> img);
    void ShowEdgedImage(std::shared_ptr<cv::Mat> processedImage);
    void PrintRadiiSlot(QString radii);
    void RunFullAnalysis();
    void ProcessElectrodesSlot();

private:
    MainWindow *mainWindow;
    std::shared_ptr<GLVisualisation> visualization;
    QImage currentImage;
    QImage originalImage;
    std::unique_ptr<LampImageAnalyser> analyser;

    QPair<QPointF, QPointF> vertices;
    int vertex_count = 0;

    void ShowImage(const QImage& image);
};
