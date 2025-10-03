#pragma once
#include <QObject>
#include "GLVisualisation.h"
#include <memory>

#include <opencv2/core.hpp>

class MainWindow;
class GLVisualisation;
class LampImageAnalyser;

class ImageProcessor : public QObject
{
    Q_OBJECT

public:
    ImageProcessor(class MainWindow *mainWindow);
    ~ImageProcessor();

    void SetVisualizationWidget(std::shared_ptr<GLVisualisation> visWidget);
    bool UploadImage(QString fileName);

private:
    void ShowImage(QImage& image);

public slots:
    void SaveCalculations();
    void SaveImage();
    void RunEdgeDetectionSlot();
    void FindVerticesSlot();
    void SelectVerticesSlot();
    void ClearVerticesSlot();

    void PaintVerticeSlot(const QPointF& imgPt);
    void ShowVerticeOnImage(std::shared_ptr<cv::Mat> img);

private:
    MainWindow *mainWindow;
    std::shared_ptr<GLVisualisation> visualization;
    QImage currentImage;
    std::unique_ptr<LampImageAnalyser> analyser;
    QImage imageWithVertices;
};
