#pragma once

#include <QMainWindow>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include "ImageProcessor.h"
#include <GLVisualisation.h>
#include <memory>

namespace Ui {
    class MainWindow;
}
    
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


public slots:
    void ShowVerticesDistanceAngle(float distance, float angle);
    void PrintLampResults
    (   std::vector<QPoint> centers, 
        std::vector<float> radii, 
        QPair<double, double> thickness,
        double lampWidth
    );

    void PrintElectrodes
    (
        QPoint coordFirst, 
        QPoint coordSecond, 
        double distance, 
        double angle
    );

private slots:
    void UploadImage();
    void ResetImageSlot();
    void AnalysisFailedSlot(AnalysisFailure failure);
    void StartManualSelection();
    void ResetManualSelection();
    void SaveImage();
    void SaveResults();
    void SaveContours();
    void BatchProcessFolder();
    void ChangeImageType();
    
private:
    Ui::MainWindow *ui;
    std::unique_ptr<ImageProcessor> imgProcessor;
    GLVisualisation * visualisation;
    const QString serialNumberDefault = "Enter serial #";
};
