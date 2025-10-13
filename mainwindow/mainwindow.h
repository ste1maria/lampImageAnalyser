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
    void PrintRadii(QString radii);
    void PrintFullResultsSlot(ResultsType results);

private slots:
    void UploadImage();
    void ResetImageSlot();
    void AnalysisFailedSlot(AnalysisFailure failure);

private:
    Ui::MainWindow *ui;
    std::unique_ptr<ImageProcessor> imgProcessor;
    std::shared_ptr<GLVisualisation> visualisation;
};
