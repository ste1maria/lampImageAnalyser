#pragma once
#include <QObject>
#include "GLVisualisation.h"
#include <memory>

class MainWindow;
class GLVisualisation;

class ImageProcessor : public QObject
{
    Q_OBJECT

public:
    ImageProcessor(class MainWindow *mainWindow);
    ~ImageProcessor();

    void SetVisualizationWidget(std::shared_ptr<GLVisualisation> visWidget);
    void UploadImage(QString fileName);

private:
    void ShowImage();

public slots:
    void SaveCalculations();
    void SaveImage();

private:
    MainWindow *mainWindow;
    std::shared_ptr<GLVisualisation> visualization;
    QImage currentImage;
};
