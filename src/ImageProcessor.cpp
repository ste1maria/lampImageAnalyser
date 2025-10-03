#include "ImageProcessor.h"
#include <QMessageBox>
#include <QFileDialog>
#include "LampImageAnalyser.h"

ImageProcessor::ImageProcessor(MainWindow *mainWindow)
    :
    mainWindow(mainWindow)
{
    this->visualization = nullptr;
    this->analyser = std::make_unique<LampImageAnalyser>();
}

ImageProcessor::~ImageProcessor()
{
}

void ImageProcessor::SetVisualizationWidget(std::shared_ptr<GLVisualisation> visWidget)
{
    this->visualization = visWidget;
}

void ImageProcessor::UploadImage(QString fileName)
{
    QImage image(fileName);
    if (image.isNull()) 
    {
        QMessageBox warningBox;
        warningBox.setText("Failed to load image.");
        warningBox.exec();
        return;
    }

    this->currentImage = image.convertToFormat(QImage::Format_Grayscale8);
    this->ShowImage();
}

void ImageProcessor::ShowImage()
{
    this->visualization->ShowImage(this->currentImage);
}

void ImageProcessor::SaveImage()
{
    QString filename = QFileDialog::getSaveFileName(
            nullptr,
            "Save Image As",
            "plasma.png",
            "Images (*.png *.bmp);;All Files (*)"
    );

    if (!filename.isEmpty()) {
        this->currentImage.save(filename);
    }
}


void ImageProcessor::SaveCalculations()
{
    QString filename = QFileDialog::getSaveFileName(
            nullptr,
            "Save Calculations As",
            "plasma_calculations.txt",
            "Text file (*.txt *.csv);;All Files (*)"
    );
    QFile data(filename);

    if (data.open(QFile::WriteOnly | QFile::Truncate))
    {
        QTextStream out(&data);
    }
}


void ImageProcessor::RunEdgeDetectionSlot()
{
    uint8_t * buffer = (uint8_t*)this->currentImage.bits();
    size_t width = this->currentImage.width();
    size_t height = this->currentImage.height();
    size_t stride = currentImage.bytesPerLine();

    this->analyser->RunEdgeDetection(height, width, buffer, stride); 
}