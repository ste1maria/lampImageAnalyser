#include "ImageProcessor.h"
#include <QMessageBox>
#include <QFileDialog>
#include "LampImageAnalyser.h"
#include <QDebug>

ImageProcessor::ImageProcessor(MainWindow *mainWindow)
    :
    mainWindow(mainWindow)
{
    this->visualization = nullptr;
    this->analyser = std::make_unique<LampImageAnalyser>();

    connect(this->analyser.get(), &LampImageAnalyser::ShowVerticeOnImage, this, &ImageProcessor::ShowVerticeOnImage);
}

ImageProcessor::~ImageProcessor()
{
}

void ImageProcessor::SetVisualizationWidget(std::shared_ptr<GLVisualisation> visWidget)
{
    this->visualization = visWidget;
}

bool ImageProcessor::UploadImage(QString fileName)
{

    QImage image(fileName);
    if (image.isNull()) 
    {
        QMessageBox warningBox;
        warningBox.setText("Failed to load image.");
        warningBox.exec();
        return false;
    }

    this->currentImage = image.convertToFormat(QImage::Format_Grayscale8);
    this->ShowImage(currentImage);
    return true;
}

void ImageProcessor::ShowImage(QImage& image)
{
    this->visualization->ShowImage(image);
    this->analyser->SetCurrentImage(image);
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
    this->analyser->RunEdgeDetection(this->currentImage); 
}

void ImageProcessor::FindVerticesSlot()
{
    this->analyser->DrawVertices(this->currentImage);
}

void ImageProcessor::SelectVerticesSlot()
{
    //this->analyser->SelectVertices();
}

void ImageProcessor::ClearVerticesSlot()
{
    //this->analyser->ClearVertices();
}

void ImageProcessor::PaintVerticeSlot(const QPointF& imgPt)
{
    this->analyser->PaintVertice(imgPt);
}

void ImageProcessor::ShowVerticeOnImage(std::shared_ptr<cv::Mat> img)
{
    this->imageWithVertices = QImage(img->data, img->cols, img->rows, img->step, QImage::Format_Grayscale8);
    this->ShowImage(this->imageWithVertices);
}