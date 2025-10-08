#include "ImageProcessor.h"
#include <QMessageBox>
#include <QFileDialog>
#include "LampImageAnalyser.h"
#include <QDebug>
#include <cmath>

ImageProcessor::ImageProcessor(MainWindow *mainWindow)
    :
    mainWindow(mainWindow)
{
    this->visualization = nullptr;
    this->analyser = std::make_unique<LampImageAnalyser>();

    connect(this->analyser.get(), &LampImageAnalyser::ShowVerticeOnImage, this, &ImageProcessor::ShowVerticeOnImage);
    connect(this->analyser.get(), &LampImageAnalyser::ImageProcessed, this, &ImageProcessor::ShowEdgedImage);
    connect(this->analyser.get(), &LampImageAnalyser::RadiiCalculated, this, &ImageProcessor::PrintRadiiSlot);
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

    this->currentImage = image.convertToFormat(QImage::Format_BGR888);
    originalImage = currentImage.copy();
    this->ShowImage(currentImage);
    return true;
}

void ImageProcessor::ShowImage(const QImage& image)
{
    this->visualization->ShowImage(image);
    this->analyser->SetCurrentImage(image);
}

void ImageProcessor::SaveImage()
{
    QString filename = QFileDialog::getSaveFileName(
            nullptr,
            "Save Image As",
            "lamp.png",
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
            "lamp_calculations.txt",
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

// void ImageProcessor::FindVerticesSlot()
// {
//     this->analyser->DrawVertices(this->currentImage);
// }

void ImageProcessor::ResetImage()
{
    vertex_count = 0;
    currentImage = originalImage.copy();
    this->ShowImage(this->currentImage);
}

void ImageProcessor::PaintVerticeSlot(const QPointF& imgPt)
{
    if (vertex_count < 2)
    {
        vertex_count++;
        this->analyser->PaintVertice(imgPt);
        if (vertex_count == 1)
        {
            vertices.first = imgPt;
        }
        else if (vertex_count == 2)
        {
            vertices.second = imgPt;
        }
    }

    if (vertex_count == 2)
    {
        // calculate distance
        QPair<float, float> result = this->CalculateDistanceAngle();
        float distance = result.first;
        float angle = result.second;
        emit DistanceCalculated(distance, angle);
    }
}

void ImageProcessor::ShowVerticeOnImage(std::shared_ptr<cv::Mat> img)
{
    cv::Mat bgr = img->clone();
    QImage qi(bgr.data, bgr.cols, bgr.rows, bgr.step, QImage::Format_BGR888);
    this->currentImage = qi.copy();        // <- make QImage own memory
    this->ShowImage(this->currentImage);
}

QPair<float, float> ImageProcessor::CalculateDistanceAngle()
{
    float distance = sqrt( pow((vertices.second.x() - vertices.first.x()), 2) +
                           pow((vertices.second.y() - vertices.first.y()), 2)
                        );
    QLineF line(vertices.first, vertices.second);
    double angle = std::abs(90.0 - line.angle());
    if (angle > 90) 
    {
        angle = 180.0 - angle;
    }

    return QPair(distance, angle);
}

void ImageProcessor::ShowEdgedImage(std::shared_ptr<cv::Mat> processedImage)
{
    this->currentImage = QImage(processedImage->data, 
        processedImage->cols,
        processedImage->rows, 
        processedImage->step,
        QImage::Format_Grayscale8
    );
    this->ShowImage(this->currentImage);
}

void ImageProcessor::PrintRadiiSlot(QString radii)
{
    emit PrintRadii(radii);
}