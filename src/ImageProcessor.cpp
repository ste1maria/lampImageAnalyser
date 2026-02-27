#include "ImageProcessor.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QDebug>
#include <cmath>
#include "mainwindow.h"

class MainWindow;

ImageProcessor::ImageProcessor(MainWindow *mainWindow)
    :
    mainWindow(mainWindow)
{
    this->visualization = nullptr;
    this->analyser = std::make_unique<LampImageAnalyser>();

    connect(this->analyser.get(), &LampImageAnalyser::ShowVerticeOnImage, this, &ImageProcessor::ShowVerticeOnImage);
    connect(this->analyser.get(), &LampImageAnalyser::ImageProcessed, this, &ImageProcessor::ShowEdgedImage);
    connect(this->analyser.get(), &LampImageAnalyser::LampWallsProcessed, this->mainWindow,  &MainWindow::PrintLampResults);
    connect(this->analyser.get(), &LampImageAnalyser::ElectrodesAnalysisFailed, this, &ImageProcessor::ElectrodesProcessingFail);
    connect(this->analyser.get(), &LampImageAnalyser::WallsAnalysisFailed, this, &ImageProcessor::WallsAnalysisFailSlot);
    connect(this->analyser.get(), &LampImageAnalyser::ElectrodesProcessed, this->mainWindow, &MainWindow::PrintElectrodes);
}

ImageProcessor::~ImageProcessor()
{
}

void ImageProcessor::SetVisualizationWidget(GLVisualisation* visWidget)
{
    this->visualization = visWidget;
}

bool ImageProcessor::IsImagePresent()
{
    return imageUploaded;
}

bool ImageProcessor::UploadImage(QString fileName)
{
    bool success = true;

    QImage image(fileName);
    if (image.isNull()) 
    {
        QMessageBox warningBox;
        warningBox.setText("Failed to load image.");
        warningBox.exec();
        return false;
    }
    if (image.height() < 400 && image.width() < 600)
    {
        emit AnalysisFailed(AnalysisFailure::FAILURE_IMG_TOO_SMALL);
        success = false;
    }
    else if (image.width() > 1200)
    {
        image = image.scaled(QSize(1200, 900), Qt::KeepAspectRatio);
    }
    this->currentImage = image.convertToFormat(QImage::Format_BGR888);
    this->originalImage = currentImage.copy();
    this->ShowImage(currentImage);
    this->analyser->ResetImage();
    this->analyser->SetOriginalImage(this->originalImage);
    this->analyser->SetCurrentImage(this->currentImage);

    return success;
}

void ImageProcessor::ShowImage(const QImage& image)
{
    this->visualization->ShowImage(image);
}

void ImageProcessor::SaveImage(QString serialNumber, bool batch)
{
    QString filenameString = QString("lamp_%1.png").arg(serialNumber);
    QString filename = filenameString;
    if (!batch)
    {
        QString filename = QFileDialog::getSaveFileName(
                nullptr,
                "Save Image As",
                QString("lamp_%1.png").arg(serialNumber),
                "Images (*.png *.bmp);;All Files (*)"
        );
    }

    QFile data(filename);

    if (!filename.isEmpty()) {
        this->currentImage.save(filename);
    }
}


void ImageProcessor::SaveCalculations(QString serialNumber, bool batch)
{
    QString filenameString = QString("lamp_%1_calc.csv").arg(serialNumber);
    QString filename = filenameString;
    if (!batch)
    {
        QString filename = QFileDialog::getSaveFileName(
                nullptr,
                "Save Calculations As",
                filenameString,
                "Text file (*.csv);;All Files (*)"
        );
    }

    QFile data(filename);

    if (data.open(QFile::WriteOnly | QFile::Truncate))
    {
        QTextStream out(&data);
        QString header = "SN,r(1),r(2),r(3),r(4),R(1),R(2),R(3),R(4),Distance electrodes,"
                            "Angle electrodes,Left wall,RightWall,Lamp width\r\n";
        QString fullRow = header + serialNumber + ',' + this->analyser->GetResultsString();
        out << fullRow;
    }
}

void ImageProcessor::SaveContours(QString serialNumber, bool batch)
{
    QString filenameString = QString("lamp_%1_contours.csv").arg(serialNumber);
    QString filename = filenameString;
    if (!batch)
    {
        QString filename = QFileDialog::getSaveFileName(
                nullptr,
                "Save Lamp Contours As",
                filenameString,
                "Text file (*.csv);;All Files (*)"
        );
    }

    QFile data(filename);

    if (data.open(QFile::WriteOnly | QFile::Truncate))
    {
        QTextStream out(&data);
        out << this->analyser->GetContoursTable();
    }
}

void ImageProcessor::RunEdgeDetectionSlot()
{
    QImage imgToProcess = this->originalImage.copy();
    this->analyser->ProcessLampWalls(imgToProcess); 
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
    this->analyser->ResetImage();
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
        QPair<float, float> result = this->analyser->CalculateDistanceAngle(vertices.first, vertices.second);
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

void ImageProcessor::ShowEdgedImage(std::shared_ptr<cv::Mat> processedImage)
{
    this->currentImage = QImage(processedImage->data, 
        processedImage->cols,
        processedImage->rows, 
        processedImage->step,
        QImage::Format_BGR888
    );
    this->ShowImage(this->currentImage);
}

void ImageProcessor::RunFullAnalysis()
{
    QImage imgToProcess = this->originalImage.copy();
    // find electrodes distance
    this->analyser->RunFullAnalysis(imgToProcess);
}

void ImageProcessor::ProcessElectrodesSlot()
{
    QImage imgToProcess = this->originalImage.copy();
    this->analyser->ProcessElectrodes(imgToProcess);
}

void ImageProcessor::ElectrodesProcessingFail()
{
    this->ResetImage();
    emit AnalysisFailed(AnalysisFailure::FAILURE_ELECTRODES);
}


void ImageProcessor::WallsAnalysisFailSlot()
{
    this->ResetImage();
    emit AnalysisFailed(AnalysisFailure::FAILURE_LAMP_CONTOURS);
}

void ImageProcessor::StartManualSelection()
{
    this->visualization->SetImageClickable(true);
}

void ImageProcessor::ResetManualSelection()
{
    this->ResetImage();
    this->visualization->SetImageClickable(false);
}

void ImageProcessor::SetImageType(ImageType type)
{
    this->analyser->SetImageType(type);
}