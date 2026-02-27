
#include <QFileDialog>
#include <QDebug>
#include <QFileInfo>
#include <QVector>
#include <QMessageBox>
#include <iostream>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <thread>


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setCentralWidget(ui->centralwidget);     

    this->imgProcessor = std::make_unique<ImageProcessor>(this);
    this->visualisation = new GLVisualisation(this);
    ui->layoutImageWidget->addWidget(this->visualisation);
    this->imgProcessor->SetVisualizationWidget(this->visualisation);

    ui->pushButtonRunAnalysis->setEnabled(false);
    ui->pushButtonManualSelection->setEnabled(false);
    ui->pushButtonReset->setEnabled(false);
    ui->pushButtonSaveResults->setEnabled(false);
    ui->pushButtonSaveContours->setEnabled(false);
    ui->pushButtonSaveImage->setEnabled(false);
    ui->lineEditSerial->setEnabled(false);

    ui->labelDistance->clear();
    ui->labelLampResults->clear();
    ui->labelElectrodesResults->clear();
    ui->lineEditSerial->setText(this->serialNumberDefault);
    ui->radioButtonXray->setChecked(true);
    this->imgProcessor->SetImageType(ImageType::XRAY);

    connect(ui->pushButtonLoadImage, &QPushButton::clicked, this, &MainWindow::UploadImage);
    connect(ui->pushButtonManualSelection, &QPushButton::clicked, this, &MainWindow::StartManualSelection);
    connect(ui->pushButtonReset, &QPushButton::clicked, this, &MainWindow::ResetImageSlot);
    connect(ui->pushButtonRunAnalysis, &QPushButton::clicked, this->imgProcessor.get(), &ImageProcessor::RunFullAnalysis);
    connect(ui->pushButtonSaveResults, &QPushButton::clicked, this, &MainWindow::SaveResults);
    connect(ui->pushButtonSaveContours, &QPushButton::clicked, this, &MainWindow::SaveContours);
    connect(ui->pushButtonBatch, &QPushButton::clicked, this, &MainWindow::BatchProcessFolder);
    connect(ui->radioButtonPhoto, &QRadioButton::toggled, this, &MainWindow::ChangeImageType);

    connect(this->visualisation, &GLVisualisation::clickedImage, this->imgProcessor.get(), &ImageProcessor::PaintVerticeSlot);
    connect(this->imgProcessor.get(), &ImageProcessor::DistanceCalculated, this, &MainWindow::ShowVerticesDistanceAngle);
    connect(this->imgProcessor.get(), &ImageProcessor::AnalysisFailed, this, &MainWindow::AnalysisFailedSlot);
    connect(ui->pushButtonSaveImage, &QPushButton::clicked, this, &MainWindow::SaveImage);
}

MainWindow::~MainWindow()
{
    delete visualisation;
}

void MainWindow::UploadImage()
{
    QString filter = "Image files (*.png *.jpg)";
    QString fileName = QFileDialog::getOpenFileName(
        this,                   // parent
        "Open Image",           // dialog title
        QDir::homePath(),       // starting directory
        filter                  // allowed extensions
    );

    if (fileName.isEmpty())
    {
        return; // user canceled
    }
    else
    {
        if (this->imgProcessor->UploadImage(fileName))
        {
            ui->pushButtonRunAnalysis->setEnabled(true);
            ui->pushButtonManualSelection->setEnabled(true);
            ui->pushButtonReset->setEnabled(true);
            ui->pushButtonSaveResults->setEnabled(true);
            ui->pushButtonSaveContours->setEnabled(true);
            ui->pushButtonSaveImage->setEnabled(true);
            ui->lineEditSerial->setEnabled(true);

            ui->labelElectrodesResults->clear();
            ui->labelLampResults->clear();
            ui->labelDistance->setText("Select two dots to calculate distance");

            QFileInfo fi(fileName);
            ui->lineEditSerial->setText(fi.baseName());
        }
    }
}

void MainWindow::ResetImageSlot()
{
    this->imgProcessor->ResetImage();
    ui->labelFullAnalysis->clear();
    ui->labelDistance->setText("Select two dots to calculate distance");
    ui->labelElectrodesResults->clear();
    ui->labelLampResults->clear();

}

void MainWindow::ShowVerticesDistanceAngle(float distance, float angle)
{
    ui->labelDistance->setText(QString("Distance: %1 px; Angle: %2°").arg(distance).arg(angle));
}


void MainWindow::PrintLampResults
(
    std::vector<QPoint> centers, 
    std::vector<float> radii, 
    QPair<double, double> thickness,
    double lampWidth
)
{
    QString resultsString;
    for (int i = 0; i < centers.size(); i++)
    {
        resultsString += QString("R(%1)=%2 mm at (%3, %4);\n\r").arg(i+1).arg(radii[i], 0, 'f', 2).arg(centers[i].x()).arg(centers[i].y());
    }
    resultsString += QString("\n\rLeft wall %1 mm, right wall %2 mm\n\r")
                            .arg(thickness.first, 0, 'f', 2)
                            .arg(thickness.second, 0, 'f', 2);
    resultsString += QString("Lamp width: %1 mm\n\r").arg(lampWidth, 0, 'f', 2);
    ui->labelLampResults->setText(resultsString);
}

void MainWindow::AnalysisFailedSlot(AnalysisFailure failure)
{
    QMessageBox failureMessage;
    QString messageText = "unknown error, sorry";
    if (failure == AnalysisFailure::FAILURE_IMG_TOO_SMALL)
    {
        ui->pushButtonRunAnalysis->setEnabled(false);
        ui->labelDistance->clear();
        messageText = "Uploaded image is too small for correct analysis.";
    }
    else if (failure == AnalysisFailure::FAILURE_ELECTRODES)
    {
        messageText = "Failed to recognise electrodes shapes. Try uploading different image.";
    }
    else if (failure == AnalysisFailure::FAILURE_LAMP_CONTOURS)
    {
        messageText = "Failed to recognise lamp or lamp walls contours. Try uploading different image.";
    }
    failureMessage.setText(messageText);
    failureMessage.exec();
}

void MainWindow::StartManualSelection()
{
    this->imgProcessor->StartManualSelection();
    ui->pushButtonManualSelection->setText("Reset selection");
    
    connect(ui->pushButtonManualSelection, &QPushButton::clicked, this, &MainWindow::ResetManualSelection);
}

void MainWindow::ResetManualSelection()
{
    this->imgProcessor->ResetManualSelection();
    ui->pushButtonManualSelection->setText("Select points (calc. distance)");
    ui->labelDistance->setText("Select two dots to calculate distance");
    
    connect(ui->pushButtonManualSelection, &QPushButton::clicked, this, &MainWindow::StartManualSelection);
}

void MainWindow::PrintElectrodes
(
    QPoint coordFirst, 
    QPoint coordSecond, 
    double distance, 
    double angle
)
{
    QString electrodesResult = 
        QString("Coordinates: (%1,%2); (%3,%4); distance: %5 mm; angle %6°")
                                            .arg(coordFirst.x()) 
                                            .arg(coordFirst.y())
                                            .arg(coordSecond.x())
                                            .arg(coordSecond.y())
                                            .arg(distance, 0, 'f', 2)
                                            .arg(angle, 0, 'f', 1);
    ui->labelElectrodesResults->setText(electrodesResult);
}

void MainWindow::SaveImage()
{
    QString serialNumber;
    serialNumber = ui->lineEditSerial->text() == this->serialNumberDefault ? "SN" : ui->lineEditSerial->text();

    this->imgProcessor->SaveImage(serialNumber);
}

void MainWindow::SaveResults()
{
    QString serialNumber;
    serialNumber = ui->lineEditSerial->text() == this->serialNumberDefault ? "SN" : ui->lineEditSerial->text();

    this->imgProcessor->SaveCalculations(serialNumber);
}

void MainWindow::SaveContours()
{
    QString serialNumber;
    serialNumber = ui->lineEditSerial->text() == this->serialNumberDefault ? "SN" : ui->lineEditSerial->text();

    this->imgProcessor->SaveContours(serialNumber);
}

void MainWindow::BatchProcessFolder()
{
    QString dirName = QFileDialog::getExistingDirectory
    (
        this, tr("Open Directory"),
        QDir::currentPath(),
        QFileDialog::ShowDirsOnly
    );

    if (dirName.isEmpty())
    {
        return; // user canceled
    }
    else
    {
        QDir dir(dirName);

        QDir::Filters filter = QDir::Files | QDir::NoDotAndDotDot;
        QFileInfoList list = dir.entryInfoList();

        for (int i = 0; i < list.size(); ++i)
        {
            QFileInfo fileInfo = list.at(i);
            QString fileName = fileInfo.fileName();
            if (!fileName.endsWith("png") && !fileName.endsWith("jpg"))
            {
                continue;
            }

            if (this->imgProcessor->UploadImage(dirName + "/" + fileName))
            {
                ui->labelElectrodesResults->clear();
                ui->labelLampResults->clear();

                QFileInfo fi(fileName);
                ui->lineEditSerial->setText(fi.baseName());

                this->imgProcessor->RunFullAnalysis();

                QString serialNumber;
                serialNumber = fi.baseName();

                this->imgProcessor->SaveCalculations(serialNumber, true);
                this->imgProcessor->SaveContours(serialNumber, true);
                this->imgProcessor->SaveImage(serialNumber, true);
            }

        }

    }
}

void MainWindow::ChangeImageType()
{
    if (ui->radioButtonPhoto->isChecked())
    {
        this->imgProcessor->SetImageType(ImageType::PHOTO);
    }
    else if (ui->radioButtonXray->isChecked())
    {
        this->imgProcessor->SetImageType(ImageType::XRAY);
    }
}
