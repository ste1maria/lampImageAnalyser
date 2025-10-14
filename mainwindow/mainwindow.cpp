
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

    ui->pushButtonCurveAnalysis->setEnabled(false);
    ui->pushButtonRunAnalysis->setEnabled(false);
    ui->pushButtonElectrodes->setEnabled(false);
    ui->pushButtonManualSelection->setEnabled(false);
    ui->pushButtonReset->setEnabled(false);
    ui->pushButtonSaveResults->setEnabled(false);
    ui->pushButtonSaveImage->setEnabled(false);
    ui->labelDistance->clear();
    ui->labelRadii->clear();

    connect(ui->pushButtonCurveAnalysis, &QPushButton::clicked, this->imgProcessor.get(), &ImageProcessor::RunEdgeDetectionSlot);
    connect(ui->pushButtonLoadImage, &QPushButton::clicked, this, &MainWindow::UploadImage);
    connect(ui->pushButtonManualSelection, &QPushButton::clicked, this, &MainWindow::StartManualSelection);
    connect(ui->pushButtonReset, &QPushButton::clicked, this, &MainWindow::ResetImageSlot);
    connect(ui->pushButtonRunAnalysis, &QPushButton::clicked, this->imgProcessor.get(), &ImageProcessor::RunFullAnalysis);
    connect(ui->pushButtonElectrodes, &QPushButton::clicked, this->imgProcessor.get(), &ImageProcessor::ProcessElectrodesSlot);

    connect(this->visualisation, &GLVisualisation::clickedImage, this->imgProcessor.get(), &ImageProcessor::PaintVerticeSlot);
    connect(this->imgProcessor.get(), &ImageProcessor::DistanceCalculated, this, &MainWindow::ShowVerticesDistanceAngle);
    connect(this->imgProcessor.get(), &ImageProcessor::PrintRadii, this, &MainWindow::PrintRadii);
    connect(this->imgProcessor.get(), &ImageProcessor::AnalysisFailed, this, &MainWindow::AnalysisFailedSlot);
    
}

MainWindow::~MainWindow()
{
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
            ui->pushButtonCurveAnalysis->setEnabled(true);
            ui->pushButtonRunAnalysis->setEnabled(true);
            ui->pushButtonElectrodes->setEnabled(true);
            ui->pushButtonManualSelection->setEnabled(true);
            ui->pushButtonReset->setEnabled(true);
            ui->pushButtonSaveResults->setEnabled(true);
            ui->pushButtonSaveImage->setEnabled(true);
            ui->labelDistance->setText("Select two dots to calculate distance");
        }
    }
}

void MainWindow::ResetImageSlot()
{
    this->imgProcessor->ResetImage();
    ui->labelFullAnalysis->clear();
    ui->labelDistance->setText("Select two dots to calculate distance");
    ui->labelElectrodes->clear();
    ui->labelRadii->clear();

}

void MainWindow::ShowVerticesDistanceAngle(float distance, float angle)
{
    ui->labelDistance->setText(QString("Distance: %1 px; Angle: %2°").arg(distance).arg(angle));
}

void MainWindow::PrintRadii(QString radii)
{
    ui->labelRadii->setText(radii);
}


void MainWindow::PrintFullResultsSlot(ResultsType results)
{
    QString resultsString;
    resultsString += QString("Radius 1 \t ");
}

void MainWindow::AnalysisFailedSlot(AnalysisFailure failure)
{
    QMessageBox failureMessage;
    QString messageText = "unknown error, sorry";
    if (failure == AnalysisFailure::FAILURE_IMG_TOO_SMALL)
    {
        ui->pushButtonCurveAnalysis->setEnabled(false);
        ui->pushButtonRunAnalysis->setEnabled(false);
        ui->pushButtonElectrodes->setEnabled(false);
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