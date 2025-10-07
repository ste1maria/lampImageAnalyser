
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
    this->visualisation = std::make_shared<GLVisualisation>(this->ui->imageWidget->GetVisualisationWidget());

    this->imgProcessor->SetVisualizationWidget(this->visualisation);

    ui->pushButtonDeselect->setEnabled(false);
    ui->pushButtonEdgeDetect->setEnabled(false);
    ui->labelDistance->clear();

    connect(ui->pushButtonEdgeDetect, &QPushButton::clicked, this->imgProcessor.get(), &ImageProcessor::RunEdgeDetectionSlot);
    connect(ui->pushButtonLoadImage, &QPushButton::clicked, this, &MainWindow::UploadImage);
    connect(ui->pushButtonDeselect, &QPushButton::clicked, this, &MainWindow::Deselect);

    connect(this->visualisation.get(), &GLVisualisation::clickedImage, this->imgProcessor.get(), &ImageProcessor::PaintVerticeSlot);
    connect(this->imgProcessor.get(), &ImageProcessor::DistanceCalculated, this, &MainWindow::ShowVerticesDistanceAngle);
    
}

MainWindow::~MainWindow()
{
}

void MainWindow::UploadImage()
{
    QString filter = "Images (*.png *.bmp *.raw)";
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
            ui->pushButtonEdgeDetect->setEnabled(true);
            ui->labelDistance->setText("Select two dots to calculate distance");
        }
    }
}

void MainWindow::Deselect()
{
    ui->pushButtonDeselect->setEnabled(false);
    ui->labelDistance->setText("Select two dots to calculate distance");
    this->imgProcessor->ClearVertices();

}

void MainWindow::ShowVerticesDistanceAngle(float distance, float angle)
{
    ui->labelDistance->setText(QString("Distance: %1 px; Angle: %2°").arg(distance).arg(angle));
    ui->pushButtonDeselect->setEnabled(true);
}