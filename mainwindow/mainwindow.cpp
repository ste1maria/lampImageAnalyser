
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

    ui->pushButtonVertices->setEnabled(false);

    connect(ui->pushButtonLoadImage, &QPushButton::clicked, this, &MainWindow::UploadImage);
    connect(ui->pushButtonVertices, &QPushButton::clicked, this, &MainWindow::SelectVertices);
    connect(ui->pushButtonDeselect, &QPushButton::clicked, this, &MainWindow::Deselect);

    connect(this->visualisation.get(), &GLVisualisation::clickedImage, this->imgProcessor.get(), &ImageProcessor::PaintVerticeSlot);
    
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
            ui->pushButtonVertices->setEnabled(true);
        }
    }
}

void MainWindow::SelectVertices()
{
    ui->pushButtonDeselect->setEnabled(true);
    this->visualisation->ActivateClicking(true);
}

void MainWindow::Deselect()
{
    ui->pushButtonDeselect->setEnabled(false);
    this->visualisation->ActivateClicking(false);
}