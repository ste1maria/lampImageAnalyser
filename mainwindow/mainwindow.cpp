
#include <QFileDialog>
#include <QDebug>
#include <QFileInfo>
#include <QVector>
#include <QMessageBox>
#include <iostream>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <thread>
#include <chrono>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setCentralWidget(ui->centralwidget); 
    
    this->imgProcessor = std::make_unique<ImageProcessor>(this);
    this->visualisation = std::make_shared<GLVisualisation>(this->ui->imageWidget->GetVisualisationWidget());

    this->imgProcessor->SetVisualizationWidget(this->visualisation);

    connect(ui->pushButtonLoadImage, &QPushButton::clicked, this, &MainWindow::UploadImage);
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
        this->imgProcessor->UploadImage(fileName);
    }
}
