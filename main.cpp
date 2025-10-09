#include "mainwindow.h"
#include <QApplication>
#include <QDebug>
#include <QProcessEnvironment>

#include <iostream>
#include <fcntl.h>
#ifndef WIN32
    #include <unistd.h>
#endif

int main(int argc, char *argv[])
{

    QCoreApplication::setAttribute(Qt::AA_UseSoftwareOpenGL);
    // Optional: fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    // Optional: fmt.setSwapInterval(1); // vsync

    QSurfaceFormat fmt;
    fmt.setRenderableType(QSurfaceFormat::OpenGL);
    fmt.setProfile(QSurfaceFormat::CoreProfile); // GL core
    fmt.setVersion(3, 0);                        // GL 3.0 core
    QSurfaceFormat::setDefaultFormat(fmt);
    
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
