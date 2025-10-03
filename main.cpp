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
    QSurfaceFormat fmt;
    fmt.setRenderableType(QSurfaceFormat::OpenGL);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setVersion(3, 3);
    // Optional: fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    // Optional: fmt.setSwapInterval(1); // vsync
    QSurfaceFormat::setDefaultFormat(fmt);

    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
