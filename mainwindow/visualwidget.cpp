#include "visualwidget.h"
#include "GLVisualisation.h"

VisualWidget::VisualWidget(QWidget *parent)
    : QWidget(parent)
{
    glWidget = new GLVisualisation(this);
    //glWidget->setGeometry(0, 0, width(), height());
}

VisualWidget::~VisualWidget()
{
    // No need to delete glWidget; Qt parent system handles it
}

void VisualWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);

    int w = width();
    int h = height();
    float targetAspect = 16.0f / 9.0f;

    int newW, newH;
    if (w / (float)h > targetAspect) {
        newH = h;
        newW = static_cast<int>(h * targetAspect);
    } else {
        newW = w;
        newH = static_cast<int>(w / targetAspect);
    }

    int x = (w - newW) / 2;
    int y = (h - newH) / 2;

    glWidget->setGeometry(rect());
}

GLVisualisation* VisualWidget::GetVisualisationWidget()
{
    return this->glWidget;
}
