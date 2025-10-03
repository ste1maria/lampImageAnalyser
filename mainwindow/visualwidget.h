#pragma once

#include <QWidget>

class GLVisualisation;

class VisualWidget : public QWidget
{
    Q_OBJECT

public:
    explicit VisualWidget(QWidget *parent = nullptr);
    ~VisualWidget();
    GLVisualisation * GetVisualisationWidget();

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    GLVisualisation *glWidget;
};