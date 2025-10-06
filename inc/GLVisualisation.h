#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QImage>
#include <QMutex>

class GLVisualisation : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit GLVisualisation(QWidget *parent = nullptr);
    ~GLVisualisation();

    void ShowImage(const QImage &image);
    void setImage(const QImage& imgIn);

signals:
    void clickedImage(const QPointF& imgPt);
    void clickedWidget(const QPoint& widgetPt);   // raw widget-space

protected:
    void initializeGL() override;   // Called once to set up OpenGL context
    void resizeGL(int w, int h) override;  // Called on resize
    void paintGL() override;       // Called when widget needs to redraw  
    void mousePressEvent(QMouseEvent* e) override;

private:
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject vao_;
    QOpenGLShaderProgram program_;
    std::unique_ptr<QOpenGLTexture> texture_;

    QImage pendingImage_;     // last image set
    bool textureDirty_ = false;
    QMutex imgMutex_;

    int imgW_ = 0, imgH_ = 0;

    void ensureTexture();
    std::optional<QPointF> widgetToImage(const QPointF& wpt) const;
};


