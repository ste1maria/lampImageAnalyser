#include "GLVisualisation.h"
#include <QOpenGLExtraFunctions>
#include <QMouseEvent>


static const char* kVertexSrc = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
uniform mat4 uMvp;
void main() {
    vUV = aUV;
    gl_Position = uMvp * vec4(aPos, 0.0, 1.0);
}
)";

static const char* kFragmentSrc = R"(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = texture(uTex, vUV);
}
)";


GLVisualisation::GLVisualisation(QWidget *parent)
    : QOpenGLWidget(parent)
{
    setMinimumSize(640, 480);
    setMouseTracking(true); 
}

GLVisualisation::~GLVisualisation()
{
    makeCurrent();
    texture_.reset();
    vbo_.destroy();
    vao_.destroy();
    doneCurrent();
}

void GLVisualisation::setImage(const QImage& img) {
    QMutexLocker lock(&imgMutex_);
    // Convert now to a GL-friendly format; mirror vertically for OpenGL coords
    pendingImage_ = img.convertToFormat(QImage::Format_RGBA8888).mirrored();
    textureDirty_ = true;
    lock.unlock();
    update(); // schedule repaint
}

void GLVisualisation::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(0.f, 0.f, 0.f, 1.f);

    // Shaders
    if (!program_.addShaderFromSourceCode(QOpenGLShader::Vertex, kVertexSrc) ||
        !program_.addShaderFromSourceCode(QOpenGLShader::Fragment, kFragmentSrc) ||
        !program_.link()) {
        qFatal("Failed to compile/link shader program: %s", program_.log().toUtf8().constData());
    }

    // Fullscreen quad (NDC), with UVs
    // aPos (x,y), aUV (u,v)
    const float quad[] = {
        //   x     y     u     v
        -1.f, -1.f,  0.f, 0.f,
         1.f, -1.f,  1.f, 0.f,
        -1.f,  1.f,  0.f, 1.f,
         1.f,  1.f,  1.f, 1.f,
    };

    vao_.create();
    vao_.bind();

    vbo_.create();
    vbo_.bind();
    vbo_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vbo_.allocate(quad, sizeof(quad));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    vbo_.release();
    vao_.release();
}

void GLVisualisation::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
}

void GLVisualisation::ensureTexture() {
    if (!textureDirty_) return;

    QImage imgCopy;
    {
        QMutexLocker lock(&imgMutex_);
        if (pendingImage_.isNull()) return;
        imgCopy = pendingImage_;  // local copy while not bound to GL
        textureDirty_ = false;
    }

    // Create or update texture
    if (!texture_) {
        texture_ = std::make_unique<QOpenGLTexture>(QOpenGLTexture::Target2D);
        texture_->setFormat(QOpenGLTexture::RGBA8_UNorm);
        texture_->setMinificationFilter(QOpenGLTexture::Linear);
        texture_->setMagnificationFilter(QOpenGLTexture::Linear);
        texture_->setWrapMode(QOpenGLTexture::ClampToEdge);
        texture_->setMipLevels(1);
        texture_->setAutoMipMapGenerationEnabled(false);
        texture_->create();
        texture_->bind();
        texture_->setSize(imgCopy.width(), imgCopy.height());
        texture_->allocateStorage();
        texture_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, imgCopy.constBits());
        texture_->release();
    } else {
        texture_->bind();
        if (texture_->width() != imgCopy.width() || texture_->height() != imgCopy.height()) {
            texture_->setSize(imgCopy.width(), imgCopy.height());
            texture_->allocateStorage();
        }
        texture_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, imgCopy.constBits());
        texture_->release();
    }
}

void GLVisualisation::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    ensureTexture();
    if (!texture_) return;

    // Just identity matrix — fill the widget completely
    QMatrix4x4 mvp; 
    mvp.setToIdentity();

    program_.bind();
    program_.setUniformValue("uMvp", mvp);
    program_.setUniformValue("uTex", 0);

    glActiveTexture(GL_TEXTURE0);
    texture_->bind();

    vao_.bind();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    vao_.release();

    texture_->release();
    program_.release();

    imgW_ = texture_->width(); 
    imgH_ = texture_->height();
}

void GLVisualisation::ShowImage(const QImage &image)
{
    this->setImage(image);
}

void GLVisualisation::mousePressEvent(QMouseEvent* e) {
    emit clickedWidget(e->pos());
    if (auto p = widgetToImage(e->position()); p.has_value())
        emit clickedImage(*p);
}

std::optional<QPointF> GLVisualisation::widgetToImage(const QPointF& wpt) const {
        if (imgW_ <= 0 || imgH_ <= 0) return std::nullopt;
        const double W = width();
        const double H = height();
        const double sx = W / imgW_;
        const double sy = H / imgH_;
        const double s  = std::min(sx, sy);              // aspect-fit
        const double dx = (W - imgW_ * s) * 0.5;         // horizontal letterbox
        const double dy = (H - imgH_ * s) * 0.5;         // vertical letterbox

        const double x_img = (wpt.x() - dx) / s;
        const double y_img = (wpt.y() - dy) / s;         // top-left origin

        if (x_img < 0 || y_img < 0 || x_img >= imgW_ || y_img >= imgH_)
            return std::nullopt; // clicked on letterbox area
        return QPointF{x_img, y_img};
    }

