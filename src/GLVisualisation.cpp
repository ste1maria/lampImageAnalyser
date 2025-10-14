#include "GLVisualisation.h"
#include <QOpenGLExtraFunctions>
#include <QOpenGLPixelTransferOptions>
#include <QMouseEvent>


static const char* kVertexSrc = R"(
#version 130
in vec2 pos;
in vec2 uv;
uniform mat4 uMvp;      // matches your C++
out vec2 v_uv;
void main() {
    v_uv = uv;
    gl_Position = uMvp * vec4(pos, 0.0, 1.0);
}
)";

static const char* kFragmentSrc = R"(
#version 130
in vec2 v_uv;
uniform sampler2D uTex; // matches your C++
out vec4 fragColor;
void main() {
    fragColor = texture(uTex, v_uv); // GLSL 1.30 uses texture()
}
)";


GLVisualisation::GLVisualisation(QWidget *parent)
    : QOpenGLWidget(parent)
{
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
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
    texture_.reset();
    update(); // schedule repaint
}

void GLVisualisation::SetImageClickable(bool isClickable)
{
    widgetClickable = isClickable;
}

void GLVisualisation::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(0.f, 0.f, 0.f, 1.f);

    if (!program_.addShaderFromSourceCode(QOpenGLShader::Vertex,   kVertexSrc) ||
        !program_.addShaderFromSourceCode(QOpenGLShader::Fragment, kFragmentSrc)) 
    {
        qFatal("Shader compile error: %s", program_.log().toUtf8().constData());
    }
    program_.bindAttributeLocation("pos", 0);
    program_.bindAttributeLocation("uv",  1);
    if (!program_.link()) 
    {
        qFatal("Link error: %s", program_.log().toUtf8().constData());
    }

    // Fullscreen quad (NDC), with UVs
    // aPos (x,y), aUV (u,v)
    const float quad[] = 
    {
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
    const float dpr = devicePixelRatioF();
    vpW_ = int(w * dpr);
    vpH_ = int(h * dpr);

    glViewport(0, 0, vpW_, vpH_);
    update(); // ← ensure paintGL() runs with new size
}

void GLVisualisation::ensureTexture()
{
    if (!textureDirty_) return;

    QImage src;
    {
        QMutexLocker lock(&imgMutex_);
        if (pendingImage_.isNull()) return;
        src = pendingImage_;
        textureDirty_ = false;
    }

    // Normalize format (flip if needed)
    QImage img = src.convertToFormat(QImage::Format_RGBA8888);
    // img = img.mirrored();

    const int w = img.width();
    const int h = img.height();

    // <<< IMPORTANT: publish the NEW size immediately so paintGL uses it
    imgW_ = w;
    imgH_ = h;

    const bool needRecreate = !texture_
                           || texture_->width()  != w
                           || texture_->height() != h;

    if (needRecreate) {
        if (texture_) { texture_->destroy(); }
        texture_ = std::make_unique<QOpenGLTexture>(QOpenGLTexture::Target2D);
        texture_->create();
        texture_->bind();
        texture_->setFormat(QOpenGLTexture::RGBA8_UNorm);
        texture_->setMinificationFilter(QOpenGLTexture::Linear);
        texture_->setMagnificationFilter(QOpenGLTexture::Linear);
        texture_->setWrapMode(QOpenGLTexture::ClampToEdge);
        texture_->setMipLevels(1);
        texture_->setAutoMipMapGenerationEnabled(false);
        texture_->setSize(w, h);
        texture_->allocateStorage(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8);
    } else {
        texture_->bind();
    }

    // Upload (respect stride)
    QOpenGLPixelTransferOptions xfer;
    xfer.setAlignment(1);
    const int rowPixels = img.bytesPerLine() / 4;
    texture_->setData(QOpenGLTexture::RGBA,
                      QOpenGLTexture::UInt8,
                      img.constBits(),
                      (rowPixels != w) ? &xfer : nullptr);

    texture_->release();
}

void GLVisualisation::paintGL()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glClear(GL_COLOR_BUFFER_BIT);

    ensureTexture();
    if (!texture_ || imgW_ <= 0 || imgH_ <= 0 || vpW_ <= 0 || vpH_ <= 0) return;

    // Ratios of image->widget in each axis (in framebuffer pixels)
    const float rx = float(imgW_) / float(vpW_);
    const float ry = float(imgH_) / float(vpH_);

    // Uniform scale that never enlarges: if image is bigger, scale down so max(rx,ry)==1
    // if image is smaller, k == 1 (no upscaling)
    const float k  = 1.0f / std::max(1.0f, std::max(rx, ry));

    // Final NDC scales: 1.0 means full widget, <1.0 means smaller (centered)
    const float sx = rx * k;
    const float sy = ry * k;

    QMatrix4x4 mvp; 
    mvp.setToIdentity();
    mvp.scale(sx, sy, 1.0f);  // centered because base quad is at origin

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
}

void GLVisualisation::ShowImage(const QImage &image)
{
    this->setImage(image);
}

void GLVisualisation::mousePressEvent(QMouseEvent* e) {
    if (widgetClickable)
    {
        emit clickedWidget(e->pos());
        if (auto p = widgetToImage(e->position()); p.has_value())
            emit clickedImage(*p);
    }
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

