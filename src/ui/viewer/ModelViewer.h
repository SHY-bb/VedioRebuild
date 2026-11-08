#pragma once
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QVector3D>
#include <QString>
#include <vector>

/// 简易 3D 查看器 — 支持三角网格与 3D Gaussian Splatting
class ModelViewer : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT
public:
    explicit ModelViewer(QWidget *parent = nullptr);

    /// 加载网格数据 (顶点 + 三角面)
    void loadMesh(const std::vector<float> &vertices,
                  const std::vector<unsigned int> &indices);
    void clearMesh();

    /// 加载 3DGS 高斯 (标准 3DGS PLY)
    bool loadGaussians(const QString &plyPath);
    void clearGaussians();
    void setShowGaussians(bool on) { showGaussians_ = on; update(); }
    bool hasGaussians() const { return gaussCount_ > 0; }
    int gaussianCount() const { return gaussCount_; }

    /// 自动适配视角
    void resetView();

    void setWireframe(bool on) { wireframe_ = on; update(); }
    void setBackgroundColor(const QColor &c) { bgColor_ = c; update(); }

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent *e) override;
    void mouseMoveEvent(QMouseEvent *e) override;
    void wheelEvent(QWheelEvent *e) override;

private:
    void compileShaders();
    void compileSplatShaders();
    QVector3D computeCentroid();
    void updateSceneBounds();
    void drawMesh();
    void drawGaussians();
    bool parseGaussianPly(const QString &path);
    void loadCameraHint(const QString &plyPath);

    // 网格数据
    std::vector<float> vertices_;
    std::vector<unsigned int> indices_;
    bool meshLoaded_ = false;

    // 高斯数据 (CPU, 用于排序)
    std::vector<float> gPos_;      // 3 * N
    std::vector<float> gAttr_;     // 16 * N (GPU 属性: pos/opacity, scale, quat, rgb)
    std::vector<float> gDepth_;    // N
    std::vector<unsigned int> gOrder_;
    int gaussCount_ = 0;
    bool showGaussians_ = true;

    // 训练相机 (可选, 来自 cameras.json)
    bool hasStartCam_ = false;
    QVector3D startCam_{0, 0, 0};
    QMatrix3x3 startRot_;      // 训练相机 world->camera 旋转
    bool hasStartRot_ = false;
    float rotationZ_ = 0;      // 滚转 (对齐训练相机的 up)
    float fovY_ = 45.0f;       // 垂直视场角 (有训练内参时按内参算)

    // 场景范围 (网格或高斯共用)
    QVector3D sceneCenter_{0, 0, 0};
    float sceneRadius_ = 1.0f;

    // 渲染 — Qt OpenGL wrappers
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer ebo_{QOpenGLBuffer::IndexBuffer};
    QOpenGLShaderProgram *shader_ = nullptr;

    // 高斯渲染资源
    QOpenGLVertexArrayObject splatVao_;
    QOpenGLBuffer splatAttrBuf_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer splatIdxBuf_{QOpenGLBuffer::IndexBuffer};
    QOpenGLShaderProgram *splatShader_ = nullptr;
    unsigned int splatAttrTex_ = 0;
    unsigned int splatOrderTex_ = 0;

    QMatrix4x4 model_, view_, projection_;

    // 交互
    bool wireframe_ = false;
    QColor bgColor_{40, 40, 45};
    QPoint lastMousePos_;
    float rotationX_ = -30.0f, rotationY_ = 30.0f;
    float zoom_ = 1.0f;
    QVector3D translation_{0, 0, 0};
    QVector3D centroid_{0, 0, 0};
};
