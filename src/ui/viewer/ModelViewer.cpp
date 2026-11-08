#include "ModelViewer.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QOpenGLContext>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

// ============================================================ 网格着色器
static const char *vertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uModel, uView, uProjection;
out vec3 vNormal, vFragPos;
void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vFragPos = wp.xyz;
    vNormal  = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uProjection * uView * wp;
}
)";

static const char *fragmentShaderSrc = R"(
#version 330 core
in vec3 vNormal, vFragPos;
uniform vec3 uLightDir, uViewPos, uColor;
uniform bool uWireframe;
out vec4 FragColor;
void main() {
    if (uWireframe) { FragColor = vec4(1,1,1,1); return; }
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uViewPos - vFragPos);
    vec3 H = normalize(L + V);
    float amb = 0.3, diff = max(dot(N,L),0.0)*0.5;
    float spec = pow(max(dot(N,H),0.0), 32.0)*0.3;
    FragColor = vec4((amb+diff)*uColor + spec*vec3(1), 1.0);
}
)";

// ============================================================ 高斯泼溅着色器
// 属性纹理: 每个高斯 16 float
//   texel0 = (px,py,pz,opacity)
//   texel1 = (sx,sy,sz,0)
//   texel2 = (qw,qx,qy,qz)
//   texel3 = (r,g,b,0)
static const char *splatVertexSrc = R"(
#version 330 core
uniform samplerBuffer uAttr;
uniform usamplerBuffer uOrder;
uniform mat4 uModel, uView, uProjection;
uniform vec2 uViewport;

out vec4 vColor;
out vec3 vConic;
out vec2 vCenter;

mat3 quatToMat(vec4 q) {
    float w = q.x, x = q.y, y = q.z, z = q.w;
    return mat3(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y),
        2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),
        2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y)
    );
}

void main() {
    const vec2 kCorner[6] = vec2[6](
        vec2(-1,-1), vec2(1,-1), vec2(-1,1),
        vec2(-1,1),  vec2(1,-1), vec2(1,1));
    int gi = int(texelFetch(uOrder, gl_VertexID / 6).r);
    vec2 aCorner = kCorner[gl_VertexID % 6];
    int base = gi * 4;
    vec4 t0 = texelFetch(uAttr, base + 0);
    vec4 t1 = texelFetch(uAttr, base + 1);
    vec4 t2 = texelFetch(uAttr, base + 2);
    vec4 t3 = texelFetch(uAttr, base + 3);

    vec3 pos   = t0.xyz;
    float opac = t0.w;
    vec3 scale = t1.xyz;
    vec4 quat  = t2;
    vec3 rgb   = t3.rgb;

    vec4 vpos = uView * uModel * vec4(pos, 1.0);
    vec4 clip = uProjection * vpos;
    if (vpos.z > -0.01 || clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // 裁掉
        vColor = vec4(0.0); vConic = vec3(0.0); vCenter = vec2(0.0);
        return;
    }

    // NDC -> 像素 (GL 坐标, y 向上)
    vec2 centerPx = (clip.xy / clip.w * 0.5 + 0.5) * uViewport;

    // 投影雅可比 (像素空间)
    float z = -vpos.z;
    float fx = uProjection[0][0] * uViewport.x * 0.5;
    float fy = uProjection[1][1] * uViewport.y * 0.5;
    mat3 J = mat3(fx / z, 0.0, -fx * vpos.x / (z * z),
                  0.0, fy / z, -fy * vpos.y / (z * z),
                  0.0, 0.0, 0.0);

    mat3 R = quatToMat(quat);
    mat3 S = mat3(scale.x, 0.0, 0.0,
                  0.0, scale.y, 0.0,
                  0.0, 0.0, scale.z);
    mat3 Sigma = R * S * transpose(S) * transpose(R);

    mat3 W = mat3(uView) * mat3(uModel);
    mat3 T = J * W;
    mat3 cov = T * Sigma * transpose(T);

    float a = cov[0][0] + 0.3;
    float b = cov[0][1];
    float c = cov[1][1] + 0.3;
    float det = a * c - b * b;
    if (det <= 1e-9) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vColor = vec4(0.0); vConic = vec3(0.0); vCenter = vec2(0.0);
        return;
    }
    vConic = vec3(c, -b, a) / det;

    float mid = 0.5 * (a + c);
    float disc = sqrt(max(0.1, mid * mid - det));
    float lambda1 = mid + disc;
    float rTrue = 3.0 * sqrt(max(lambda1, 0.0));
    float radius = clamp(ceil(rTrue), 1.0, 256.0);
    // 半径被截断时按面积补偿不透明度, 否则超大高斯会把整屏糊平
    float comp = (rTrue > 256.0) ? (256.0 / rTrue) * (256.0 / rTrue) : 1.0;
    opac *= comp;

    vec2 offset = aCorner * radius;
    vec2 px = centerPx + offset;
    vec2 ndc = px / uViewport * 2.0 - 1.0;

    vCenter = centerPx;
    vColor = vec4(rgb, opac);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

static const char *splatFragmentSrc = R"(
#version 330 core
in vec4 vColor;
in vec3 vConic;
in vec2 vCenter;
out vec4 FragColor;
void main() {
    vec2 d = gl_FragCoord.xy - vCenter;
    float power = -0.5 * (vConic.x * d.x * d.x +
                          2.0 * vConic.y * d.x * d.y +
                          vConic.z * d.y * d.y);
    if (power > 0.0) discard;
    float alpha = vColor.a * exp(power);
    if (alpha < 1.0 / 255.0) discard;
    FragColor = vec4(vColor.rgb * alpha, alpha);   // 预乘 alpha
}
)";

static const float SH_C0 = 0.28209479177387814f;

ModelViewer::ModelViewer(QWidget *parent) : QOpenGLWidget(parent) {
    setMinimumSize(400, 300);
    setFocusPolicy(Qt::StrongFocus);
}

void ModelViewer::loadMesh(const std::vector<float> &verts,
                            const std::vector<unsigned int> &inds) {
    vertices_ = verts;
    indices_ = inds;
    meshLoaded_ = true;

    makeCurrent();
    if (getenv("WR_DEBUG_VIEW"))
        fprintf(stderr, "[load] ctx=%p vao=%d vbo=%d verts=%zu inds=%zu\n",
                (void *)context(), (int)vao_.isCreated(), (int)vbo_.isCreated(),
                vertices_.size(), indices_.size());
    vao_.bind();
    vbo_.bind();
    vbo_.allocate(vertices_.data(),
                  static_cast<int>(vertices_.size() * sizeof(float)));
    ebo_.bind();
    ebo_.allocate(indices_.data(),
                  static_cast<int>(indices_.size() * sizeof(unsigned int)));
    vao_.release();
    doneCurrent();

    centroid_ = computeCentroid();
    updateSceneBounds();
    update();
}

void ModelViewer::clearMesh() {
    vertices_.clear();
    indices_.clear();
    meshLoaded_ = false;
    update();
}

void ModelViewer::clearGaussians() {
    gaussCount_ = 0;
    gPos_.clear();
    gAttr_.clear();
    gDepth_.clear();
    gOrder_.clear();
    update();
}

QVector3D ModelViewer::computeCentroid() {
    if (vertices_.empty()) return {0, 0, 0};
    QVector3D c;
    for (size_t i = 0; i + 2 < vertices_.size(); i += 6)
        c += QVector3D(vertices_[i], vertices_[i + 1], vertices_[i + 2]);
    return c / static_cast<float>(vertices_.size() / 6);
}

void ModelViewer::updateSceneBounds() {
    // 用分位数估计场景范围, 避免稀疏点离群值把包围盒撑大
    std::vector<float> px, py, pz;
    const size_t approx = vertices_.size() / 6 + gaussCount_;
    const size_t stride = std::max<size_t>(1, approx / 150000);
    px.reserve(approx / stride + 1);
    py.reserve(approx / stride + 1);
    pz.reserve(approx / stride + 1);

    size_t k = 0;
    for (size_t i = 0; i + 2 < vertices_.size(); i += 6, ++k) {
        if (k % stride) continue;
        px.push_back(vertices_[i]);
        py.push_back(vertices_[i + 1]);
        pz.push_back(vertices_[i + 2]);
    }
    for (int i = 0; i < gaussCount_; ++i) {
        if (static_cast<size_t>(i) % stride) continue;
        px.push_back(gPos_[i * 3]);
        py.push_back(gPos_[i * 3 + 1]);
        pz.push_back(gPos_[i * 3 + 2]);
    }
    if (getenv("WR_DEBUG_VIEW")) {
        float mnx = 1e9f, mxx = -1e9f, mny = 1e9f, mxy = -1e9f;
        for (size_t i = 0; i < px.size(); ++i) {
            mnx = std::min(mnx, px[i]); mxx = std::max(mxx, px[i]);
            mny = std::min(mny, py[i]); mxy = std::max(mxy, py[i]);
        }
        fprintf(stderr, "[bounds] n=%zu x[%.2f,%.2f] y[%.2f,%.2f] first=(%.2f,%.2f,%.2f)\n",
                px.size(), mnx, mxx, mny, mxy,
                vertices_.size() > 2 ? vertices_[0] : 0.f,
                vertices_.size() > 2 ? vertices_[1] : 0.f,
                vertices_.size() > 2 ? vertices_[2] : 0.f);
    }
    if (px.empty()) { sceneCenter_ = {0, 0, 0}; sceneRadius_ = 1.0f; return; }

    // 取 10%~90% 分位构成包围盒: 对离群高斯稳健, 又能把主体放满画面
    auto pct = [](std::vector<float> v, float q) {
        if (v.empty()) return 0.f;
        const size_t k = std::min(v.size() - 1, size_t(v.size() * q));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    };
    const QVector3D lo(pct(px, 0.10f), pct(py, 0.10f), pct(pz, 0.10f));
    const QVector3D hi(pct(px, 0.90f), pct(py, 0.90f), pct(pz, 0.90f));
    sceneCenter_ = (lo + hi) * 0.5f;
    sceneRadius_ = std::max((hi - lo).length() * 0.62f, 1e-4f);
}

void ModelViewer::resetView() {
    translation_ = {0, 0, 0};
    if (hasStartCam_) {
        const QVector3D d = startCam_ - sceneCenter_;
        const float distWorld = d.length();

        if (hasStartRot_) {
            // 世界->相机的完整姿态: 行 = (right, -down, -forward)
            // 相机坐标系 (COLMAP): x 右, y 下, z 朝前
            const QVector3D rightW(startRot_(0, 0), startRot_(0, 1), startRot_(0, 2));
            const QVector3D downW (startRot_(1, 0), startRot_(1, 1), startRot_(1, 2));
            const QVector3D fwdW  (startRot_(2, 0), startRot_(2, 1), startRot_(2, 2));
            const QVector3D m0 = rightW;
            const QVector3D m1 = -downW;
            const QVector3D m2 = -fwdW;

            // M = Rz(roll) * Rx(pitch) * Ry(yaw) 的欧拉角反解
            const float cp = std::hypot(m2.z(), m2.x());
            rotationX_ = float(qRadiansToDegrees(std::asin(std::clamp(m2.y(), -1.0f, 1.0f))));
            rotationY_ = float(qRadiansToDegrees(std::atan2(-m2.x(), m2.z())));
            rotationZ_ = (cp > 1e-5f)
                ? float(qRadiansToDegrees(std::atan2(-m0.y(), m1.y()))) : 0.0f;
        } else {
            const float rho = std::hypot(d.x(), d.z());
            rotationY_ = -float(qRadiansToDegrees(std::atan2(d.x(), d.z())));
            rotationX_ = float(qRadiansToDegrees(std::atan2(d.y(), rho)));
            rotationZ_ = 0.0f;
        }

        zoom_ = 2.8f * sceneRadius_ / std::max(distWorld, 1e-6f);
        zoom_ = std::clamp(zoom_, 0.05f, 20.0f);
        if (getenv("WR_DEBUG_VIEW"))
            fprintf(stderr, "[cam] rot=(%.1f,%.1f,%.1f) zoom=%.3f dist=%.2f R=%.2f\n",
                    rotationX_, rotationY_, rotationZ_, zoom_, distWorld,
                    sceneRadius_);
    } else {
        rotationX_ = -30; rotationY_ = 30; rotationZ_ = 0; zoom_ = 1;
    }
    update();
}

void ModelViewer::loadCameraHint(const QString &plyPath) {
    hasStartCam_ = false;
    const QString cj = QFileInfo(plyPath).dir().filePath("cameras.json");
    QFile f(cj);
    if (!f.open(QIODevice::ReadOnly)) return;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    const QJsonArray arr = doc.object().value("cameras").toArray();
    if (arr.isEmpty()) return;

    std::vector<QVector3D> cs;
    cs.reserve(arr.size());
    for (const auto &v : arr) {
        const QJsonArray p = v.toArray();
        if (p.size() < 3) continue;
        cs.emplace_back(float(p[0].toDouble()), float(p[1].toDouble()),
                        float(p[2].toDouble()));
    }
    if (cs.empty()) return;

    const QJsonArray rots = doc.object().value("rotations").toArray();
    const QJsonArray ints = doc.object().value("intrinsics").toArray();

    // 选最靠近相机中位位置的那台, 视觉上最稳
    std::vector<float> xs, ys, zs;
    for (const auto &c : cs) { xs.push_back(c.x()); ys.push_back(c.y()); zs.push_back(c.z()); }
    auto med = [](std::vector<float> v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    int pick = 0;
    if (!getenv("WR_CAM0")) {
        const QVector3D m(med(xs), med(ys), med(zs));
        float best = std::numeric_limits<float>::max();
        for (size_t i = 0; i < cs.size(); ++i) {
            const float d = (cs[i] - m).lengthSquared();
            if (d < best) { best = d; pick = int(i); }
        }
    }
    startCam_ = cs[size_t(pick)];
    hasStartCam_ = true;

    if (pick < ints.size()) {
        const QJsonArray k = ints[pick].toArray();
        if (k.size() >= 6 && k[1].toDouble() > 1.0) {
            const float fy = float(k[1].toDouble());
            const float hh = float(k[5].toDouble());
            fovY_ = float(qRadiansToDegrees(2.0 * std::atan(hh / (2.0 * fy))));
        }
    }
    if (pick < rots.size()) {
        const QJsonArray r = rots[pick].toArray();
        if (r.size() == 9) {
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    startRot_(row, col) = float(r[row * 3 + col].toDouble());
            hasStartRot_ = true;
        }
    }
}

void ModelViewer::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(bgColor_.redF(), bgColor_.greenF(), bgColor_.blueF(), 1.0f);
    glEnable(GL_DEPTH_TEST);

    // ---- 网格 VAO
    vao_.create();
    vao_.bind();
    vbo_.create();
    vbo_.bind();
    vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    vbo_.allocate(nullptr, 0);
    ebo_.create();
    ebo_.bind();
    ebo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    ebo_.allocate(nullptr, 0);
    int stride = 6 * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    vao_.release();

    compileShaders();

    // ---- 高斯 VAO
    splatVao_.create();
    splatVao_.bind();

    splatAttrBuf_.create();
    splatAttrBuf_.bind();
    splatAttrBuf_.setUsagePattern(QOpenGLBuffer::StaticDraw);

    splatIdxBuf_.create();
    splatIdxBuf_.bind();
    splatIdxBuf_.setUsagePattern(QOpenGLBuffer::DynamicDraw);

    splatVao_.release();

    glGenTextures(1, &splatAttrTex_);
    glGenTextures(1, &splatOrderTex_);
    compileSplatShaders();
}

void ModelViewer::compileShaders() {
    shader_ = new QOpenGLShaderProgram(this);
    shader_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSrc);
    shader_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSrc);
    if (!shader_->link())
        fprintf(stderr, "[viewer] 网格着色器链接失败: %s\n",
                qPrintable(shader_->log()));
}

void ModelViewer::compileSplatShaders() {
    splatShader_ = new QOpenGLShaderProgram(this);
    splatShader_->addShaderFromSourceCode(QOpenGLShader::Vertex, splatVertexSrc);
    splatShader_->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                          splatFragmentSrc);
    if (!splatShader_->link())
        fprintf(stderr, "[viewer] 高斯着色器链接失败: %s\n",
                qPrintable(splatShader_->log()));
}

void ModelViewer::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    projection_.setToIdentity();
    projection_.perspective(fovY_, float(w) / std::max(h, 1), 0.01f, 200.0f);
}

void ModelViewer::paintGL() {
    // Qt 合成时可能留下 scissor/颜色掩码等状态, 不重置会导致 glClear 只清局部,
    // 上一帧的窗口内容 (比如字体图集) 就会残留在背景里
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glClearColor(0.157f, 0.157f, 0.176f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    projection_.setToIdentity();
    projection_.perspective(fovY_, float(width()) / std::max(height(), 1), 0.01f, 200.0f);

    const bool hasG = (gaussCount_ > 0 && showGaussians_ && splatShader_ &&
                       splatShader_->isLinked());
    if (!hasG && (!meshLoaded_ || !shader_ || !shader_->isLinked()))
        return;

    // ---- 相机: 把场景缩放到单位球, 固定距离观察
    float invRadius = 1.0f / std::max(sceneRadius_, 1e-6f);
    model_.setToIdentity();
    model_.translate(translation_);
    model_.rotate(rotationX_, 1, 0, 0);
    model_.rotate(rotationY_, 0, 1, 0);
    model_.rotate(rotationZ_, 0, 0, 1);
    model_.scale(invRadius);
    model_.translate(-sceneCenter_);

    view_.setToIdentity();
    const float dist = 2.8f / std::max(zoom_, 0.01f);
    view_.lookAt({0, 0, dist}, {0, 0, 0}, {0, 1, 0});

    if (getenv("WR_DEBUG_VIEW"))
        fprintf(stderr, "[view] mesh=%d inds=%zu gauss=%d radius=%.3f center=(%.2f,%.2f,%.2f) vao=%d vbo=%d\n",
                (int)meshLoaded_, indices_.size(), gaussCount_, sceneRadius_,
                sceneCenter_.x(), sceneCenter_.y(), sceneCenter_.z(),
                (int)vao_.isCreated(), (int)vbo_.isCreated());

    if (hasG)
        drawGaussians();
    else
        drawMesh();
}

void ModelViewer::drawMesh() {
    shader_->bind();
    shader_->setUniformValue("uModel", model_);
    shader_->setUniformValue("uView", view_);
    shader_->setUniformValue("uProjection", projection_);
    shader_->setUniformValue("uLightDir", QVector3D(0.5f, 1.0f, 0.8f));
    shader_->setUniformValue("uViewPos", QVector3D(0, 0, 5));
    shader_->setUniformValue("uColor", QVector3D(0.4f, 0.7f, 0.9f));
    shader_->setUniformValue("uWireframe", wireframe_);

    QOpenGLVertexArrayObject::Binder bind(&vao_);
    if (wireframe_) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, static_cast<int>(indices_.size()),
                   GL_UNSIGNED_INT, nullptr);
    if (wireframe_) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    shader_->release();
}

void ModelViewer::drawGaussians() {
    // ---- 深度排序 (远 → 近)
    const QMatrix4x4 mv = view_ * model_;
    const float *m = mv.constData();
    for (int i = 0; i < gaussCount_; ++i) {
        const float x = gPos_[i * 3], y = gPos_[i * 3 + 1], z = gPos_[i * 3 + 2];
        gDepth_[i] = m[2] * x + m[6] * y + m[10] * z + m[14];
    }
    std::sort(gOrder_.begin(), gOrder_.end(),
              [this](unsigned int a, unsigned int b) {
                  return gDepth_[a] < gDepth_[b];   // 越小越远
              });

    // 先 orphan 再写: 否则上一帧还可能在该缓冲上绘制, 会导致同一个高斯的
    // 6 个顶点读到不同索引 (表现为满屏拖影/错位色块)
    splatIdxBuf_.bind();
    const int orderBytes = static_cast<int>(gOrder_.size() * sizeof(unsigned int));
    splatIdxBuf_.allocate(nullptr, orderBytes);
    splatIdxBuf_.write(0, gOrder_.data(), orderBytes);

    // ---- 绑定属性纹理 (绘制后恢复 GL 状态, 避免破坏 Qt 的字体纹理)
    GLint prevActive = GL_TEXTURE0, prevAttrTex = 0, prevOrderTex = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActive);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &prevAttrTex);
    splatAttrBuf_.bind();
    glBindTexture(GL_TEXTURE_BUFFER, splatAttrTex_);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, splatAttrBuf_.bufferId());

    glActiveTexture(GL_TEXTURE1);
    glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &prevOrderTex);
    splatIdxBuf_.bind();
    glBindTexture(GL_TEXTURE_BUFFER, splatOrderTex_);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, splatIdxBuf_.bufferId());
    glActiveTexture(GL_TEXTURE0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    splatShader_->bind();
    splatShader_->setUniformValue("uModel", model_);
    splatShader_->setUniformValue("uView", view_);
    splatShader_->setUniformValue("uProjection", projection_);
    splatShader_->setUniformValue("uViewport",
                                  QVector2D(float(width()), float(height())));
    splatShader_->setUniformValue("uAttr", 0);
    splatShader_->setUniformValue("uOrder", 1);

    {
        QOpenGLVertexArrayObject::Binder bind(&splatVao_);
        glDrawArrays(GL_TRIANGLES, 0, 6 * gaussCount_);
    }

    splatShader_->release();
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, static_cast<GLuint>(prevOrderTex));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, static_cast<GLuint>(prevAttrTex));
    glActiveTexture(static_cast<GLenum>(prevActive));
}

bool ModelViewer::loadGaussians(const QString &plyPath) {
    if (!parseGaussianPly(plyPath)) return false;

    makeCurrent();
    splatAttrBuf_.bind();
    splatAttrBuf_.allocate(gAttr_.data(),
                           static_cast<int>(gAttr_.size() * sizeof(float)));
    splatIdxBuf_.bind();
    splatIdxBuf_.allocate(gOrder_.data(),
                          static_cast<int>(gOrder_.size() *
                                           sizeof(unsigned int)));
    splatAttrBuf_.release();
    doneCurrent();

    loadCameraHint(plyPath);
    if (gPos_.size() >= 6)
        centroid_ = QVector3D(gPos_[0], gPos_[1], gPos_[2]);
    updateSceneBounds();
    showGaussians_ = true;
    update();
    return true;
}

bool ModelViewer::parseGaussianPly(const QString &path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;

    // ---- 解析头部
    int vertexCount = 0;
    bool binaryLE = false;
    std::vector<QString> props;
    while (!f.atEnd()) {
        const QByteArray line = f.readLine().trimmed();
        if (line.startsWith("format")) {
            if (line.contains("binary_little_endian")) binaryLE = true;
        } else if (line.startsWith("element vertex")) {
            vertexCount = line.mid(QString("element vertex").size()).trimmed().toInt();
        } else if (line.startsWith("element")) {
            // 只读顶点
        } else if (line.startsWith("property float")) {
            props.push_back(QString::fromLatin1(line.mid(14).trimmed()));
        } else if (line == "end_header") {
            break;
        }
    }
    if (!binaryLE || vertexCount <= 0 || props.empty()) {
        fprintf(stderr, "[viewer] 仅支持 binary_little_endian PLY\n");
        return false;
    }

    const int nProps = static_cast<int>(props.size());
    const int stride = nProps * 4;
    std::vector<char> raw(static_cast<size_t>(vertexCount) * stride);
    if (f.read(raw.data(), qint64(raw.size())) != qint64(raw.size())) {
        fprintf(stderr, "[viewer] PLY 数据不完整\n");
        return false;
    }
    f.close();

    auto idx = [&](const char *name) {
        for (int i = 0; i < nProps; ++i)
            if (props[i] == QLatin1String(name)) return i;
        return -1;
    };
    const int ix = idx("x"), iy = idx("y"), iz = idx("z");
    const int ir = idx("f_dc_0"), ig = idx("f_dc_1"), ib = idx("f_dc_2");
    const int io = idx("opacity");
    const int is0 = idx("scale_0"), is1 = idx("scale_1"), is2 = idx("scale_2");
    const int iq0 = idx("rot_0"), iq1 = idx("rot_1");
    const int iq2 = idx("rot_2"), iq3 = idx("rot_3");
    if (ix < 0 || iy < 0 || iz < 0 || io < 0 || is0 < 0 || iq0 < 0) {
        fprintf(stderr, "[viewer] PLY 缺少 3DGS 属性\n");
        return false;
    }

    gPos_.assign(static_cast<size_t>(vertexCount) * 3, 0.f);
    gAttr_.assign(static_cast<size_t>(vertexCount) * 16, 0.f);
    gDepth_.assign(vertexCount, 0.f);
    gOrder_.resize(vertexCount);

    for (int i = 0; i < vertexCount; ++i) {
        const float *v = reinterpret_cast<const float *>(
            raw.data() + static_cast<size_t>(i) * stride);
        const float px = (ix >= 0) ? v[ix] : 0.f;
        const float py = (iy >= 0) ? v[iy] : 0.f;
        const float pz = (iz >= 0) ? v[iz] : 0.f;
        const float op = 1.f / (1.f + std::exp(-v[io]));
        const float sx = std::exp(v[is0]);
        const float sy = std::exp(v[is1]);
        const float sz = std::exp(v[is2]);

        float qw = v[iq0], qx = (iq1 >= 0) ? v[iq1] : 0.f;
        float qy = (iq2 >= 0) ? v[iq2] : 0.f, qz = (iq3 >= 0) ? v[iq3] : 0.f;
        const float qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (qn > 1e-8f) { qw /= qn; qx /= qn; qy /= qn; qz /= qn; }

        float r = 0.5f, gg = 0.5f, b = 0.5f;
        if (ir >= 0) r = std::clamp(0.5f + SH_C0 * v[ir], 0.f, 1.f);
        if (ig >= 0) gg = std::clamp(0.5f + SH_C0 * v[ig], 0.f, 1.f);
        if (ib >= 0) b = std::clamp(0.5f + SH_C0 * v[ib], 0.f, 1.f);

        gPos_[i * 3] = px; gPos_[i * 3 + 1] = py; gPos_[i * 3 + 2] = pz;

        float *a = &gAttr_[static_cast<size_t>(i) * 16];
        a[0] = px; a[1] = py; a[2] = pz; a[3] = op;
        a[4] = sx; a[5] = sy; a[6] = sz; a[7] = 0.f;
        a[8] = qw; a[9] = qx; a[10] = qy; a[11] = qz;
        a[12] = r; a[13] = gg; a[14] = b; a[15] = 0.f;

        gOrder_[i] = static_cast<unsigned int>(i);
    }

    gaussCount_ = vertexCount;
    fprintf(stderr, "[viewer] 已加载 %d 个高斯: %s\n", gaussCount_,
            qPrintable(path));
    return true;
}

void ModelViewer::mousePressEvent(QMouseEvent *e) { lastMousePos_ = e->pos(); }

void ModelViewer::mouseMoveEvent(QMouseEvent *e) {
    QPoint d = e->pos() - lastMousePos_;
    if (e->buttons() & Qt::LeftButton) {
        rotationY_ += d.x() * 0.5f;
        rotationX_ += d.y() * 0.5f;
    } else if (e->buttons() & Qt::MiddleButton) {
        translation_ += QVector3D(d.x() * 0.01f, -d.y() * 0.01f, 0);
    }
    lastMousePos_ = e->pos();
    update();
}

void ModelViewer::wheelEvent(QWheelEvent *e) {
    zoom_ *= (e->angleDelta().y() > 0) ? 1.1f : 0.9f;
    update();
}
