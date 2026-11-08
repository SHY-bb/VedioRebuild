#pragma once
#include <QObject>
#include <QProcess>
#include <QString>

/// 管线执行器 — 调用 Python 重建脚本 + 管理流程
class PipelineRunner : public QObject {
    Q_OBJECT
public:
    enum Stage {
        Idle = 0,
        ExtractingFrames,
        RunningSfM,
        RunningMvs,
        BuildingMesh,
        TrainingGS,
        Exporting,
        Finished,
        Failed
    };
    Q_ENUM(Stage)

    /// 重建模式
    enum class Mode {
        Standard,   ///< 旧版 pycolmap (CPU MVS, 快速预览)
        Dense,      ///< COLMAP CUDA PatchMatch 稠密 MVS (百万点)
        DepthTsdf   ///< DepthAnything V3 深度 + Open3D TSDF
    };
    Q_ENUM(Mode)

    /// SfM 接口模式（先选路线，再选具体引擎）
    enum class SfmInterface {
        Geometric,  ///< 几何优化：显式匹配 + 三角化 + BA（COLMAP / GLOMAP）
        Neural      ///< 神经前馈：端到端回归相机/深度（VGGT 已接入）
    };
    Q_ENUM(SfmInterface)

    /// SfM 引擎（隶属于某个接口模式）
    enum class SfmEngine {
        Incremental,  ///< COLMAP 增量式 mapper（逐张注册，鲁棒）
        Glomap,       ///< GLOMAP 全局式（一次解全部位姿，快，适合大规模）
        Vggt,         ///< VGGT 前馈式（已接入，4GB 低显存）
        VggtOmega     ///< VGGT-Ω 前馈式（需 >=6GB + gated 权重，暂置灰）
    };
    Q_ENUM(SfmEngine)

    explicit PipelineRunner(QObject *parent = nullptr);

    // ── 可配置路径解析 ──────────────────────────────────
    // 优先级: 显式调用 setter > 环境变量 > 编译期默认值。
    // 所有缺省值都可用 README「配置项」中的环境变量覆盖。

    /// Python 虚拟环境根目录 (WR_PYTHON_ENV, 默认 /opt/sam-venv)
    static QString defaultPythonEnv();
    /// 传给子进程的附加库搜索路径 (WR_EXTRA_LIB_PATH)
    static QString defaultExtraLibPath();
    /// 脚本目录 (WR_SCRIPTS_DIR, 否则自动探测 build/ 与 install/ 布局)
    static QString resolveScriptDir();
    /// 模型目录 (WR_MODELS_DIR, 否则自动探测)
    static QString resolveModelDir();

    /// 设置 Python 虚拟环境和脚本路径
    void setPythonEnv(const QString &venvPath) { venvPath_ = venvPath; }
    void setScript(const QString &scriptPath) { scriptPath_ = scriptPath; }
    /// 新脚本目录 (reconstruct_dense.py / depth_tsdf.py / train_3dgs.py)
    void setScriptDir(const QString &dir) { scriptDir_ = dir; }
    void setMode(Mode m) { mode_ = m; }
    void setSfmInterface(SfmInterface i) { sfmInterface_ = i; }
    SfmInterface sfmInterface() const { return sfmInterface_; }
    void setSfmEngine(SfmEngine e) { sfmEngine_ = e; }
    SfmEngine sfmEngine() const { return sfmEngine_; }
    void setDenseImageSize(int px) { denseImageSize_ = px; }
    void setGsIterations(int iters) { gsIters_ = iters; }

    /// 启动完整流程
    void run(const QString &videoPath, const QString &outputDir,
             int fps = 2, int maxSize = 1600);

    /// 质量提升: 基于已有 COLMAP 结果训练 3DGS
    void runQualityEnhancement(const QString &outputDir);

    /// 中止当前流程
    void cancel();

    Stage currentStage() const { return stage_; }
    QString lastError() const { return error_; }

signals:
    void stageChanged(Stage stage, const QString &message);
    void progressUpdate(int percent, const QString &detail);
    void finished(bool success, const QString &outputPath);
    void consoleOutput(const QString &line);

private slots:
    void onProcessOutput();
    void onProcessFinished(int exitCode, QProcess::ExitStatus status);

private:
    void startProcess(const QStringList &args);
    QString findColmapModelDir(const QString &outputDir) const;
    Stage stage_ = Idle;
    Mode mode_ = Mode::Dense;
    SfmInterface sfmInterface_ = SfmInterface::Geometric;
    SfmEngine sfmEngine_ = SfmEngine::Incremental;
    QString venvPath_;
    QString scriptPath_;
    QString scriptDir_;
    QString error_;
    int denseImageSize_ = 1000;
    int gsIters_ = 7000;
    QProcess *process_ = nullptr;
};
