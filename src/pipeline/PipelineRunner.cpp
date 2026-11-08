#include "PipelineRunner.h"
#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <functional>
#include <iostream>

namespace {

/// 环境变量优先，空串视为未设置
QString envOr(const char *key, const QString &fallback) {
    const QString v = qEnvironmentVariable(key).trimmed();
    return v.isEmpty() ? fallback : v;
}

/// 在若干候选目录中挑第一个满足条件的
QString firstMatchingDir(const QStringList &candidates,
                         const std::function<bool(const QString &)> &accept) {
    for (const QString &c : candidates) {
        if (accept(c)) return c;
    }
    return {};
}

/// 可执行文件所在目录的上一级（build/ 或 bin/ 的父目录）
QString appParentDir() {
    return QDir::cleanPath(QDir(QCoreApplication::applicationDirPath()).filePath(".."));
}

QString sfmInterfaceArg(PipelineRunner::SfmInterface i) {
    return i == PipelineRunner::SfmInterface::Neural ? "neural" : "geometric";
}

QString sfmEngineArg(PipelineRunner::SfmEngine e) {
    switch (e) {
    case PipelineRunner::SfmEngine::Glomap: return "glomap";
    case PipelineRunner::SfmEngine::Vggt: return "vggt";
    case PipelineRunner::SfmEngine::VggtOmega: return "vggt_omega";
    case PipelineRunner::SfmEngine::Incremental:
    default: return "incremental";
    }
}

} // namespace

PipelineRunner::PipelineRunner(QObject *parent) : QObject(parent) {}

// ══════════════════════════════════════════════════════════
//  路径解析: 环境变量 > 自动探测 > 编译期默认值
// ══════════════════════════════════════════════════════════

QString PipelineRunner::defaultPythonEnv() {
    return envOr("WR_PYTHON_ENV", "/opt/sam-venv");
}

QString PipelineRunner::defaultExtraLibPath() {
    return envOr("WR_EXTRA_LIB_PATH",
                 "/opt/opencv/4.11.0/lib:/opt/qt/6.8.0/lib:/opt/pcl/1.14.1/lib");
}

QString PipelineRunner::resolveScriptDir() {
    const QString fromEnv = qEnvironmentVariable("WR_SCRIPTS_DIR").trimmed();
    if (!fromEnv.isEmpty()) return fromEnv;

    const QString base = appParentDir();
    const QStringList candidates = {
        base + "/scripts",                        // 源码树: <root>/build/WorldRebuild
        base + "/share/WorldRebuild/scripts",     // 安装:   <prefix>/bin/WorldRebuild
        QCoreApplication::applicationDirPath() + "/scripts",
    };
    const QString found = firstMatchingDir(candidates, [](const QString &d) {
        return QFileInfo::exists(d + "/reconstruct_dense.py");
    });
    if (!found.isEmpty()) return found;

    // 都没找到时按布局给出最可能的路径，方便错误提示
    return QFileInfo(base + "/share/WorldRebuild").isDir()
               ? base + "/share/WorldRebuild/scripts"
               : base + "/scripts";
}

QString PipelineRunner::resolveModelDir() {
    const QString fromEnv = qEnvironmentVariable("WR_MODELS_DIR").trimmed();
    if (!fromEnv.isEmpty()) return fromEnv;

    const QString base = appParentDir();
    const QStringList candidates = {
        base + "/models",                         // 源码树
        base + "/share/WorldRebuild/models",      // 安装
        QCoreApplication::applicationDirPath() + "/models",
    };
    const QString found = firstMatchingDir(candidates, [](const QString &d) {
        return QFileInfo(d + "/DA3-SMALL").isDir() ||
               QFileInfo(d + "/VGGT-1B").isDir();
    });
    if (!found.isEmpty()) return found;

    return QFileInfo(base + "/share/WorldRebuild").isDir()
               ? base + "/share/WorldRebuild/models"
               : base + "/models";
}

void PipelineRunner::run(const QString &videoPath, const QString &outputDir,
                          int fps, int maxSize) {
    if (venvPath_.isEmpty()) {
        error_ = "未设置 Python 环境";
        stage_ = Failed;
        emit stageChanged(Failed, error_);
        emit finished(false, {});
        return;
    }

    QString script;
    QStringList args;

    if (mode_ == Mode::Standard) {
        script = scriptPath_;
        args << script << videoPath << "-o" << outputDir
             << "--fps" << QString::number(fps)
             << "--max-size" << QString::number(maxSize);
        stage_ = RunningSfM;
        emit stageChanged(RunningSfM, "启动重建流程 (标准 CPU MVS)...");
    } else {
        script = scriptDir_ + "/reconstruct_dense.py";
        args << script << videoPath << "-o" << outputDir
             << "--fps" << QString::number(fps)
             << "--max-size" << QString::number(maxSize)
             << "--dense-size" << QString::number(denseImageSize_)
             << "--dense-method"
             << (mode_ == Mode::Dense ? "colmap" : "da3")
             << "--sfm-interface" << sfmInterfaceArg(sfmInterface_)
             << "--sfm" << sfmEngineArg(sfmEngine_);
        stage_ = ExtractingFrames;
        emit stageChanged(ExtractingFrames,
                          mode_ == Mode::Dense
                              ? "启动稠密重建 (COLMAP CUDA PatchMatch)..."
                              : "启动深度重建 (DepthAnything V3 + TSDF)...");
    }

    if (!QFileInfo::exists(script)) {
        error_ = "找不到脚本: " + script;
        stage_ = Failed;
        emit stageChanged(Failed, error_);
        emit finished(false, {});
        return;
    }

    startProcess(args);
}

void PipelineRunner::runQualityEnhancement(const QString &outputDir) {
    if (venvPath_.isEmpty() || scriptDir_.isEmpty()) {
        emit stageChanged(Failed, "未设置 Python 环境或脚本目录");
        return;
    }

    // DA3 模式的 sparse_txt 来自去畸变模型, 图像必须跟着用 undistorted/images,
    // 否则"带畸变图像 + 针孔内参"会让 3DGS 学出来的边缘发虚
    QString imagesDir = outputDir + "/undistorted/images";
    if (!QFileInfo::exists(imagesDir)) imagesDir = outputDir + "/images";
    if (!QFileInfo::exists(imagesDir)) {
        emit stageChanged(Failed, "缺少图像目录: " + imagesDir);
        return;
    }
    QString modelDir = findColmapModelDir(outputDir);
    if (modelDir.isEmpty()) {
        emit stageChanged(Failed, "缺少 COLMAP 模型 (sparse_txt / sparse / sfm)");
        return;
    }

    QString gsOut = outputDir + "/gs";
    QString script = scriptDir_ + "/train_3dgs.py";
    if (!QFileInfo::exists(script)) {
        emit stageChanged(Failed, "找不到脚本: " + script);
        return;
    }

    QStringList args;
    args << script << "--images" << imagesDir
         << "--model" << modelDir
         << "-o" << gsOut
         << "--iters" << QString::number(gsIters_)
         << "--max-size" << "800";

    stage_ = TrainingGS;
    emit stageChanged(TrainingGS, "开始 3DGS 训练 (" +
                                      QString::number(gsIters_) + " 迭代)...");
    startProcess(args);
}

void PipelineRunner::cancel() {
    if (process_ && process_->state() != QProcess::NotRunning) {
        process_->kill();
        process_->waitForFinished(3000);
    }
    stage_ = Idle;
    emit stageChanged(Idle, "已取消");
}

QString PipelineRunner::findColmapModelDir(const QString &outputDir) const {
    // 1. 优先 TXT 模型 (train_3dgs.py 直接可用)
    const QString txt = outputDir + "/sparse_txt";
    if (QFileInfo::exists(txt + "/cameras.txt")) return txt;

    // 2. 稠密管线的 sparse/<N>
    QDir sparse(outputDir + "/sparse");
    if (sparse.exists()) {
        const auto subs = sparse.entryList(QDir::Dirs | QDir::NoDotAndDotDot,
                                           QDir::Name);
        for (const auto &s : subs) {
            const QString p = sparse.filePath(s);
            if (QFileInfo::exists(p + "/cameras.bin") ||
                QFileInfo::exists(p + "/cameras.txt"))
                return p;
        }
    }

    // 3. 旧版 pycolmap 管线 sfm/sparse/0
    const QString legacy = outputDir + "/sfm/sparse/0";
    if (QFileInfo::exists(legacy + "/cameras.bin") ||
        QFileInfo::exists(legacy + "/cameras.txt"))
        return legacy;

    return {};
}

void PipelineRunner::startProcess(const QStringList &args) {
    delete process_;
    process_ = new QProcess(this);

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    const QString pythonEnv = venvPath_.isEmpty() ? defaultPythonEnv() : venvPath_;
    QString pythonBin = pythonEnv + "/bin/python3";
    env.insert("PATH", pythonEnv + "/bin:" + env.value("PATH", ""));

    // 附加库路径: 允许通过 WR_EXTRA_LIB_PATH 覆盖 (为空则不追加)
    const QString extraLib = defaultExtraLibPath();
    if (!extraLib.isEmpty()) {
        env.insert("LD_LIBRARY_PATH",
                   extraLib + ":" + env.value("LD_LIBRARY_PATH", ""));
    }

    // ── 把外部工具 / 权重位置注入子进程 ──────────────────
    // 这些变量 Python 脚本本身也支持；在这里兜底设默认值，
    // 使 GUI 启动时无需依赖 shell 环境。
    const QString home = QDir::homePath();
    const QString models = resolveModelDir();

    const auto setDefault = [&env](const char *key, const QString &value) {
        if (env.value(key).trimmed().isEmpty()) env.insert(key, value);
    };
    setDefault("COLMAP_BIN", home + "/.local/colmap-install/bin/colmap");
    setDefault("GLOMAP_BIN", home + "/.local/glomap-install/bin/glomap");
    setDefault("VGGT_DIR", home + "/.local/vggt-src/vggt");
    setDefault("VGGT_OMEGA_DIR", home + "/.local/vggt-src/vggt-omega");
    if (!models.isEmpty()) {
        setDefault("DA3_MODEL_DIR", models + "/DA3-SMALL");
        setDefault("VGGT_CHECKPOINT", models + "/VGGT-1B/model.safetensors");
        setDefault("VGGT_OMEGA_CHECKPOINT",
                   models + "/VGGT-Omega/vggt_omega_1b_512.pt");
    }

    // Python 侧不要继承 Qt 的插件路径 (避免影响 cv2)
    env.remove("QT_PLUGIN_PATH");
    process_->setProcessEnvironment(env);
    process_->setWorkingDirectory(scriptDir_.isEmpty()
                                      ? QFileInfo(scriptPath_).absolutePath()
                                      : scriptDir_);

    connect(process_, &QProcess::readyReadStandardOutput,
            this, &PipelineRunner::onProcessOutput);
    connect(process_, &QProcess::readyReadStandardError,
            this, &PipelineRunner::onProcessOutput);
    connect(process_,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &PipelineRunner::onProcessFinished);

    std::cout << "[Pipeline] " << pythonBin.toStdString() << " "
              << args.join(" ").toStdString() << std::endl;

    process_->start(pythonBin, args);
}

void PipelineRunner::onProcessOutput() {
    while (process_->canReadLine()) {
        QString line = QString::fromUtf8(process_->readLine()).trimmed();
        if (line.isEmpty()) continue;
        emit consoleOutput(line);

        // ---- 阶段解析
        if (line.contains("抽帧")) {
            stage_ = ExtractingFrames;
            emit stageChanged(stage_, line);
        } else if (line.contains("特征提取") || line.contains("特征匹配") ||
                   line.contains("增量 SfM") || line.contains("全局 SfM") ||
                   line.contains("前馈 SfM") || line.contains("神经前馈") ||
                   line.contains("GLOMAP") || line.contains("VGGT") ||
                   line.contains("复用已有 SfM")) {
            stage_ = RunningSfM;
            emit stageChanged(stage_, line);
        } else if (line.contains("PatchMatch") || line.contains("TSDF") ||
                   line.contains("立体融合") || line.contains("DA3")) {
            stage_ = RunningMvs;
            emit stageChanged(stage_, line);
        } else if (line.contains("Poisson 网格") || line.contains("网格")) {
            stage_ = BuildingMesh;
            emit stageChanged(stage_, line);
        } else if (line.contains("训练") && line.contains("迭代")) {
            stage_ = TrainingGS;
            emit stageChanged(stage_, line);
        } else if (line.contains("iter ") || line.contains("it ")) {
            emit progressUpdate(-1, line);
        } else if (line.contains("导出")) {
            stage_ = Exporting;
            emit stageChanged(stage_, line);
        }
    }
}

void PipelineRunner::onProcessFinished(int exitCode,
                                        QProcess::ExitStatus status) {
    if (status == QProcess::CrashExit || exitCode != 0) {
        stage_ = Failed;
        error_ = "脚本异常退出 (exit=" + QString::number(exitCode) + ")";
        emit stageChanged(Failed, error_);
        emit finished(false, {});
    } else {
        stage_ = Finished;
        emit stageChanged(Finished, "完成!");
        emit finished(true, QString());
    }
}
