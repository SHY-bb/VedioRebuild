#include "ui/MainWindow.h"
#include "pipeline/MeshProcessor.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QGroupBox>
#include <QFormLayout>
#include <QMessageBox>
#include <QDateTime>
#include <QStatusBar>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QApplication>
#include <QToolBar>
#include <QTime>
#include <QTextDocument>
#include <QComboBox>
#include <QStandardItemModel>
#include <QSignalBlocker>
#include <QDesktopServices>
#include <QUrl>
#include <iostream>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("WorldRebuild — 视频转三维");
    resize(1400, 800);

    runner_ = new PipelineRunner(this);
    // 路径全部可配置: 环境变量 > 自动探测 > 默认值 (见 README「配置项」)
    runner_->setPythonEnv(PipelineRunner::defaultPythonEnv());
    const QString scriptDir = PipelineRunner::resolveScriptDir();
    runner_->setScriptDir(scriptDir);
    // 旧版 CPU 预览脚本, 已收进 scripts/ (不再依赖笔记库路径)
    runner_->setScript(scriptDir + "/reconstruct_cpu.py");

    // 启动时打印实际生效的路径，便于排查环境问题
    std::cout << "[Config] python   = " << runner_->defaultPythonEnv().toStdString()
              << "\n[Config] scripts  = " << scriptDir.toStdString()
              << "\n[Config] models   = "
              << PipelineRunner::resolveModelDir().toStdString() << std::endl;

    setupUi();
    setupConnections();
    setState(false);
}

void MainWindow::setupUi() {
    auto *fileMenu = menuBar()->addMenu("文件(&F)");
    auto *openAct = fileMenu->addAction("打开视频...");
    connect(openAct, &QAction::triggered, this, &MainWindow::onSelectVideo);

    auto *viewMenu = menuBar()->addMenu("视图(&V)");
    auto *resetAct = viewMenu->addAction("重置视角");
    // viewer_ 稍后创建, 在下面连接

    auto *toolbar = addToolBar("主工具栏");

    videoPathEdit_ = new QLineEdit;
    videoPathEdit_->setPlaceholderText("选择输入视频...");
    videoPathEdit_->setMinimumWidth(300);
    toolbar->addWidget(videoPathEdit_);

    browseBtn_ = new QPushButton("浏览");
    toolbar->addWidget(browseBtn_);
    toolbar->addSeparator();

    runBtn_ = new QPushButton("▶ 开始重建");
    runBtn_->setStyleSheet("color: green; font-weight: bold;");
    toolbar->addWidget(runBtn_);

    cancelBtn_ = new QPushButton("■ 停止");
    cancelBtn_->setEnabled(false);
    toolbar->addWidget(cancelBtn_);
    toolbar->addSeparator();

    qualityBtn_ = new QPushButton("🔧 提升质量");
    qualityBtn_->setToolTip("3DGS 训练 / 稠密重建");
    toolbar->addWidget(qualityBtn_);

    wireframeBtn_ = new QPushButton("线框");
    wireframeBtn_->setCheckable(true);
    toolbar->addWidget(wireframeBtn_);

    splatBtn_ = new QPushButton("高斯");
    splatBtn_->setCheckable(true);
    splatBtn_->setChecked(true);
    splatBtn_->setToolTip("显示 3DGS 高斯泼溅 / 切回三角网格");
    toolbar->addWidget(splatBtn_);

    auto *paramGroup = new QGroupBox("参数");
    auto *paramLayout = new QFormLayout(paramGroup);

    modeCombo_ = new QComboBox;
    modeCombo_->addItem("稠密重建 (COLMAP CUDA)", "dense");
    modeCombo_->addItem("深度重建 (DA3 + TSDF)", "da3");
    modeCombo_->addItem("快速预览 (CPU MVS)", "standard");
    modeCombo_->setToolTip(
        "稠密: 百万级点云, 细节最好\n"
        "深度: 弱纹理更稳, 显存占用低\n"
        "快速: 旧版 pycolmap, 最快但稀疏");
    paramLayout->addRow("重建模式:", modeCombo_);

    sfmInterfaceCombo_ = new QComboBox;
    sfmInterfaceCombo_->addItem("几何优化 (COLMAP / GLOMAP)", "geometric");
    sfmInterfaceCombo_->addItem("神经前馈 (VGGT)", "neural");
    sfmInterfaceCombo_->setToolTip(
        "几何优化: 显式特征匹配 + 三角化 + Bundle Adjustment\n"
        "神经前馈: 端到端回归相机/深度 (VGGT 已接入; VGGT-Ω 需 ≥6GB)");
    paramLayout->addRow("接口模式:", sfmInterfaceCombo_);

    sfmCombo_ = new QComboBox;
    paramLayout->addRow("SfM 引擎:", sfmCombo_);
    refreshSfmEngines();

    fpsSpin_ = new QSpinBox;
    fpsSpin_->setRange(1, 10);
    fpsSpin_->setValue(2);
    paramLayout->addRow("抽帧帧率:", fpsSpin_);

    maxSizeSpin_ = new QSpinBox;
    maxSizeSpin_->setRange(640, 3840);
    maxSizeSpin_->setSingleStep(160);
    maxSizeSpin_->setValue(1600);
    paramLayout->addRow("最大边长:", maxSizeSpin_);

    wireframeCheck_ = new QCheckBox("线框模式");
    paramLayout->addRow("", wireframeCheck_);

    splatCheck_ = new QCheckBox("显示高斯 (3DGS)");
    splatCheck_->setChecked(true);
    paramLayout->addRow("", splatCheck_);

    meshInfoLabel_ = new QLabel("未加载模型");
    meshInfoLabel_->setStyleSheet("color: gray;");
    paramLayout->addRow("模型:", meshInfoLabel_);

    viewer_ = new ModelViewer;
    viewer_->setMinimumSize(600, 400);

    // 视图→重置视角 连接到 viewer
    connect(resetAct, &QAction::triggered, viewer_, &ModelViewer::resetView);

    auto *leftPanel = new QWidget;
    auto *leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->addWidget(paramGroup);

    auto *logGroup = new QGroupBox("运行日志");
    auto *logLayout = new QVBoxLayout(logGroup);
    logView_ = new QTextEdit;
    logView_->setReadOnly(true);
    logView_->document()->setMaximumBlockCount(500);
    logLayout->addWidget(logView_);

    progressBar_ = new QProgressBar;
    progressBar_->setRange(0, 100);
    logLayout->addWidget(progressBar_);
    leftLayout->addWidget(logGroup);

    auto *splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(leftPanel);
    splitter->addWidget(viewer_);
    splitter->setStretchFactor(0, 2);
    splitter->setStretchFactor(1, 5);
    setCentralWidget(splitter);

    statusLabel_ = new QLabel("就绪");
    statusBar()->addWidget(statusLabel_);
}

void MainWindow::setupConnections() {
    connect(browseBtn_, &QPushButton::clicked, this, &MainWindow::onSelectVideo);
    connect(runBtn_, &QPushButton::clicked, this, &MainWindow::onRun);
    connect(cancelBtn_, &QPushButton::clicked, this, &MainWindow::onCancel);
    connect(qualityBtn_, &QPushButton::clicked, this, &MainWindow::onQualityEnhance);
    // 快速预览走 pycolmap，不支持 SfM 引擎选择，整组置灰
    connect(modeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                const bool standard =
                    modeCombo_->currentData().toString() == "standard";
                if (standard) {
                    const int gi = sfmInterfaceCombo_->findData("geometric");
                    if (gi >= 0 && sfmInterfaceCombo_->currentIndex() != gi) {
                        QSignalBlocker blocker(sfmInterfaceCombo_);
                        sfmInterfaceCombo_->setCurrentIndex(gi);
                        refreshSfmEngines();
                    }
                }
                updateSfmControls();
            });

    // 接口模式切换时，重建引擎列表
    connect(sfmInterfaceCombo_,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
                refreshSfmEngines();
                const bool neural =
                    sfmInterfaceCombo_->currentData().toString() == "neural";
                updateStatusBar(neural ? "接口模式: 神经前馈 (VGGT)"
                                       : "接口模式: 几何优化");
            });
    connect(wireframeCheck_, &QCheckBox::toggled, viewer_, &ModelViewer::setWireframe);
    connect(wireframeBtn_, &QPushButton::toggled, viewer_, &ModelViewer::setWireframe);
    connect(splatCheck_, &QCheckBox::toggled, viewer_, &ModelViewer::setShowGaussians);
    connect(splatBtn_, &QPushButton::toggled, viewer_, &ModelViewer::setShowGaussians);
    connect(runner_, &PipelineRunner::stageChanged, this, &MainWindow::onPipelineStageChanged);
    connect(runner_, &PipelineRunner::finished, this, &MainWindow::onPipelineFinished);
    connect(runner_, &PipelineRunner::consoleOutput, this, &MainWindow::onConsoleOutput);
}

void MainWindow::refreshSfmEngines() {
    const QString iface = sfmInterfaceCombo_
                              ? sfmInterfaceCombo_->currentData().toString()
                              : QStringLiteral("geometric");
    sfmCombo_->clear();

    if (iface == "neural") {
        // VGGT 已接入；VGGT-Ω 需要 gated 权重且 >=6GB 显存，暂置灰
        sfmCombo_->addItem("VGGT (前馈)", "vggt");
        sfmCombo_->addItem("VGGT-Ω (需 ≥6GB / gated)", "vggt_omega");
        if (auto *model =
                qobject_cast<QStandardItemModel *>(sfmCombo_->model())) {
            if (model->rowCount() > 1)
                model->item(1)->setEnabled(false);
        }
        sfmCombo_->setToolTip(
            "VGGT: 1B 前馈模型，4GB 显存自动降分辨率/帧数\n"
            "VGGT-Ω: 官方峰值显存约 6GB，且权重需申请，暂不可用");
    } else {
        sfmCombo_->addItem("增量式 (COLMAP)", "incremental");
        sfmCombo_->addItem("全局式 (GLOMAP)", "glomap");
        sfmCombo_->setToolTip(
            "增量式: 逐张注册, 鲁棒, 适合小规模/视频\n"
            "全局式: 一次解全部位姿, 快, 适合大规模/无序照片\n"
            "仅对稠密/深度重建生效 (快速预览不支持)");
    }

    updateSfmControls();
}

void MainWindow::updateSfmControls() {
    const bool standard = modeCombo_->currentData().toString() == "standard";
    sfmInterfaceCombo_->setEnabled(!standard);
    // 神经模式下的 VGGT 可用；VGGT-Ω item 单独置灰
    sfmCombo_->setEnabled(!standard);
}

void MainWindow::onSelectVideo() {
    QString path = QFileDialog::getOpenFileName(
        this, "选择视频", {},
        "视频 (*.mp4 *.avi *.mov *.mkv *.webm);;所有文件 (*)");
    if (!path.isEmpty()) videoPathEdit_->setText(path);
}

void MainWindow::onRun() {
    QString video = videoPathEdit_->text();
    if (video.isEmpty() || !QFileInfo::exists(video)) {
        QMessageBox::warning(this, "错误", "请先选择一个有效的视频文件");
        return;
    }
    const QString mode = modeCombo_->currentData().toString();
    if (mode == "standard")
        runner_->setMode(PipelineRunner::Mode::Standard);
    else if (mode == "da3")
        runner_->setMode(PipelineRunner::Mode::DepthTsdf);
    else
        runner_->setMode(PipelineRunner::Mode::Dense);

    const QString iface = sfmInterfaceCombo_->currentData().toString();
    const QString engine = sfmCombo_->currentData().toString();
    if (mode != "standard" && iface == "neural" && engine == "vggt_omega") {
        QMessageBox::information(
            this, "VGGT-Ω 暂不可用",
            "VGGT-Ω 官方峰值显存约 6GB，且权重需要申请访问。\n\n"
            "本机 4GB 显存请使用 VGGT；或换 >=6GB 显卡并准备 gated 权重后启用。");
        return;
    }

    runner_->setSfmInterface(
        iface == "neural" ? PipelineRunner::SfmInterface::Neural
                          : PipelineRunner::SfmInterface::Geometric);
    if (engine == "glomap") {
        runner_->setSfmEngine(PipelineRunner::SfmEngine::Glomap);
    } else if (engine == "vggt") {
        runner_->setSfmEngine(PipelineRunner::SfmEngine::Vggt);
    } else if (engine == "vggt_omega") {
        runner_->setSfmEngine(PipelineRunner::SfmEngine::VggtOmega);
    } else {
        runner_->setSfmEngine(PipelineRunner::SfmEngine::Incremental);
    }

    currentOutputDir_ = QFileInfo(video).absolutePath() + "/" +
                        QFileInfo(video).baseName() + "_rebuild";
    lastRunWasQuality_ = false;
    setState(true);
    logView_->clear();
    runner_->run(video, currentOutputDir_,
                 fpsSpin_->value(), maxSizeSpin_->value());
}

void MainWindow::onCancel() {
    runner_->cancel();
    setState(false);
    updateStatusBar("已取消");
}

void MainWindow::onQualityEnhance() {
    if (currentOutputDir_.isEmpty() || !QFileInfo::exists(currentOutputDir_)) {
        QMessageBox::information(this, "提示", "请先完成一次重建");
        return;
    }
    auto ret = QMessageBox::question(
        this, "3DGS 训练",
        "将对当前重建结果训练 3D Gaussian Splatting。\n"
        "4GB 显存约需 20-40 分钟，是否继续？");
    if (ret != QMessageBox::Yes) return;

    lastRunWasQuality_ = true;
    setState(true);
    runner_->runQualityEnhancement(currentOutputDir_);
}

void MainWindow::onPipelineStageChanged(PipelineRunner::Stage stage,
                                         const QString &msg) {
    updateStatusBar(msg);
    logView_->append("[" + QTime::currentTime().toString("hh:mm:ss") + "] " + msg);
    switch (stage) {
    case PipelineRunner::ExtractingFrames: progressBar_->setValue(5); break;
    case PipelineRunner::RunningSfM: progressBar_->setValue(25); break;
    case PipelineRunner::RunningMvs: progressBar_->setValue(55); break;
    case PipelineRunner::BuildingMesh: progressBar_->setValue(80); break;
    case PipelineRunner::TrainingGS: progressBar_->setValue(50); break;
    case PipelineRunner::Exporting: progressBar_->setValue(95); break;
    case PipelineRunner::Finished: progressBar_->setValue(100); break;
    default: break;
    }
}

void MainWindow::onPipelineFinished(bool success, const QString & /*unused*/) {
    setState(false);
    if (!success) {
        updateStatusBar("流程失败");
        return;
    }

    if (lastRunWasQuality_) {
        const QString gs = currentOutputDir_ + "/gs/gaussians.ply";
        if (QFileInfo::exists(gs)) {
            updateStatusBar("3DGS 训练完成: " + gs);
            logView_->append("3DGS 输出:\n  " + gs +
                             "\n  " + currentOutputDir_ + "/gs/gaussians.splat" +
                             "\n  预览图: " + currentOutputDir_ + "/gs/preview_*.png");
            meshInfoLabel_->setText("3DGS: " + QFileInfo(gs).fileName());
            if (viewer_->loadGaussians(gs)) {
                viewer_->resetView();
                updateStatusBar(QString("3DGS 已加载: %1 个高斯")
                                    .arg(viewer_->gaussianCount()));
            }
        }
        return;
    }

    updateStatusBar("重建完成");
    QStringList cand = {currentOutputDir_ + "/mesh_da3/mesh.ply",
                        currentOutputDir_ + "/mesh/mesh.ply",
                        currentOutputDir_ + "/export/model.obj",
                        currentOutputDir_ + "/export/model.ply"};
    for (const auto &p : cand) {
        if (QFileInfo::exists(p)) { loadAndDisplayMesh(p); break; }
    }
}

void MainWindow::onConsoleOutput(const QString &line) {
    logView_->append(line);
}

bool MainWindow::saveViewerShot(const QString &path) {
    const QImage img = viewer_->grabFramebuffer();
    return !img.isNull() && img.save(path);
}

void MainWindow::openModel(const QString &path) {
    loadAndDisplayMesh(path);
}

void MainWindow::openGaussians(const QString &path) {
    if (viewer_->loadGaussians(path)) {
        viewer_->resetView();
        meshInfoLabel_->setText("3DGS: " + QFileInfo(path).fileName());
        updateStatusBar("已加载 3DGS: " + path);
    } else {
        updateStatusBar("高斯文件加载失败: " + path);
    }
}

void MainWindow::setSfmInterfaceByName(const QString &name) {
    const int idx = sfmInterfaceCombo_->findData(name);
    if (idx >= 0) {
        sfmInterfaceCombo_->setCurrentIndex(idx);
        updateStatusBar("SfM 接口: " + sfmInterfaceCombo_->currentText());
    } else {
        updateStatusBar("未知 SfM 接口: " + name +
                        " (可选 geometric/neural)");
    }
}

void MainWindow::setSfmEngineByName(const QString &name) {
    // 先根据引擎推断接口模式，保证两级下拉一致
    const QString iface =
        (name == "vggt" || name == "vggt_omega") ? "neural" : "geometric";
    const int ii = sfmInterfaceCombo_->findData(iface);
    if (ii >= 0 && sfmInterfaceCombo_->currentIndex() != ii)
        sfmInterfaceCombo_->setCurrentIndex(ii);  // 触发 refreshSfmEngines()

    const int idx = sfmCombo_->findData(name);
    if (idx >= 0) {
        sfmCombo_->setCurrentIndex(idx);
        updateStatusBar("SfM 接口: " + sfmInterfaceCombo_->currentText() +
                        " / 引擎: " + sfmCombo_->currentText());
    } else {
        updateStatusBar("未知 SfM 引擎: " + name +
                        " (可选 incremental/glomap/vggt/vggt_omega)");
    }
}

void MainWindow::loadAndDisplayMesh(const QString &meshPath) {
    if (!QFileInfo::exists(meshPath)) { updateStatusBar("模型文件不存在"); return; }
    updateStatusBar("加载模型: " + meshPath);
    QApplication::processEvents();

    wr::MeshProcessor mp;
    mp.setStatusCallback([this](const std::string &msg) {
        updateStatusBar(QString::fromStdString(msg));
    });
    if (!mp.loadMesh(meshPath.toStdString())) { updateStatusBar("加载失败"); return; }

    lastMeshPath_ = meshPath;
    meshInfoLabel_->setText(
        QString("%1 顶点, %2 面").arg(mp.vertexCount()).arg(mp.faceCount()));

    auto rd = mp.toRenderData();
    viewer_->loadMesh(rd.vertices, rd.indices);
    viewer_->resetView();

    updateStatusBar(QString("已加载: %1 顶点, %2 面")
                        .arg(mp.vertexCount()).arg(mp.faceCount()));
}

void MainWindow::setState(bool running) {
    videoPathEdit_->setEnabled(!running);
    browseBtn_->setEnabled(!running);
    runBtn_->setEnabled(!running);
    cancelBtn_->setEnabled(running);
    qualityBtn_->setEnabled(!running);
    fpsSpin_->setEnabled(!running);
    maxSizeSpin_->setEnabled(!running);
    modeCombo_->setEnabled(!running);
    sfmCombo_->setEnabled(!running &&
                          modeCombo_->currentData().toString() != "standard");
    progressBar_->setValue(running ? 0 : 100);
}

void MainWindow::updateStatusBar(const QString &msg) {
    statusLabel_->setText(msg);
}