#pragma once
#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QProgressBar>
#include <QLabel>
#include <QCheckBox>
#include <QSpinBox>
#include <QComboBox>

#include "pipeline/PipelineRunner.h"
#include "ui/viewer/ModelViewer.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

    /// 命令行加载
    void openModel(const QString &path);
    void openGaussians(const QString &path);
    bool saveViewerShot(const QString &path);

    /// 命令行/外部设置 SfM 接口模式 ("geometric" | "neural")
    void setSfmInterfaceByName(const QString &name);
    /// 命令行/外部设置 SfM 引擎
    /// ("incremental" | "glomap" | "vggt" | "vggt_omega")
    void setSfmEngineByName(const QString &name);

private slots:
    void onSelectVideo();
    void onRun();
    void onCancel();
    void onQualityEnhance();
    void onPipelineStageChanged(PipelineRunner::Stage stage, const QString &msg);
    void onPipelineFinished(bool success, const QString &outputPath);
    void onConsoleOutput(const QString &line);
    void loadAndDisplayMesh(const QString &meshPath);

private:
    void setupUi();
    void setupConnections();
    void refreshSfmEngines();
    void updateSfmControls();
    void setState(bool running);
    void updateStatusBar(const QString &msg);

    // UI 控件
    QLineEdit *videoPathEdit_ = nullptr;
    QPushButton *browseBtn_ = nullptr;
    QPushButton *runBtn_ = nullptr;
    QPushButton *cancelBtn_ = nullptr;
    QPushButton *qualityBtn_ = nullptr;
    QPushButton *wireframeBtn_ = nullptr;
    QPushButton *splatBtn_ = nullptr;
    QSpinBox *fpsSpin_ = nullptr;
    QSpinBox *maxSizeSpin_ = nullptr;
    QComboBox *modeCombo_ = nullptr;
    QComboBox *sfmInterfaceCombo_ = nullptr;
    QTextEdit *logView_ = nullptr;
    QProgressBar *progressBar_ = nullptr;
    QLabel *statusLabel_ = nullptr;
    QLabel *meshInfoLabel_ = nullptr;
    ModelViewer *viewer_ = nullptr;
    QCheckBox *wireframeCheck_ = nullptr;
    QCheckBox *splatCheck_ = nullptr;
    QComboBox *sfmCombo_ = nullptr;

    // 管线组件 (PipelineRunner 是 Qt 桥接层)
    PipelineRunner *runner_ = nullptr;

    // 状态
    QString currentOutputDir_;
    QString lastMeshPath_;
    bool lastRunWasQuality_ = false;
};