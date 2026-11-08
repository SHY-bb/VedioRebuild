#include <QApplication>
#include <QSurfaceFormat>
#include <QStringList>
#include <QTimer>
#include <QScreen>
#include "ui/MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("WorldRebuild");
    app.setApplicationVersion("1.0.0");

    // 启用 OpenGL 3.3 Core Profile
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(fmt);

    MainWindow window;
    window.show();

    const QStringList args = app.arguments();

    // 可选: --sfm-interface geometric|neural 预选接口模式
    const int ifaceIdx = args.indexOf("--sfm-interface");
    if (ifaceIdx > 0 && ifaceIdx + 1 < args.size())
        window.setSfmInterfaceByName(args.at(ifaceIdx + 1));

    // 可选: --sfm incremental|glomap|vggt|vggt_omega 预选 SfM 引擎
    const int sfmIdx = args.indexOf("--sfm");
    if (sfmIdx > 0 && sfmIdx + 1 < args.size())
        window.setSfmEngineByName(args.at(sfmIdx + 1));

    // 可选: WorldRebuild <model.ply|mesh.obj>  [--splat]
    if (args.size() > 1 && !args.at(1).startsWith("-")) {
        const QString path = args.at(1);
        if (args.contains("--splat") || path.endsWith(".splat") ||
            path.contains("gaussian"))
            window.openGaussians(path);
        else
            window.openModel(path);
    }

    // 可选: --shot-gl <png> 只保存 3D 视图 (避开窗口合成)
    const int glIdx = args.indexOf("--shot-gl");
    if (glIdx > 0 && glIdx + 1 < args.size()) {
        const QString out = args.at(glIdx + 1);
        QTimer::singleShot(3000, &app, [&window, out]() {
            window.saveViewerShot(out);
            QApplication::quit();
        });
        return app.exec();
    }

    // 可选: --shot <png> 渲染一帧后截图退出 (自动化验证)
    const int shotIdx = args.indexOf("--shot");
    if (shotIdx > 0 && shotIdx + 1 < args.size()) {
        const QString out = args.at(shotIdx + 1);
        QTimer::singleShot(3500, &app, [&window, out]() {
            window.grab().save(out);
            QApplication::quit();
        });
    }

    return app.exec();
}
