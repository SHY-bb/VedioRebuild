#include "TestRunner.h"
#include "pipeline/FrameExtractor.h"
#include "pipeline/MeshProcessor.h"
#include <iostream>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

namespace wr::test {

bool testFrameExtractor(const std::string &videoPath) {
    std::cout << "\n=== Test: FrameExtractor ===\n";

    if (videoPath.empty() || !fs::exists(videoPath)) {
        std::cout << "  ⚠️  跳过 (无测试视频)\n";
        return true;
    }

    FrameExtractor fe;
    fe.setFps(2);
    fe.setMaxSize(800);

    std::string tmpDir = "/tmp/wr_test_frames";
    std::vector<std::string> frames;

    int progressCalls = 0;
    fe.setProgressCallback([&](int cur, int total) { progressCalls++; });

    bool ok = fe.extract(videoPath, tmpDir, frames);
    assert(ok && "帧提取失败");
    assert(!frames.empty() && "未生成任何帧");
    assert(progressCalls > 0 && "进度回调未调用");

    std::cout << "  ✅ 通过 (" << frames.size() << " 帧, "
              << progressCalls << " 次回调)\n";

    fs::remove_all(tmpDir);
    return true;
}

bool testMeshProcessor(const std::string &meshPath) {
    std::cout << "\n=== Test: MeshProcessor ===\n";

    if (meshPath.empty() || !fs::exists(meshPath)) {
        std::cout << "  ⚠️  跳过 (无测试网格)\n";
        return true;
    }

    MeshProcessor mp;
    std::string lastMsg;
    mp.setStatusCallback([&](const std::string &msg) { lastMsg = msg; });

    // 加载
    bool loaded = mp.loadMesh(meshPath);
    assert(loaded && "网格加载失败");
    assert(mp.vertexCount() > 0);
    assert(!lastMsg.empty());

    // 去噪
    mp.filterOutliers(20, 2.0);
    std::cout << "  去噪后: " << mp.vertexCount() << " 顶点\n";

    // 保存副本
    std::string tmpPath = "/tmp/wr_test_output.ply";
    bool saved = mp.save(tmpPath);
    assert(saved && "保存失败");

    fs::remove(tmpPath);
    std::cout << "  ✅ 通过\n";
    return true;
}

bool testFullPipeline(const std::string &videoPath,
                       const std::string &outputDir) {
    std::cout << "\n=== Test: Full Pipeline ===\n";

    if (videoPath.empty() || !fs::exists(videoPath)) {
        std::cout << "  ⚠️  跳过 (无测试视频)\n";
        return true;
    }

    // 1. 抽帧
    FrameExtractor fe;
    fe.setFps(2);
    std::vector<std::string> frames;
    bool ok = fe.extract(videoPath, outputDir + "/images", frames);
    assert(ok && "抽帧失败");

    std::cout << "  ✅ 抽帧: " << frames.size() << " 帧\n";
    std::cout << "  💡 完整测试需运行 reconstruct.py 完成 SfM+MVS\n";
    return true;
}

int runAll(const std::string &sampleVideo, const std::string &sampleMesh) {
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  WorldRebuild 测试套件" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    int passed = 0, total = 3;
    if (testFrameExtractor(sampleVideo)) passed++;
    if (testMeshProcessor(sampleMesh)) passed++;
    if (testFullPipeline(sampleVideo, "/tmp/wr_test")) passed++;

    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  结果: " << passed << "/" << total << " 通过\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    return passed == total ? 0 : 1;
}

} // namespace wr::test