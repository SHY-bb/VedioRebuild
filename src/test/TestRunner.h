#pragma once
#include <string>
#include <vector>
#include <functional>

/// 测试套件 — 验证算法层各组件
namespace wr::test {

/// 测试帧提取 (需要一小段测试视频)
bool testFrameExtractor(const std::string &videoPath);

/// 测试网格加载与处理 (需要 PLY/OBJ 文件)
bool testMeshProcessor(const std::string &meshPath);

/// 测试完整管线 (抽帧 + 调用重建)
bool testFullPipeline(const std::string &videoPath, const std::string &outputDir);

/// 运行所有测试
int runAll(const std::string &sampleVideo = "",
           const std::string &sampleMesh = "");

} // namespace wr::test