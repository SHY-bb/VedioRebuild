#pragma once
#include <string>
#include <vector>
#include <functional>

namespace wr {

/// 视频帧抽取器 — 纯 C++, 无 Qt 依赖
class FrameExtractor {
public:
    using ProgressCallback = std::function<void(int current, int total)>;

    void setFps(int fps) { fps_ = fps; }
    void setMaxSize(int px) { maxSize_ = px; }
    void setProgressCallback(ProgressCallback cb) { onProgress_ = std::move(cb); }

    /// 抽取视频帧, 返回输出目录和帧文件列表
    bool extract(const std::string &videoPath,
                 const std::string &outputDir,
                 std::vector<std::string> &outFrames);

    int fps() const { return fps_; }
    int maxSize() const { return maxSize_; }

private:
    int fps_ = 2;
    int maxSize_ = 1600;
    ProgressCallback onProgress_;
};

} // namespace wr