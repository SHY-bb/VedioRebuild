#include "FrameExtractor.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace wr {

bool FrameExtractor::extract(const std::string &videoPath,
                              const std::string &outputDir,
                              std::vector<std::string> &outFrames) {
    outFrames.clear();
    namespace fs = std::filesystem;

    fs::path out(outputDir);
    std::error_code ec;
    fs::remove_all(out, ec);
    fs::create_directories(out, ec);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[FrameExtractor] 无法打开: " << videoPath << std::endl;
        return false;
    }

    double videoFps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int interval = std::max(1, static_cast<int>(videoFps / fps_));

    std::cout << "[FrameExtractor] FPS:" << videoFps << " 帧数:" << totalFrames
              << " 间隔:" << interval << std::endl;

    int saved = 0;
    for (int i = 0; i < totalFrames; ++i) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        if (i % interval != 0) continue;

        int h = frame.rows, w = frame.cols;
        if (std::max(w, h) > maxSize_) {
            double scale = static_cast<double>(maxSize_) / std::max(w, h);
            cv::resize(frame, frame,
                       cv::Size(static_cast<int>(w * scale),
                                static_cast<int>(h * scale)));
        }

        std::ostringstream ss;
        ss << outputDir << "/" << std::setw(4) << std::setfill('0') << saved << ".jpg";
        std::string filename = ss.str();
        cv::imwrite(filename, frame);
        outFrames.push_back(filename);
        saved++;

        if (onProgress_) onProgress_(i, totalFrames);
    }
    cap.release();

    std::cout << "[FrameExtractor] 抽取 " << saved << " 帧" << std::endl;
    return saved > 0;
}

} // namespace wr