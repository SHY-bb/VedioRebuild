/// 测试入口
#include "test/TestRunner.h"
#include <iostream>

int main(int argc, char *argv[]) {
    std::string video = argc > 1 ? argv[1] : "";
    std::string mesh = argc > 2 ? argv[2] : "";
    return wr::test::runAll(video, mesh);
}