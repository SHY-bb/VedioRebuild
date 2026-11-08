cmake_minimum_required(VERSION 3.20)
project(WorldRebuild VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ── 依赖安装前缀 ─────────────────────────────
# 默认从 /opt 读取本地编译的四套库；换机器时可用
#   cmake -DWR_PREFIX_ROOT=/your/prefix ..        覆盖
#   WR_PREFIX_ROOT=/your/prefix cmake ..          覆盖（环境变量）
if(NOT DEFINED WR_PREFIX_ROOT OR WR_PREFIX_ROOT STREQUAL "")
    if(DEFINED ENV{WR_PREFIX_ROOT} AND NOT "$ENV{WR_PREFIX_ROOT}" STREQUAL "")
        set(WR_PREFIX_ROOT "$ENV{WR_PREFIX_ROOT}")
    else()
        set(WR_PREFIX_ROOT "/opt")
    endif()
endif()
message(STATUS "WR_PREFIX_ROOT = ${WR_PREFIX_ROOT}")

set(CMAKE_PREFIX_PATH
    "${WR_PREFIX_ROOT}/qt/6.8.0;${WR_PREFIX_ROOT}/opencv/4.11.0;${WR_PREFIX_ROOT}/pcl/1.14.1;${WR_PREFIX_ROOT}/cgal/5.6"
    CACHE STRING "Library install prefixes")

# 安装后的可执行文件用 $ORIGIN 相对定位库，避免写死构建机的绝对路径
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../lib64")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# ── Qt6 ──────────────────────────────────────
find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGLWidgets Concurrent)

# ── OpenCV ───────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui imgcodecs)

# ── PCL ──────────────────────────────────────
find_package(PCL 1.14 REQUIRED COMPONENTS common io filters surface kdtree features search)

# ── OpenGL ───────────────────────────────────
find_package(OpenGL REQUIRED)

# ══════════════════════════════════════════════
#  核心算法层 (无 Qt 依赖)
# ══════════════════════════════════════════════
set(ALGO_SOURCES
    src/pipeline/FrameExtractor.cpp
    src/pipeline/MeshProcessor.cpp
)
set(ALGO_HEADERS
    src/pipeline/FrameExtractor.h
    src/pipeline/MeshProcessor.h
)

# ══════════════════════════════════════════════
#  UI 层 (Qt)
# ══════════════════════════════════════════════
set(UI_SOURCES
    src/ui/MainWindow.cpp
    src/ui/viewer/ModelViewer.cpp
    src/pipeline/PipelineRunner.cpp
)
set(UI_HEADERS
    src/ui/MainWindow.h
    src/ui/viewer/ModelViewer.h
    src/pipeline/PipelineRunner.h
)

# ══════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════
add_executable(WorldRebuild
    src/main.cpp
    ${ALGO_SOURCES} ${ALGO_HEADERS}
    ${UI_SOURCES} ${UI_HEADERS}
)

target_include_directories(WorldRebuild PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(WorldRebuild PRIVATE
    Qt6::Core Qt6::Widgets Qt6::OpenGLWidgets
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    OpenGL::GL
)

target_compile_definitions(WorldRebuild PRIVATE ${PCL_DEFINITIONS})

# ══════════════════════════════════════════════
#  测试程序 (命令行, 纯 C++)
# ══════════════════════════════════════════════
add_executable(WorldRebuild_test
    src/test/main.cpp
    src/test/TestRunner.cpp
    ${ALGO_SOURCES}
)

target_include_directories(WorldRebuild_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(WorldRebuild_test PRIVATE
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

target_compile_definitions(WorldRebuild_test PRIVATE ${PCL_DEFINITIONS})

# ── 安装 ─────────────────────────────────────
# 布局与 resolveScriptDir()/resolveModelDir() 的探测顺序一致:
#   <prefix>/bin/WorldRebuild
#   <prefix>/share/WorldRebuild/{scripts,models,resources}
install(TARGETS WorldRebuild WorldRebuild_test RUNTIME DESTINATION bin)
install(DIRECTORY resources/ DESTINATION share/WorldRebuild)
install(DIRECTORY scripts/ DESTINATION share/WorldRebuild/scripts
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE)
# 权重体积很大，默认不安装；需要完整打包时加 -DWR_INSTALL_MODELS=ON
option(WR_INSTALL_MODELS "把 models/ 权重一并安装 (体积数 GB)" OFF)
if(WR_INSTALL_MODELS)
    install(DIRECTORY models/ DESTINATION share/WorldRebuild/models)
endif()