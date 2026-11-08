#pragma once
#include <string>
#include <functional>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

namespace wr {

/// PCL 网格处理器 — 纯 C++ 无 Qt 依赖
class MeshProcessor {
public:
    using StatusCallback = std::function<void(const std::string &msg)>;

    MeshProcessor();

    /// 加载 PLY/OBJ
    bool loadMesh(const std::string &path);

    /// 统计滤波去离群
    void filterOutliers(int nbNeighbors = 20, double stdRatio = 2.0);

    /// 网格简化
    void simplify(int targetFaces);

    /// MLS 平滑
    void smooth(double radius = 0.02);

    /// 保存
    bool save(const std::string &path);

    /// 从网格提取顶点+索引 (给 3D 查看器)
    struct RenderData {
        std::vector<float> vertices;       // xyz + normal (交错)
        std::vector<unsigned int> indices; // 三角形索引
    };
    RenderData toRenderData() const;

    // 访问
    size_t vertexCount() const;
    size_t faceCount() const;
    const pcl::PolygonMesh &mesh() const { return mesh_; }

    void setStatusCallback(StatusCallback cb) { onStatus_ = std::move(cb); }

private:
    pcl::PolygonMesh mesh_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    StatusCallback onStatus_;
};

} // namespace wr