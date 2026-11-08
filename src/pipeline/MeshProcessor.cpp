#include "MeshProcessor.h"
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdint>
#include <vector>

namespace {

/// PLY 字段类型字节数 (PCL 只认 float, Open3D 默认写 double, 所以要自己读)
int plyTypeSize(const std::string &t) {
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "int" || t == "uint" || t == "float" ||
        t == "int32" || t == "uint32" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    return 0;
}

double readNumber(const char *p, const std::string &type) {
    if (type == "float" || type == "float32") { float v; std::memcpy(&v, p, 4); return v; }
    if (type == "double" || type == "float64") { double v; std::memcpy(&v, p, 8); return v; }
    if (type == "int" || type == "int32") { int32_t v; std::memcpy(&v, p, 4); return v; }
    if (type == "uint" || type == "uint32") { uint32_t v; std::memcpy(&v, p, 4); return v; }
    if (type == "short" || type == "int16") { int16_t v; std::memcpy(&v, p, 2); return v; }
    if (type == "ushort" || type == "uint16") { uint16_t v; std::memcpy(&v, p, 2); return v; }
    if (type == "char" || type == "int8") { int8_t v; std::memcpy(&v, p, 1); return v; }
    if (type == "uchar" || type == "uint8") { uint8_t v; std::memcpy(&v, p, 1); return v; }
    return 0.0;
}

struct PlyField { std::string name; std::string type; int offset = 0; };

/// 读取任意精度的二进制 PLY (支持 double / float 顶点与三角面)
bool loadPlyGeneric(const std::string &path, pcl::PolygonMesh &mesh) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    std::string line;
    bool binaryLE = false;
    size_t nv = 0, nf = 0;
    std::vector<PlyField> fields;
    std::string faceCountType, faceIndexType;
    bool inVertex = false, inFace = false;

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        if (kw == "format") {
            std::string fmt;
            ss >> fmt;
            binaryLE = (fmt == "binary_little_endian");
        } else if (kw == "element") {
            std::string name;
            size_t n = 0;
            ss >> name >> n;
            inVertex = (name == "vertex");
            inFace = (name == "face");
            if (inVertex) nv = n;
            if (inFace) nf = n;
        } else if (kw == "property") {
            std::string t;
            ss >> t;
            if (t == "list") {
                if (inFace) ss >> faceCountType >> faceIndexType;
            } else if (inVertex) {
                PlyField f;
                f.type = t;
                ss >> f.name;
                f.offset = fields.empty() ? 0
                                          : fields.back().offset +
                                                plyTypeSize(fields.back().type);
                fields.push_back(f);
            }
        } else if (kw == "end_header") {
            break;
        }
    }

    if (!binaryLE || nv == 0 || fields.empty()) return false;

    auto findField = [&](const char *name) -> const PlyField * {
        for (const auto &f : fields)
            if (f.name == name) return &f;
        return nullptr;
    };
    const PlyField *fx = findField("x");
    const PlyField *fy = findField("y");
    const PlyField *fz = findField("z");
    if (!fx || !fy || !fz) return false;

    const size_t stride = fields.back().offset + plyTypeSize(fields.back().type);
    std::vector<char> vbuf(nv * stride);
    if (!in.read(vbuf.data(), static_cast<std::streamsize>(vbuf.size())))
        return false;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.resize(nv);
    for (size_t i = 0; i < nv; ++i) {
        const char *base = vbuf.data() + i * stride;
        cloud[i].x = static_cast<float>(readNumber(base + fx->offset, fx->type));
        cloud[i].y = static_cast<float>(readNumber(base + fy->offset, fy->type));
        cloud[i].z = static_cast<float>(readNumber(base + fz->offset, fz->type));
    }
    pcl::toPCLPointCloud2(cloud, mesh.cloud);

    mesh.polygons.clear();
    if (nf > 0 && !faceIndexType.empty()) {
        const int cntSize = plyTypeSize(faceCountType);
        const int idxSize = plyTypeSize(faceIndexType);
        std::vector<char> fbuf(nf * 32);   // 上限保留
        in.read(fbuf.data(), static_cast<std::streamsize>(fbuf.size()));
        const size_t got = static_cast<size_t>(in.gcount());
        size_t pos = 0;
        for (size_t i = 0; i < nf && pos + cntSize <= got; ++i) {
            const uint8_t cnt = *reinterpret_cast<const uint8_t *>(fbuf.data() + pos);
            pos += cntSize;
            pcl::Vertices poly;
            for (uint8_t k = 0; k < cnt && pos + idxSize <= got; ++k) {
                uint32_t idx = 0;
                if (idxSize == 4) std::memcpy(&idx, fbuf.data() + pos, 4);
                else if (idxSize == 2) { uint16_t v; std::memcpy(&v, fbuf.data() + pos, 2); idx = v; }
                pos += idxSize;
                poly.vertices.push_back(idx);
            }
            if (poly.vertices.size() >= 3) mesh.polygons.push_back(poly);
        }
    }
    return true;
}

} // namespace

namespace wr {

MeshProcessor::MeshProcessor() : cloud_(new pcl::PointCloud<pcl::PointXYZ>) {}

bool MeshProcessor::loadMesh(const std::string &path) {
    mesh_ = pcl::PolygonMesh();
    bool ok = false;

    if (path.size() >= 4) {
        std::string ext = path.substr(path.size() - 4);
        // 自己解析: PCL 的 PLY 读取器不支持 double 精度 (Open3D 默认输出),
        // 且字段不匹配时仍返回"成功", 导致点云全为 0
        if (ext == ".ply")
            ok = loadPlyGeneric(path, mesh_);
        else if (ext == ".obj")
            ok = (pcl::io::loadOBJFile(path, mesh_) >= 0);
    }
    if (!ok) ok = (pcl::io::loadPLYFile(path, mesh_) >= 0);

    if (!ok) {
        if (onStatus_) onStatus_("加载失败: " + path);
        return false;
    }

    pcl::fromPCLPointCloud2(mesh_.cloud, *cloud_);
    if (onStatus_) onStatus_("已加载网格: " + std::to_string(cloud_->size()) +
                             " 顶点, " + std::to_string(mesh_.polygons.size()) + " 面");
    return true;
}

void MeshProcessor::filterOutliers(int nbNeighbors, double stdRatio) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_);
    sor.setMeanK(nbNeighbors);
    sor.setStddevMulThresh(stdRatio);

    pcl::PointCloud<pcl::PointXYZ> filtered;
    sor.filter(filtered);

    size_t removed = cloud_->size() - filtered.size();
    *cloud_ = filtered;
    pcl::toPCLPointCloud2(*cloud_, mesh_.cloud);

    if (onStatus_)
        onStatus_("去噪: 移除 " + std::to_string(removed) + " 点, 保留 " +
                  std::to_string(cloud_->size()) + " 点");
}

void MeshProcessor::simplify(int targetFaces) {
    if ((int)cloud_->size() > targetFaces * 2) {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud_);
        vg.setLeafSize(0.01f, 0.01f, 0.01f);
        pcl::PointCloud<pcl::PointXYZ> ds;
        vg.filter(ds);
        *cloud_ = ds;
    }

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setInputCloud(cloud_);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);
    ne.compute(*normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud_, *normals, *cloudWithNormals);

    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;
    gp3.setSearchRadius(0.05);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setNormalConsistency(false);
    gp3.setInputCloud(cloudWithNormals);
    gp3.reconstruct(triangles);

    mesh_ = triangles;
    pcl::toPCLPointCloud2(*cloud_, mesh_.cloud);

    if (onStatus_)
        onStatus_("简化完成: " + std::to_string(mesh_.polygons.size()) + " 面");
}

void MeshProcessor::smooth(double radius) {
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
    mls.setInputCloud(cloud_);
    mls.setSearchRadius(radius);
    mls.setPolynomialOrder(2);
    mls.setComputeNormals(true);

    pcl::PointCloud<pcl::PointXYZ> smoothed;
    mls.process(smoothed);
    *cloud_ = smoothed;
    pcl::toPCLPointCloud2(*cloud_, mesh_.cloud);

    if (onStatus_) onStatus_("MLS 平滑完成");
}

bool MeshProcessor::save(const std::string &path) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".ply")
        pcl::io::savePLYFileBinary(path, mesh_);
    else
        pcl::io::savePLYFileBinary(path + ".ply", mesh_);
    if (onStatus_) onStatus_("已保存: " + path);
    return true;
}

MeshProcessor::RenderData MeshProcessor::toRenderData() const {
    RenderData rd;
    const size_t n = cloud_->points.size();
    rd.vertices.resize(n * 6, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        const auto &pt = cloud_->points[i];
        rd.vertices[i * 6 + 0] = pt.x;
        rd.vertices[i * 6 + 1] = pt.y;
        rd.vertices[i * 6 + 2] = pt.z;
    }

    // 三角形索引 (多边形按扇形拆分)
    for (const auto &poly : mesh_.polygons) {
        for (size_t k = 1; k + 1 < poly.vertices.size(); ++k) {
            rd.indices.push_back(poly.vertices[0]);
            rd.indices.push_back(poly.vertices[k]);
            rd.indices.push_back(poly.vertices[k + 1]);
        }
    }

    // 面积加权顶点法线: 否则所有顶点法线都一样, 渲染出来是纯色剪影
    std::vector<float> nrm(n * 3, 0.0f);
    auto crossAdd = [&](unsigned int a, unsigned int b, unsigned int c) {
        if (a >= n || b >= n || c >= n) return;
        const float *pa = &rd.vertices[a * 6];
        const float *pb = &rd.vertices[b * 6];
        const float *pc = &rd.vertices[c * 6];
        const float ux = pb[0] - pa[0], uy = pb[1] - pa[1], uz = pb[2] - pa[2];
        const float vx = pc[0] - pa[0], vy = pc[1] - pa[1], vz = pc[2] - pa[2];
        const float nx = uy * vz - uz * vy;
        const float ny = uz * vx - ux * vz;
        const float nz = ux * vy - uy * vx;   // 未归一化 = 2*面积, 天然加权
        nrm[a * 3 + 0] += nx; nrm[a * 3 + 1] += ny; nrm[a * 3 + 2] += nz;
        nrm[b * 3 + 0] += nx; nrm[b * 3 + 1] += ny; nrm[b * 3 + 2] += nz;
        nrm[c * 3 + 0] += nx; nrm[c * 3 + 1] += ny; nrm[c * 3 + 2] += nz;
    };
    for (size_t t = 0; t + 2 < rd.indices.size(); t += 3)
        crossAdd(rd.indices[t], rd.indices[t + 1], rd.indices[t + 2]);

    for (size_t i = 0; i < n; ++i) {
        float x = nrm[i * 3 + 0], y = nrm[i * 3 + 1], z = nrm[i * 3 + 2];
        const float len = std::sqrt(x * x + y * y + z * z);
        if (len > 1e-12f) { x /= len; y /= len; z /= len; }
        else { x = 0.0f; y = 0.0f; z = 1.0f; }
        rd.vertices[i * 6 + 3] = x;
        rd.vertices[i * 6 + 4] = y;
        rd.vertices[i * 6 + 5] = z;
    }
    return rd;
}

size_t MeshProcessor::vertexCount() const { return cloud_->size(); }
size_t MeshProcessor::faceCount() const { return mesh_.polygons.size(); }

} // namespace wr