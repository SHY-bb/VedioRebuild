#!/usr/bin/env python3
"""
手机视频 → 三维重建 全流程脚本
================================
阶段: 抽帧 → 特征提取 → 特征匹配 → SfM → MVS → 网格 → 导出

依赖: pycolmap 4.2.1 + Open3D 0.19 + OpenCV 4.11
"""

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap


# ============================================================
# 阶段 1: 视频抽帧
# ============================================================
def extract_frames(video_path: str, output_dir: str, fps: int = 2, max_size: int = 1600):
    """OpenCV 抽帧 + 降分辨率 (适配 4GB VRAM)"""
    out = Path(output_dir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / video_fps if video_fps > 0 else 0

    print(f"\n{'='*50}")
    print(f"📹 阶段 1: 视频抽帧")
    print(f"{'='*50}")
    print(f"  视频: {video_path}")
    print(f"  时长: {duration:.1f}s | FPS: {video_fps:.1f}")

    interval = max(1, int(video_fps / fps))
    saved = 0

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            h, w = frame.shape[:2]
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            cv2.imwrite(str(out / f"{saved:04d}.jpg"), frame)
            saved += 1

    cap.release()
    print(f"  ✅ 抽取 {saved} 帧 (间隔 {interval} 帧)")
    return saved


# ============================================================
# 阶段 2+3: COLMAP SfM (特征 → 匹配 → 增量重建)
# ============================================================
def run_colmap_sfm(image_dir: str, output_dir: str):
    """pycolmap 全自动 SfM + 稀疏点云"""
    images = Path(image_dir)
    out = Path(output_dir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / "database").mkdir(exist_ok=True)

    db_path = out / "database" / "database.db"

    print(f"\n{'='*50}")
    print(f"🔍 阶段 2: 特征提取 (SIFT)")
    print(f"{'='*50}")
    t0 = time.time()

    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images),
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
    print(f"  ✅ 完成 ({time.time() - t0:.0f}s)")

    print(f"\n{'='*50}")
    print(f"🔗 阶段 3: 特征匹配 (Exhaustive)")
    print(f"{'='*50}")
    t0 = time.time()

    pycolmap.match_exhaustive(database_path=str(db_path))

    match_time = time.time() - t0
    print(f"  ✅ 完成 ({match_time:.0f}s)")

    print(f"\n{'='*50}")
    print(f"🏗️  阶段 4: 增量 SfM 重建")
    print(f"{'='*50}")
    t0 = time.time()

    sparse_dir = out / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    reconstructions = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images),
        output_path=str(sparse_dir),
    )

    sfm_time = time.time() - t0
    total_time = sfm_time + match_time

    if not reconstructions:
        raise RuntimeError("SfM 重建失败 — 没有找到有效的相机位姿")

    rec = list(reconstructions.values())[0]
    print(f"  ✅ SfM 完成 ({total_time:.0f}s)")
    print(f"  注册图像: {rec.num_reg_images()}")
    print(f"  三维点: {len(rec.points3D)}")

    return db_path, sparse_dir


# ============================================================
# 阶段 5: MVS 稠密重建
# ============================================================
def run_colmap_mvs(image_dir: str, sparse_dir: str, output_dir: str):
    """pycolmap MVS → 稠密点云"""
    images = Path(image_dir)
    sparse = Path(sparse_dir) / "0"
    dense = Path(output_dir) / "dense"
    shutil.rmtree(dense, ignore_errors=True)
    dense.mkdir(parents=True)

    print(f"\n{'='*50}")
    print(f"🔬 阶段 5: 稠密重建 (MVS)")
    print(f"{'='*50}")
    print(f"  此步骤使用 GPU, 注意 4GB 显存限制...")

    t0 = time.time()

    # 图像去畸变
    print(f"  → 去畸变图像...")
    pycolmap.undistort_images(
        output_path=str(dense),
        input_path=str(sparse),
        image_path=str(images),
    )

    # PatchMatch 立体匹配 (pip版 pycolmap 可能不带 CUDA)
    try:
        print(f"  → PatchMatch 立体匹配 (GPU)...")
        pycolmap.patch_match_stereo(
            workspace_path=str(dense),
            options=pycolmap.PatchMatchOptions(),
        )

        print(f"  → 立体融合...")
        pycolmap.stereo_fusion(
            output_path=str(dense / "fused.ply"),
            workspace_path=str(dense),
            input_type="geometric",
        )

        elapsed = time.time() - t0
        print(f"  ✅ MVS 完成 ({elapsed:.0f}s)")
        return dense / "fused.ply"

    except Exception as e:
        print(f"  ⚠️  MVS 跳过 (pip版不带CUDA)")
        print(f"  → 降级: 直接用稀疏点云建网格")
        return None


# ============================================================
# 阶段 6: 网格重建
# ============================================================
def reconstruct_mesh(ply_path: str, output_dir: str, depth: int = 9):
    """Open3D Poisson 曲面重建"""
    out = Path(output_dir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    print(f"\n{'='*50}")
    print(f"🧊 阶段 6: 网格重建 (Poisson)")
    print(f"{'='*50}")

    t0 = time.time()

    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_orig = len(pcd.points)
    print(f"  原始点云: {n_orig:,} 点")

    # 降采样
    if n_orig > 500_000:
        voxel = 0.005
        pcd = pcd.voxel_down_sample(voxel)
        print(f"  降采样 ({voxel*1000:.0f}mm): {len(pcd.points):,} 点")

    # 法线估计
    print(f"  → 估计法线...")
    pcd.estimate_normals()

    # 方向一致性
    pcd.orient_normals_consistent_tangent_plane(30)

    # Poisson 重建
    print(f"  → Poisson 重建 (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # 去低密度
    thresh = np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(densities < thresh)

    # 清理
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    elapsed = time.time() - t0
    print(f"  ✅ 网格完成 ({elapsed:.0f}s)")
    print(f"  顶点: {len(mesh.vertices):,} | 面: {len(mesh.triangles):,}")

    mesh_path = out / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    return mesh, mesh_path


# ============================================================
# 阶段 7: 简化 + 导出
# ============================================================
def simplify_and_export(mesh_path: str, output_dir: str, target: int = 150_000):
    """网格简化 + 多格式导出"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"📦 阶段 7: 导出")
    print(f"{'='*50}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    n_faces = len(mesh.triangles)

    if n_faces > target:
        print(f"  → 简化 {n_faces:,} → {target:,} 面...")
        mesh = mesh.simplify_quadric_decimation(target)

    # 导出
    o3d.io.write_triangle_mesh(str(out / "model.obj"), mesh)
    o3d.io.write_triangle_mesh(str(out / "model.ply"), mesh)
    print(f"  ✅ model.obj + model.ply → {out.resolve()}")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="手机视频 → 三维重建")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("-o", "--output", default="./reconstruction", help="输出目录")
    parser.add_argument("--fps", type=int, default=2, help="抽帧帧率 (默认 2)")
    parser.add_argument("--max-size", type=int, default=1600, help="图像最大边长 (默认 1600)")
    parser.add_argument("--depth", type=int, default=9, help="Poisson 深度 (默认 9)")
    parser.add_argument("--faces", type=int, default=150000, help="目标面数 (默认 15万)")
    parser.add_argument("--skip-frames", action="store_true", help="跳过抽帧")
    args = parser.parse_args()

    out = Path(args.output)
    images_dir = out / "images"

    print(f"\n{'#'*50}")
    print(f"#  手机视频 → 三维重建")
    print(f"#  输入: {args.video}")
    print(f"#  输出: {out.resolve()}")
    print(f"{'#'*50}")

    # 1. 抽帧
    if not args.skip_frames:
        n = extract_frames(args.video, str(images_dir), args.fps, args.max_size)
        if n < 5:
            print("❌ 帧数太少, 已退出")
            return

    # 2-4. SfM
    db_path, sparse_dir = run_colmap_sfm(str(images_dir), str(out / "sfm"))

    # 5. MVS
    ply_path = run_colmap_mvs(str(images_dir), str(sparse_dir), str(out / "sfm"))

    # 如 MVS 不可用, 从稀疏重建导出点云
    if ply_path is None:
        rec = pycolmap.Reconstruction(str(sparse_dir / "0"))
        sparse_ply = out / "mesh" / "sparse_cloud.ply"
        sparse_ply.parent.mkdir(parents=True, exist_ok=True)
        rec.export_PLY(str(sparse_ply))
        ply_path = sparse_ply
        print("  → 使用稀疏点云 (5k~6k 点)")

    # 6. 网格
    mesh, mesh_path = reconstruct_mesh(str(ply_path), str(out / "mesh"), args.depth)

    # 7. 导出
    simplify_and_export(str(mesh_path), str(out / "export"), args.faces)

    print(f"\n{'#'*50}")
    print(f"#  ✅ 全部完成!")
    print(f"#  最终模型: {out.resolve() / 'export' / 'model.obj'}")
    print(f"{'#'*50}\n")


if __name__ == "__main__":
    main()