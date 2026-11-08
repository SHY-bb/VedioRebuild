#!/usr/bin/env python3
"""
reconstruct_dense.py — 基于 COLMAP CUDA CLI 的高质量稠密重建

流程:
  video → 抽帧 → SfM → 去畸变 → PatchMatch MVS (CUDA) → 稠密点云 → 网格
  SfM 分两种接口模式:
    * 几何优化: COLMAP 增量式 / GLOMAP 全局式
    * 神经前馈: VGGT (已接入, 4GB 低显存适配) / VGGT-Ω (需 >=6GB 且权重 gated)

相对旧版 (pycolmap) 的区别:
  * 稠密 MVS 使用源码编译的 CUDA 版 colmap CLI (百万级点云)
  * 网格优先用 Open3D Poisson (COLMAP 未编 CGAL 时)

用法:
  python reconstruct_dense.py video.mp4 -o out_dir --fps 2 --max-size 1600
  python reconstruct_dense.py video.mp4 -o out_dir --dense-method da3   # DA3 深度路线
  python reconstruct_dense.py video.mp4 -o out_dir --sfm glomap         # 全局式 SfM
  python reconstruct_dense.py video.mp4 -o out_dir --sfm vggt           # 神经前馈 VGGT
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- 配置
DEFAULT_COLMAP = os.environ.get(
    "COLMAP_BIN",
    os.path.expanduser("~/.local/colmap-install/bin/colmap"),
)
# 全局式 SfM（与 COLMAP 共用 database.db，输出同格式 sparse）
DEFAULT_GLOMAP = os.environ.get(
    "GLOMAP_BIN",
    os.path.expanduser("~/.local/glomap-install/bin/glomap"),
)
# 模型权重统一放在项目 models/ 下，避免依赖用户 home 目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VGGT_CKPT = os.environ.get(
    "VGGT_CHECKPOINT",
    str(PROJECT_ROOT / "models" / "VGGT-1B" / "model.safetensors"),
)
# 4GB 显存保守设置
GPU_INDEX = "0"


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


def run(cmd, logfile=None, env=None, check=True):
    """运行子进程并透传输出"""
    pretty = " ".join(str(c) for c in cmd)
    log(f"  $ {pretty}", logfile)
    p = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if p.stdout:
        for line in p.stdout.splitlines():
            log(f"    {line}", logfile)
    if check and p.returncode != 0:
        raise RuntimeError(f"命令失败 ({p.returncode}): {pretty}")
    return p.returncode


# ---------------------------------------------------------------- 抽帧
def extract_frames(video: str, out_dir: Path, fps: int, max_size: int) -> Path:
    import cv2

    img_dir = out_dir / "images"
    shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")

    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(round(vfps / fps)))

    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            h, w = frame.shape[:2]
            if max(w, h) > max_size:
                s = max_size / max(w, h)
                frame = cv2.resize(frame, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(img_dir / f"{saved:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        idx += 1
    cap.release()
    print(f"  ✅ 抽帧 {saved} 张 (视频 {vfps:.1f}fps, 间隔 {interval})")
    if saved < 3:
        raise RuntimeError("抽帧太少，无法重建")
    return img_dir


# ---------------------------------------------------------------- SfM
def pick_best_model(sparse: Path) -> Path:
    """mapper 可能产生多个子模型，选注册图像最多的那个"""
    models = sorted([p for p in sparse.iterdir() if p.is_dir()])
    if not models:
        raise RuntimeError(f"SfM 失败：{sparse} 下无相机模型")

    def _registered(model_dir) -> int:
        img_bin = model_dir / "images.bin"
        if not img_bin.exists():
            return len(list((model_dir / "images").glob("*")))
        import struct
        with open(img_bin, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]

    best = max(models, key=_registered)
    for m in models:
        if m != best:
            print(f"  (忽略子模型 {m.name}: {_registered(m)} 张)")
    n_img = _registered(best)
    if n_img < 3:
        raise RuntimeError(f"SfM 失败：最大模型仅注册 {n_img} 张图像")
    print(f"  ✅ 使用模型 {best.name}, 注册图像 {n_img}")
    return best


def neural_sfm(img_dir: Path, out: Path, logf, engine: str = "vggt",
               colmap: str = DEFAULT_COLMAP, vggt_dir: str = "",
               vggt_checkpoint: str = "", resolution: int = 0,
               max_frames: int = 0, conf_thres: float = 5.0,
               max_points: int = 100000, dtype: str = "auto"):
    """神经前馈 SfM：调用 scripts/vggt_sfm.py，输出 COLMAP 二进制模型。"""
    log("\n==== 阶段: 神经前馈 SfM (VGGT) ====", logf)
    log(f"  引擎: {engine}", logf)
    if engine == "vggt_omega":
        log("  (VGGT-Ω 需要 gated 权重和 >=6GB 显存；若未准备会给出明确错误)", logf)

    script = Path(__file__).resolve().parent / "vggt_sfm.py"
    if not script.exists():
        raise RuntimeError(f"找不到 VGGT 适配器: {script}")

    cmd = [
        sys.executable, script,
        "--images", img_dir,
        "--output", out,
        "--engine", engine,
        "--colmap", colmap,
        "--resolution", str(resolution),
        "--max-frames", str(max_frames),
        "--conf-thres", str(conf_thres),
        "--max-points", str(max_points),
        "--dtype", dtype,
    ]
    if vggt_dir:
        cmd += ["--vggt-dir", vggt_dir]
    if vggt_checkpoint:
        cmd += ["--checkpoint", vggt_checkpoint]

    run(cmd, logf)
    model = pick_best_model(out / "sparse")
    return None, model


def colmap_sfm(colmap: str, img_dir: Path, out: Path, logf,
               sfm_engine: str = "incremental", glomap: str = ""):
    db = out / "database.db"
    sparse = out / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    log("\n==== 阶段: COLMAP 特征提取 ====", logf)
    run([colmap, "feature_extractor",
         "--database_path", db,
         "--image_path", img_dir,
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", "OPENCV",
         "--SiftExtraction.use_gpu", "1",
         "--SiftExtraction.gpu_index", GPU_INDEX,
         "--SiftExtraction.max_image_size", "2000"], logf)

    log("\n==== 阶段: 特征匹配 (Exhaustive) ====", logf)
    run([colmap, "exhaustive_matcher",
         "--database_path", db,
         "--SiftMatching.use_gpu", "1",
         "--SiftMatching.gpu_index", GPU_INDEX], logf)

    if sfm_engine == "glomap":
        # 全局式 SfM：读同一个 database.db，输出 COLMAP 格式 sparse/0
        bin_path = glomap or DEFAULT_GLOMAP
        if not Path(bin_path).exists() and shutil.which(bin_path) is None:
            log("\n[错误] 找不到 glomap 可执行文件: " + str(bin_path), logf)
            log("  安装: https://github.com/colmap/glomap", logf)
            log("  或用 GLOMAP_BIN / --glomap 指定路径", logf)
            raise FileNotFoundError("找不到 glomap: " + str(bin_path))
        log("\n==== 阶段: 全局 SfM (GLOMAP) ====", logf)
        run([bin_path, "mapper",
             "--database_path", db,
             "--image_path", img_dir,
             "--output_path", sparse], logf)
    else:
        log("\n==== 阶段: 增量 SfM ====", logf)
        run([colmap, "mapper",
             "--database_path", db,
             "--image_path", img_dir,
             "--output_path", sparse,
             "--Mapper.ba_global_function_tolerance", "0.000001"], logf)

    models = sorted([p for p in sparse.iterdir() if p.is_dir()])
    if not models:
        raise RuntimeError("SfM 失败：无相机模型")

    best = pick_best_model(sparse)
    return db, best


# ---------------------------------------------------------------- 稠密 MVS
def colmap_dense(colmap: str, img_dir: Path, sparse_model: Path,
                 out: Path, logf, max_image_size: int):
    dense = out / "dense"
    shutil.rmtree(dense, ignore_errors=True)
    dense.mkdir(parents=True)

    log("\n==== 阶段: 图像去畸变 ====", logf)
    run([colmap, "image_undistorter",
         "--image_path", img_dir,
         "--input_path", sparse_model,
         "--output_path", dense,
         "--output_type", "COLMAP",
         "--max_image_size", str(max_image_size)], logf)

    log("\n==== 阶段: PatchMatch 立体匹配 (CUDA) ====", logf)
    # 4GB 显存: 降低图像尺寸 + 单卡
    run([colmap, "patch_match_stereo",
         "--workspace_path", dense,
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true",
         "--PatchMatchStereo.gpu_index", GPU_INDEX,
         "--PatchMatchStereo.max_image_size", str(max_image_size),
         "--PatchMatchStereo.window_radius", "5"], logf)

    log("\n==== 阶段: 立体融合 (稠密点云) ====", logf)
    fused = dense / "fused.ply"
    run([colmap, "stereo_fusion",
         "--workspace_path", dense,
         "--workspace_format", "COLMAP",
         "--input_type", "geometric",
         "--output_path", fused], logf)

    if not fused.exists() or fused.stat().st_size == 0:
        raise RuntimeError("稠密点云为空")
    print(f"  ✅ 稠密点云: {fused} ({fused.stat().st_size/1e6:.1f} MB)")
    return fused


# ---------------------------------------------------------------- 网格
def poisson_mesh(ply: Path, out: Path, depth: int, faces: int):
    import numpy as np
    import open3d as o3d

    log(f"\n==== 阶段: Poisson 网格 (depth={depth}) ====")
    pcd = o3d.io.read_point_cloud(str(ply))
    n = len(pcd.points)
    print(f"  输入点云: {n:,} 点")
    if n == 0:
        raise RuntimeError("点云为空")

    # 去离群点
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(50)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(~keep)
    mesh = mesh.simplify_quadric_decimation(faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    mesh_dir = out / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    ply_out = mesh_dir / "mesh.ply"
    obj_out = mesh_dir / "mesh.obj"
    o3d.io.write_triangle_mesh(str(ply_out), mesh)
    o3d.io.write_triangle_mesh(str(obj_out), mesh)
    print(f"  ✅ 网格: {len(mesh.vertices):,} 顶点 / {len(mesh.triangles):,} 面")
    return mesh_dir


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="视频 → 稠密三维重建 (COLMAP CUDA)")
    ap.add_argument("video")
    ap.add_argument("-o", "--output", default="./reconstruction_dense")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=1600)
    ap.add_argument("--dense-size", type=int, default=1200,
                    help="MVS 图像尺寸 (4GB 显存建议 1000-1400)")
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--faces", type=int, default=200000)
    ap.add_argument("--colmap", default=DEFAULT_COLMAP)
    ap.add_argument("--dense-method", choices=["colmap", "da3", "none"],
                    default="colmap")
    ap.add_argument("--sfm-interface", choices=["geometric", "neural"],
                    default=None,
                    help="SfM 接口模式: 几何优化(COLMAP/GLOMAP) 或 神经前馈(VGGT); "
                         "不填时按 --sfm 推断")
    ap.add_argument("--sfm",
                    choices=["incremental", "glomap", "vggt", "vggt_omega"],
                    default="incremental",
                    help="SfM 引擎: incremental/glomap (几何) 或 vggt/vggt_omega (神经)")
    ap.add_argument("--glomap", default=DEFAULT_GLOMAP,
                    help="glomap 可执行路径 (默认读 GLOMAP_BIN)")
    ap.add_argument("--vggt-dir", default=os.environ.get(
        "VGGT_DIR", os.path.expanduser("~/.local/vggt-src/vggt")),
        help="VGGT 源码目录")
    ap.add_argument("--vggt-checkpoint", default=DEFAULT_VGGT_CKPT,
                    help="VGGT-1B 权重 (默认 models/VGGT-1B/model.safetensors)")
    ap.add_argument("--vggt-resolution", type=int, default=0,
                    help="VGGT 输入分辨率 (0=按显存自动; 14 的倍数)")
    ap.add_argument("--vggt-max-frames", type=int, default=0,
                    help="VGGT 最多使用帧数 (0=按显存自动)")
    ap.add_argument("--vggt-conf-thres", type=float, default=5.0,
                    help="VGGT 深度置信度阈值")
    ap.add_argument("--vggt-max-points", type=int, default=100000,
                    help="写入 COLMAP 的稀疏点上限")
    ap.add_argument("--vggt-dtype", choices=["auto", "float16", "bfloat16", "float32"],
                    default="auto", help="VGGT 推理精度")
    ap.add_argument("--da3-model", default="", help="DA3 权重目录 (默认项目 models/DA3-SMALL)")
    ap.add_argument("--process-res", type=int, default=504, help="DA3 处理分辨率")
    ap.add_argument("--max-frames", type=int, default=0, help="DA3 最多使用的帧数")
    ap.add_argument("--skip-frames", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="复用已有 SfM 结果 (跳过特征提取/匹配/建图)")
    args = ap.parse_args()

    geometric_engines = {"incremental", "glomap"}
    neural_engines = {"vggt", "vggt_omega"}
    if args.sfm_interface is None:
        args.sfm_interface = "neural" if args.sfm in neural_engines else "geometric"
    if args.sfm_interface == "geometric" and args.sfm not in geometric_engines:
        ap.error("几何优化接口只支持 --sfm incremental|glomap")
    if args.sfm_interface == "neural" and args.sfm not in neural_engines:
        ap.error("神经前馈接口只支持 --sfm vggt|vggt_omega")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "pipeline.log", "w")

    t0 = time.time()
    try:
        if not args.skip_frames:
            log("==== 阶段: 抽帧 ====", logf)
            img_dir = extract_frames(args.video, out, args.fps, args.max_size)
        else:
            img_dir = out / "images"

        if not Path(args.colmap).exists():
            raise FileNotFoundError(
                f"找不到 colmap: {args.colmap}\n请先编译安装或设置 COLMAP_BIN")

        sparse_root = out / "sparse"
        if args.resume and sparse_root.exists() and any(sparse_root.iterdir()):
            log("==== 阶段: 复用已有 SfM 模型 ====", logf)
            model = pick_best_model(sparse_root)
        elif args.sfm_interface == "neural":
            try:
                _, model = neural_sfm(
                    img_dir, out, logf, engine=args.sfm, colmap=args.colmap,
                    vggt_dir=args.vggt_dir,
                    vggt_checkpoint=args.vggt_checkpoint,
                    resolution=args.vggt_resolution,
                    max_frames=args.vggt_max_frames,
                    conf_thres=args.vggt_conf_thres,
                    max_points=args.vggt_max_points,
                    dtype=args.vggt_dtype,
                )
            except RuntimeError as e:
                log(f"\n[错误] {e}", logf)
                raise SystemExit(2)
        else:
            _, model = colmap_sfm(args.colmap, img_dir, out, logf,
                                  sfm_engine=args.sfm, glomap=args.glomap)

        fused = None
        if args.dense_method == "colmap":
            fused = colmap_dense(args.colmap, img_dir, model, out, logf,
                                 args.dense_size)
        elif args.dense_method == "da3":
            # ---- DA3 深度 + TSDF 融合 (与 COLMAP MVS 二选一)
            from depth_tsdf import run_tsdf, DEFAULT_MODEL_DIR
            log("\n==== 阶段: DA3 单目深度 + Open3D TSDF ====", logf)
            summary = run_tsdf(
                img_dir, model, out,
                da3_model=args.da3_model or DEFAULT_MODEL_DIR,
                process_res=args.process_res,
                max_frames=args.max_frames,
                faces=args.faces,
                colmap_bin=args.colmap,
            )
            (out / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False))
            log(f"\n🎉 完成 (DA3+TSDF)，耗时 {summary['elapsed_s']}s", logf)
            return
        else:
            log("\n==== 跳过稠密, 使用稀疏点云 ====", logf)
            fused = None

        if fused is None:
            # 稀疏点云导出
            fused = out / "sparse.ply"
            run([args.colmap, "model_converter",
                 "--input_path", model,
                 "--output_path", fused,
                 "--output_type", "PLY"], logf)

        mesh_dir = poisson_mesh(fused, out, args.depth, args.faces)

        # 导出 TXT 模型 (3DGS 训练 / 后续处理使用)
        try:
            from depth_tsdf import export_colmap_txt
            log("\n==== 阶段: 导出 COLMAP TXT 模型 ====", logf)
            export_colmap_txt(args.colmap, model, out / "sparse_txt")
        except Exception as e:
            log(f"  (TXT 导出失败: {e})", logf)

        summary = {
            "images": len(list(img_dir.glob("*.jpg"))),
            "model": str(model),
            "dense": str(fused),
            "mesh": str(mesh_dir / "mesh.ply"),
            "elapsed_s": round(time.time() - t0, 1),
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        log(f"\n🎉 完成，耗时 {summary['elapsed_s']}s", logf)
        log(json.dumps(summary, indent=2, ensure_ascii=False), logf)
    finally:
        logf.close()


if __name__ == "__main__":
    main()
