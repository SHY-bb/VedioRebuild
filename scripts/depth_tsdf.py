#!/usr/bin/env python3
"""
depth_tsdf.py — DepthAnything V3 单目深度 + COLMAP 位姿 → Open3D TSDF 融合

思路:
  COLMAP SfM 提供准确的相机位姿与内参 (尺度一致)
  DA3 提供每帧高精度稠密深度 (相对尺度)
  逐帧把 DA3 深度对齐到 COLMAP 尺度 (利用稀疏三维点做稳健线性拟合)
  再用 Open3D ScalableTSDFVolume 融合出致密表面

相比 COLMAP PatchMatch MVS:
  * 弱纹理 / 反光 / 少纹理区域更稳
  * 显存占用极低 (DA3-SMALL 峰值 ~320MB)
  * 缺点: 单目深度有局部误差, 依赖 COLMAP 位姿精度

用法:
  python depth_tsdf.py --images out/images --model out/sparse/0 -o out
  python depth_tsdf.py --images out/images --model out/sparse/0 -o out \
      --voxel 0.004 --max-frames 120
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL_DIR = os.environ.get(
    "DA3_MODEL_DIR",
    str(Path(__file__).resolve().parent.parent / "models" / "DA3-SMALL"),
)


def log(msg):
    print(msg, flush=True)


# ============================================================ COLMAP 文本模型解析
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras_txt(path):
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            cam_id = int(p[0])
            model = p[1]
            w, h = int(p[2]), int(p[3])
            params = [float(x) for x in p[4:]]
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:  # PINHOLE / OPENCV / OPENCV_FISHEYE ...
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            cams[cam_id] = dict(model=model, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return cams


def read_images_txt(path):
    """返回 [(name, cam_id, R(w2c), t(w2c), {point3d_id: (x,y)})]"""
    images = []
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        p = line.split()
        if len(p) < 10:
            i += 1
            continue
        img_id = int(p[0])
        qvec = [float(x) for x in p[1:5]]
        tvec = np.array([float(x) for x in p[5:8]])
        cam_id = int(p[8])
        name = " ".join(p[9:])
        R = qvec2rotmat(qvec)
        obs = {}
        i += 1
        if i < len(lines):
            for tok in lines[i].split():
                pass
            vals = lines[i].split()
            for k in range(0, len(vals) - 2, 3):
                x, y, pid = float(vals[k]), float(vals[k + 1]), int(vals[k + 2])
                if pid > 0:
                    obs[pid] = (x, y)
        i += 1
        images.append(dict(id=img_id, name=name, cam_id=cam_id,
                           R=R, t=tvec, obs=obs))
    return images


def read_points3d_txt(path):
    pts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            pts[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return pts


# ============================================================ DA3 深度估计
class DA3Estimator:
    def __init__(self, model_dir, device="cuda"):
        import torch
        from depth_anything_3.api import DepthAnything3
        self.torch = torch
        self.device = device
        log(f"  加载 DA3: {model_dir}")
        self.model = DepthAnything3.from_pretrained(model_dir).to(device)
        self.model.eval()

    def depth(self, image_bgr, process_res=504):
        """输入 BGR uint8, 返回 (H,W) float32 相对深度"""
        import cv2
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred = self.model.inference([rgb], process_res=process_res)
        d = pred.depth[0]
        # 恢复到输入分辨率
        h, w = image_bgr.shape[:2]
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        return d.astype(np.float32)


# ============================================================ 深度-尺度对齐
def fit_scale_shift(da3_d, target_z, min_pts=8):
    """稳健拟合 target_z ≈ a * da3_d + b, 返回 (a, b)"""
    if len(da3_d) < min_pts:
        return None
    a, b = np.polyfit(da3_d, target_z, 1)
    # 2 轮迭代剔除离群点
    for _ in range(2):
        pred = a * da3_d + b
        resid = np.abs(pred - target_z)
        thr = max(np.median(resid) * 3.0, 1e-6)
        keep = resid < thr
        if keep.sum() < min_pts:
            break
        a, b = np.polyfit(da3_d[keep], target_z[keep], 1)
    return float(a), float(b)


# ============================================================ TSDF
def fuse_tsdf(images_dir, model_dir, out_dir, depth_maps, cameras, poses,
              voxel_length, sdf_trunc, max_depth):
    import open3d as o3d

    log(f"\n==== TSDF 融合 (voxel={voxel_length:.4f}, trunc={sdf_trunc:.4f}) ====")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    import cv2
    n = 0
    for name, depth, cam, pose in zip(
            [p["name"] for p in poses], depth_maps, cameras, poses):
        img_path = Path(images_dir) / name
        color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if color is None:
            continue
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        d = depth.copy()
        if cam["w"] != d.shape[1] or cam["h"] != d.shape[0]:
            d = cv2.resize(d, (cam["w"], cam["h"]), interpolation=cv2.INTER_NEAREST)
            color = cv2.resize(color, (cam["w"], cam["h"]),
                               interpolation=cv2.INTER_AREA)

        # 无效深度置 0, 超过上限剔除
        d[~np.isfinite(d)] = 0
        d[d <= 0] = 0
        d[d > max_depth] = 0

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            cam["w"], cam["h"], cam["fx"], cam["fy"], cam["cx"], cam["cy"])

        # COLMAP w2c -> 4x4
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = pose["R"]
        w2c[:3, 3] = pose["t"]

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(d.astype(np.float32))),
            depth_scale=1.0, depth_trunc=float(max_depth),
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)
        n += 1
        if n % 10 == 0:
            log(f"  已融合 {n} 帧")

    log(f"  融合完成: {n} 帧")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


# ============================================================ 主流程
def export_colmap_txt(colmap_bin, model_dir, txt_out):
    """把 COLMAP 二进制模型导出为 TXT, 供本脚本/3DGS 使用"""
    txt_out = Path(txt_out)
    txt_out.mkdir(parents=True, exist_ok=True)
    subprocess_run([colmap_bin, "model_converter",
                    "--input_path", str(model_dir),
                    "--output_path", str(txt_out),
                    "--output_type", "TXT"])
    return txt_out


def subprocess_run(cmd):
    import subprocess
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}\n{p.stderr}")
    return p


def run_tsdf(images_dir, model_dir, out_dir, da3_model=DEFAULT_MODEL_DIR,
             process_res=504, max_frames=0, voxel=0.0, voxel_ratio=0.03,
             faces=200000, max_depth_factor=2.0, device="cuda",
             colmap_bin=None):
    """DA3 深度 + COLMAP 位姿 → TSDF 网格。
    model_dir 可为二进制或 TXT; 内部会先去畸变并统一为 TXT。"""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    colmap_bin = colmap_bin or os.environ.get(
        "COLMAP_BIN", os.path.expanduser("~/.local/colmap-install/bin/colmap"))

    # ---- 去畸变 (TSDF 只能处理针孔相机)
    undist = out / "undistorted"
    if not (undist / "sparse" / "cameras.bin").exists() and \
       not (undist / "sparse" / "cameras.txt").exists():
        log("\n==== 阶段: 图像去畸变 ====")
        undist.mkdir(parents=True, exist_ok=True)
        subprocess_run([colmap_bin, "image_undistorter",
                        "--image_path", str(images_dir),
                        "--input_path", str(model_dir),
                        "--output_path", str(undist),
                        "--output_type", "COLMAP"])
    images_dir = undist / "images"
    txt_dir = out / "sparse_txt"
    if not (txt_dir / "cameras.txt").exists():
        export_colmap_txt(colmap_bin, undist / "sparse", txt_dir)
    model_dir = txt_dir

    log("==== 读取 COLMAP 模型 (去畸变后) ====")
    cams = read_cameras_txt(model_dir / "cameras.txt")
    imgs = read_images_txt(model_dir / "images.txt")
    pts3d = read_points3d_txt(model_dir / "points3D.txt")
    log(f"  相机 {len(cams)} / 图像 {len(imgs)} / 三维点 {len(pts3d):,}")

    if max_frames and len(imgs) > max_frames:
        step = len(imgs) / max_frames
        imgs = [imgs[int(i * step)] for i in range(max_frames)]
        log(f"  采样 {len(imgs)} 帧")

    log("\n==== DA3 深度估计 ====")
    est = DA3Estimator(da3_model, device)
    import cv2

    depth_maps, cameras, poses, all_depths = [], [], [], []
    aligned = degraded = 0

    for k, im in enumerate(imgs):
        ipath = Path(images_dir) / im["name"]
        bgr = cv2.imread(str(ipath), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        raw = est.depth(bgr, process_res)
        # DA3 按 process_res 输出低分辨率深度, 必须放大回原图尺寸,
        # 否则下面用原图坐标采样会错位 (尺度拟合全错)
        if raw.shape[:2] != bgr.shape[:2]:
            raw = cv2.resize(raw, (bgr.shape[1], bgr.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

        cam = cams[im["cam_id"]]
        us, vs, zs, ds = [], [], [], []
        for pid, (u, v) in im["obs"].items():
            X = pts3d.get(pid)
            if X is None:
                continue
            Xc = im["R"] @ X + im["t"]
            z = Xc[2]
            if z <= 0:
                continue
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < raw.shape[1] and 0 <= vi < raw.shape[0]):
                continue
            us.append(u); vs.append(v); zs.append(z); ds.append(raw[vi, ui])

        zs = np.asarray(zs, dtype=np.float64)
        ds = np.asarray(ds, dtype=np.float64)
        fit = fit_scale_shift(ds, zs, min_pts=8)

        if fit is not None:
            a, b = fit
            d_aligned = a * raw + b
            aligned += 1
        else:
            ratio = np.median(zs) / max(np.median(ds), 1e-8) if len(zs) else 1.0
            d_aligned = raw * ratio
            degraded += 1
        d_aligned = np.clip(d_aligned, 1e-6, None).astype(np.float32)
        # 轻度中值滤波, 抑制单目深度的椒盐噪声
        d_aligned = cv2.medianBlur(d_aligned, 5)
        dump = os.environ.get("DA3_DUMP_DEPTH")
        if dump:
            Path(dump).mkdir(parents=True, exist_ok=True)
            np.save(Path(dump) / (Path(im["name"]).stem + ".npy"), d_aligned)
        depth_maps.append(d_aligned)
        cameras.append(cam)
        poses.append(im)
        all_depths.append(d_aligned)

        if os.environ.get("DA3_DEBUG") and fit is not None:
            resid = np.abs(a * ds + b - zs)
            log(f"  帧{k}: 对齐点 {len(zs)} 残差中位 "
                f"{np.median(resid):.3f} ({100*np.median(resid)/max(np.median(zs),1e-6):.1f}%) "
                f"scale={a:.3f} shift={b:.3f}")
        if (k + 1) % 10 == 0:
            log(f"  {k+1}/{len(imgs)} 帧完成 (对齐 {aligned}, 退化 {degraded})")

    if not depth_maps:
        raise RuntimeError("没有可用帧")

    med = float(np.median(np.concatenate([d[d > 0][::50] for d in all_depths])))
    vx = voxel if voxel > 0 else med * voxel_ratio
    sdf_trunc = vx * 3.0
    max_depth = med * max_depth_factor
    log(f"\n  中位深度 {med:.4f}  体素 {vx:.5f}  截断 {max_depth:.4f}")

    mesh = fuse_tsdf(images_dir, model_dir, out, depth_maps, cameras, poses,
                     vx, sdf_trunc, max_depth)

    log("\n==== 网格后处理 ====")
    log(f"  原始: {len(mesh.vertices):,} 顶点 / {len(mesh.triangles):,} 面")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    # 去除小碎片 (仅保留累计 98% 面的连通块, 最多 30 块)
    try:
        labels = np.asarray(mesh.cluster_connected_triangles()[0])
        if labels.size:
            uniq, cnt = np.unique(labels, return_counts=True)
            order = np.argsort(-cnt)
            keep, acc = [], 0
            for i in order[:30]:
                keep.append(uniq[i])
                acc += cnt[i]
                if acc > cnt.sum() * 0.98:
                    break
            mask = np.isin(labels, keep)
            if not mask.all():
                mesh.remove_triangles_by_mask(~mask)
                mesh.remove_unreferenced_vertices()
                log(f"  去除碎片后: {len(mesh.vertices):,} 顶点 / "
                    f"{len(mesh.triangles):,} 面 ({len(keep)} 块)")
    except Exception as e:
        log(f"  (连通块过滤跳过: {e})")

    if len(mesh.triangles) > faces:
        mesh = mesh.simplify_quadric_decimation(faces)
    mesh.compute_vertex_normals()

    mesh_dir = out / "mesh_da3"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    import open3d as o3d
    o3d.io.write_triangle_mesh(str(mesh_dir / "mesh.ply"), mesh)
    o3d.io.write_triangle_mesh(str(mesh_dir / "mesh.obj"), mesh)

    summary = {
        "frames_used": len(depth_maps),
        "frames_aligned": aligned,
        "frames_degraded": degraded,
        "median_depth": round(med, 5),
        "voxel": round(vx, 6),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "mesh": str(mesh_dir / "mesh.ply"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "summary_da3.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"\n🎉 DA3+TSDF 完成 {summary['elapsed_s']}s")
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser(description="DA3 深度 + TSDF 稠密重建")
    ap.add_argument("--images", required=True, help="图像目录 (与 COLMAP 一致)")
    ap.add_argument("--model", required=True, help="COLMAP 稀疏模型目录 (TXT 格式)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--da3-model", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = 全部")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="TSDF 体素 (0=自动按场景尺度)")
    ap.add_argument("--voxel-ratio", type=float, default=0.03,
                    help="自动体素 = 中位深度 * 该比例")
    ap.add_argument("--faces", type=int, default=200000)
    ap.add_argument("--max-depth-factor", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--colmap", default=os.environ.get(
        "COLMAP_BIN", os.path.expanduser("~/.local/colmap-install/bin/colmap")))
    args = ap.parse_args()

    run_tsdf(args.images, args.model, args.output, args.da3_model,
             args.process_res, args.max_frames, args.voxel, args.voxel_ratio,
             args.faces, args.max_depth_factor, args.device, args.colmap)


if __name__ == "__main__":
    main()
