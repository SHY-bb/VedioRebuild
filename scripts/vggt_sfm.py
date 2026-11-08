#!/usr/bin/env python3
"""
vggt_sfm.py — VGGT 前馈式 SfM 适配器

输入:
    <images>/frame_0001.jpg ...（由 reconstruct_dense.py 抽帧得到）

输出:
    <output>/sparse/0/{cameras,images,points3D}.bin  (COLMAP 二进制)
    <output>/sparse_vggt_text/                       (可调试的文本模型)

设计要点:
  * 只构建 VGGT 的 aggregator + camera_head + depth_head，跳过 point/track head
  * 权重流式加载为 FP16/BF16，避免 5GB FP32 权重压爆 4GB 显存
  * 4GB 显存默认自动降分辨率/帧数；OOM 时逐级回退
  * 直接写 COLMAP 文本格式，再用本机 COLMAP model_converter 转 BIN，
    绕开 pycolmap 3.x/4.x 的 Image/Rig/Frame API 差异
  * 不依赖 LightGlue/pyceres，也不做 BA（前馈位姿直接使用）
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 必须在 import torch 之前设置，减少碎片化显存
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_VGGT_DIR = os.path.expanduser("~/.local/vggt-src/vggt")
DEFAULT_VGGT_CKPT = str(MODELS_DIR / "VGGT-1B" / "model.safetensors")
DEFAULT_OMEGA_DIR = os.path.expanduser("~/.local/vggt-src/vggt-omega")
DEFAULT_OMEGA_CKPT = str(MODELS_DIR / "VGGT-Omega" / "vggt_omega_1b_512.pt")
DEFAULT_COLMAP = os.environ.get(
    "COLMAP_BIN", os.path.expanduser("~/.local/colmap-install/bin/colmap")
)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VGGT feed-forward SfM -> COLMAP")
    ap.add_argument("--images", required=True, help="图像目录（只包含图像）")
    ap.add_argument("--output", required=True, help="输出目录（通常是 out/）")
    ap.add_argument("--engine", choices=["vggt", "vggt_omega"], default="vggt")
    ap.add_argument("--colmap", default=DEFAULT_COLMAP, help="COLMAP 可执行文件")
    ap.add_argument("--vggt-dir", default=os.environ.get("VGGT_DIR", DEFAULT_VGGT_DIR))
    ap.add_argument("--checkpoint", default=os.environ.get("VGGT_CHECKPOINT", DEFAULT_VGGT_CKPT))
    ap.add_argument("--omega-dir", default=os.environ.get("VGGT_OMEGA_DIR", DEFAULT_OMEGA_DIR))
    ap.add_argument("--omega-checkpoint", default=os.environ.get("VGGT_OMEGA_CHECKPOINT", DEFAULT_OMEGA_CKPT))
    ap.add_argument("--resolution", type=int, default=0,
                    help="0=按显存自动；必须是 14 的倍数（518/448/392/336/280/224）")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0=按显存自动；超过时均匀抽帧")
    ap.add_argument("--conf-thres", type=float, default=5.0,
                    help="VGGT 深度置信度阈值（越小点越多）")
    ap.add_argument("--max-points", type=int, default=100000,
                    help="写入 COLMAP 的稀疏点上限")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"],
                    default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def select_images(image_dir: Path, max_frames: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    if len(paths) < 2:
        raise RuntimeError(f"VGGT 至少需要 2 张图像，实际 {len(paths)} 张: {image_dir}")
    if max_frames and len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in sorted(set(idx.tolist()))]
    return paths


def auto_plan(num_images: int, free_gb: float, resolution: int, max_frames: int):
    """根据显存与图像数量给出低显存默认配置。"""
    if resolution <= 0:
        if free_gb < 3.2:
            resolution = 336
        elif free_gb < 4.2:
            resolution = 392
        elif free_gb < 5.5:
            resolution = 448
        else:
            resolution = 518
    # 确保是 14 的倍数，且不小于 224
    resolution = max(224, int(round(resolution / 14.0)) * 14)

    if max_frames <= 0:
        if free_gb < 3.2:
            max_frames = min(num_images, 8)
        elif free_gb < 4.2:
            max_frames = min(num_images, 16)
        elif free_gb < 5.5:
            max_frames = min(num_images, 24)
        else:
            max_frames = min(num_images, 32)
    max_frames = max(2, min(max_frames, num_images))
    return resolution, max_frames


def _get_target(model, key: str):
    parts = key.split(".")
    try:
        mod = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
        return getattr(mod, parts[-1])
    except (AttributeError, KeyError):
        return None


def load_state_streaming(model, checkpoint: Path) -> None:
    """逐 tensor 加载，避免同时持有 5GB state_dict + 5GB model。"""
    model_keys = set(model.state_dict().keys())
    loaded = skipped = 0
    loaded_keys = set()

    if checkpoint.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                if key not in model_keys:
                    skipped += 1
                    continue
                target = _get_target(model, key)
                if target is None:
                    skipped += 1
                    continue
                tensor = f.get_tensor(key)
                if tuple(tensor.shape) != tuple(target.shape):
                    log(f"  [warn] shape mismatch, skip {key}: {tuple(tensor.shape)} vs {tuple(target.shape)}")
                    skipped += 1
                    del tensor
                    continue
                target.data.copy_(tensor.to(dtype=target.dtype))
                loaded += 1
                loaded_keys.add(key)
                del tensor
    else:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        for key, tensor in state.items():
            if key not in model_keys:
                skipped += 1
                continue
            target = _get_target(model, key)
            if target is None or tuple(tensor.shape) != tuple(target.shape):
                skipped += 1
                continue
            target.data.copy_(tensor.to(dtype=target.dtype))
            loaded += 1
            loaded_keys.add(key)
        del state
        gc.collect()

    missing = sorted(model_keys - loaded_keys)
    log(f"  权重加载: {loaded} 个 tensor，跳过 {skipped} 个（point/track head 等）")
    if missing:
        log(f"  [warn] 模型缺少 {len(missing)} 个 key，例如: {missing[:5]}")
    if loaded == 0:
        raise RuntimeError(f"没有从 {checkpoint} 加载到任何权重")
    gc.collect()


def load_vggt_model(vggt_dir: Path, checkpoint: Path, dtype, device: str):
    if not vggt_dir.exists():
        raise RuntimeError(f"VGGT 源码目录不存在: {vggt_dir}")
    if not checkpoint.exists():
        raise RuntimeError(
            f"VGGT 权重不存在: {checkpoint}\n"
            f"  下载: HF_ENDPOINT=https://hf-mirror.com hf download facebook/VGGT-1B "
            f"model.safetensors --local-dir {checkpoint.parent}"
        )

    sys.path.insert(0, str(vggt_dir))
    from vggt.models.vggt import VGGT

    log("  构建 VGGT (aggregator + camera + depth, 跳过 point/track head)...")
    # 只要相机和深度，point/track head 不实例化，显存更省
    model = VGGT(enable_point=False, enable_track=False)
    model.eval()
    model.to(dtype=dtype)
    load_state_streaming(model, checkpoint)
    model.to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return model


def run_inference(model, paths: list[Path], resolution: int, dtype, device: str,
                  frames_chunk_size: int = 4):
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    log(f"  预处理 {len(paths)} 张图像 -> {resolution}x{resolution}")
    images, original_coords = load_and_preprocess_images_square(
        [str(p) for p in paths], target_size=resolution
    )
    images = images.to(device=device, dtype=dtype)

    log("  VGGT 前向推理...")
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda")):
            aggregated, patch_start = model.aggregator(images[None])
            pose_enc = model.camera_head(aggregated)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )
            depth_map, depth_conf = model.depth_head(
                aggregated, images[None], patch_start,
                frames_chunk_size=max(1, int(frames_chunk_size)),
            )

    depth_map = depth_map.squeeze(0).float().cpu().numpy()
    depth_conf = depth_conf.squeeze(0).float().cpu().numpy()
    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()
    rgb = (images.detach().float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    rgb = np.transpose(rgb, (0, 2, 3, 1))  # S,H,W,3
    coords = original_coords.cpu().numpy()

    del aggregated, pose_enc, images
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return extrinsic, intrinsic, depth_map, depth_conf, rgb, coords


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def _quat_wxyz(R: np.ndarray):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(_orthonormalize(R)).as_quat()
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / max(n, 1e-12)


def export_colmap_text(
    out: Path,
    paths: list[Path],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    rgb: np.ndarray,
    coords: np.ndarray,
    conf_thres: float,
    max_points: int,
) -> tuple[Path, int]:
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    log("  反投影深度到世界坐标...")
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    S, H, W, _ = points_3d.shape
    points_xyf = create_pixel_coordinate_grid(S, H, W)

    # 置信度 + 有效深度 + 去掉方形 padding 区域
    finite_mask = np.isfinite(points_3d).all(axis=-1)
    depth_ok = depth_map[..., 0] > 1e-6
    conf_ok = depth_conf >= conf_thres
    effective_conf_thres = float(conf_thres)
    if int((finite_mask & depth_ok & conf_ok).sum()) < 1000:
        # 低分辨率下 VGGT 置信度整体偏低，绝对阈值 5.0 可能一个点都留不下。
        # 退回按分位数取点，保证稀疏模型可用。
        vals = depth_conf[np.isfinite(depth_conf)]
        if vals.size:
            # 先取最大值的一半（比 P50 更能过滤低置信点）；仍太少再退回 P50
            fallback = max(1.0, 0.5 * float(np.nanmax(vals)))
            if int((finite_mask & depth_ok & (depth_conf >= fallback)).sum()) < 1000:
                fallback = max(1.0, float(np.percentile(vals, 50)))
            effective_conf_thres = min(float(conf_thres), fallback)
            conf_ok = depth_conf >= effective_conf_thres
            log(
                f"  [warn] conf>={conf_thres:g} 点数过少，自动阈值 -> "
                f"{effective_conf_thres:.3f} (0.5*max / P50)"
            )
    log(
        f"  depth: min={np.nanmin(depth_map):.4g} max={np.nanmax(depth_map):.4g} "
        f"mean={np.nanmean(depth_map):.4g} | conf: min={np.nanmin(depth_conf):.4g} "
        f"max={np.nanmax(depth_conf):.4g} mean={np.nanmean(depth_conf):.4g}"
    )
    log(
        f"  masks: finite={int(finite_mask.sum())}/{finite_mask.size} "
        f"depth>0={int(depth_ok.sum())} conf>={effective_conf_thres:g}={int(conf_ok.sum())}"
    )
    conf_mask = finite_mask & depth_ok & conf_ok
    for i in range(S):
        x1, y1, x2, y2, _, _ = coords[i]
        conf_mask[i] &= (
            (points_xyf[i, :, :, 0] >= x1)
            & (points_xyf[i, :, :, 0] <= x2)
            & (points_xyf[i, :, :, 1] >= y1)
            & (points_xyf[i, :, :, 1] <= y2)
        )

    log(f"  coords[0]: {coords[0].tolist()} | after padding+conf: {int(conf_mask.sum())}")
    conf_mask = randomly_limit_trues(conf_mask, int(max_points))
    if conf_mask.sum() < 32:
        raise RuntimeError(
            f"VGGT 有效点太少 ({int(conf_mask.sum())})，可降低 --conf-thres 或提高分辨率"
        )

    pts = points_3d[conf_mask].astype(np.float64)
    xyf = points_xyf[conf_mask].astype(np.float64)
    cols = rgb[conf_mask]

    # 从方形 padded 模型坐标映射回原图坐标
    fi = xyf[:, 2].astype(np.int32)
    x1 = coords[fi, 0].astype(np.float64)
    y1 = coords[fi, 1].astype(np.float64)
    x2 = coords[fi, 2].astype(np.float64)
    y2 = coords[fi, 3].astype(np.float64)
    ow = coords[fi, 4].astype(np.float64)
    oh = coords[fi, 5].astype(np.float64)
    sx = ow / np.maximum(x2 - x1, 1e-6)
    sy = oh / np.maximum(y2 - y1, 1e-6)
    u = (xyf[:, 0] - x1) * sx
    v = (xyf[:, 1] - y1) * sy

    per_frame: list[list[tuple[float, float, int]]] = [[] for _ in range(S)]
    point_lines: list[str] = []
    for pid in range(len(pts)):
        fidx = int(fi[pid])
        p2d_idx = len(per_frame[fidx])
        per_frame[fidx].append((float(u[pid]), float(v[pid]), pid + 1))
        r, g, b = (int(c) for c in cols[pid][:3])
        point_lines.append(
            f"{pid + 1} {pts[pid,0]:.8f} {pts[pid,1]:.8f} {pts[pid,2]:.8f} "
            f"{r} {g} {b} 0.0 {fidx + 1} {p2d_idx}"
        )

    text_dir = out / "sparse_vggt_text"
    shutil.rmtree(text_dir, ignore_errors=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    cam_lines = []
    img_lines = []
    for i, path in enumerate(paths):
        x1i, y1i, x2i, y2i, wi, hi = coords[i]
        sx_i = wi / max(x2i - x1i, 1e-6)
        sy_i = hi / max(y2i - y1i, 1e-6)
        fx = intrinsic[i, 0, 0] * sx_i
        fy = intrinsic[i, 1, 1] * sy_i
        cx = (intrinsic[i, 0, 2] - x1i) * sx_i
        cy = (intrinsic[i, 1, 2] - y1i) * sy_i
        cam_lines.append(
            f"{i + 1} PINHOLE {int(round(wi))} {int(round(hi))} "
            f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}"
        )

        qw, qx, qy, qz = _quat_wxyz(extrinsic[i, :3, :3])
        tx, ty, tz = extrinsic[i, :3, 3]
        img_lines.append(
            f"{i + 1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
            f"{tx:.10f} {ty:.10f} {tz:.10f} {i + 1} {path.name}"
        )
        img_lines.append(
            " ".join(f"{x:.4f} {y:.4f} {pid}" for x, y, pid in per_frame[i])
        )
        img_lines.append("")  # 每个图像后空行

    (text_dir / "cameras.txt").write_text(
        "# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        + "\n".join(cam_lines) + "\n", encoding="utf-8"
    )
    (text_dir / "images.txt").write_text(
        "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "\n".join(img_lines), encoding="utf-8"
    )
    (text_dir / "points3D.txt").write_text(
        "# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
        + "\n".join(point_lines) + "\n", encoding="utf-8"
    )

    log(f"  COLMAP 文本模型: {text_dir} ({len(point_lines)} 点, {S} 图)")
    return text_dir, len(point_lines)


def convert_to_binary(colmap: str, text_dir: Path, out: Path) -> Path:
    bin_dir = out / "sparse" / "0"
    shutil.rmtree(bin_dir.parent, ignore_errors=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap, "model_converter",
        "--input_path", str(text_dir),
        "--output_path", str(bin_dir),
        "--output_type", "BIN",
    ]
    log("  " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.stdout:
        for line in p.stdout.splitlines():
            log("    " + line)
    if p.returncode != 0:
        raise RuntimeError(f"COLMAP model_converter 失败 ({p.returncode})")
    if not (bin_dir / "images.bin").exists():
        raise RuntimeError(f"转换后缺少 images.bin: {bin_dir}")
    return bin_dir


def run_omega(args, paths: list[Path], resolution: int, dtype, device: str):
    """VGGT-Ω 预留路径。

    官方 VGGT-Ω 需要 torch>=2.6、gated 权重；512 版官方峰值显存约 6GB，
    4GB 卡无法直接运行。这里保留清晰的错误信息，避免用户误以为已支持。
    """
    raise RuntimeError(
        "VGGT-Ω 尚未在当前环境启用。\n"
        "  原因: 官方权重需要申请访问；VGGT-Omega-1B-512 在 416x624 输入下官方峰值显存约 6GB，"
        "本机 RTX 3050 4GB 不足以直接运行。\n"
        f"  如果你已获得权重: 放到 {args.omega_checkpoint}，并安装 torch>=2.6 的独立环境后，"
        "再在 vggt_sfm.py 的 run_omega() 中接入。\n"
        "  当前请使用 VGGT 引擎（已适配 4GB 低显存路径）。"
    )


def main() -> int:
    args = parse_args()
    t0 = time.time()
    np.random.seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；VGGT 需要 NVIDIA GPU")

    if args.dtype == "auto":
        if device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[args.dtype]

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.images).resolve()

    free_gb = 0.0
    if device.startswith("cuda"):
        free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        log(f"GPU: {torch.cuda.get_device_name(0)} | free {free_gb:.2f} GiB | dtype {dtype}")

    all_paths = select_images(image_dir, 0)
    resolution, max_frames = auto_plan(
        len(all_paths), free_gb, args.resolution, args.max_frames
    )
    log(f"输入 {len(all_paths)} 张 -> 计划 {max_frames} 张 @ {resolution}px")

    model = None
    if args.engine == "vggt":
        log("\n加载 VGGT 模型（只加载一次，后续 OOM 只调整输入）...")
        model = load_vggt_model(
            Path(args.vggt_dir), Path(args.checkpoint), dtype, device
        )

    # OOM 回退：先按 14 的倍数降分辨率，再降帧数；限制尝试次数
    attempts = [(resolution, max_frames)]
    for r in (448, 392, 336, 280, 224):
        if r < resolution:
            attempts.append((r, max_frames))
    if max_frames > 8:
        attempts.append((resolution, max(8, max_frames // 2)))
        attempts.append((resolution, max(6, max_frames // 3)))
    seen = set()
    attempts = [a for a in attempts if not (a in seen or seen.add(a))][:6]

    last_err = None
    for attempt_idx, (res, mf) in enumerate(attempts, 1):
        paths = select_images(image_dir, mf)
        log(f"\n[尝试 {attempt_idx}/{len(attempts)}] {len(paths)} 张 @ {res}px")
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        try:
            if args.engine == "vggt":
                extrinsic, intrinsic, depth_map, depth_conf, rgb, coords = run_inference(
                    model, paths, res, dtype, device
                )
            else:
                run_omega(args, paths, res, dtype, device)
                raise AssertionError("unreachable")
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda error" not in msg:
                raise
            last_err = e
            log(f"  [OOM] {e}")
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    else:
        raise RuntimeError(
            "VGGT 在多次降级后仍然显存不足。\n"
            f"  最后错误: {last_err}\n"
            "  可手动降低: --vggt-resolution 336 --vggt-max-frames 8"
        )

    log("\n  导出 COLMAP 模型...")
    text_dir, n_points = export_colmap_text(
        out, paths, extrinsic, intrinsic, depth_map, depth_conf, rgb, coords,
        args.conf_thres, args.max_points,
    )
    bin_dir = convert_to_binary(args.colmap, text_dir, out)

    peak_alloc = peak_reserved = 0.0
    if device.startswith("cuda"):
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

    summary = {
        "engine": args.engine,
        "checkpoint": str(args.checkpoint),
        "images_total": len(all_paths),
        "images_used": len(paths),
        "resolution": res,
        "points": n_points,
        "dtype": str(dtype).replace("torch.", ""),
        "elapsed_s": round(time.time() - t0, 2),
        "peak_allocated_gb": round(peak_alloc, 2),
        "peak_reserved_gb": round(peak_reserved, 2),
        "model": str(bin_dir),
    }
    (out / "vggt_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("\n✅ VGGT SfM 完成")
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except RuntimeError as e:
        log(f"\n[VGGT 错误] {e}")
        sys.exit(2)
