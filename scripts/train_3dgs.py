#!/usr/bin/env python3
"""
train_3dgs.py — 用 gsplat 从 COLMAP 结果训练 3D Gaussian Splatting

针对 4GB 显存优化:
  * 图像默认降采样到 800px
  * 高斯数量上限 (默认 30 万)
  * SH 阶数 0 (仅基础颜色, 省显存)
  * 单卡, 单视图每步

输入: COLMAP 稀疏模型 (TXT 格式) + 图像目录
输出: gaussians.ply (标准 3DGS 格式), gaussians.splat, preview_*.png

用法:
  python train_3dgs.py --images out/images --model out/sparse_txt -o out/gs
  python train_3dgs.py --images out/images --model out/sparse_txt -o out/gs --iters 7000
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


# ============================================================ COLMAP TXT 解析
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        cid, model, w, h = int(p[0]), p[1], int(p[2]), int(p[3])
        pr = [float(x) for x in p[4:]]
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = pr[0]; cx, cy = pr[1], pr[2]
        else:
            fx, fy, cx, cy = pr[0], pr[1], pr[2], pr[3]
        out[cid] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return out


def read_images(path):
    lines = [ln.rstrip("\n") for ln in open(path)]
    out = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1; continue
        p = s.split()
        if len(p) < 10:
            i += 1; continue
        q = [float(x) for x in p[1:5]]
        t = np.array([float(x) for x in p[5:8]])
        out.append(dict(name=" ".join(p[9:]), cam_id=int(p[8]),
                        R=qvec2rotmat(q), t=t))
        i += 2  # 跳过 2D 观测行
    return out


def read_points3d(path):
    xyz, rgb = [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.uint8)


# ============================================================ 工具
def ssim(img1, img2, window_size=11):
    """简化 SSIM (单通道三通道分别计算取均值)"""
    c = img1.shape[1]
    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()
    win = (g[:, None] @ g[None, :]).expand(c, 1, window_size, window_size)
    mu1 = F.conv2d(img1, win, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(img2, win, padding=window_size // 2, groups=c)
    s1 = F.conv2d(img1 * img1, win, padding=window_size // 2, groups=c) - mu1 ** 2
    s2 = F.conv2d(img2 * img2, win, padding=window_size // 2, groups=c) - mu2 ** 2
    s12 = F.conv2d(img1 * img2, win, padding=window_size // 2, groups=c) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return ssim_map.mean()


def knn_scale(xyz, k=3):
    """每个点到第 3 近邻的距离 (分块计算)"""
    n = len(xyz)
    out = np.zeros(n, np.float32)
    t = torch.from_numpy(xyz)
    chunk = 2048
    for i in range(0, n, chunk):
        d = torch.cdist(t[i:i + chunk], t)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = 1e9
        vals, _ = torch.topk(d, k, dim=1, largest=False)
        out[i:i + chunk] = vals[:, -1].numpy()
    return out


# ============================================================ 高斯模型
class Gaussians(nn.Module):
    def __init__(self, means, colors, scales, sh_degree=0):
        super().__init__()
        n = len(means)
        self.means = nn.Parameter(torch.from_numpy(means).float())
        self.scales = nn.Parameter(torch.log(torch.from_numpy(scales).float()))
        self.quats = nn.Parameter(torch.zeros(n, 4))
        self.quats.data[:, 0] = 1.0
        self.opacities = nn.Parameter(torch.full((n,), -2.0))  # logit(0.12)
        self.sh_degree = sh_degree
        k = (sh_degree + 1) ** 2
        sh = torch.zeros(n, k, 3)
        sh[:, 0, :] = torch.from_numpy(
            (colors.astype(np.float32) / 255.0 - 0.5) / SH_C0)
        self.sh = nn.Parameter(sh)

    @property
    def num(self):
        return len(self.means)

    def params(self):
        return [self.means, self.scales, self.quats, self.opacities, self.sh]


# ============================================================ 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", required=True, help="COLMAP TXT 模型目录")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--max-size", type=int, default=800)
    ap.add_argument("--sh-degree", type=int, default=0)
    ap.add_argument("--max-gaussians", type=int, default=300000)
    ap.add_argument("--max-scale", type=float, default=0.05,
                    help="高斯最大尺度(归一化场景单位), 0 表示不限制")
    ap.add_argument("--densify-until", type=float, default=0.8,
                    help="占总迭代比例")
    ap.add_argument("--grad-thresh", type=float, default=3e-5)
    ap.add_argument("--opacity-reset", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("==== 读取 COLMAP 模型 ====")
    model_dir = Path(args.model)
    if not (model_dir / "cameras.txt").exists():
        # 二进制模型 → TXT
        import subprocess
        colmap_bin = os.environ.get(
            "COLMAP_BIN", os.path.expanduser("~/.local/colmap-install/bin/colmap"))
        txt = Path(args.output) / "sparse_txt"
        txt.mkdir(parents=True, exist_ok=True)
        print(f"  转换二进制模型 -> {txt}")
        subprocess.run([colmap_bin, "model_converter",
                        "--input_path", str(model_dir),
                        "--output_path", str(txt),
                        "--output_type", "TXT"], check=True,
                       capture_output=True)
        model_dir = txt

    cams = read_cameras(model_dir / "cameras.txt")
    imgs = read_images(model_dir / "images.txt")
    xyz, rgb = read_points3d(model_dir / "points3D.txt")
    print(f"  图像 {len(imgs)}  初始点 {len(xyz):,}")

    # ---- 场景归一化 (质心 + 尺度)
    centers = np.stack([-im["R"].T @ im["t"] for im in imgs])
    c0 = centers.mean(axis=0)
    radius = np.linalg.norm(centers - c0, axis=1).max()
    scene_scale = 1.0 / max(radius, 1e-6)
    xyz_n = (xyz - c0) * scene_scale
    print(f"  场景归一化: 中心 {c0.round(3)}, 尺度 {scene_scale:.4f}")

    # ---- 读取图像 & 内参
    import cv2
    data = []  # (name, image tensor HWC, K 4x4, viewmat)
    for im in imgs:
        p = Path(args.images) / im["name"]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h0, w0 = bgr.shape[:2]
        sc = args.max_size / max(h0, w0) if max(h0, w0) > args.max_size else 1.0
        if sc < 1.0:
            bgr = cv2.resize(bgr, (int(w0 * sc), int(h0 * sc)),
                             interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        cam = cams[im["cam_id"]]
        K = torch.tensor([
            [cam["fx"] * sc, 0, cam["cx"] * sc],
            [0, cam["fy"] * sc, cam["cy"] * sc],
            [0, 0, 1]], dtype=torch.float32, device=dev)
        R = torch.from_numpy(im["R"]).float().to(dev)
        t = torch.from_numpy(scene_scale * (im["R"] @ c0 + im["t"])).float().to(dev)
        vm = torch.eye(4, device=dev)
        vm[:3, :3] = R
        vm[:3, 3] = t
        img = torch.from_numpy(bgr[:, :, ::-1].copy()).float().to(dev) / 255.0
        data.append((im["name"], img, K, vm, w, h))

    if len(data) < 3:
        raise RuntimeError("可用图像不足")
    print(f"  加载 {len(data)} 帧, 分辨率 {data[0][4]}x{data[0][5]}")

    # ---- 初始化高斯
    print("==== 初始化高斯 ====")
    if len(xyz_n) < 100:
        raise RuntimeError("稀疏点太少")
    nn_d = knn_scale(xyz_n, k=3)
    scales = np.maximum(nn_d * 0.5, 1e-4)[:, None] * np.ones((1, 3))
    g = Gaussians(xyz_n, rgb, scales, args.sh_degree).to(dev)
    print(f"  初始高斯 {g.num:,}")

    # ---- 优化器
    lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh": 2.5e-3}
    opts = {
        "means": torch.optim.Adam([g.means], lr=lrs["means"]),
        "scales": torch.optim.Adam([g.scales], lr=lrs["scales"]),
        "quats": torch.optim.Adam([g.quats], lr=lrs["quats"]),
        "opacities": torch.optim.Adam([g.opacities], lr=lrs["opacities"]),
        "sh": torch.optim.Adam([g.sh], lr=lrs["sh"]),
    }
    from gsplat import rasterization

    densify_until = int(args.iters * args.densify_until)
    ema_grad = torch.zeros(g.num, device=dev)
    ema_count = torch.zeros(g.num, device=dev)
    best_psnr = 0.0

    print(f"==== 训练 {args.iters} 迭代 ====")
    for it in range(args.iters):
        # 学习率衰减 (means)
        lr = lrs["means"] * (0.01 ** (it / args.iters))
        for grp in opts["means"].param_groups:
            grp["lr"] = lr

        _, img, K, vm, w, h = data[np.random.randint(len(data))]
        gt = img.permute(2, 0, 1)[None]

        rgb_r, alpha, meta = rasterization(
            g.means, F.normalize(g.quats, dim=-1), torch.exp(g.scales),
            torch.sigmoid(g.opacities), g.sh, vm[None], K[None], w, h,
            sh_degree=args.sh_degree, near_plane=0.01, far_plane=100.0,
            absgrad=(it < densify_until), packed=False,
            rasterize_mode="antialiased")
        pred = rgb_r.permute(0, 3, 1, 2)  # (1,3,H,W)

        l1 = F.l1_loss(pred, gt)
        ssim_val = ssim(pred, gt)
        loss = 0.8 * l1 + 0.2 * (1 - ssim_val)
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        loss.backward()

        for o in opts.values():
            o.step()

        # 限制最大尺度: 过大的高斯会在渲染时铺满屏幕形成拖影
        if args.max_scale > 0:
            with torch.no_grad():
                g.scales.data.clamp_(max=math.log(args.max_scale))

        # ---- 致密化统计
        if it < densify_until and it > 100 and it % 100 == 0:
            with torch.no_grad():
                m2d = meta["means2d"]
                if m2d.dim() == 3:
                    ag = m2d.absgrad
                    if ag is None:
                        ag = m2d.grad
                    grad = ag.norm(dim=-1)[0]
                    radii = meta["radii"][0]
                    if radii.dim() > 1:
                        radii = radii[:, 0]
                else:  # packed
                    ids = meta["gaussian_ids"]
                    grad = torch.zeros(g.num, device=dev)
                    ag = meta["means2d"].absgrad
                    grad[ids] = ag.norm(dim=-1)
                    radii = torch.zeros(g.num, device=dev)
                    radii[ids] = meta["radii"][:, 0]

                vis = radii > 0
                score = torch.where(vis, grad, torch.zeros_like(grad))
                n_cand = int((score > args.grad_thresh).sum())
                # 每轮最多增长 8%, 防止显存爆炸
                n_add = min(n_cand, max(1, int(g.num * 0.10)))
                if os.environ.get("GS_DEBUG"):
                    print(f"    [densify] vis={int(vis.sum())} "
                          f"grad max={grad.max():.2e} cand={n_cand} add={n_add}")

                if n_add > 0:
                    idx = torch.topk(score, n_add).indices
                    big = torch.exp(g.scales).max(dim=1).values > 0.01
                    clone_idx = idx[~big[idx]]
                    split_idx = idx[big[idx]]
                    if len(clone_idx):
                        g = _append(g, clone_idx, split=False)
                    if len(split_idx):
                        g = _append(g, split_idx, split=True)

                # 剪枝
                keep = torch.sigmoid(g.opacities) > 0.005
                if g.num > args.max_gaussians:
                    _, top = torch.topk(torch.sigmoid(g.opacities),
                                        args.max_gaussians)
                    keep = torch.zeros(g.num, dtype=torch.bool, device=dev)
                    keep[top] = True
                if keep.sum() < g.num:
                    g = _select(g, keep)
                g = g.to(dev)
                # 重建优化器 (参数被替换)
                opts = {
                    "means": torch.optim.Adam([g.means], lr=lr),
                    "scales": torch.optim.Adam([g.scales], lr=lrs["scales"]),
                    "quats": torch.optim.Adam([g.quats], lr=lrs["quats"]),
                    "opacities": torch.optim.Adam([g.opacities], lr=lrs["opacities"]),
                    "sh": torch.optim.Adam([g.sh], lr=lrs["sh"]),
                }

        # ---- 不透明度重置
        if args.opacity_reset > 0 and it > 0 and it % args.opacity_reset == 0 \
                and it < densify_until:
            with torch.no_grad():
                g.opacities.fill_(-4.6)
            opts["opacities"] = torch.optim.Adam([g.opacities],
                                                 lr=lrs["opacities"])

        if it % 200 == 0 or it == args.iters - 1:
            psnr = -10 * torch.log10(F.mse_loss(pred.clamp(0, 1), gt)).item()
            best_psnr = max(best_psnr, psnr)
            print(f"  it {it:5d}/{args.iters}  loss {loss.item():.4f}  "
                  f"psnr {psnr:.2f}  gauss {g.num:,}  "
                  f"显存 {torch.cuda.max_memory_allocated()/2**30:.2f}G")
        if it % 500 == 0:
            torch.cuda.empty_cache()

    # ---- 导出
    print("\n==== 导出 ====")
    means = g.means.detach().cpu().numpy()
    scales = torch.exp(g.scales).detach().cpu().numpy()
    quats = F.normalize(g.quats.detach(), dim=-1).cpu().numpy()
    opac = torch.sigmoid(g.opacities).detach().cpu().numpy()
    sh = g.sh.detach().cpu().numpy()  # (N,K,3)

    _save_ply(out / "gaussians.ply", means / scene_scale + c0,
              scales / scene_scale, quats, opac, sh)

    # 训练相机位置 (世界坐标), 供查看器默认从训练视角观察
    import json
    with open(out / "cameras.json", "w") as fp:
        json.dump({
            "center": [float(v) for v in c0],
            "scale": float(1.0 / scene_scale),
            "cameras": [[float(v) for v in c] for c in centers],
            # 每台相机的 world->camera 旋转 (行主序 9 个数), 查看器用来对齐滚转
            "rotations": [[float(v) for v in im["R"].reshape(-1)] for im in imgs],
            "resolution": [int(data[0][4]), int(data[0][5])],
            # 每台相机的内参 [fx, fy, cx, cy, w, h], 查看器据此设置视场角
            "intrinsics": [[float(v) for v in
                            (cams[im["cam_id"]]["fx"], cams[im["cam_id"]]["fy"],
                             cams[im["cam_id"]]["cx"], cams[im["cam_id"]]["cy"],
                             cams[im["cam_id"]]["w"], cams[im["cam_id"]]["h"])]
                           for im in imgs],
        }, fp, indent=1)
    _save_splat(out / "gaussians.splat", means, scales, quats, opac, sh, scene_scale, c0)
    _render_previews(out, g, data, args, dev)

    summary = {
        "gaussians": int(g.num),
        "iters": args.iters,
        "sh_degree": args.sh_degree,
        "psnr": round(best_psnr, 2),
        "elapsed_s": round(time.time() - t_start, 1),
        "ply": str(out / "gaussians.ply"),
    }
    (out / "summary_gs.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _append(g, idx, split):
    """克隆/分裂高斯, 返回新 Gaussians"""
    with torch.no_grad():
        m = g.means[idx]
        sc = g.scales[idx]
        q = g.quats[idx]
        o = g.opacities[idx]
        sh = g.sh[idx]
        if split:
            # 在原位置附近生成两个, 尺度缩小
            ofs = torch.randn_like(m) * torch.exp(sc) * 2
            m1, m2 = m + ofs, m - ofs
            sc_new = sc - math.log(1.6)
            nm = torch.cat([m1, m2])
            nsc = torch.cat([sc_new, sc_new])
            nq = torch.cat([q, q]); no = torch.cat([o, o])
            nsh = torch.cat([sh, sh])
        else:
            nm, nsc = m, sc
            nq, no, nsh = q, o, sh
        new = Gaussians.__new__(Gaussians)
        nn.Module.__init__(new)
        new.means = nn.Parameter(torch.cat([g.means.detach(), nm]))
        new.scales = nn.Parameter(torch.cat([g.scales.detach(), nsc]))
        new.quats = nn.Parameter(torch.cat([g.quats.detach(), nq]))
        new.opacities = nn.Parameter(torch.cat([g.opacities.detach(), no]))
        new.sh = nn.Parameter(torch.cat([g.sh.detach(), nsh]))
        new.sh_degree = g.sh_degree
        return new


def _select(g, keep):
    new = Gaussians.__new__(Gaussians)
    nn.Module.__init__(new)
    new.means = nn.Parameter(g.means.detach()[keep])
    new.scales = nn.Parameter(g.scales.detach()[keep])
    new.quats = nn.Parameter(g.quats.detach()[keep])
    new.opacities = nn.Parameter(g.opacities.detach()[keep])
    new.sh = nn.Parameter(g.sh.detach()[keep])
    new.sh_degree = g.sh_degree
    return new


def _save_ply(path, means, scales, quats, opac, sh):
    """标准 3DGS PLY (含 f_rest 占位)"""
    n = len(means)
    k = sh.shape[1]
    rest = np.zeros((n, 45), np.float32)
    props = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(45)]
    props += ["opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    # 标准 3DGS 的 PLY 存 opacity logit, 查看器再 sigmoid
    p = np.clip(opac, 1e-6, 1 - 1e-6)
    opac_logit = np.log(p / (1 - p)).astype(np.float32)
    attrs = np.concatenate([
        means, np.zeros((n, 3), np.float32), sh[:, 0, :],
        rest,
        opac_logit[:, None],
        np.log(scales),
        quats,
    ], axis=1).astype(np.float32)

    with open(path, "wb") as f:
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        for p in props:
            header += f"property float {p}\n"
        header += "end_header\n"
        f.write(header.encode())
        f.write(attrs.tobytes())
    print(f"  PLY: {path} ({path.stat().st_size/1e6:.1f} MB, {n:,} 高斯)")


def _save_splat(path, means, scales, quats, opac, sh, s, c0):
    """antimatter15/splat 格式: 每高斯 32 字节 (含坐标反归一化)"""
    n = len(means)
    pos = (means / s + c0).astype(np.float32)   # 反归一化回 COLMAP 坐标
    sc = scales.astype(np.float32)
    color = np.clip((sh[:, 0, :] * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    alpha = (opac * 255).astype(np.uint8)
    q = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9)
    rot = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)

    out = np.zeros((n, 32), np.uint8)
    out[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
    out[:, 12:24] = sc.view(np.uint8).reshape(n, 12)
    out[:, 24:27] = color
    out[:, 27] = alpha
    out[:, 28:32] = rot
    with open(path, "wb") as f:
        f.write(out.tobytes())
    print(f"  SPLAT: {path} ({path.stat().st_size/1e6:.1f} MB)")


def _render_previews(out, g, data, args, dev):
    from gsplat import rasterization
    import cv2
    picks = [0, len(data) // 3, 2 * len(data) // 3, len(data) - 1]
    for n, i in enumerate(picks):
        _, img, K, vm, w, h = data[i]
        with torch.no_grad():
            rgb, alpha, _ = rasterization(
                g.means, F.normalize(g.quats, dim=-1), torch.exp(g.scales),
                torch.sigmoid(g.opacities), g.sh, vm[None], K[None], w, h,
                sh_degree=args.sh_degree, near_plane=0.01, far_plane=100.0)
        arr = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out / f"preview_{n}.png"), arr[:, :, ::-1])
        cv2.imwrite(str(out / f"gt_{n}.png"),
                    (img.cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1])
    print(f"  预览图 -> {out}/preview_*.png")


if __name__ == "__main__":
    main()
