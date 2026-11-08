#!/usr/bin/env python3
"""
evaluate_sfm.py — 评估两个 COLMAP 模型的相机位姿差异

用途：
  1. 用 COLMAP / GLOMAP 作为参考，评估 VGGT、VGGT+BA 的位姿精度；
  2. 对比 BA 前后、不同参数下的 ATE / RPE / 旋转误差；
  3. 没有真值时，至少给出模型规模、track、连通性和相对位姿差异。

注意：
  - 单目 SfM 是 up-to-scale 的，默认用 Sim(3) 对齐（Umeyama）后再算误差；
  - 这不是“真值评估”。没有 GT 时，参考模型只是伪真值；
  - 不要在 BA 自己优化的 track 上证明 BA 好，最好用独立真值或留出 track。

用法：
  /opt/sam-venv/bin/python scripts/evaluate_sfm.py \
    --pred /tmp/vggt-ba/sparse \
    --gt   /tmp/colmap-gt/sparse/0 \
    --out  /tmp/eval.json

  # 只看模型统计
  /opt/sam-venv/bin/python scripts/evaluate_sfm.py --pred /tmp/vggt-ba/sparse
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pycolmap


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def natural_key(s: str):
    """1.jpg, 2.jpg, 10.jpg 按数字顺序排，而不是字典序。"""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def rotation_angle_deg(R: np.ndarray) -> float:
    """两个旋转之间的大地测量角（度）。R 应是 R_pred^T R_gt 或反向。"""
    c = (np.trace(R) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    return math.degrees(math.acos(c))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    求 s, R, t 使 dst ≈ s R src + t。
    src, dst: (N, 3)。返回 (s, R, t)。
    """
    if src.shape != dst.shape or src.shape[0] < 2:
        raise ValueError("Umeyama 至少需要 2 个对应点")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src0 = src - mu_src
    dst0 = dst - mu_dst

    cov = (dst0.T @ src0) / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U @ Vt)
    R = U @ D @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src0 * src0, axis=1))
        if var_src < 1e-15:
            raise ValueError("源点退化，无法估计 Sim(3) 尺度")
        s = float(np.trace(np.diag(S) @ D) / var_src)
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def summarize(values):
    if values is None:
        return None
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return None
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "rmse": float(np.sqrt(np.mean(a * a))),
    }


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self):
        groups = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return list(groups.values())


# ---------------------------------------------------------------------------
# 读取模型
# ---------------------------------------------------------------------------

def load_poses(model_dir: Path):
    """
    返回 images: name -> {
      image_id, R_cw (world->camera), t_cw, C (camera center in world),
      R_wc (camera->world), width, height, fx, fy
    }
    """
    rec = pycolmap.Reconstruction(str(model_dir))
    images = {}
    for image_id, img in rec.images.items():
        if not img.has_pose:
            continue
        T_cw = img.cam_from_world()
        R_cw = np.asarray(T_cw.rotation.matrix(), dtype=np.float64)
        t_cw = np.asarray(T_cw.translation, dtype=np.float64)
        C = -R_cw.T @ t_cw
        R_wc = R_cw.T

        cam = img.camera
        params = list(cam.params)
        if cam.model in (pycolmap.CameraModelId.SIMPLE_PINHOLE,
                         pycolmap.CameraModelId.SIMPLE_RADIAL,
                         pycolmap.CameraModelId.RADIAL):
            fx = fy = float(params[0]) if params else 0.0
        else:
            fx = float(params[0]) if len(params) > 0 else 0.0
            fy = float(params[1]) if len(params) > 1 else fx

        name = Path(img.name).as_posix()
        images[name] = {
            "image_id": int(image_id),
            "R_cw": R_cw,
            "t_cw": t_cw,
            "C": C,
            "R_wc": R_wc,
            "width": int(cam.width),
            "height": int(cam.height),
            "fx": fx,
            "fy": fy,
        }
    return rec, images


def match_images(pred, gt):
    """优先按完整相对路径匹配；失败再按 basename 匹配。

    返回 (common, mode, pred_use, gt_use)。basename 模式下会把两个模型
    重映射成同一组 basename 键，避免调用方用 tuple 索引。
    """
    common = sorted(set(pred) & set(gt), key=natural_key)
    if len(common) >= 2:
        return common, "path", pred, gt

    pred_base = {}
    gt_base = {}
    for k in pred:
        pred_base.setdefault(Path(k).name, []).append(k)
    for k in gt:
        gt_base.setdefault(Path(k).name, []).append(k)
    common_base = sorted(set(pred_base) & set(gt_base), key=natural_key)
    # basename 有重名时直接拒绝，避免错配
    if any(len(pred_base[x]) != 1 or len(gt_base[x]) != 1 for x in common_base):
        raise RuntimeError("basename 存在重名，无法可靠匹配图像")

    pred_use = {x: pred[pred_base[x][0]] for x in common_base}
    gt_use = {x: gt[gt_base[x][0]] for x in common_base}
    return common_base, "basename", pred_use, gt_use


def align_centers(pred_centers, gt_centers, mode):
    if mode == "none":
        return 1.0, np.eye(3), np.zeros(3)
    s, R, t = umeyama(pred_centers, gt_centers, with_scale=(mode == "sim3"))
    return s, R, t


# ---------------------------------------------------------------------------
# 模型统计
# ---------------------------------------------------------------------------

def model_stats(rec: pycolmap.Reconstruction):
    track_lengths = []
    cross_view_points = 0
    pair_counts = Counter()
    image_ids = set(rec.images.keys())

    for point in rec.points3D.values():
        elements = point.track.elements
        length = len(elements)
        track_lengths.append(length)
        imgs = sorted({int(e.image_id) for e in elements})
        if len(imgs) >= 2:
            cross_view_points += 1
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    pair_counts[(imgs[i], imgs[j])] += 1

    uf = UnionFind(image_ids)
    for (a, b), c in pair_counts.items():
        if c >= 1:
            uf.union(a, b)
    comps = sorted(uf.components(), key=len, reverse=True)

    n_points = len(track_lengths)
    n_obs = int(sum(track_lengths))
    hist = Counter(track_lengths)
    errors = [float(p.error) for p in rec.points3D.values() if p.has_error]

    return {
        "images": len(rec.images),
        "images_with_pose": sum(1 for img in rec.images.values() if img.has_pose),
        "points": n_points,
        "observations": n_obs,
        "mean_track_length": (n_obs / n_points) if n_points else 0.0,
        "median_track_length": float(np.median(track_lengths)) if track_lengths else 0.0,
        "max_track_length": int(max(track_lengths)) if track_lengths else 0,
        "track_length_hist": {str(k): int(v) for k, v in sorted(hist.items())},
        "cross_view_points": cross_view_points,
        "cross_view_pct": (100.0 * cross_view_points / n_points) if n_points else 0.0,
        "components": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "pairs_ge5": sum(1 for c in pair_counts.values() if c >= 5),
        "pairs_ge20": sum(1 for c in pair_counts.values() if c >= 20),
        "mean_reproj_error_px": float(np.mean(errors)) if errors else None,
        "median_reproj_error_px": float(np.median(errors)) if errors else None,
    }


# ---------------------------------------------------------------------------
# 位姿评估
# ---------------------------------------------------------------------------

def evaluate_poses(pred_imgs, gt_imgs, common, align_mode, deltas):
    pred_centers = np.stack([pred_imgs[k]["C"] for k in common])
    gt_centers = np.stack([gt_imgs[k]["C"] for k in common])

    s, R_align, t_align = align_centers(pred_centers, gt_centers, align_mode)
    pred_centers_aligned = (s * (R_align @ pred_centers.T)).T + t_align

    # 世界坐标系旋转 R 后，camera->world 的旋转也要左乘 R
    R_wc_pred_aligned = [R_align @ pred_imgs[k]["R_wc"] for k in common]
    R_cw_pred_aligned = [x.T for x in R_wc_pred_aligned]
    R_cw_gt = [gt_imgs[k]["R_cw"] for k in common]
    C_gt = [gt_imgs[k]["C"] for k in common]

    # 绝对轨迹误差 ATE
    ate_errors = np.linalg.norm(pred_centers_aligned - gt_centers, axis=1)
    gt_center_mean = gt_centers.mean(axis=0)
    traj_scale = float(np.sqrt(np.mean(np.sum((gt_centers - gt_center_mean) ** 2, axis=1))))
    if traj_scale < 1e-12:
        traj_scale = 1.0

    # 绝对旋转误差
    rot_errors = []
    per_image = []
    for idx, k in enumerate(common):
        R_err = R_wc_pred_aligned[idx] @ gt_imgs[k]["R_wc"].T
        ang = rotation_angle_deg(R_err)
        rot_errors.append(ang)
        per_image.append({
            "name": k,
            "center_error": float(ate_errors[idx]),
            "rotation_error_deg": float(ang),
        })

    per_image.sort(key=lambda x: x["center_error"], reverse=True)
    ate = summarize(ate_errors)
    if ate is not None:
        ate["rmse_norm"] = ate["rmse"] / traj_scale
        ate["mean_norm"] = ate["mean"] / traj_scale
        ate["median_norm"] = ate["median"] / traj_scale

    # 相对位姿误差 RPE
    rpe = {}
    for delta in deltas:
        if delta <= 0 or delta >= len(common):
            continue
        rot_delta = []
        trans_delta = []
        trans_rel = []
        dir_delta = []
        for i in range(len(common) - delta):
            j = i + delta

            # 使用对齐后的 camera->world，避免把 pred 原始 R 混进来
            R_ij_pred = R_cw_pred_aligned[i] @ R_wc_pred_aligned[j]

            R_ij_gt = R_cw_gt[i] @ gt_imgs[common[j]]["R_wc"]

            t_ij_pred = R_cw_pred_aligned[i] @ (pred_centers_aligned[j] - pred_centers_aligned[i])
            t_ij_gt = R_cw_gt[i] @ (C_gt[j] - C_gt[i])

            rot_delta.append(rotation_angle_deg(R_ij_pred @ R_ij_gt.T))
            te = float(np.linalg.norm(t_ij_pred - t_ij_gt))
            trans_delta.append(te)
            denom = float(np.linalg.norm(t_ij_gt))
            trans_rel.append(te / denom if denom > 1e-12 else 0.0)
            n1 = np.linalg.norm(t_ij_pred)
            n2 = np.linalg.norm(t_ij_gt)
            if n1 > 1e-12 and n2 > 1e-12:
                c = float(np.clip(np.dot(t_ij_pred, t_ij_gt) / (n1 * n2), -1.0, 1.0))
                dir_delta.append(math.degrees(math.acos(c)))

        rpe[str(delta)] = {
            "pairs": len(rot_delta),
            "rotation_deg": summarize(rot_delta),
            "translation": summarize(trans_delta),
            "translation_relative": summarize(trans_rel),
            "translation_direction_deg": summarize(dir_delta),
        }

    return {
        "common_images": len(common),
        "alignment": {
            "mode": align_mode,
            "scale": float(s),
            "rotation": R_align.tolist(),
            "translation": t_align.tolist(),
        },
        "ate": ate,
        "rotation_deg": summarize(rot_errors),
        "rpe": rpe,
        "per_image_worst": per_image[:10],
        "trajectory_scale": traj_scale,
    }


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------

def print_stats(label, stats):
    print(f"{label}:")
    print(f"  图像: {stats['images']}（有pose {stats['images_with_pose']}） | "
          f"3D点: {stats['points']} | 观测: {stats['observations']}")
    print(f"  平均 track: {stats['mean_track_length']:.3f} | "
          f"中位数: {stats['median_track_length']:.1f} | "
          f"最大: {stats['max_track_length']}")
    print(f"  跨视图点: {stats['cross_view_points']} "
          f"({stats['cross_view_pct']:.2f}%) | 连通分量: {stats['components']} | "
          f"最大分量: {stats['largest_component']}")
    print(f"  图像对 >=5 track: {stats['pairs_ge5']} | >=20: {stats['pairs_ge20']}")
    if stats["mean_reproj_error_px"] is not None:
        print(f"  平均重投影误差: {stats['mean_reproj_error_px']:.3f}px | "
              f"中位数: {stats['median_reproj_error_px']:.3f}px")
    hist = stats["track_length_hist"]
    keys = [int(k) for k in hist]
    if keys:
        top = sorted(hist.items(), key=lambda kv: int(kv[0]))[:8]
        print("  track 长度分布: " + ", ".join(f"{k}:{v}" for k, v in top))


def print_eval(result):
    print("=" * 72)
    print(f"共同图像: {result['common_images']}")
    al = result["alignment"]
    print(f"对齐: {al['mode']} | scale={al['scale']:.6f}")
    ate = result["ate"]
    if ate:
        print(f"ATE: RMSE={ate['rmse']:.4f} | mean={ate['mean']:.4f} | "
              f"median={ate['median']:.4f} | P95={ate['p95']:.4f} | max={ate['max']:.4f}")
        print(f"     normalized RMSE={ate['rmse_norm']:.4%} | "
              f"median={ate['median_norm']:.4%}")
    rot = result["rotation_deg"]
    if rot:
        print(f"绝对旋转误差: mean={rot['mean']:.3f}° | median={rot['median']:.3f}° | "
              f"P95={rot['p95']:.3f}° | max={rot['max']:.3f}°")

    print("RPE:")
    for delta, r in result["rpe"].items():
        rr = r["rotation_deg"]
        tt = r["translation"]
        tr = r["translation_relative"]
        dd = r["translation_direction_deg"]
        line = (f"  Δ={delta} 帧 | pairs={r['pairs']} | "
                f"rot mean={rr['mean']:.3f}° median={rr['median']:.3f}° | "
                f"trans mean={tt['mean']:.4f} median={tt['median']:.4f} | "
                f"trans_rel median={tr['median']:.2%}")
        if dd:
            line += f" | dir median={dd['median']:.2f}°"
        print(line)

    worst = result.get("per_image_worst") or []
    if worst:
        print("误差最大的图像:")
        for x in worst[:5]:
            print(f"  {x['name']}: center={x['center_error']:.4f}, "
                  f"rot={x['rotation_error_deg']:.2f}°")


def main():
    ap = argparse.ArgumentParser(description="评估 COLMAP 模型位姿（ATE/RPE）与 track 统计")
    ap.add_argument("--pred", required=True, help="预测模型目录（COLMAP BIN/TXT）")
    ap.add_argument("--gt", default="", help="参考/真值模型目录；省略则只打印统计")
    ap.add_argument("--out", default="", help="JSON 输出路径")
    ap.add_argument("--align", choices=["sim3", "se3", "none"], default="sim3",
                    help="对齐方式；单目默认 sim3")
    ap.add_argument("--deltas", default="1,5,10",
                    help="RPE 帧间隔，逗号分隔")
    args = ap.parse_args()

    pred_dir = Path(args.pred).resolve()
    if not pred_dir.exists():
        raise RuntimeError(f"pred 不存在: {pred_dir}")

    pred_rec, pred_imgs = load_poses(pred_dir)
    pred_stats = model_stats(pred_rec)

    print("=" * 72)
    print(f"Pred: {pred_dir}")
    print_stats("Pred 模型统计", pred_stats)

    gt_dir = Path(args.gt).resolve() if args.gt else None
    gt_rec = gt_imgs = gt_stats = None
    if gt_dir:
        if not gt_dir.exists():
            raise RuntimeError(f"gt 不存在: {gt_dir}")
        gt_rec, gt_imgs = load_poses(gt_dir)
        gt_stats = model_stats(gt_rec)
        print()
        print(f"GT: {gt_dir}")
        print_stats("GT 模型统计", gt_stats)

    result = {
        "pred": str(pred_dir),
        "gt": str(gt_dir) if gt_dir else None,
        "pred_stats": pred_stats,
        "gt_stats": gt_stats,
    }

    if gt_imgs is not None:
        common, match_mode, pred_use, gt_use = match_images(pred_imgs, gt_imgs)
        if len(common) < 2:
            raise RuntimeError(f"共同图像不足（{len(common)}），无法评估位姿")
        deltas = []
        for x in args.deltas.split(","):
            x = x.strip()
            if x:
                deltas.append(int(x))
        pose_result = evaluate_poses(pred_use, gt_use, common, args.align, deltas)
        pose_result["match_mode"] = match_mode
        result["pose"] = pose_result
        print()
        print_eval(pose_result)
    else:
        print("\n未提供 --gt，只输出模型统计，不计算 ATE/RPE。")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
