#!/usr/bin/env python3
"""
check_colmap_tracks.py — 检查 COLMAP 模型里有没有跨视图 track

判断依据不是“点数多少”，而是每个 3D 点的 track 里是否出现 >=2 个不同 image_id。
track 长度 = 1 时，BA 没有跨视图约束；track 长度 >=2 才能真正把相机“绑”起来。

用法:
    python scripts/check_colmap_tracks.py --model out/sparse/0
    python scripts/check_colmap_tracks.py --model out/sparse/0 --json /tmp/tracks.json
    python scripts/check_colmap_tracks.py --model out/sparse/0 --min-shared 5 --top 10
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="检查 COLMAP 模型的跨视图 track")
    ap.add_argument("--model", required=True, help="COLMAP 模型目录（BIN 或 TXT）")
    ap.add_argument("--colmap", default=os.path.expanduser("~/.local/colmap-install/bin/colmap"),
                    help="COLMAP 可执行文件（BIN 模型转 TXT 时需要）")
    ap.add_argument("--min-shared", type=int, default=1,
                    help="判定两张图“连通”所需的最少共享 track 数（默认 1）")
    ap.add_argument("--top", type=int, default=10, help="打印共享 track 最多的前 N 对图像")
    ap.add_argument("--json", default="", help="可选：把统计写到 JSON")
    return ap.parse_args()


def load_text_model(model_dir: Path, colmap_bin: str) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """返回含 cameras.txt/images.txt/points3D.txt 的目录；BIN 模型自动转换。"""
    if (model_dir / "cameras.txt").exists() and (model_dir / "images.txt").exists():
        return model_dir, None
    if not (model_dir / "cameras.bin").exists():
        raise RuntimeError(f"既不是 TXT 也不是 BIN 模型: {model_dir}")

    colmap = shutil.which(colmap_bin) or (colmap_bin if Path(colmap_bin).exists() else None)
    if not colmap:
        raise RuntimeError(
            f"BIN 模型需要 colmap model_converter，但找不到: {colmap_bin}\n"
            "  可加 --colmap /path/to/colmap，或先手动转成 TXT"
        )
    tmp = tempfile.TemporaryDirectory(prefix="colmap_tracks_")
    tmp_path = Path(tmp.name)
    cmd = [str(colmap), "model_converter",
           "--input_path", str(model_dir),
           "--output_path", str(tmp_path),
           "--output_type", "TXT"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        tmp.cleanup()
        raise RuntimeError("colmap model_converter 失败:\n" + (p.stdout or ""))
    return tmp_path, tmp


def parse_images(path: Path):
    """image_id -> {name, point_ids}；point_ids[idx] 是该图第 idx 个 2D 点对应的 3D 点。"""
    images = {}
    lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if not l.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        p = lines[i].split()
        if len(p) < 10:
            raise RuntimeError(f"images.txt 格式异常: {lines[i]!r}")
        image_id = int(p[0])
        name = p[9]
        i += 1
        point_ids = []
        if i < len(lines) and lines[i].strip():
            vals = lines[i].split()
            if len(vals) % 3 != 0:
                raise RuntimeError(f"images.txt 2D 点格式异常: {lines[i]!r}")
            for j in range(0, len(vals), 3):
                point_ids.append(int(vals[j + 2]))
            i += 1
        images[image_id] = {"name": name, "point_ids": point_ids}
    return images


def parse_points3d(path: Path):
    """point_id -> {xyz, rgb, error, pairs: [(image_id, point2d_idx)]}"""
    points = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = line.split()
        if len(p) < 8 or (len(p) - 8) % 2 != 0:
            raise RuntimeError(f"points3D.txt 格式异常: {line!r}")
        pid = int(p[0])
        pairs = [(int(p[j]), int(p[j + 1])) for j in range(8, len(p), 2)]
        points[pid] = {
            "xyz": tuple(float(x) for x in p[1:4]),
            "rgb": tuple(int(x) for x in p[4:7]),
            "error": float(p[7]),
            "pairs": pairs,
        }
    return points


def percentile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


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


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model).resolve()
    if not model_dir.exists():
        raise RuntimeError(f"模型不存在: {model_dir}")

    text_dir, tmp = load_text_model(model_dir, args.colmap)
    try:
        images = parse_images(text_dir / "images.txt")
        points = parse_points3d(text_dir / "points3D.txt")
    finally:
        if tmp is not None:
            tmp.cleanup()

    n_images = len(images)
    n_points = len(points)
    if n_images == 0 or n_points == 0:
        raise RuntimeError("模型为空（0 图或 0 点）")

    # 1) track 长度 + 有效性
    track_lengths = []
    cross_view_points = 0
    cross_view_observations = 0
    invalid_pairs = 0
    duplicate_image_tracks = 0
    image_track_counts = Counter()       # image_id -> track 长度 >=2 的点数
    pair_counts = Counter()              # (img_a, img_b) -> 共享 track 数

    for pid, point in points.items():
        pairs = point["pairs"]
        img_ids = [img for img, _ in pairs]
        track_len = len(pairs)
        track_lengths.append(track_len)

        if len(set(img_ids)) != len(img_ids):
            duplicate_image_tracks += 1

        # 校验 track 里的 (image_id, point2d_idx) 能否对上 images.txt
        for img_id, idx in pairs:
            if img_id not in images:
                invalid_pairs += 1
                continue
            point_ids = images[img_id]["point_ids"]
            if idx < 0 or idx >= len(point_ids) or point_ids[idx] != pid:
                invalid_pairs += 1

        unique_imgs = sorted(set(img_ids))
        if len(unique_imgs) >= 2:
            cross_view_points += 1
            cross_view_observations += track_len
            for img in unique_imgs:
                image_track_counts[img] += 1
            for a_i in range(len(unique_imgs)):
                for b_i in range(a_i + 1, len(unique_imgs)):
                    pair_counts[(unique_imgs[a_i], unique_imgs[b_i])] += 1

    n_obs = sum(track_lengths)
    mean_track = n_obs / n_points if n_points else 0.0
    sorted_lengths = sorted(track_lengths)
    len1 = sum(1 for x in track_lengths if x == 1)
    pct_cross = 100.0 * cross_view_points / n_points if n_points else 0.0

    # 2) 图像连通分量（按 min_shared 建边）
    uf = UnionFind(images.keys())
    for (a, b), c in pair_counts.items():
        if c >= args.min_shared:
            uf.union(a, b)
    comps = uf.components()
    comps.sort(key=len, reverse=True)

    total_pairs = n_images * (n_images - 1) // 2
    pairs_ge1 = len(pair_counts)
    pairs_ge5 = sum(1 for c in pair_counts.values() if c >= 5)
    pairs_ge20 = sum(1 for c in pair_counts.values() if c >= 20)

    # 3) 打印
    print("=" * 72)
    print(f"模型: {model_dir}")
    print(f"图像: {n_images} | 3D 点: {n_points} | 观测: {n_obs}")
    print(f"平均 track 长度: {mean_track:.3f} | 中位数: {percentile(sorted_lengths, 0.5):.1f} "
          f"| P95: {percentile(sorted_lengths, 0.95):.1f} "
          f"| 最大: {max(track_lengths) if track_lengths else 0}")
    print()
    print("track 长度分布:")
    hist = Counter(track_lengths)
    for k in range(1, min(max(hist), 10) + 1):
        c = hist.get(k, 0)
        print(f"  {k:>2} 视图: {c:>8} 点 ({100.0 * c / n_points:5.1f}%)")
    if max(hist) > 10:
        c = sum(v for k, v in hist.items() if k > 10)
        print(f"  >10 视图: {c:>8} 点 ({100.0 * c / n_points:5.1f}%)")
    print()
    print(f"跨视图 track (>=2 个不同 image_id): {cross_view_points} 点 "
          f"({pct_cross:.2f}%)，覆盖 {cross_view_observations} 个观测")
    print(f"长度为 1 的 track: {len1} 点 ({100.0 * len1 / n_points:.1f}%)")
    print(f"无效 track 观测: {invalid_pairs} | 同一图像重复出现的 track: {duplicate_image_tracks}")
    print()
    print(f"图像对共享 >=1 track: {pairs_ge1}/{total_pairs} "
          f"({100.0 * pairs_ge1 / total_pairs if total_pairs else 0:.1f}%)")
    print(f"图像对共享 >=5 track: {pairs_ge5} | >=20 track: {pairs_ge20}")
    print(f"图像连通分量 (min_shared={args.min_shared}): {len(comps)}")
    for idx, comp in enumerate(comps[:5], 1):
        names = [images[i]["name"] for i in comp[:3]]
        suffix = " ..." if len(comp) > 3 else ""
        print(f"  分量 {idx}: {len(comp)} 图  {', '.join(names)}{suffix}")
    if len(comps) > 5:
        print(f"  ... 另有 {len(comps) - 5} 个分量")

    if args.top > 0 and pair_counts:
        print()
        print(f"共享 track 最多的前 {args.top} 对图像:")
        for (a, b), c in pair_counts.most_common(args.top):
            print(f"  {images[a]['name']} <-> {images[b]['name']}: {c} tracks")

    # 4) 结论
    print()
    print("=" * 72)
    if cross_view_points == 0:
        print("结论: ❌ 没有跨视图 track。")
        print("  BA 没有把不同相机联系起来的观测；bundle_adjuster 跑得起来也没有多视图约束。")
        print("  典型原因: 每个 3D 点只从单帧深度反投影生成（track 长度 1），没有做特征匹配/三角化。")
        verdict = "no_cross_view_tracks"
    elif pct_cross < 1.0:
        print("结论: ⚠️ 跨视图 track 很少（<1% 的点）。")
        print("  BA 可能只能约束少数相机对，位姿提升有限；先检查匹配/三角化是否失败。")
        verdict = "few_cross_view_tracks"
    elif len(comps) > 1:
        print("结论: ⚠️ 有跨视图 track，但图像图不连通。")
        print(f"  {len(comps)} 个连通分量；BA 只能在分量内优化，分量之间需要 Sim(3)/刚体对齐。")
        verdict = "disconnected"
    elif pairs_ge5 == 0:
        print("结论: ⚠️ 有跨视图 track，但每对相机共享的 track 很少（没有 >=5）。")
        print("  BA 约束较弱，容易出现漂移；建议增加匹配或使用更鲁棒的初始化。")
        verdict = "weak_constraints"
    else:
        print("结论: ✅ 存在可用的跨视图 track。")
        print("  可以跑 BA；建议关注无效观测、track 长度分布和连通分量数量。")
        verdict = "ok"

    if invalid_pairs:
        print("  ⚠️ 存在无效 track 观测，跑 BA 前建议先修复或重新三角化。")
    if duplicate_image_tracks:
        print("  ⚠️ 存在同一图像重复观测的 track；BA 仍可运行，但建议检查重复特征/三角化。")

    if args.json:
        out = {
            "model": str(model_dir),
            "images": n_images,
            "points": n_points,
            "observations": n_obs,
            "mean_track_length": mean_track,
            "median_track_length": percentile(sorted_lengths, 0.5),
            "p95_track_length": percentile(sorted_lengths, 0.95),
            "max_track_length": max(track_lengths) if track_lengths else 0,
            "track_length_hist": dict(sorted(hist.items())),
            "cross_view_points": cross_view_points,
            "cross_view_pct": pct_cross,
            "length1_points": len1,
            "invalid_pairs": invalid_pairs,
            "duplicate_image_tracks": duplicate_image_tracks,
            "pairs_ge1": pairs_ge1,
            "pairs_ge5": pairs_ge5,
            "pairs_ge20": pairs_ge20,
            "total_pairs": total_pairs,
            "components_min_shared": len(comps),
            "min_shared": args.min_shared,
            "top_pairs": [
                {"a": images[a]["name"], "b": images[b]["name"], "tracks": c}
                for (a, b), c in pair_counts.most_common(args.top)
            ],
            "verdict": verdict,
        }
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nJSON: {args.json}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(2)
