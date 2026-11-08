# WorldRebuild

手机视频 → 三维模型 / 3D 高斯泼溅（3DGS）的桌面重建与实时查看工具。

C++17 + Qt 6 + OpenGL 3.3 负责交互与可视化；Python 负责调用 COLMAP、DepthAnything V3、
VGGT、3DGS 等算法。UI 层与算法层严格分离：`pipeline/` 中的算法类不依赖 Qt，
Qt 只出现在 `ui/` 与进程桥接处。

---

## 1. 功能特性

### 三种重建模式

| 模式 | 后端 | 产物 | 适用场景 |
|---|---|---|---|
| **快速预览** | pycolmap（CPU MVS） | 粗网格 | 几十秒内看效果、验脚本 |
| **稠密重建** | COLMAP 3.11 + CUDA PatchMatch | 稠密点云 + Poisson 网格 | 追求几何精度，百万点级 |
| **深度重建** | DepthAnything V3 + Open3D TSDF | 融合网格 | 弱纹理、低显存（4 GB 可跑） |

### 四种 SfM 引擎

- **COLMAP 增量式**（`--sfm incremental`）：逐张注册，鲁棒，默认。
- **GLOMAP 全局式**（`--sfm glomap`）：一次解全部位姿，适合大规模序列。
- **VGGT 前馈式**（`--sfm vggt`）：端到端回归相机/深度，无需特征匹配。
- **VGGT-Ω**（`--sfm vggt_omega`）：需 gated 权重，暂未启用。

### 质量提升：3DGS

对已有 COLMAP 结果训练 3D 高斯，输出标准 3DGS PLY（62 个 float 属性，opacity 存 logit）
与 `.splat`，并导出 `cameras.json`（位置 / 完整旋转 / 内参），查看器可还原原始机位视角。

### 查看器

- 网格渲染：缺失法线时按三角形叉积生成面积加权顶点法线，避免"纯色剪影"。
- 泼溅渲染：`gl_VertexID` 展开 6 顶点四边形（兼容 Mesa 软/硬渲染），CPU 深度排序 + 整数
  buffer texture（`usamplerBuffer`），超大高斯按 `(256/r)²` 做能量补偿。
- 自由视角（轨道 / 平移 / 缩放）、重置视角、截图导出。
- 可直接打开外部 `.ply` / `.splat` / `.obj`。

### 命令行 / 自动化

```bash
WorldRebuild                       # GUI
WorldRebuild model.ply             # 直接看图
WorldRebuild gaussians.ply --splat # 直接看泼溅
WorldRebuild --shot-gl out.png     # 无窗口渲染一帧（自动化验收）
WorldRebuild --sfm vggt --sfm-interface neural   # 预选引擎
```

---

## 2. 目录结构

```text
WorldRebuild/
├── CMakeLists.txt              构建脚本（ALGO / UI 分层，两个 target）
├── run.sh                      启动脚本（注入库路径与环境变量）
├── config.env.example          本机配置模板（复制为 config.env）
├── src/
│   ├── main.cpp                入口：GUI / 打开模型 / 截图
│   ├── ui/
│   │   ├── MainWindow.{h,cpp}  参数栏、模式选择、日志、阶段进度、产物探测
│   │   └── viewer/
│   │       └── ModelViewer.{h,cpp}  网格 + 3DGS 泼溅渲染、取景、相机恢复
│   ├── pipeline/               ← 算法编排层（除 QProcess 桥外无 Qt 依赖）
│   │   ├── FrameExtractor.{h,cpp}   视频抽帧（OpenCV）
│   │   ├── MeshProcessor.{h,cpp}    PLY/OBJ 读取、滤波、简化、平滑、法线
│   │   └── PipelineRunner.{h,cpp}   路径解析、参数拼装、子进程管理、阶段解析
│   └── test/                   命令行测试（抽帧 / 网格 / 端到端）
├── scripts/                    Python 算法脚本
│   ├── reconstruct_cpu.py      快速预览：pycolmap 全流程
│   ├── reconstruct_dense.py    抽帧 → SfM → 稠密 MVS / DA3 分派 → Poisson
│   ├── depth_tsdf.py           去畸变 → DA3 推理 → 尺度对齐 → TSDF
│   ├── vggt_sfm.py             VGGT / VGGT-Ω 前馈 SfM → COLMAP 格式
│   ├── train_3dgs.py           3DGS 训练 + PLY/splat/cameras.json 导出
│   ├── check_colmap_tracks.py  检查 track 连通性（诊断 SfM 质量）
│   └── evaluate_sfm.py         位姿精度评估（对照 GT）
├── models/                     开源模型权重（不入库，见 models/README.md）
│   ├── README.md               权重清单与下载命令
│   ├── DA3-SMALL/              DepthAnything V3 Small（~131 MB）
│   └── VGGT-1B/                VGGT-1B（~4.7 GB）
└── resources/                  Qt 资源
```

---

## 3. 环境要求

| 项目 | 版本 / 说明 |
|---|---|
| 系统 | Ubuntu 24.04（其他 Linux 发行版未验证） |
| 编译器 | GCC 11+ / C++17 |
| GPU | NVIDIA，显存 ≥ 4 GB（本机 RTX 3050 Laptop 4 GB 验证通过） |
| CUDA | 12.4（`nvcc` 与驱动匹配） |
| 磁盘 | 建议 ≥ 20 GB（依赖 + 权重 + 中间产物） |

> 4 GB 显存下推荐组合：**DA3-SMALL + 3DGS 800 px / 30 万高斯上限**。更大的深度模型
> （MEDIUM 需 6–8 GB、LARGE 需 10–12 GB）会 OOM，不建议在本机替换。

---

## 4. 依赖清单

### 4.1 C++（默认安装到 `/opt`）

| 库 | 版本 | 用途 |
|---|---|---|
| Qt | 6.8.0 | 窗口、控件、OpenGL 上下文 |
| OpenCV | 4.11.0 | 视频解码、抽帧、图像缩放 |
| PCL | 1.14.1 | 点云滤波、法线、网格简化/平滑 |
| CGAL | 5.6 | COLMAP 的 Delaunay/网格相关组件 |
| OpenGL | 3.3 Core | 渲染 |
| COLMAP | 3.11.1（CUDA 编译） | SfM + PatchMatch 稠密 MVS |

默认前缀布局：

```text
/opt/qt/6.8.0
/opt/opencv/4.11.0
/opt/pcl/1.14.1
/opt/cgal/5.6
```

换机器时可用 `-DWR_PREFIX_ROOT=/your/prefix` 或环境变量 `WR_PREFIX_ROOT` 指向别处。

### 4.2 Python（默认 `/opt/sam-venv`）

| 包 | 版本 | 用途 |
|---|---|---|
| torch / torchvision | 2.5.1+cu124 / 0.20.1+cu124 | 推理与训练 |
| gsplat | 1.5.3 | 3DGS 光栅化 |
| depth-anything-3 | 0.1.1 | DA3 单目深度 |
| pycolmap | 4.2.1 | 快速预览管线 |
| open3d | 0.19.0 | Poisson / TSDF |
| opencv-python | 4.11.0.86 | 图像 IO |
| numpy | 2.x | — |
| safetensors / huggingface_hub | — | 权重加载 / 下载 |

### 4.3 外部可执行文件

| 工具 | 默认路径 | 用途 |
|---|---|---|
| COLMAP | `~/.local/colmap-install/bin/colmap` | SfM / MVS / 模型格式转换 |
| GLOMAP（可选） | `~/.local/glomap-install/bin/glomap` | 全局式 SfM |
| VGGT 源码（可选） | `~/.local/vggt-src/vggt` | 前馈 SfM |

---

## 5. 环境搭建

### 5.1 CUDA

```bash
nvcc --version        # 应为 12.4.x
nvidia-smi
```

### 5.2 C++ 依赖（源码编译到 /opt）

四个库均从源码构建，典型方式：

```bash
# 以 Qt 为例（需先装好系统依赖：build-essential cmake ninja-build 及 X11/OpenGL 开发包）
cmake -S qt-everywhere-src-6.8.0 -B build-qt -G Ninja \
      -DCMAKE_INSTALL_PREFIX=/opt/qt/6.8.0 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build-qt -j"$(nproc)"
cmake --install build-qt
```

OpenCV / PCL / CGAL 同理，分别安装到 `/opt/opencv/4.11.0`、`/opt/pcl/1.14.1`、
`/opt/cgal/5.6`。安装后确认：

```bash
ls /opt/qt/6.8.0/lib/cmake/Qt6 \
   /opt/opencv/4.11.0/lib/cmake/opencv4 \
   /opt/pcl/1.14.1/share/pcl-1.14 \
   /opt/cgal/5.6/lib/cmake/CGAL
```

### 5.3 COLMAP（必须开启 CUDA，否则没有 PatchMatch）

```bash
cmake -S colmap-3.11.1 -B build-colmap -G Ninja \
      -DCMAKE_INSTALL_PREFIX="$HOME/.local/colmap-install" \
      -DCMAKE_BUILD_TYPE=Release \
      -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
      -DCGAL_ENABLED=OFF -DTESTS_ENABLED=OFF \
      -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DFETCH_POSELIB=ON
cmake --build build-colmap -j"$(nproc)"
cmake --install build-colmap

"$HOME/.local/colmap-install/bin/colmap" -h | head -1
# 期望输出包含 "with CUDA"
```

> `-DCMAKE_CUDA_ARCHITECTURES` 按显卡算力填写（RTX 30 系 = 86，40 系 = 89）。
> 若系统缺依赖且无 sudo，可用 `apt download` + `dpkg -x` 解包到本地目录后通过
> `CMAKE_PREFIX_PATH` 引入。

### 5.4 Python 虚拟环境

```bash
python3 -m venv /opt/sam-venv
source /opt/sam-venv/bin/activate

# PyTorch（按 CUDA 版本选择官方源）
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# 其余依赖（国内可用清华镜像）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  gsplat==1.5.3 open3d==0.19.0 pycolmap==4.2.1 \
  opencv-python==4.11.0.86 numpy safetensors huggingface_hub
```

DepthAnything V3 通常以源码 / wheel 形式安装（`depth-anything-3`），确保
`python -c "import depth_anything_3"` 能通过。

### 5.5 模型权重

权重体积大，**不随仓库分发**，按 [`models/README.md`](models/README.md) 下载到 `models/`：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 国内镜像

hf download depth-anything/Depth-Anything-V3-Small --local-dir models/DA3-SMALL
hf download facebook/VGGT-1B model.safetensors config.json --local-dir models/VGGT-1B
```

放好后应存在：

```text
models/DA3-SMALL/{config.json, model.safetensors}
models/VGGT-1B/{config.json, model.safetensors}
```

---

## 6. 构建

```bash
cd WorldRebuild
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

产物：

```text
build/WorldRebuild        GUI
build/WorldRebuild_test   命令行测试
```

测试：

```bash
./build/WorldRebuild_test
```

安装（可选）：

```bash
cmake --install build --prefix /your/prefix
# 需要连同权重一起打包时：
cmake -S . -B build -DWR_INSTALL_MODELS=ON
```

安装布局：

```text
<prefix>/bin/WorldRebuild
<prefix>/share/WorldRebuild/{scripts, resources, models}
```

---

## 7. 运行

```bash
./run.sh                 # 推荐：自动设置库路径与默认环境变量
# 或
LD_LIBRARY_PATH=/opt/opencv/4.11.0/lib:/opt/qt/6.8.0/lib:/opt/pcl/1.14.1/lib \
QT_PLUGIN_PATH=/opt/qt/6.8.0/plugins \
./build/WorldRebuild
```

启动时会打印实际生效的路径，便于排查：

```text
[Config] python   = /opt/sam-venv
[Config] scripts  = /path/to/WorldRebuild/scripts
[Config] models   = /path/to/WorldRebuild/models
```

### 本机配置

```bash
cp config.env.example config.env   # 按需修改，已被 .gitignore 忽略
./run.sh
```

---

## 8. 配置项

所有路径都遵循同一优先级：**显式设置 > 环境变量 > 自动探测 > 编译期默认值**。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `WR_PREFIX_ROOT` | `/opt` | C++ 依赖根目录 |
| `WR_PYTHON_ENV` | `/opt/sam-venv` | Python 虚拟环境 |
| `WR_SCRIPTS_DIR` | 自动探测 | 脚本目录 |
| `WR_MODELS_DIR` | 自动探测 | 模型根目录 |
| `WR_EXTRA_LIB_PATH` | `/opt/{opencv,qt,pcl}/.../lib` | 注入子进程的 `LD_LIBRARY_PATH` |
| `COLMAP_BIN` | `~/.local/colmap-install/bin/colmap` | COLMAP 可执行文件 |
| `GLOMAP_BIN` | `~/.local/glomap-install/bin/glomap` | GLOMAP 可执行文件 |
| `VGGT_DIR` | `~/.local/vggt-src/vggt` | VGGT 源码 |
| `VGGT_CHECKPOINT` | `<models>/VGGT-1B/model.safetensors` | VGGT 权重 |
| `VGGT_OMEGA_DIR` | `~/.local/vggt-src/vggt-omega` | VGGT-Ω 源码 |
| `VGGT_OMEGA_CHECKPOINT` | `<models>/VGGT-Omega/vggt_omega_1b_512.pt` | VGGT-Ω 权重 |
| `DA3_MODEL_DIR` | `<models>/DA3-SMALL` | DA3 权重目录 |

脚本目录自动探测顺序：

1. `$WR_SCRIPTS_DIR`
2. `<可执行文件>/../scripts`（源码树 `build/`）
3. `<可执行文件>/../share/WorldRebuild/scripts`（安装布局）
4. `<可执行文件>/scripts`

模型目录同理（`../models` → `../share/WorldRebuild/models`）。

---

## 9. 三种模式的调用链

### 快速预览（CPU MVS）

```text
MainWindow::onRun
  → PipelineRunner::run(Mode::Standard)
    → scripts/reconstruct_cpu.py
       抽帧 → pycolmap 特征/匹配/BA → CPU MVS → Poisson → 导出
```

### 稠密重建（COLMAP CUDA）

```text
Mode::Dense
  → scripts/reconstruct_dense.py --dense-method colmap [--sfm incremental|glomap|vggt]
       抽帧 → SfM → model_converter(→TXT) → image_undistorter
            → patch_match_stereo → stereo_fusion → Poisson → mesh.ply
```

### 深度重建（DA3 + TSDF）

```text
Mode::DepthTsdf
  → scripts/reconstruct_dense.py --dense-method da3
       → scripts/depth_tsdf.py
           去畸变 → DA3 逐帧深度 → 与 COLMAP 位姿做鲁棒 scale/shift 拟合
           → TSDF 融合（voxel = 中位深度 3%，sdf_trunc = 3×voxel）→ mesh_da3/mesh.ply
```

### 质量提升（3DGS）

```text
MainWindow::onQualityEnhance
  → PipelineRunner::runQualityEnhancement
    → scripts/train_3dgs.py --images <undistorted/images 优先> --model sparse_txt
         → gaussians.ply（logit opacity）+ gaussians.splat + cameras.json
```

---

## 10. 输出目录

```text
output/
├── images/                 抽帧结果
├── database.db             COLMAP 特征库
├── sparse/0/               COLMAP 稀疏模型（BIN）
├── sparse_txt/             转出的 TXT 模型（3DGS 用）
├── undistorted/            image_undistorter 输出
│   ├── images/
│   └── sparse/
├── depth/                  DA3 深度图（可选）
├── mesh/mesh.ply           Poisson 网格（稠密路线）
├── mesh_da3/mesh.ply       TSDF 网格（深度路线）
├── summary.json            各阶段耗时与统计
└── gs/
    ├── gaussians.ply       标准 3DGS PLY
    ├── gaussians.splat     查看器格式
    ├── cameras.json        位置 / 旋转 / 内参
    └── preview_*.png       训练过程预览
```

---

## 11. 已知限制

- **DA3 深度误差约 3–10%**，TSDF 网格精度不会优于 voxel 尺寸。
- 3DGS 背景轻微重影来自模型本身的大尺度、低不透明度高斯，可通过增加迭代、补拍环绕
  视频或收紧 `--max-scale` 缓解。
- 4 GB 显存下 3DGS 采用 800 px、30 万高斯上限；更高质量需要更大显存或更多视角。
- COLMAP 相机模型为 `OPENCV`，做 TSDF 前必须先 `image_undistorter`。
- 该 Mesa 环境不支持 `glDrawArraysInstanced` + `glVertexAttribDivisor`，查看器改用
  `gl_VertexID` 展开；整数 buffer texture 必须配 `usamplerBuffer`。
- PCL 的 `loadPLYFile` 对 `property double` 会静默返回全零点云，因此走自研
  `loadPlyGeneric`。

---

## 12. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| 启动报 `cannot open shared object file` | 没设 `LD_LIBRARY_PATH`，用 `./run.sh` 或检查 `WR_EXTRA_LIB_PATH` |
| `找不到脚本: .../scripts/xxx.py` | 设置了错误的 `WR_SCRIPTS_DIR`，或安装时未装 `scripts/` |
| `colmap ... with CUDA` 缺失 | COLMAP 编译时未加 `-DCUDA_ENABLED=ON` |
| PatchMatch 报 CUDA OOM | 降低 `--dense-size`（默认 1000） |
| DA3 推理 OOM | 确认使用 `DA3-SMALL`；关闭其他占用显存的进程 |
| 3DGS 训练 OOM | 降低 `--max-size` 或迭代数 |
| 查看器画面全黑 | 模型无顶点法线且三角化失败，检查 PLY 是否含 `face` |
| 泼溅透明度异常 | PLY 的 opacity 应为 **logit**，查看器再做 sigmoid |

---

## 13. 说明

- 本项目不包含模型权重，请遵守各上游模型（DepthAnything V3、VGGT、COLMAP 等）的许可证。
- 训练 / 重建产物（`output/`、`build/`、`models/` 中的权重）默认不纳入版本管理。
