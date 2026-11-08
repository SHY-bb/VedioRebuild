# Open-source Models & Weights
# =============================

所有推理模型和权重统一放在本目录，避免依赖 `~/.local` 或临时缓存。

## 当前目录

```text
models/
├── README.md
├── DA3-SMALL/              # DepthAnything V3 Small
│   ├── config.json
│   └── model.safetensors   # ~131 MB
└── VGGT-1B/                # VGGT-1B 前馈 SfM
    ├── config.json
    └── model.safetensors   # ~4.7 GB
```

`VGGT-Ω` 目前未安装权重；如以后申请到 gated checkpoint，按下面路径放置：

```text
models/VGGT-Omega/vggt_omega_1b_512.pt
```

项目代码默认读取：

| 用途 | 默认路径 | 覆盖环境变量 |
|---|---|---|
| 模型根目录 | 项目 `models/` | `WR_MODELS_DIR` |
| VGGT-1B 权重 | `models/VGGT-1B/model.safetensors` | `VGGT_CHECKPOINT` |
| VGGT 源码 | `~/.local/vggt-src/vggt` | `VGGT_DIR` |
| VGGT-Ω 权重 | `models/VGGT-Omega/vggt_omega_1b_512.pt` | `VGGT_OMEGA_CHECKPOINT` |
| DA3-SMALL | `models/DA3-SMALL` | `DA3_MODEL_DIR` |

> 模型根目录由 C++ 侧 `PipelineRunner::resolveModelDir()` 解析：优先 `WR_MODELS_DIR`，
> 否则依次尝试 `<可执行文件>/../models` 与 `<可执行文件>/../share/WorldRebuild/models`。
> 因此源码树（`build/`）和安装布局（`<prefix>/bin`）都能自动找到权重。
>
> VGGT 源码是 Python 包，不属于模型权重；目前仍放在 `~/.local/vggt-src/vggt`。
> 如果希望整个项目完全自包含，可以把源码也移到 `third_party/vggt`，并同步修改 `VGGT_DIR`。

## 下载命令

### VGGT-1B

```bash
HF_ENDPOINT=https://hf-mirror.com hf download facebook/VGGT-1B \
  model.safetensors config.json \
  --local-dir models/VGGT-1B
```

### VGGT-Ω（需要 gated 访问）

```bash
# 先在 Hugging Face 申请 facebook/VGGT-Omega 访问权限
hf download facebook/VGGT-Omega \
  vggt_omega_1b_512.pt \
  --local-dir models/VGGT-Omega
```

### DepthAnything V3 Small

```bash
HF_ENDPOINT=https://hf-mirror.com hf download depth-anything/Depth-Anything-V3-Small \
  --local-dir models/DA3-SMALL
```

## 注意

- 这些文件很大，不要提交到 Git；如使用 Git，请在 `.gitignore` 中忽略 `*.safetensors`、`*.pt`、`*.pth`、`*.ckpt`、`*.onnx`。
- `CMakeLists.txt` 的 `install(DIRECTORY models/ ...)` 会把整个 `models/` 安装到 `share/WorldRebuild/models`，打包时注意体积。
- 移动权重后，旧的 `~/.local/vggt-weights` 路径不再使用；如其他脚本仍引用旧路径，请改到 `models/VGGT-1B` 或设置 `VGGT_CHECKPOINT`。
