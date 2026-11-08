#!/usr/bin/env bash
# WorldRebuild 启动脚本
# 所有路径均可用环境变量覆盖，默认值对应本机 /opt 编译布局。
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 可选：从 config.env 读取本机配置（该文件不入库，参见 config.env.example）
if [[ -f "$APP_DIR/config.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$APP_DIR/config.env"
    set +a
fi

# 依赖前缀（Qt / OpenCV / PCL 安装根目录）
WR_PREFIX_ROOT="${WR_PREFIX_ROOT:-/opt}"

# Python 虚拟环境
export WR_PYTHON_ENV="${WR_PYTHON_ENV:-/opt/sam-venv}"

# 额外库搜索路径（一般不用改）
export WR_EXTRA_LIB_PATH="${WR_EXTRA_LIB_PATH:-${WR_PREFIX_ROOT}/opencv/4.11.0/lib:${WR_PREFIX_ROOT}/qt/6.8.0/lib:${WR_PREFIX_ROOT}/pcl/1.14.1/lib}"

# COLMAP / GLOMAP / VGGT（存在才设置，避免覆盖用户已配好的值）
export COLMAP_BIN="${COLMAP_BIN:-${HOME}/.local/colmap-install/bin/colmap}"
export GLOMAP_BIN="${GLOMAP_BIN:-${HOME}/.local/glomap-install/bin/glomap}"
export VGGT_DIR="${VGGT_DIR:-${HOME}/.local/vggt-src/vggt}"

export LD_LIBRARY_PATH="${WR_EXTRA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:-${WR_PREFIX_ROOT}/qt/6.8.0/plugins}"

cd "$APP_DIR/build"
exec ./WorldRebuild "$@"
