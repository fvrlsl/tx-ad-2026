#!/bin/bash
# 纯数据检查启动脚本 —— 不做训练，只分析数据并打印到 log/stdout
# 用法: bash run_inspect.sh [--max_rows 100000] [额外参数...]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/inspect_data.py" \
    --max_rows 100000 \
    "$@"
