#!/bin/bash
# 本地 demo 数据训练启动脚本
# 使用 .venv312 Python 环境（Python 3.12 + PyTorch 2.11）
# 运行方式：在 tx-ad-2026/ 目录下执行，或直接 bash tx-ad-2026/run_local.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${SCRIPT_DIR}/../.venv312"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 输出目录（放在 tx-ad-2026/ 下）
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
LOG_DIR="${SCRIPT_DIR}/logs"
TF_DIR="${SCRIPT_DIR}/tf_events"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${TF_DIR}"

export TRAIN_TF_EVENTS_PATH="${TF_DIR}"

"${VENV}/bin/python3" -u "${SCRIPT_DIR}/train.py" \
    --data_dir       "${SCRIPT_DIR}" \
    --schema_path    "${SCRIPT_DIR}/schema.json" \
    --ckpt_dir       "${CKPT_DIR}" \
    --log_dir        "${LOG_DIR}" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries    2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers    0 \
    --batch_size     64 \
    --num_epochs     5 \
    --patience       3 \
    --d_model        64 \
    --emb_dim        16 \
    --num_hyformer_blocks 2 \
    --valid_ratio    0.2 \
    --eval_every_n_steps 0 \
    --device         cpu \
    "$@"
