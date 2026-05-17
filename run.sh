#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ════════════════════════════════════════════════════════════════════════════
# Sequence Encoder 选型说明（基于线上数据分析 2026-05-07）
#
# 各域 p90 序列长度：seq_a=136, seq_b=1920, seq_c=1109, seq_d=3793
#
# ● transformer（standard self-attention，O(T²)）
#   seq_a 适用（T≤136），seq_b/c/d 会比 longer(K=64) 慢 200~800x，不推荐
#
# ● longer（Top-K most-recent + self-attention，O(K²)，默认）
#   只保留最近 K 条行为做 attention，K=64 下 AUC 接近全量，推荐起点
#   若 AUC 比 transformer 低 >0.002，尝试 K=128
#
# ● swiglu（纯 FFN，无 attention，O(T)）
#   最快，但无序列内交互，精度通常最低，适合超长序列的快速实验基线
#
# 调参建议：
#   1. 先用 longer K=64 跑一个完整 epoch 确认 AUC
#   2. 对比 K=32 / K=128 的 AUC 差距（二分搜索）
#   3. K 超过 128 后收益递减，通常无需继续增大
# ════════════════════════════════════════════════════════════════════════════

# ── Active config: LongerEncoder K=64（推荐，速度/精度最优平衡）────────────
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_encoder_type longer \
    --seq_top_k 64 \
    --use_target_aware_topk \
    --pre_topk 256 \
    --use_amp \
    "$@"

# ── Alternative A: LongerEncoder K=128（精度优先，速度约为 K=64 的 1/4）───
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --seq_encoder_type longer \
#     --seq_top_k 128 \
#     "$@"

# ── Alternative B: LongerEncoder K=32（速度优先，快速验证实验）─────────────
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --seq_encoder_type longer \
#     --seq_top_k 32 \
#     "$@"

# ── Alternative C: GroupNSTokenizer driven by ns_groups.json ────────────────
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --seq_encoder_type longer \
#     --seq_top_k 64 \
#     "$@"
