"""纯数据检查脚本（自包含，无外部本地依赖）—— 不做训练，只分析数据并打印到 stdout/log。

用法:
    python inspect_data.py [--data_dir <目录>] [--max_rows N]

环境变量（与 train.py 保持一致，可直接复用平台注入的变量）:
    TRAIN_DATA_PATH   训练数据目录（*.parquet + schema.json）
    TRAIN_LOG_PATH    日志目录
"""

import argparse
import glob
import json
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════
# 配置常量
# ════════════════════════════════════════════════════════════════════════════

KNOWN_TS_FIDS = {39, 67, 27, 26}
OUTLIER_IQR_FACTOR = 3.0
CORR_THRESHOLD = 0.90
SEQ_LEN_SAMPLE_CAP = 500_000  # 序列长度水库采样上限

# ════════════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════════════

class ScalarAccumulator:
    """流式累积标量特征统计，逐文件 update，最终 flush 产出结果。

    内存策略：水库采样，最多保留 sample_cap 个值用于分位数估计；
    同时精确累积 count/sum/sum_sq 用于精确 mean/std 计算。
    """

    SAMPLE_CAP = 500_000

    def __init__(self) -> None:
        self.total = 0
        self.null_count = 0
        self.valid_count = 0
        self.running_sum = 0.0
        self.running_sum_sq = 0.0
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.sample: List[float] = []  # 水库采样，最多 SAMPLE_CAP 个

    def update(self, series: pd.Series) -> None:
        self.total += len(series)
        self.null_count += int(series.isna().sum())
        valid = series.dropna().values.astype(np.float64)
        if len(valid) == 0:
            return
        self.valid_count += len(valid)
        self.running_sum += float(valid.sum())
        self.running_sum_sq += float((valid ** 2).sum())
        self.global_min = min(self.global_min, float(valid.min()))
        self.global_max = max(self.global_max, float(valid.max()))
        # 水库采样
        if len(self.sample) < self.SAMPLE_CAP:
            self.sample.extend(valid.tolist())
        else:
            keep_prob = self.SAMPLE_CAP / self.valid_count
            mask = np.random.random(len(valid)) < keep_prob
            new_vals = valid[mask].tolist()
            if new_vals:
                replace_indices = np.random.randint(0, self.SAMPLE_CAP, len(new_vals))
                for idx, val in zip(replace_indices, new_vals):
                    self.sample[idx] = val

    def flush(self) -> dict:
        stats: dict = {
            'total': self.total,
            'null_count': self.null_count,
            'null_rate': f'{self.null_count / max(1, self.total):.2%}',
            'valid_count': self.valid_count,
        }
        if self.valid_count == 0:
            stats['note'] = '全部为空'
            return stats
        # 精确 mean/std（全量累积值）
        mean = self.running_sum / self.valid_count
        variance = self.running_sum_sq / self.valid_count - mean ** 2
        std = float(np.sqrt(max(0.0, variance)))
        # 分位数用采样近似
        sample_arr = np.array(self.sample, dtype=np.float64)
        stats.update({
            'min':  round(self.global_min, 6),
            'max':  round(self.global_max, 6),
            'mean': round(mean, 6),
            'std':  round(std,  6),
            'p50':  round(float(np.percentile(sample_arr, 50)), 6),
            'p90':  round(float(np.percentile(sample_arr, 90)), 6),
            'p99':  round(float(np.percentile(sample_arr, 99)), 6),
        })
        return stats


class StringAccumulator:
    """流式累积字符串特征统计，逐文件合并 Counter。
    
    超过 counter_cap 条目时截断，只保留当前 top-K，控制内存。
    """

    def __init__(self, top_k: int = 10000, counter_cap: int = 1_000_000) -> None:
        self.top_k = top_k
        self.counter_cap = counter_cap
        self.total = 0
        self.null_count = 0
        self.counter: Counter = Counter()
        self.truncated = False

    def update(self, series: pd.Series) -> None:
        self.total += len(series)
        self.null_count += int(series.isna().sum())
        valid = series.dropna()
        self.counter.update(valid.astype(str).tolist())
        # 超过上限时截断到 top-K，释放内存
        if len(self.counter) > self.counter_cap:
            self.truncated = True
            self.counter = Counter(dict(self.counter.most_common(self.top_k)))

    def flush(self) -> dict:
        unique_count = len(self.counter)
        top_freq = dict(self.counter.most_common(self.top_k))
        return {
            'total': self.total,
            'null_count': self.null_count,
            'null_rate': f'{self.null_count / max(1, self.total):.2%}',
            'unique_count': unique_count,
            'freq_truncated': self.truncated,
            'freq_note': f'unique 超过上限，仅展示 top {self.top_k}' if self.truncated else '',
            'top_freq': {str(k): int(v) for k, v in top_freq.items()},
        }


class ArrayAccumulator:
    """流式累积序列特征统计。
    
    内存策略：
    - 长度列表：每行只记录一个 int，极省内存
    - 元素值：均匀水库采样，最多保留 sample_cap 个，用于 min/max/mean/std 估计
    - 整数元素：额外保留 top-K Counter
    """

    def __init__(self, top_k: int = 20, sample_cap: int = 500_000) -> None:
        self.top_k = top_k
        self.sample_cap = sample_cap
        self.total = 0
        self.null_count = 0
        self.lengths: List[int] = []
        self.elem_sample: List[float] = []
        self.elem_total_count = 0
        self.elem_counter: Counter = Counter()
        self.is_int_elem: Optional[bool] = None  # 延迟判断

    def update(self, series: pd.Series) -> None:
        self.total += len(series)
        self.null_count += int(series.isna().sum())
        for raw in series.dropna():
            arr = raw.tolist() if isinstance(raw, np.ndarray) else list(raw)
            self.lengths.append(len(arr))
            if not arr:
                continue
            flat = np.array(arr, dtype=np.float64)
            self.elem_total_count += len(flat)
            # 水库采样：以 sample_cap/elem_total_count 的概率保留
            if len(self.elem_sample) < self.sample_cap:
                self.elem_sample.extend(flat.tolist())
            else:
                # 简化版：每次只随机替换，保证采样均匀
                keep_prob = self.sample_cap / self.elem_total_count
                mask = np.random.random(len(flat)) < keep_prob
                new_samples = flat[mask].tolist()
                if new_samples:
                    replace_indices = np.random.randint(0, self.sample_cap, len(new_samples))
                    for idx, val in zip(replace_indices, new_samples):
                        self.elem_sample[idx] = val
            # 判断是否整数型（只判断一次）
            if self.is_int_elem is None and len(flat) > 0:
                self.is_int_elem = bool(np.all(flat == flat.astype(np.int64)))
            if self.is_int_elem:
                self.elem_counter.update(flat.astype(np.int64).tolist())
                if len(self.elem_counter) > 200_000:
                    self.elem_counter = Counter(dict(self.elem_counter.most_common(self.top_k)))

    def flush(self) -> dict:
        result: dict = {
            'total': self.total,
            'null_count': self.null_count,
            'null_rate': f'{self.null_count / max(1, self.total):.2%}',
        }
        if not self.lengths:
            result['note'] = '全部为空'
            return result
        lengths_arr = np.array(self.lengths)
        result['length_stats'] = {
            'mean': round(float(lengths_arr.mean()), 2),
            'std':  round(float(lengths_arr.std()),  2),
            'min':  int(lengths_arr.min()),
            'p50':  int(np.percentile(lengths_arr, 50)),
            'p75':  int(np.percentile(lengths_arr, 75)),
            'p90':  int(np.percentile(lengths_arr, 90)),
            'p95':  int(np.percentile(lengths_arr, 95)),
            'p99':  int(np.percentile(lengths_arr, 99)),
            'max':  int(lengths_arr.max()),
            'zero_len_rate': f'{(lengths_arr == 0).mean():.2%}',
        }
        if self.elem_sample:
            sample_arr = np.array(self.elem_sample, dtype=np.float64)
            elem_stats: dict = {
                'total_elements': self.elem_total_count,
                'sample_size': len(sample_arr),
                'min':  round(float(sample_arr.min()),  6),
                'max':  round(float(sample_arr.max()),  6),
                'mean': round(float(sample_arr.mean()), 6),
                'std':  round(float(sample_arr.std()),  6),
            }
            if self.is_int_elem and self.elem_counter:
                top_elements = self.elem_counter.most_common(self.top_k)
                elem_stats['top_elements'] = {str(k): v for k, v in top_elements}
            result['element_stats'] = elem_stats
        return result


def scalar_stats(series: pd.Series) -> dict:
    """单批标量统计（兼容旧调用路径）。"""
    acc = ScalarAccumulator()
    acc.update(series)
    return acc.flush()


def string_stats(series: pd.Series, top_k: int = 10000, freq_cap: int = 100_000) -> dict:
    """单批字符串统计（兼容旧调用路径）。"""
    acc = StringAccumulator(top_k=top_k, counter_cap=freq_cap)
    acc.update(series)
    return acc.flush()


def seq_stats(series: pd.Series, top_k: int = 20) -> dict:
    """单批序列统计（兼容旧调用路径）。"""
    acc = ArrayAccumulator(top_k=top_k)
    acc.update(series)
    return acc.flush()


def detect_outliers_iqr(series: pd.Series) -> Tuple[float, int]:
    valid = series.dropna()
    valid = valid[valid > 0]
    if len(valid) < 10:
        return 0.0, 0
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - OUTLIER_IQR_FACTOR * iqr
    upper = q3 + OUTLIER_IQR_FACTOR * iqr
    outlier_count = int(((valid < lower) | (valid > upper)).sum())
    return outlier_count / len(valid), outlier_count


def is_likely_timestamp(series: pd.Series) -> bool:
    valid = series[series > 0].dropna()
    if len(valid) == 0:
        return False
    ts_min, ts_max = 1577836800, 1893456000
    in_range = ((valid >= ts_min) & (valid <= ts_max)).mean()
    return float(in_range) > 0.8

# ════════════════════════════════════════════════════════════════════════════
# 综合建议 & 摘要打印
# ════════════════════════════════════════════════════════════════════════════


def generate_suggestions(report: dict) -> List[str]:
    suggestions = []
    label = report.get('label_distribution', {})
    pos = label.get('positive(label_type=2)', 0)
    neg = label.get('negative(label_type=1)', 0)
    if neg > 0 and pos > 0 and neg / pos > 20:
        suggestions.append(f'⚠️ 正负样本严重不均衡（1:{neg/pos:.0f}），建议使用 focal loss 或正样本过采样')
    for col, info in report.get('missing_rate', {}).items():
        rate = float(info.get('missing_rate', '0%').replace('%', '')) / 100
        if rate > 0.5:
            suggestions.append(f'⚠️ {col} 缺失率 {info["missing_rate"]}，建议删除或填充')
    high_corr = report.get('feature_correlation', {}).get('high_corr_pairs', [])
    if high_corr:
        suggestions.append(
            f'⚠️ 发现 {len(high_corr)} 对高度相关特征（≥{report["feature_correlation"]["corr_threshold"]}），'
            f'建议删除其中一个：' + ', '.join(f'{p["col_a"]}↔{p["col_b"]}' for p in high_corr[:3])
        )
    for domain, info in report.get('timestamp_fid_detect', {}).items():
        ts_fids = info.get('auto_detected_ts_fids', [])
        if ts_fids and info.get('schema_ts_fid') is None:
            fid_list = [str(t['fid']) for t in ts_fids]
            suggestions.append(f'⚠️ {domain} 检测到时间戳 fid={fid_list}，但 schema ts_fid=None')
    for domain, info in report.get('seq_domain_coverage', {}).items():
        rate = float(info.get('user_coverage', '100%').replace('%', '')) / 100
        if rate < 0.5:
            suggestions.append(f'⚠️ {domain} 用户覆盖率仅 {info["user_coverage"]}，建议评估是否值得保留')
    severe_outliers = [col for col, info in report.get('outlier_detection', {}).items()
                       if info.get('severity') in ('轻度', '严重')]
    if severe_outliers:
        suggestions.append(f'⚠️ 以下特征存在明显异常值，建议 clip：{severe_outliers[:5]}')
    if not suggestions:
        suggestions.append('✅ 未发现明显数据质量问题')
    return suggestions


def print_summary(report: dict) -> None:
    print('\n' + '═' * 60)
    print('                    分析摘要')
    print('═' * 60)

    # Label 分布
    label = report.get('label_distribution', {})
    if label:
        print(f'\n📊 Label 分布:')
        print(f'   正样本: {label.get("positive(label_type=2)", 0):,}')
        print(f'   负样本: {label.get("negative(label_type=1)", 0):,}')
        print(f'   正样本率: {label.get("pos_rate", "N/A")}')
        print(f'   正负比: {label.get("neg_pos_ratio", "N/A")}')

    # 特征统计汇总（按类型分组）
    feat_stats = report.get('feat_stats', {})
    if feat_stats:
        scalar_cols = [(col, s) for col, s in feat_stats.items() if s.get('type') in ('int', 'float')]
        string_cols = [(col, s) for col, s in feat_stats.items() if s.get('type') == 'string']
        array_cols  = [(col, s) for col, s in feat_stats.items() if s.get('type') == 'array']

        # 标量特征：高 null_rate 预警
        high_null = [(col, s['null_rate']) for col, s in scalar_cols
                     if float(s.get('null_rate', '0%').replace('%', '')) > 10]
        print(f'\n📈 标量特征（共 {len(scalar_cols)} 列）:')
        if high_null:
            print(f'   ⚠️  null_rate > 10% 的列（共 {len(high_null)} 个）:')
            for col, rate in high_null[:10]:
                print(f'      {col}: {rate}')
        else:
            print('   ✅ 无高缺失率标量列')
        for col, s in scalar_cols[:20]:
            print(f'   {col}: min={s.get("min")}  max={s.get("max")}  '
                  f'mean={s.get("mean")}  std={s.get("std")}  '
                  f'p50={s.get("p50")}  p90={s.get("p90")}  p99={s.get("p99")}  null={s.get("null_rate")}')
        if len(scalar_cols) > 20:
            print(f'   ... 还有 {len(scalar_cols) - 20} 列，详见上方 [REPORT] scalar_user_int / scalar_item_int / scalar_dense 日志')

        # 字符串特征：高基数预警
        print(f'\n🔤 字符串特征（共 {len(string_cols)} 列）:')
        for col, s in string_cols:
            unique_count = s.get('unique_count', 0)
            trunc = '（截断）' if s.get('freq_truncated') else ''
            top3 = list(s.get('top_freq', {}).items())[:3]
            top3_str = '  '.join(f'{k}:{v}' for k, v in top3)
            warn = '⚠️ 高基数 ' if unique_count > 100_000 else '   '
            print(f'   {warn}{col}: unique={unique_count:,}  null={s.get("null_rate")}  top3{trunc}: {top3_str}')

        # 序列特征：长度分布 + 元素值域
        print(f'\n📏 序列/数组特征（共 {len(array_cols)} 列）:')
        for col, s in array_cols:
            ls = s.get('length_stats', {})
            es = s.get('element_stats', {})
            print(f'   {col}:')
            print(f'      长度: mean={ls.get("mean")}  '
                  f'p50={ls.get("p50")}  p75={ls.get("p75")}  '
                  f'p90={ls.get("p90")}  p95={ls.get("p95")}  p99={ls.get("p99")}  '
                  f'max={ls.get("max")}  zero_rate={ls.get("zero_len_rate")}')
            if es:
                print(f'      元素值: min={es.get("min")}  max={es.get("max")}  '
                      f'mean={es.get("mean")}  std={es.get("std")}')

    # 特征相关性
    print(f'\n🔗 特征相关性:')
    corr = report.get('feature_correlation', {})
    pairs = corr.get('high_corr_pairs', [])
    print(f'   发现 {len(pairs)} 对高度相关特征（≥{corr.get("corr_threshold", 0.9)}）')
    for pair in pairs[:3]:
        print(f'   → {pair["col_a"]} ↔ {pair["col_b"]}（r={pair["pearson"]}）')

    # 异常值
    print(f'\n⚡ 异常值（IQR法）:')
    outliers = report.get('outlier_detection', {})
    severe = [(c, i) for c, i in outliers.items() if i.get('severity') in ('轻度', '严重')]
    if severe:
        for col, info in severe[:5]:
            print(f'   {col}: {info["outlier_rate"]} ({info["severity"]})')
    else:
        print('   未发现明显异常值')

    # 时间戳 fid 检测
    print(f'\n🕐 时间戳 fid 检测:')
    for domain, info in report.get('timestamp_fid_detect', {}).items():
        ts_fids = info.get('auto_detected_ts_fids', [])
        if ts_fids:
            fid_list = [str(t['fid']) for t in ts_fids]
            print(f'   {domain}: 检测到时间戳 fid={fid_list}，schema ts_fid={info["schema_ts_fid"]}')

    # Vocab 覆盖率
    print(f'\n📦 Vocab 覆盖率（训练集实际出现 / schema 声明）:')
    for domain, fid_info in report.get('vocab_coverage', {}).items():
        print(f'   {domain}:')
        for fid_key, info in fid_info.items():
            if info.get('is_ts'):
                continue
            actual = info.get('actual_unique_ids', 0)
            schema_vs = info.get('schema_vocab', 0)
            rate = info.get('coverage_rate', 'N/A')
            print(f'     {fid_key}: 实际={actual:,} / schema={schema_vs:,} ({rate})')

    # 跨域重叠
    print(f'\n🔀 跨域 item 重叠:')
    overlap_info = report.get('cross_domain_overlap', {})
    if overlap_info:
        for pair, info in overlap_info.items():
            print(f'   {pair}: 重叠={info["overlap"]:,} ({info["overlap_rate_a"]} of A), {info["suggestion"]}')
    else:
        print('   无法分析（所有域的 item fid 均超过 vocab 上限）')

    # 综合建议
    print(f'\n💡 综合建议:')
    for s in report.get('summary_suggestions', []):
        print(f'   {s}')
    print('\n' + '═' * 60)


def _infer_col_category(series: pd.Series) -> str:
    """根据第一个有效值推断列的类别：scalar_float / scalar_int / string / array / unknown。"""
    first_valid = series.dropna().iloc[0] if series.notna().any() else None
    if first_valid is None:
        return 'unknown'
    if isinstance(first_valid, (np.ndarray, list)):
        return 'array'
    if isinstance(first_valid, str):
        return 'string'
    if isinstance(first_valid, (float, np.floating)):
        return 'scalar_float'
    return 'scalar_int'


def run_analysis(data_path: str, schema_path: str, output_dir: str,
                 max_rows: Optional[int], export_rows: Optional[int],
                 preview_rows: int = 10000) -> None:
    import json as _json
    os.makedirs(output_dir, exist_ok=True)

    def _log_json(tag: str, data: dict) -> None:
        print(f'\n[REPORT] ===== {tag} =====')
        print(_json.dumps(data, ensure_ascii=False, indent=2, default=str))
        sys.stdout.flush()

    # ── 1. 读取 schema（需要用来分类列） ──────────────────────────────────
    print(f'[1/8] 读取 schema: {schema_path}')
    with open(schema_path) as f:
        schema = json.load(f)

    # ── 2. 第一个文件：推断所有列的类别，初始化 Accumulator ───────────────
    data_dir = os.path.dirname(data_path)
    parquet_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not parquet_files:
        parquet_files = [data_path]
    print(f'[2/8] 读取数据目录: {data_dir}（共 {len(parquet_files)} 个 parquet 文件，全量流式统计）')

    first_df = pd.read_parquet(parquet_files[0])
    col_names = list(first_df.columns)
    col_categories: Dict[str, str] = {col: _infer_col_category(first_df[col]) for col in col_names}

    # 按类别分组，用于后续分析模块分类
    user_int_cols = [c for c in col_names if c.startswith('user_int_feats')]
    item_int_cols = [c for c in col_names if c.startswith('item_int_feats')]
    dense_cols    = [c for c in col_names if c.startswith('user_dense_feats')]
    scalar_user_int = [c for c in user_int_cols if col_categories[c] in ('scalar_int', 'scalar_float')]
    array_user_int  = [c for c in user_int_cols if col_categories[c] == 'array']
    scalar_item_int = [c for c in item_int_cols if col_categories[c] in ('scalar_int', 'scalar_float')]
    array_item_int  = [c for c in item_int_cols if col_categories[c] == 'array']
    scalar_dense    = [c for c in dense_cols    if col_categories[c] in ('scalar_int', 'scalar_float')]
    all_scalar_numeric = scalar_user_int + scalar_item_int + scalar_dense
    print(f'      列数: {len(col_names)}  user_int: {len(scalar_user_int)}标量+{len(array_user_int)}array  '
          f'item_int: {len(scalar_item_int)}标量+{len(array_item_int)}array  dense: {len(scalar_dense)}')

    # 初始化每列对应的 Accumulator
    accumulators: Dict[str, object] = {}
    for col, cat in col_categories.items():
        if cat in ('scalar_int', 'scalar_float'):
            accumulators[col] = ScalarAccumulator()
        elif cat == 'string':
            accumulators[col] = StringAccumulator()
        elif cat == 'array':
            accumulators[col] = ArrayAccumulator()
        # unknown 跳过

    # ── 3. 流式逐文件 update，读完即释放；达到 max_rows 后提前停止 ─────────
    total_rows = 0
    for file_idx, pf_path in enumerate(parquet_files):
        if max_rows is not None and total_rows >= max_rows:
            break
        if file_idx == 0:
            pf_df = first_df  # 第一个文件已读
        else:
            pf_df = pd.read_parquet(pf_path)
        # 如果本文件超出 max_rows 上限，截取需要的行数
        if max_rows is not None and total_rows + len(pf_df) > max_rows:
            pf_df = pf_df.iloc[:max_rows - total_rows]
        total_rows += len(pf_df)
        for col, acc in accumulators.items():
            acc.update(pf_df[col])
        print(f'      [{file_idx + 1}/{len(parquet_files)}] {os.path.basename(pf_path)}'
              f'（{len(pf_df):,} 行），累计 {total_rows:,} 行')
        sys.stdout.flush()
        del pf_df
        if max_rows is not None and total_rows >= max_rows:
            print(f'      已达到 max_rows={max_rows:,} 上限，提前停止读取')
            break

    limit_note = f'（限制 {max_rows:,} 行）' if max_rows else '（全量）'
    print(f'      读取完成：{total_rows:,} 行，{len(col_names)} 列 {limit_note}')

    # ── 4. flush 所有 Accumulator 得到 feat_stats ─────────────────────────
    feat_stats: Dict[str, dict] = {}
    for col, acc in accumulators.items():
        cat = col_categories[col]
        col_type = 'float' if cat == 'scalar_float' else ('int' if cat == 'scalar_int' else cat)
        feat_stats[col] = {'type': col_type, **acc.flush()}
    # unknown 列
    for col, cat in col_categories.items():
        if cat == 'unknown':
            feat_stats[col] = {'type': 'unknown', 'note': '全部为空'}

    # ── 5. 打印 FEAT STATS ────────────────────────────────────────────────
    print(f'\n[FEAT STATS] ======== 特征统计（共 {len(col_names)} 列，{total_rows:,} 行）========')
    for col in col_names:
        stats = feat_stats.get(col, {})
        col_type = stats.get('type', '?')
        if col_type in ('int', 'float'):
            print(f'[FEAT STATS] {col} ({col_type}): '
                  f'null={stats.get("null_rate", "N/A")}  '
                  f'min={stats.get("min", "N/A")}  max={stats.get("max", "N/A")}  '
                  f'mean={stats.get("mean", "N/A")}  std={stats.get("std", "N/A")}  '
                  f'p50={stats.get("p50", "N/A")}  p90={stats.get("p90", "N/A")}  p99={stats.get("p99", "N/A")}')
        elif col_type == 'string':
            unique_count = stats.get('unique_count', 0)
            top3 = list(stats.get('top_freq', {}).items())[:3]
            top3_str = '  '.join(f'{k}:{v}' for k, v in top3)
            trunc = '（截断）' if stats.get('freq_truncated') else ''
            print(f'[FEAT STATS] {col} (string): '
                  f'null={stats.get("null_rate", "N/A")}  unique={unique_count:,}  top3{trunc}: {top3_str}')
        elif col_type == 'array':
            ls = stats.get('length_stats', {})
            es = stats.get('element_stats', {})
            elem_str = (f'  elem_min={es.get("min", "N/A")}  elem_max={es.get("max", "N/A")}  '
                        f'elem_mean={es.get("mean", "N/A")}  elem_std={es.get("std", "N/A")}') if es else ''
            print(f'[FEAT STATS] {col} (array): '
                  f'null={stats.get("null_rate", "N/A")}  '
                  f'len_mean={ls.get("mean", "N/A")}  '
                  f'p50={ls.get("p50", "N/A")}  p75={ls.get("p75", "N/A")}  '
                  f'p90={ls.get("p90", "N/A")}  p95={ls.get("p95", "N/A")}  p99={ls.get("p99", "N/A")}  '
                  f'max={ls.get("max", "N/A")}  zero_rate={ls.get("zero_len_rate", "N/A")}{elem_str}')
        else:
            print(f'[FEAT STATS] {col}: {stats}')
    print('[FEAT STATS] ======================================================\n')
    sys.stdout.flush()

    # ── 6. 流式逐文件计算 label 分布 + 序列覆盖率 + 序列长度 ──────────────
    # seq_lengths 采用水库采样，最多保留 SEQ_LEN_SAMPLE_CAP 个值，避免 OOM
    print('[3/8] 流式计算 label 分布 + 序列覆盖率 + 序列长度...')
    label_counter: Counter = Counter()
    # domain -> 采样长度列表（最多 SEQ_LEN_SAMPLE_CAP 个）
    seq_lengths: Dict[str, List[int]] = {}
    # domain -> 已见总行数（用于水库采样概率计算）
    seq_lengths_seen: Dict[str, int] = {}
    seq_has_data: Dict[str, int] = {}       # domain -> 有序列的用户数
    seq_no_data: Dict[str, int] = {}        # domain -> 无序列的用户数
    vocab_sets: Dict[str, Dict[int, set]] = {}    # domain -> fid -> id set
    ts_flat_samples: Dict[str, Dict[int, list]] = {}  # domain -> fid -> 采样值

    # 构建 seq 域的元信息
    seq_domain_meta: Dict[str, dict] = {}
    for domain, cfg in schema.get('seq', {}).items():
        prefix = cfg['prefix']
        features = cfg['features']
        ref_fid = next((fid for fid, _ in features if fid not in KNOWN_TS_FIDS), features[0][0])
        ref_col = f'{prefix}_{ref_fid}'
        seq_domain_meta[domain] = {
            'prefix': prefix, 'features': features,
            'ref_fid': ref_fid, 'ref_col': ref_col,
            'ts_fid': cfg.get('ts_fid'),
        }
        seq_lengths[domain] = []
        seq_lengths_seen[domain] = 0
        seq_has_data[domain] = 0
        seq_no_data[domain] = 0
        vocab_sets[domain] = {}
        ts_flat_samples[domain] = {}
        for fid, _ in features:
            vocab_sets[domain][fid] = set()
            ts_flat_samples[domain][fid] = []

    seq_rows_read = 0
    for pf_path in parquet_files:
        if max_rows is not None and seq_rows_read >= max_rows:
            break
        pf_df = pd.read_parquet(pf_path)
        if max_rows is not None and seq_rows_read + len(pf_df) > max_rows:
            pf_df = pf_df.iloc[:max_rows - seq_rows_read]
        seq_rows_read += len(pf_df)

        # label 分布
        if 'label_type' in pf_df.columns:
            label_counter.update(pf_df['label_type'].dropna().astype(int).tolist())

        # 序列统计
        for domain, meta in seq_domain_meta.items():
            ref_col = meta['ref_col']
            if ref_col not in pf_df.columns:
                continue
            for raw in pf_df[ref_col].dropna():
                arr = raw if isinstance(raw, np.ndarray) else np.array(raw)
                valid_len = int((arr > 0).sum())
                seq_lengths_seen[domain] += 1
                # 水库采样：保证内存上限为 SEQ_LEN_SAMPLE_CAP 个长度值
                seen = seq_lengths_seen[domain]
                if len(seq_lengths[domain]) < SEQ_LEN_SAMPLE_CAP:
                    seq_lengths[domain].append(valid_len)
                else:
                    replace_pos = int(np.random.randint(0, seen))
                    if replace_pos < SEQ_LEN_SAMPLE_CAP:
                        seq_lengths[domain][replace_pos] = valid_len
                if valid_len > 0:
                    seq_has_data[domain] += 1
                else:
                    seq_no_data[domain] += 1

            # vocab 覆盖率 + 时间戳检测：对每个 fid 列
            for fid, vocab_size in meta['features']:
                col = f'{meta["prefix"]}_{fid}'
                if col not in pf_df.columns:
                    continue
                flat_parts = []
                for raw in pf_df[col].dropna():
                    arr = raw if isinstance(raw, np.ndarray) else np.array(raw)
                    valid = arr[arr > 0]
                    flat_parts.append(valid)
                if not flat_parts:
                    continue
                flat = np.concatenate(flat_parts).astype(np.int64)
                vocab_sets[domain][fid].update(flat.tolist())
                # 时间戳检测：只采样最多 1000 个值
                if len(ts_flat_samples[domain][fid]) < 1000:
                    ts_flat_samples[domain][fid].extend(
                        flat[:max(0, 1000 - len(ts_flat_samples[domain][fid]))].tolist()
                    )

        del pf_df

    # 整理 label 分布
    pos = label_counter.get(2, 0)
    neg = label_counter.get(1, 0)
    label_distribution = {
        'positive(label_type=2)': pos,
        'negative(label_type=1)': neg,
        'total': sum(label_counter.values()),
        'pos_rate': f'{pos / max(1, pos + neg):.4%}',
        'neg_pos_ratio': f'1:{neg / max(1, pos):.1f}',
        'value_counts': {str(k): v for k, v in label_counter.items()},
    }

    # 整理序列覆盖率
    seq_domain_coverage = {}
    for domain in seq_domain_meta:
        has = seq_has_data[domain]
        no = seq_no_data[domain]
        total_seq = has + no
        seq_domain_coverage[domain] = {
            'user_coverage': f'{has / max(1, total_seq):.2%}',
            'users_with_seq': has,
            'users_without_seq': no,
        }

    # 整理序列长度统计
    seq_length_stats = {}
    for domain, lengths_list in seq_lengths.items():
        if not lengths_list:
            continue
        lengths_arr = np.array(lengths_list)
        meta = seq_domain_meta[domain]
        seq_length_stats[domain] = {
            'ref_fid': meta['ref_fid'],
            'mean_len': round(float(lengths_arr.mean()), 1),
            'p50': int(np.percentile(lengths_arr, 50)),
            'p90': int(np.percentile(lengths_arr, 90)),
            'p99': int(np.percentile(lengths_arr, 99)),
            'max_len': int(lengths_arr.max()),
            'zero_seq_rate': f'{(lengths_arr == 0).mean():.2%}',
            'seq_max_lens_suggestion': int(np.percentile(lengths_arr, 90)),
        }

    # 整理 vocab 覆盖率
    vocab_coverage = {}
    for domain, meta in seq_domain_meta.items():
        domain_result = {}
        for fid, vocab_size in meta['features']:
            unique_count = len(vocab_sets[domain][fid])
            domain_result[f'fid_{fid}'] = {
                'schema_vocab': vocab_size,
                'actual_unique_ids': unique_count,
                'coverage_rate': f'{unique_count / max(1, vocab_size):.4%}',
                'is_ts': fid in KNOWN_TS_FIDS,
                'note': '时间戳，跳过 Embedding' if fid in KNOWN_TS_FIDS else (
                    '实际词表远小于 schema，Embedding 表可大幅压缩'
                    if unique_count < vocab_size * 0.01 else ''
                ),
            }
        vocab_coverage[domain] = domain_result

    # 整理时间戳 fid 检测
    timestamp_fid_detect = {}
    ts_min, ts_max = 1577836800, 1893456000
    for domain, meta in seq_domain_meta.items():
        detected = []
        for fid, vocab_size in meta['features']:
            samples = ts_flat_samples[domain][fid]
            if not samples:
                continue
            arr = np.array(samples)
            in_range = ((arr >= ts_min) & (arr <= ts_max)).mean()
            if float(in_range) > 0.8:
                detected.append({'fid': fid, 'vocab': vocab_size, 'in_known_ts_fids': fid in KNOWN_TS_FIDS})
        timestamp_fid_detect[domain] = {
            'schema_ts_fid': meta['ts_fid'],
            'auto_detected_ts_fids': detected,
        }

    # 整理跨域 item 重叠
    domain_id_sets: Dict[str, set] = {}
    domain_fid_used: Dict[str, int] = {}
    for domain, meta in seq_domain_meta.items():
        for fid, vocab_size in meta['features']:
            if fid in KNOWN_TS_FIDS or vocab_size > 100_000_000:
                continue
            id_set = vocab_sets[domain][fid]
            if id_set:
                domain_id_sets[domain] = id_set
                domain_fid_used[domain] = fid
                break
    cross_domain_overlap = {}
    domains = list(domain_id_sets.keys())
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            da, db = domains[i], domains[j]
            set_a, set_b = domain_id_sets[da], domain_id_sets[db]
            overlap = len(set_a & set_b)
            cross_domain_overlap[f'{da}x{db}'] = {
                'fid_a': domain_fid_used[da],
                'fid_b': domain_fid_used[db],
                'unique_a': len(set_a),
                'unique_b': len(set_b),
                'overlap': overlap,
                'overlap_rate_a': f'{overlap / max(1, len(set_a)):.2%}',
                'overlap_rate_b': f'{overlap / max(1, len(set_b)):.2%}',
                'suggestion': 'Embedding 可考虑共享'
                    if overlap / max(1, min(len(set_a), len(set_b))) > 0.3
                    else '各域商品池差异较大，建议独立 Embedding',
            }

    # ── 7. 标量特征的相关性 + 异常值（基于 Accumulator 采样值近似）─────────
    # 用各 ScalarAccumulator 里已收集的 values 列表构建小 DataFrame 做相关性和异常值分析
    print('[4/8] 相关性分析 + 异常值检测（基于流式采样值）...')
    scalar_sample_dict = {}
    for col in all_scalar_numeric:
        acc = accumulators.get(col)
        if acc and isinstance(acc, ScalarAccumulator) and acc.values:
            sample = np.concatenate(acc.values)
            # 最多取 100000 个采样，避免内存过大
            if len(sample) > 100_000:
                indices = np.random.choice(len(sample), 100_000, replace=False)
                sample = sample[indices]
            scalar_sample_dict[col] = sample
    scalar_sample_df = pd.DataFrame(scalar_sample_dict) if scalar_sample_dict else pd.DataFrame()

    # 相关性
    if len(scalar_sample_df.columns) >= 2:
        valid_cols = [c for c in scalar_sample_df.columns
                      if scalar_sample_df[c].notna().mean() > 0.2]
        corr_matrix = scalar_sample_df[valid_cols].corr(method='pearson')
        high_corr_pairs = []
        for i in range(len(valid_cols)):
            for j in range(i + 1, len(valid_cols)):
                c_val = abs(corr_matrix.iloc[i, j])
                if c_val >= CORR_THRESHOLD:
                    high_corr_pairs.append({
                        'col_a': valid_cols[i], 'col_b': valid_cols[j],
                        'pearson': round(float(corr_matrix.iloc[i, j]), 4),
                        'suggestion': '高度相关，考虑删除其中一个',
                    })
        feature_correlation = {
            'analyzed_cols': len(valid_cols),
            'high_corr_pairs': high_corr_pairs,
            'corr_threshold': CORR_THRESHOLD,
        }
    else:
        feature_correlation = {'note': '有效列不足，跳过相关性分析'}

    # 异常值（IQR 法）
    outlier_detection = {}
    for col in all_scalar_numeric:
        if col not in scalar_sample_dict:
            continue
        values = scalar_sample_dict[col]
        valid = values[~np.isnan(values)]
        valid = valid[valid > 0]
        if len(valid) < 10:
            continue
        q1, q3 = np.percentile(valid, 25), np.percentile(valid, 75)
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_FACTOR * iqr
        upper = q3 + OUTLIER_IQR_FACTOR * iqr
        outlier_count = int(((valid < lower) | (valid > upper)).sum())
        rate = outlier_count / len(valid)
        if rate > 0:
            severity = '严重' if rate > 0.01 else ('轻度' if rate > 0.0001 else '极少')
            outlier_detection[col] = {
                'outlier_count': outlier_count,
                'outlier_rate': f'{rate:.4%}',
                'severity': severity,
            }

    # ── 8. 缺失率汇总（从 feat_stats 里提取高缺失率列）─────────────────────
    missing_rate = {
        col: {'missing_count': feat_stats[col]['null_count'],
              'missing_rate': feat_stats[col]['null_rate']}
        for col in all_scalar_numeric
        if col in feat_stats and float(feat_stats[col].get('null_rate', '0%').replace('%', '')) > 1
    }

    # ── 9. 组装 report 并打印 ────────────────────────────────────────────
    print('[5/8] 打印各模块统计结果...')
    report = {
        'label_distribution':  label_distribution,
        'seq_domain_coverage': seq_domain_coverage,
        'seq_length_stats':    seq_length_stats,
        'missing_rate':        missing_rate,
        'feature_correlation': feature_correlation,
        'outlier_detection':   outlier_detection,
        'timestamp_fid_detect': timestamp_fid_detect,
        'vocab_coverage':      vocab_coverage,
        'cross_domain_overlap': cross_domain_overlap,
        'feat_stats':          feat_stats,
    }

    _log_json('label_distribution',  report['label_distribution'])
    _log_json('seq_domain_coverage', report['seq_domain_coverage'])
    _log_json('seq_length_stats',    report['seq_length_stats'])
    _log_json('missing_rate',        report['missing_rate'])
    _log_json('feature_correlation', report['feature_correlation'])
    _log_json('outlier_detection',   report['outlier_detection'])
    _log_json('timestamp_fid_detect', report['timestamp_fid_detect'])
    _log_json('vocab_coverage',       report['vocab_coverage'])
    _log_json('cross_domain_overlap', report['cross_domain_overlap'])

    print('[6/8] 生成综合建议...')
    report['summary_suggestions'] = generate_suggestions(report)
    _log_json('summary_suggestions', {'suggestions': report['summary_suggestions']})

    print_summary(report)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='纯数据检查（不训练）')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='训练数据目录，含 *.parquet + schema.json（环境变量 TRAIN_DATA_PATH 优先）')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='schema.json 路径，不指定则从 data_dir 自动查找')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志 / 报告输出目录（环境变量 TRAIN_LOG_PATH 优先）')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='限制分析行数，None=全量（建议先用 100000 快速检查）')
    parser.add_argument('--export_rows', type=int, default=0,
                        help='导出前 N 行到 parquet（0=不导出）')
    args = parser.parse_args()
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.log_dir  = os.environ.get('TRAIN_LOG_PATH',  args.log_dir) or '.'
    return args


def setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, 'inspect_data.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding='utf-8'),
        ],
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)

    if not args.data_dir:
        logging.error('❌ 请通过 --data_dir 或环境变量 TRAIN_DATA_PATH 指定数据目录')
        sys.exit(1)
    if not os.path.isdir(args.data_dir):
        logging.error(f'❌ data_dir 不存在或不是目录: {args.data_dir}')
        sys.exit(1)

    parquet_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    if not parquet_files:
        logging.error(f'❌ data_dir 下未找到 *.parquet 文件: {args.data_dir}')
        sys.exit(1)

    data_path = parquet_files[0]
    logging.info(f'找到 parquet 文件（共 {len(parquet_files)} 个），使用第一个: {data_path}')

    schema_path = args.schema_path or os.path.join(args.data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        logging.error(f'❌ schema.json 未找到: {schema_path}')
        sys.exit(1)

    output_dir = os.path.join(args.log_dir, 'data_analysis')
    export_rows = args.export_rows if args.export_rows > 0 else None

    logging.info('=' * 70)
    logging.info('[inspect] ===== 纯数据检查模式（不训练）=====')
    logging.info(f'[inspect]   data_path  : {data_path}')
    logging.info(f'[inspect]   schema     : {schema_path}')
    logging.info(f'[inspect]   output_dir : {output_dir}')
    logging.info(f'[inspect]   max_rows   : {args.max_rows}')
    logging.info(f'[inspect]   export_rows: {export_rows}')
    logging.info('=' * 70)

    run_analysis(
        data_path=data_path,
        schema_path=schema_path,
        output_dir=output_dir,
        max_rows=args.max_rows,
        export_rows=export_rows,
    )

    logging.info(f'[inspect] ✅ 数据检查完成，报告已保存至: {output_dir}')


if __name__ == '__main__':
    main()
