"""数据前置清洗分析脚本

用法:
    python data_analysis.py --data_path <parquet路径> [--schema_path <schema.json路径>]
                            [--output_dir <输出目录>] [--max_rows <最大读取行数>]
                            [--export_rows 100000]

示例:
    # 分析 demo 数据，输出前10万行供离线训练
    python data_analysis.py --data_path demo_1000.parquet --export_rows 10000

    # 分析前100万条
    python data_analysis.py --data_path /data/train.parquet --max_rows 1000000
"""

import argparse
import json
import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────────────────────────────────

# 已知时间戳 fid（vocab 在 Unix 时间戳量级，不应参与 Embedding 建表）
KNOWN_TS_FIDS = {39, 67, 27, 26}

# 异常值判定阈值
OUTLIER_IQR_FACTOR = 3.0     # IQR 方法倍数
OUTLIER_SIGMA_FACTOR = 3.0   # 3σ 方法倍数

# 相关性去重阈值（Pearson 相关系数绝对值超过此值认为高度相关）
CORR_THRESHOLD = 0.90


# ────────────────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────────────────

def detect_column_type(series: pd.Series) -> str:
    """判断列类型：scalar_int / scalar_float / array / string。"""
    if series.dtype == object:
        sample = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
        if sample is None:
            return 'unknown'
        if isinstance(sample, (np.ndarray, list)):
            return 'array'
        return 'string'
    if pd.api.types.is_float_dtype(series):
        return 'scalar_float'
    return 'scalar_int'


def scalar_stats(series: pd.Series, top_k: int = 10) -> dict:
    """对标量数值列计算统计信息。"""
    valid = series.replace(0, np.nan).dropna()
    total = len(series)
    missing = (series <= 0).sum()

    stats = {
        'total': total,
        'missing(<=0)': int(missing),
        'missing_rate': f'{missing / total:.2%}',
        'valid_count': len(valid),
    }
    if len(valid) == 0:
        stats['note'] = '全部缺失'
        return stats

    stats.update({
        'min': float(valid.min()),
        'max': float(valid.max()),
        'mean': float(valid.mean()),
        'std': float(valid.std()),
        'p50': float(valid.quantile(0.5)),
        'p90': float(valid.quantile(0.9)),
        'p99': float(valid.quantile(0.99)),
        'top10_values': valid.value_counts().head(top_k).to_dict(),
    })
    return stats


def array_col_stats(series: pd.Series, top_k: int = 10) -> dict:
    """对 array 列按位置展开计算统计。"""
    arrays = series.dropna().tolist()
    if not arrays:
        return {'note': '全部缺失'}

    # 检查是否是行为序列（值很大，可能是时间戳或全局 id）
    max_dim = max(len(a) for a in arrays)
    result = {'array_dim': max_dim, 'columns': {}}

    all_flat = np.concatenate(arrays)
    all_flat_valid = all_flat[all_flat > 0]

    result['global'] = {
        'total_values': len(all_flat),
        'valid(>0)': len(all_flat_valid),
        'missing_rate': f'{(len(all_flat) - len(all_flat_valid)) / max(1, len(all_flat)):.2%}',
        'min': float(all_flat_valid.min()) if len(all_flat_valid) else 'N/A',
        'max': float(all_flat_valid.max()) if len(all_flat_valid) else 'N/A',
        'mean': float(all_flat_valid.mean()) if len(all_flat_valid) else 'N/A',
        'top10_values': dict(Counter(all_flat_valid.astype(int).tolist()).most_common(top_k)),
    }
    return result


def detect_outliers_iqr(series: pd.Series) -> Tuple[float, int]:
    """IQR 方法检测异常值，返回 (异常占比, 异常数量)。"""
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
    """启发式判断一列是否为 Unix 时间戳（2020~2030 年范围）。"""
    valid = series[series > 0].dropna()
    if len(valid) == 0:
        return False
    ts_min, ts_max = 1577836800, 1893456000  # 2020-01-01 ~ 2030-01-01
    in_range = ((valid >= ts_min) & (valid <= ts_max)).mean()
    return float(in_range) > 0.8


# ────────────────────────────────────────────────────────────────────────────
# 各分析模块
# ────────────────────────────────────────────────────────────────────────────

def analyze_label(df: pd.DataFrame) -> dict:
    """分析 label 分布。"""
    result = {}
    if 'label_type' in df.columns:
        vc = df['label_type'].value_counts().to_dict()
        pos = vc.get(2, 0)
        neg = vc.get(1, 0)
        result = {
            'positive(label_type=2)': pos,
            'negative(label_type=1)': neg,
            'total': len(df),
            'pos_rate': f'{pos / max(1, len(df)):.4%}',
            'neg_pos_ratio': f'1:{neg / max(1, pos):.1f}',
            'value_counts': vc,
        }
    return result


def analyze_missing(df: pd.DataFrame, scalar_cols: List[str]) -> dict:
    """统计标量列的缺失值率（<=0 视为缺失）。"""
    result = {}
    for col in scalar_cols:
        total = len(df)
        missing = (df[col] <= 0).sum()
        rate = missing / total
        if rate > 0.01:  # 只报告缺失率 > 1% 的列
            result[col] = {'missing_count': int(missing), 'missing_rate': f'{rate:.2%}'}
    return result


def analyze_seq_length(df: pd.DataFrame, schema: dict) -> dict:
    """分析各行为域序列实际有效长度分布（有效 = 元素 > 0）。"""
    result = {}
    for domain, cfg in schema.get('seq', {}).items():
        prefix = cfg['prefix']
        # 找非时间戳的第一个 fid 作为长度参考
        ref_fid = None
        for fid, vs in cfg['features']:
            if fid not in KNOWN_TS_FIDS:
                ref_fid = fid
                break
        if ref_fid is None:
            ref_fid = cfg['features'][0][0]
        col = f'{prefix}_{ref_fid}'
        if col not in df.columns:
            continue
        # 用实际有效元素数（>0）作为序列长度
        lengths = df[col].apply(
            lambda x: int((x > 0).sum()) if isinstance(x, np.ndarray) else 0
        )
        total_len = df[col].apply(
            lambda x: len(x) if isinstance(x, np.ndarray) else 0
        )
        result[domain] = {
            'ref_fid': ref_fid,
            'mean_len': round(float(lengths.mean()), 1),
            'p50': int(lengths.quantile(0.5)),
            'p90': int(lengths.quantile(0.9)),
            'p99': int(lengths.quantile(0.99)),
            'max_len': int(lengths.max()),
            'max_array_len': int(total_len.max()),   # parquet 里存储的最大原始长度
            'zero_seq_rate': f'{(lengths == 0).mean():.2%}',
            'seq_max_lens_suggestion': int(lengths.quantile(0.9)),  # 推荐 seq_max_lens 配置值
        }
    return result


def analyze_vocab_coverage(df: pd.DataFrame, schema: dict) -> dict:
    """分析各行为域的 vocab 覆盖情况：训练集中实际出现了多少 unique id。
    
    这决定了真实需要的 Embedding 表大小（往往远小于 schema 标注的 vocab_size）。
    """
    result = {}
    for domain, cfg in schema.get('seq', {}).items():
        prefix = cfg['prefix']
        domain_result = {}
        for fid, vs in cfg['features']:
            col = f'{prefix}_{fid}'
            if col not in df.columns:
                continue
            all_vals = df[col].dropna()
            flat = np.concatenate(all_vals.tolist()) if len(all_vals) else np.array([])
            flat_valid = flat[flat > 0]
            unique_count = len(np.unique(flat_valid))
            domain_result[f'fid_{fid}'] = {
                'schema_vocab': vs,
                'actual_unique_ids': unique_count,
                'coverage_rate': f'{unique_count / max(1, vs):.4%}',
                'is_ts': fid in KNOWN_TS_FIDS,
                'note': '时间戳，跳过 Embedding' if fid in KNOWN_TS_FIDS else (
                    '实际词表远小于 schema，Embedding 表可大幅压缩' if unique_count < vs * 0.01 else ''
                ),
            }
        result[domain] = domain_result
    return result


def analyze_cross_domain_overlap(df: pd.DataFrame, schema: dict) -> dict:
    """分析跨行为域的 item_id 重叠情况。
    
    重叠高 = 用户在不同场景看到了同一批商品 = Embedding 可以共享
    重叠低 = 各域商品池差异大 = Embedding 最好独立
    """
    # 只取非时间戳且 vocab 不超大的 fid 来做重叠分析
    domain_id_sets: Dict[str, set] = {}
    domain_fid_used: Dict[str, int] = {}

    for domain, cfg in schema.get('seq', {}).items():
        prefix = cfg['prefix']
        for fid, vs in cfg['features']:
            if fid in KNOWN_TS_FIDS:
                continue
            if vs > 100_000_000:  # 超大 vocab 不做重叠分析
                continue
            col = f'{prefix}_{fid}'
            if col not in df.columns:
                continue
            all_vals = df[col].dropna()
            flat = np.concatenate(all_vals.tolist()) if len(all_vals) else np.array([])
            flat_valid = set(flat[flat > 0].astype(int).tolist())
            if flat_valid:
                domain_id_sets[domain] = flat_valid
                domain_fid_used[domain] = fid
                break  # 每域只取第一个合适的 fid

    result = {}
    domains = list(domain_id_sets.keys())
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            da, db = domains[i], domains[j]
            set_a, set_b = domain_id_sets[da], domain_id_sets[db]
            overlap = len(set_a & set_b)
            result[f'{da}×{db}'] = {
                'fid_a': domain_fid_used[da],
                'fid_b': domain_fid_used[db],
                'unique_a': len(set_a),
                'unique_b': len(set_b),
                'overlap': overlap,
                'overlap_rate_a': f'{overlap / max(1, len(set_a)):.2%}',
                'overlap_rate_b': f'{overlap / max(1, len(set_b)):.2%}',
                'suggestion': 'Embedding 可考虑共享' if overlap / max(1, min(len(set_a), len(set_b))) > 0.3
                              else '各域商品池差异较大，建议独立 Embedding',
            }
    return result


def analyze_scalar_features(df: pd.DataFrame, cols: List[str]) -> dict:
    """统计所有标量数值特征。"""
    result = {}
    for col in cols:
        result[col] = scalar_stats(df[col])
    return result


def analyze_array_features(df: pd.DataFrame, cols: List[str]) -> dict:
    """统计所有 array 特征。"""
    result = {}
    for col in cols:
        result[col] = array_col_stats(df[col])
    return result


def analyze_correlation(df: pd.DataFrame, scalar_cols: List[str]) -> dict:
    """Pearson 相关性分析，找出高度相关的特征对。"""
    # 只取有效行
    sub = df[scalar_cols].copy()
    sub = sub.replace(0, np.nan)
    # 过滤掉缺失率 > 80% 的列
    valid_cols = [c for c in scalar_cols if sub[c].notna().mean() > 0.2]
    if len(valid_cols) < 2:
        return {'note': '有效列不足，跳过相关性分析'}

    corr_matrix = sub[valid_cols].corr(method='pearson')
    high_corr_pairs = []
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            c = abs(corr_matrix.iloc[i, j])
            if c >= CORR_THRESHOLD:
                high_corr_pairs.append({
                    'col_a': valid_cols[i],
                    'col_b': valid_cols[j],
                    'pearson': round(float(corr_matrix.iloc[i, j]), 4),
                    'suggestion': '高度相关，考虑删除其中一个',
                })

    return {
        'analyzed_cols': len(valid_cols),
        'high_corr_pairs': high_corr_pairs,
        'corr_threshold': CORR_THRESHOLD,
    }


def analyze_outliers(df: pd.DataFrame, scalar_cols: List[str]) -> dict:
    """检测标量特征的异常值。"""
    result = {}
    for col in scalar_cols:
        rate, count = detect_outliers_iqr(df[col])
        if rate > 0:
            severity = '严重' if rate > 0.01 else ('轻度' if rate > 0.0001 else '极少')
            result[col] = {
                'outlier_count': count,
                'outlier_rate': f'{rate:.4%}',
                'severity': severity,
                'suggestion': '剔除或替换为用户均值' if rate <= 0.0001 else '需进一步分析',
            }
    return result


def analyze_timestamp_fids(df: pd.DataFrame, schema: dict) -> dict:
    """自动识别序列中的时间戳 fid。"""
    result = {}
    for domain, cfg in schema.get('seq', {}).items():
        prefix = cfg['prefix']
        ts_fid_schema = cfg.get('ts_fid')
        detected = []
        for fid, vs in cfg['features']:
            col = f'{prefix}_{fid}'
            if col not in df.columns:
                continue
            all_vals = df[col].dropna()
            flat = np.concatenate(all_vals.tolist()) if len(all_vals) else np.array([])
            flat_valid = flat[flat > 0]
            if len(flat_valid) == 0:
                continue
            # 自动检测
            if is_likely_timestamp(pd.Series(flat_valid)):
                detected.append({
                    'fid': fid,
                    'vocab': vs,
                    'in_known_ts_fids': fid in KNOWN_TS_FIDS,
                    'suggestion': '确认为时间戳，应设置为 ts_fid 而非普通 sideinfo',
                })
        result[domain] = {
            'schema_ts_fid': ts_fid_schema,
            'auto_detected_ts_fids': detected,
        }
    return result


def analyze_seq_domain_coverage(df: pd.DataFrame, schema: dict) -> dict:
    """分析各域的用户覆盖率（有多少用户该域全为空）。"""
    result = {}
    for domain, cfg in schema.get('seq', {}).items():
        prefix = cfg['prefix']
        first_fid = cfg['features'][0][0]
        col = f'{prefix}_{first_fid}'
        if col not in df.columns:
            continue
        has_seq = df[col].apply(lambda x: bool((x > 0).any()) if isinstance(x, np.ndarray) else False)
        result[domain] = {
            'user_coverage': f'{has_seq.mean():.2%}',
            'users_with_seq': int(has_seq.sum()),
            'users_without_seq': int((~has_seq).sum()),
        }
    return result


# ────────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────────

def run_analysis(data_path: str, schema_path: str, output_dir: str,
                 max_rows: Optional[int], export_rows: Optional[int]) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取数据
    print(f'[1/8] 读取数据: {data_path}')
    df = pd.read_parquet(data_path)
    total_rows = len(df)
    print(f'      总行数: {total_rows:,}，列数: {len(df.columns)}')

    if max_rows and total_rows > max_rows:
        df = df.head(max_rows)
        print(f'      截取前 {max_rows:,} 行用于分析')

    # 2. 导出离线训练数据
    if export_rows:
        export_path = os.path.join(output_dir, f'offline_train_{export_rows}.parquet')
        df.head(export_rows).to_parquet(export_path, index=False)
        print(f'[2/8] 已导出前 {export_rows:,} 行 → {export_path}')
    else:
        print(f'[2/8] 跳过导出（未指定 --export_rows）')

    # 3. 读取 schema
    print(f'[3/8] 读取 schema: {schema_path}')
    with open(schema_path) as f:
        schema = json.load(f)

    # 4. 列分类
    user_int_cols = [c for c in df.columns if c.startswith('user_int_feats')]
    item_int_cols = [c for c in df.columns if c.startswith('item_int_feats')]
    dense_cols    = [c for c in df.columns if c.startswith('user_dense_feats')]
    seq_cols      = [c for c in df.columns if any(
                        c.startswith(cfg['prefix'])
                        for cfg in schema.get('seq', {}).values())]

    scalar_user_int = [c for c in user_int_cols if df[c].dtype != object]
    array_user_int  = [c for c in user_int_cols if df[c].dtype == object]
    scalar_item_int = [c for c in item_int_cols if df[c].dtype != object]
    array_item_int  = [c for c in item_int_cols if df[c].dtype == object]
    scalar_dense    = [c for c in dense_cols if df[c].dtype != object]

    all_scalar_numeric = scalar_user_int + scalar_item_int + scalar_dense

    print(f'      user_int: {len(scalar_user_int)} 标量 + {len(array_user_int)} array')
    print(f'      item_int: {len(scalar_item_int)} 标量 + {len(array_item_int)} array')
    print(f'      dense: {len(scalar_dense)} 列，seq: {len(seq_cols)} 列')

    # 5. 各模块分析
    report = {}

    print('[4/8] 分析 label 分布 + 序列覆盖率...')
    report['label_distribution']   = analyze_label(df)
    report['seq_domain_coverage']  = analyze_seq_domain_coverage(df, schema)
    report['seq_length_stats']     = analyze_seq_length(df, schema)

    print('[5/8] 分析特征统计（缺失值 / 数值分布）...')
    report['missing_rate']         = analyze_missing(df, all_scalar_numeric)
    report['scalar_user_int']      = analyze_scalar_features(df, scalar_user_int)
    report['scalar_item_int']      = analyze_scalar_features(df, scalar_item_int)
    report['scalar_dense']         = analyze_scalar_features(df, scalar_dense)
    report['array_user_int']       = analyze_array_features(df, array_user_int)
    report['array_item_int']       = analyze_array_features(df, array_item_int)

    print('[6/8] 相关性分析...')
    report['feature_correlation']  = analyze_correlation(df, all_scalar_numeric)

    print('[7/8] 异常值检测 + 时间戳 fid 识别...')
    report['outlier_detection']    = analyze_outliers(df, all_scalar_numeric)
    report['timestamp_fid_detect'] = analyze_timestamp_fids(df, schema)

    print('[7.5/8] vocab 覆盖率 + 跨域 item 重叠分析...')
    report['vocab_coverage']        = analyze_vocab_coverage(df, schema)
    report['cross_domain_overlap']  = analyze_cross_domain_overlap(df, schema)

    print('[8/8] 生成综合建议...')
    report['summary_suggestions']  = generate_suggestions(report)

    # 6. 输出报告
    report_path = os.path.join(output_dir, 'analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n✅ 报告已保存: {report_path}')

    # 7. 打印摘要
    print_summary(report)


def generate_suggestions(report: dict) -> List[str]:
    """根据分析结果生成综合清洗建议。"""
    suggestions = []

    # label 不均衡
    label = report.get('label_distribution', {})
    pos = label.get('positive(label_type=2)', 0)
    neg = label.get('negative(label_type=1)', 0)
    if neg > 0 and pos > 0 and neg / pos > 20:
        suggestions.append(
            f'⚠️ 正负样本严重不均衡（1:{neg/pos:.0f}），建议使用 focal loss 或正样本过采样'
        )

    # 缺失率高的特征
    for col, info in report.get('missing_rate', {}).items():
        rate_str = info.get('missing_rate', '0%').replace('%', '')
        rate = float(rate_str) / 100
        if rate > 0.5:
            suggestions.append(f'⚠️ {col} 缺失率 {info["missing_rate"]}，建议删除或填充')

    # 高度相关特征
    high_corr = report.get('feature_correlation', {}).get('high_corr_pairs', [])
    if high_corr:
        suggestions.append(
            f'⚠️ 发现 {len(high_corr)} 对高度相关特征（≥{report["feature_correlation"]["corr_threshold"]}），'
            f'建议删除其中一个：' + ', '.join(f'{p["col_a"]}↔{p["col_b"]}' for p in high_corr[:3])
        )

    # 时间戳 fid 未配置
    for domain, info in report.get('timestamp_fid_detect', {}).items():
        ts_fids = info.get('auto_detected_ts_fids', [])
        if ts_fids and info.get('schema_ts_fid') is None:
            fid_list = [str(t['fid']) for t in ts_fids]
            suggestions.append(
                f'⚠️ {domain} 检测到时间戳 fid={fid_list}，但 schema ts_fid=None，'
                f'建议在 schema.json 中配置 ts_fid 以避免将时间戳当作 item_id'
            )

    # 序列为空的用户比例
    for domain, info in report.get('seq_domain_coverage', {}).items():
        rate_str = info.get('user_coverage', '100%').replace('%', '')
        rate = float(rate_str) / 100
        if rate < 0.5:
            suggestions.append(
                f'⚠️ {domain} 用户覆盖率仅 {info["user_coverage"]}，'
                f'unified 模型对该域的 token 大量为 padding，建议评估是否值得保留'
            )

    # 异常值
    severe_outliers = [
        col for col, info in report.get('outlier_detection', {}).items()
        if info.get('severity') in ('轻度', '严重')
    ]
    if severe_outliers:
        suggestions.append(
            f'⚠️ 以下特征存在明显异常值，建议 clip 或替换为用户均值：{severe_outliers[:5]}'
        )

    if not suggestions:
        suggestions.append('✅ 未发现明显数据质量问题')

    return suggestions


def print_summary(report: dict) -> None:
    """打印核心摘要到控制台。"""
    print('\n' + '═' * 60)
    print('                    分析摘要')
    print('═' * 60)

    label = report.get('label_distribution', {})
    if label:
        print(f'\n📊 Label 分布:')
        print(f'   正样本: {label.get("positive(label_type=2)", 0):,}')
        print(f'   负样本: {label.get("negative(label_type=1)", 0):,}')
        print(f'   正样本率: {label.get("pos_rate", "N/A")}')
        print(f'   正负比: {label.get("neg_pos_ratio", "N/A")}')

    print(f'\n📏 序列长度 (p90):')
    for domain, info in report.get('seq_length_stats', {}).items():
        print(f'   {domain}: p90={info.get("p90", "N/A")}, p99={info.get("p99", "N/A")}, '
              f'空序列率={info.get("zero_seq_rate", "N/A")}')

    print(f'\n🔗 特征相关性:')
    corr = report.get('feature_correlation', {})
    pairs = corr.get('high_corr_pairs', [])
    print(f'   发现 {len(pairs)} 对高度相关特征（≥{corr.get("corr_threshold", 0.9)}）')
    for pair in pairs[:3]:
        print(f'   → {pair["col_a"]} ↔ {pair["col_b"]}（r={pair["pearson"]}）')

    print(f'\n⚡ 异常值（IQR法）:')
    outliers = report.get('outlier_detection', {})
    severe = [(c, i) for c, i in outliers.items() if i.get('severity') in ('轻度', '严重')]
    if severe:
        for col, info in severe[:5]:
            print(f'   {col}: {info["outlier_rate"]} ({info["severity"]})')
    else:
        print('   未发现明显异常值')

    print(f'\n🕐 时间戳 fid 检测:')
    for domain, info in report.get('timestamp_fid_detect', {}).items():
        ts_fids = info.get('auto_detected_ts_fids', [])
        if ts_fids:
            fid_list = [str(t['fid']) for t in ts_fids]
            print(f'   {domain}: 检测到时间戳 fid={fid_list}，schema ts_fid={info["schema_ts_fid"]}')

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

    print(f'\n🔀 跨域 item 重叠:')
    overlap_info = report.get('cross_domain_overlap', {})
    if overlap_info:
        for pair, info in overlap_info.items():
            print(f'   {pair}: 重叠={info["overlap"]:,} ({info["overlap_rate_a"]} of A), {info["suggestion"]}')
    else:
        print('   无法分析（所有域的 item fid 均超过 vocab 上限）')

    print(f'\n💡 综合建议:')
    for s in report.get('summary_suggestions', []):
        print(f'   {s}')

    print('\n' + '═' * 60)


# ────────────────────────────────────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据前置清洗分析')
    parser.add_argument('--data_path', type=str, required=True, help='parquet 数据路径')
    parser.add_argument('--schema_path', type=str, default=None, help='schema.json 路径')
    parser.add_argument('--output_dir', type=str, default='data_analysis', help='输出目录')
    parser.add_argument('--max_rows', type=int, default=None, help='最大分析行数（None=全部）')
    parser.add_argument('--export_rows', type=int, default=None, help='导出前 N 行供离线训练')
    args = parser.parse_args()

    # schema 默认和 data 同目录
    if args.schema_path is None:
        args.schema_path = os.path.join(os.path.dirname(args.data_path), 'schema.json')

    run_analysis(
        data_path=args.data_path,
        schema_path=args.schema_path,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        export_rows=args.export_rows,
    )
