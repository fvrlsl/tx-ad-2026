# TAAC2026 基础操作说明

## 环境准备

### 安装依赖

```bash
pip install pyarrow pandas numpy scikit-learn
```

### 目录结构

```
race/
├── tx-ad-2026/
│   ├── demo_1000.parquet   # 数据集文件（~39MB）
│   └── README.md           # 官方数据集说明
├── data_analysis.md        # 字段含义分析文档
├── quickstart.md           # 本文档
├── train.py                # 模型训练代码
└── test.py                 # 数据读取示例
```

---

## 数据读取

### 基础读取

```python
import pandas as pd

df = pd.read_parquet("tx-ad-2026/demo_1000.parquet")

print(df.shape)    # (1000, 120)
print(df.columns)  # 所有列名
```

### 查看前 N 条数据

```python
# 查看关键列的前 10 条
key_cols = ['user_id', 'item_id', 'label_type', 'label_time', 'timestamp']
print(df[key_cols].head(10))
```

### 查看数据基本信息

```python
# 数据类型
print(df.dtypes)

# 空值统计
print(df.isnull().sum())

# 数值列统计摘要
print(df.describe())
```

---

## 特征分类访问

数据集共 120 列，分为 6 类特征，可按如下方式分组访问：

```python
# ID & Label 列（5列）
id_label_cols = ['user_id', 'item_id', 'label_type', 'label_time', 'timestamp']

# 标量 User Int Features（35列）
scalar_user_int_cols = [
    'user_int_feats_1', 'user_int_feats_3', 'user_int_feats_4',
    'user_int_feats_48', 'user_int_feats_49', 'user_int_feats_50',
    'user_int_feats_51', 'user_int_feats_52', 'user_int_feats_53',
    'user_int_feats_54', 'user_int_feats_55', 'user_int_feats_56',
    'user_int_feats_57', 'user_int_feats_58', 'user_int_feats_59',
    'user_int_feats_82', 'user_int_feats_86',
] + [f'user_int_feats_{i}' for i in range(92, 110)]

# 数组 User Int Features（11列）
array_user_int_cols = [
    'user_int_feats_15', 'user_int_feats_60', 'user_int_feats_62',
    'user_int_feats_63', 'user_int_feats_64', 'user_int_feats_65',
    'user_int_feats_66', 'user_int_feats_80', 'user_int_feats_89',
    'user_int_feats_90', 'user_int_feats_91',
]

# User Dense Features（10列）
user_dense_cols = [
    'user_dense_feats_61', 'user_dense_feats_62', 'user_dense_feats_63',
    'user_dense_feats_64', 'user_dense_feats_65', 'user_dense_feats_66',
    'user_dense_feats_87', 'user_dense_feats_89', 'user_dense_feats_90',
    'user_dense_feats_91',
]

# 标量 Item Int Features（13列）
scalar_item_int_cols = [
    'item_int_feats_5', 'item_int_feats_6', 'item_int_feats_7',
    'item_int_feats_8', 'item_int_feats_9', 'item_int_feats_10',
    'item_int_feats_12', 'item_int_feats_13', 'item_int_feats_16',
    'item_int_feats_81', 'item_int_feats_83', 'item_int_feats_84',
    'item_int_feats_85',
]

# 数组 Item Int Features（1列）
array_item_int_cols = ['item_int_feats_11']

# Domain Sequence Features（45列）
domain_a_cols = [f'domain_a_seq_{i}' for i in range(38, 47)]
domain_b_cols = [f'domain_b_seq_{i}' for i in range(67, 80)] + ['domain_b_seq_88']
domain_c_cols = [f'domain_c_seq_{i}' for i in range(27, 38)] + ['domain_c_seq_47']
domain_d_cols = [f'domain_d_seq_{i}' for i in range(17, 27)]
domain_seq_cols = domain_a_cols + domain_b_cols + domain_c_cols + domain_d_cols
```

---

## 常用数据操作

### 按 label_type 筛选

```python
# label_type=1：曝光/点击样本（876条）
df_click = df[df['label_type'] == 1]

# label_type=2：转化/深度行为样本（124条）
df_convert = df[df['label_type'] == 2]

print(f"点击样本: {len(df_click)}, 转化样本: {len(df_convert)}")
```

### 处理数组类型列

```python
import numpy as np

# 获取单行的数组特征
row = df.iloc[0]
user_embedding = np.array(row['user_dense_feats_61'])  # 256维向量
print(f"用户 Embedding 维度: {user_embedding.shape}")

# 获取行为序列（过滤掉 0 值）
seq = np.array(row['domain_a_seq_38'])
active_items = seq[seq != 0]
print(f"Domain A 有效行为数: {len(active_items)}")
```

### 处理空值

```python
# 查看各列空值数量
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])

# 对标量列填充空值
df[scalar_user_int_cols] = df[scalar_user_int_cols].fillna(-1)
df[scalar_item_int_cols] = df[scalar_item_int_cols].fillna(-1)
```

### 提取用户 Embedding

```python
import numpy as np

# 提取所有用户的 256 维 Embedding 矩阵
user_embeddings_256 = np.stack(df['user_dense_feats_61'].values)
print(f"用户 Embedding 矩阵形状: {user_embeddings_256.shape}")  # (1000, 256)

# 提取所有用户的 320 维 Embedding 矩阵
user_embeddings_320 = np.stack(df['user_dense_feats_87'].values)
print(f"用户 Embedding 矩阵形状: {user_embeddings_320.shape}")  # (1000, 320)
```

---

## 标签说明

| label_type | 数量 | 占比 | 推测含义 |
|---|---|---|---|
| 1 | 876 | 87.6% | 曝光/点击 |
| 2 | 124 | 12.4% | 转化/深度行为（如购买） |

> ⚠️ 注意：数据集存在**类别不平衡**问题（约 7:1），训练时需要考虑加权采样或调整损失函数权重。

---

## 注意事项

1. **路径问题**：数据文件位于 `tx-ad-2026/demo_1000.parquet`，运行脚本时需在项目根目录 `race/` 下执行。
2. **数组列处理**：`object` 类型列存储的是 Python 列表或 numpy 数组，使用前需用 `np.array()` 转换。
3. **空值处理**：`user_int_feats_54` 空值率高达 36.8%，`user_int_feats_58/59` 空值率 15%，使用前需填充。
4. **序列稀疏性**：Domain Sequence 列大量元素为 0，建议只取非零元素作为有效行为。
5. **Embedding 对齐**：`user_int_feats_{fid}` 和 `user_dense_feats_{fid}` 共享相同 `{fid}` 时联合描述同一实体，使用时注意对齐。
