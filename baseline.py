import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. 数据加载
# ─────────────────────────────────────────────

DATA_PATH = "demo_1000.parquet"

df = pd.read_parquet(DATA_PATH)
print(f"数据集形状: {df.shape}")


# ─────────────────────────────────────────────
# 2. 特征列定义
# ─────────────────────────────────────────────

# 标量 User Int Features（35列）
scalar_user_int_cols = [
    "user_int_feats_1", "user_int_feats_3", "user_int_feats_4",
    "user_int_feats_48", "user_int_feats_49", "user_int_feats_50",
    "user_int_feats_51", "user_int_feats_52", "user_int_feats_53",
    "user_int_feats_54", "user_int_feats_55", "user_int_feats_56",
    "user_int_feats_57", "user_int_feats_58", "user_int_feats_59",
    "user_int_feats_82", "user_int_feats_86",
] + [f"user_int_feats_{i}" for i in range(92, 110)]

# 标量 Item Int Features（13列）
scalar_item_int_cols = [
    "item_int_feats_5", "item_int_feats_6", "item_int_feats_7",
    "item_int_feats_8", "item_int_feats_9", "item_int_feats_10",
    "item_int_feats_12", "item_int_feats_13", "item_int_feats_16",
    "item_int_feats_81", "item_int_feats_83", "item_int_feats_84",
    "item_int_feats_85",
]

# 用户 Embedding 列（固定维度，可直接展开）
user_embedding_cols = ["user_dense_feats_61", "user_dense_feats_87"]

# Domain Sequence 列（稀疏序列，取非零元素数量作为统计特征）
domain_a_cols = [f"domain_a_seq_{i}" for i in range(38, 47)]
domain_b_cols = [f"domain_b_seq_{i}" for i in range(67, 80)] + ["domain_b_seq_88"]
domain_c_cols = [f"domain_c_seq_{i}" for i in range(27, 38)] + ["domain_c_seq_47"]
domain_d_cols = [f"domain_d_seq_{i}" for i in range(17, 27)]
domain_seq_cols = domain_a_cols + domain_b_cols + domain_c_cols + domain_d_cols


# ─────────────────────────────────────────────
# 3. 特征工程
# ─────────────────────────────────────────────

def extract_scalar_features(dataframe: pd.DataFrame) -> np.ndarray:
    """提取并填充标量整型特征（用户 + 广告），空值填 -1。"""
    all_scalar_cols = scalar_user_int_cols + scalar_item_int_cols
    scalar_features = dataframe[all_scalar_cols].fillna(-1).values.astype(np.float32)
    return scalar_features


def extract_embedding_features(dataframe: pd.DataFrame) -> np.ndarray:
    """展开固定维度的用户 Embedding 向量（256 + 320 = 576 维）。"""
    embedding_parts = []
    for col in user_embedding_cols:
        embedding_matrix = np.stack(dataframe[col].values).astype(np.float32)
        embedding_parts.append(embedding_matrix)
    return np.concatenate(embedding_parts, axis=1)


def extract_sequence_statistics(dataframe: pd.DataFrame) -> np.ndarray:
    """
    将稀疏 Domain Sequence 列压缩为统计特征：
    每列取非零元素数量（即有效行为次数）。
    """
    sequence_stats = []
    for col in domain_seq_cols:
        non_zero_counts = dataframe[col].apply(
            lambda sequence: int(np.count_nonzero(sequence)) if sequence is not None else 0
        ).values.astype(np.float32)
        sequence_stats.append(non_zero_counts.reshape(-1, 1))
    return np.concatenate(sequence_stats, axis=1)


def build_feature_matrix(dataframe: pd.DataFrame) -> np.ndarray:
    """拼接所有特征：标量特征 + 用户 Embedding + 序列统计特征。"""
    scalar_features = extract_scalar_features(dataframe)
    embedding_features = extract_embedding_features(dataframe)
    sequence_features = extract_sequence_statistics(dataframe)

    feature_matrix = np.concatenate(
        [scalar_features, embedding_features, sequence_features], axis=1
    )
    print(f"特征矩阵形状: {feature_matrix.shape}")
    return feature_matrix


# ─────────────────────────────────────────────
# 4. 标签处理
# ─────────────────────────────────────────────

def build_labels(dataframe: pd.DataFrame) -> np.ndarray:
    """
    将 label_type 转换为二分类标签：
    - label_type=1 → 0（曝光/点击）
    - label_type=2 → 1（转化/深度行为）
    """
    return (dataframe["label_type"] == 2).astype(int).values


# ─────────────────────────────────────────────
# 5. 模型训练与评估
# ─────────────────────────────────────────────

def train_and_evaluate(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """划分训练/测试集，训练模型并输出评估指标。"""
    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")
    print(f"训练集正样本比例: {y_train.mean():.3f}, 测试集正样本比例: {y_test.mean():.3f}")

    # 特征标准化（对 Embedding 列影响较大）
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # ── 模型一：逻辑回归（基线）──
    print("\n─── 逻辑回归（基线）───")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    lr_model.fit(x_train_scaled, y_train)
    lr_predictions = lr_model.predict(x_test_scaled)
    lr_probabilities = lr_model.predict_proba(x_test_scaled)[:, 1]

    print(classification_report(y_test, lr_predictions, target_names=["点击", "转化"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, lr_probabilities):.4f}")

    # ── 模型二：GBDT（梯度提升树）──
    print("\n─── GBDT（梯度提升树）───")
    gbdt_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=random_state,
    )
    # GBDT 对标量特征效果好，仅使用标量 + 序列统计特征（去掉高维 Embedding）
    scalar_dim = len(scalar_user_int_cols) + len(scalar_item_int_cols)
    sequence_dim = len(domain_seq_cols)
    x_train_gbdt = np.concatenate(
        [x_train[:, :scalar_dim], x_train[:, -sequence_dim:]], axis=1
    )
    x_test_gbdt = np.concatenate(
        [x_test[:, :scalar_dim], x_test[:, -sequence_dim:]], axis=1
    )

    gbdt_model.fit(x_train_gbdt, y_train)
    gbdt_predictions = gbdt_model.predict(x_test_gbdt)
    gbdt_probabilities = gbdt_model.predict_proba(x_test_gbdt)[:, 1]

    print(classification_report(y_test, gbdt_predictions, target_names=["点击", "转化"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, gbdt_probabilities):.4f}")

    # ── 特征重要性（GBDT）──
    scalar_feature_names = scalar_user_int_cols + scalar_item_int_cols + domain_seq_cols
    feature_importances = pd.Series(
        gbdt_model.feature_importances_,
        index=scalar_feature_names,
    ).sort_values(ascending=False)

    print("\n─── Top 10 重要特征（GBDT）───")
    print(feature_importances.head(10))


# ─────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────

if __name__ == "__main__":
    feature_matrix = build_feature_matrix(df)
    labels = build_labels(df)

    print(f"\n标签分布: 点击={int((labels == 0).sum())}, 转化={int((labels == 1).sum())}")

    train_and_evaluate(feature_matrix, labels)
