# TAAC2026 项目文件说明

## 目录结构

```
race/
├── tx-ad-2026/                  # 官方提供的数据集与基线代码
│   ├── demo_1000.parquet        # 数据集（1000条样本，~39MB）
│   ├── README.md                # 官方数据集说明
│   ├── dataset.py               # 数据加载与特征处理
│   ├── model.py                 # PCVRHyFormer 模型定义
│   ├── trainer.py               # 训练器（训练循环、验证、早停、保存）
│   ├── train.py                 # 训练入口（命令行参数解析、主流程）
│   ├── test.py                  # 数据读取示例脚本
│   ├── utils.py                 # 工具函数（日志、早停、随机种子、Focal Loss）
│   ├── ns_groups.json           # NS Token 特征分组配置（示例）
│   └── run.sh                   # 一键启动训练的 Shell 脚本
├── data_analysis.md             # 字段含义分析文档
├── quickstart.md                # 基础操作说明
├── project_overview.md          # 本文档
└── train.py                     # 自定义训练代码（sklearn 基线）
```

---

## 文件详细说明

### `tx-ad-2026/` 官方基线代码

#### `dataset.py` — 数据加载与特征处理
基于 PyTorch `IterableDataset` 实现的高性能 Parquet 数据加载器。

**核心类**：`PCVRParquetDataset`
- 直接流式读取 `.parquet` 文件的 Row Group，支持多文件、多 worker 并行
- 按 `schema.json` 解析特征布局，将原始列转换为模型所需的 Tensor 格式
- 支持变长整型/浮点数组的 Padding、序列截断（`seq_max_lens`）
- 内置 OOB（越界）检测与 clip，防止 Embedding 索引越界
- 支持时间差分桶（Time Bucket），将行为时间差映射为离散 Embedding ID

**核心函数**：`get_pcvr_data()`
- 按 `valid_ratio` 将 Row Group 末尾部分划为验证集
- 返回 `(train_loader, valid_loader, train_dataset)` 三元组

---

#### `model.py` — PCVRHyFormer 模型
基于 **HyFormer** 架构的广告转化率预估模型（PCVR = Post-Click Conversion Rate）。

**核心组件**：
| 组件 | 说明 |
|---|---|
| `GroupNSTokenizer` | 按特征分组，将每组 Embedding 拼接后投影为一个 NS Token |
| `RankMixerNSTokenizer` | RankMixer 风格：所有 Embedding 拼接后均匀切分为 N 个 Token |
| `MultiSeqQueryGenerator` | 为每个序列域独立生成 Query Token |
| `MultiSeqHyFormerBlock` | 核心 Transformer Block，支持多序列并行注意力 |
| `PCVRHyFormer` | 完整模型：NS Tokenizer → 序列 Embedding → HyFormer Block → 分类头 |

**前向流程**：
1. 用户/广告整型特征 → NS Tokenizer → NS Tokens
2. 用户 Dense 特征（Embedding 向量）→ 线性投影 → Dense Token
3. 各域行为序列 → Embedding + 时间桶 → Seq Tokens
4. Query Generator 生成 Q Tokens
5. 多层 HyFormerBlock 处理
6. 输出投影 → 分类头 → logits

**两种 NS Tokenizer 模式**：
- `group`：需要 `ns_groups.json` 配置分组，每组输出 1 个 Token
- `rankmixer`（默认）：无需分组配置，自动将所有 Embedding 切分为指定数量的 Token

---

#### `trainer.py` — 训练器
`PCVRHyFormerRankingTrainer` 实现完整训练循环。

**关键特性**：
- **双优化器**：Embedding 参数用 `Adagrad`（`sparse_lr=0.05`），其余参数用 `AdamW`
- **损失函数**：支持 `BCEWithLogitsLoss`（默认）和 `Focal Loss`（处理类别不平衡）
- **评估指标**：AUC-ROC + Binary LogLoss
- **早停**：基于验证集 AUC，支持 `patience` 配置
- **稀疏参数冷重启**：每 epoch 结束后对高基数 Embedding 重新初始化，缓解过拟合
- **Checkpoint 管理**：按 `global_step` 命名，自动保留最优模型，附带 `schema.json`、`ns_groups.json`、`train_config.json`

---

#### `train.py`（官方）— 训练入口
命令行参数解析 + 主流程编排，支持通过环境变量覆盖路径配置。

**主要参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | 必填 | 训练数据目录（含 `.parquet` 和 `schema.json`） |
| `--ckpt_dir` | 必填 | Checkpoint 输出目录 |
| `--batch_size` | 256 | 批大小 |
| `--lr` | 1e-4 | AdamW 学习率 |
| `--num_epochs` | 999 | 最大 epoch 数（通常由早停终止） |
| `--patience` | 5 | 早停耐心值 |
| `--d_model` | 64 | 模型隐层维度 |
| `--emb_dim` | 64 | Embedding 维度 |
| `--num_hyformer_blocks` | 2 | HyFormer Block 层数 |
| `--loss_type` | bce | 损失函数（`bce` / `focal`） |
| `--ns_tokenizer_type` | rankmixer | NS Tokenizer 类型（`rankmixer` / `group`） |
| `--seq_max_lens` | a:256,b:256,c:512,d:512 | 各域序列最大长度 |

**环境变量**（优先级高于命令行）：
- `TRAIN_DATA_PATH`：训练数据目录
- `TRAIN_CKPT_PATH`：Checkpoint 输出目录
- `TRAIN_LOG_PATH`：日志目录

---

#### `run.sh` — 一键启动脚本
封装了两套训练配置，直接运行即可启动训练：

```bash
cd tx-ad-2026
bash run.sh --data_dir /path/to/data --ckpt_dir /path/to/ckpt --log_dir /path/to/log
```

**默认配置（RankMixer 模式）**：
- `--ns_tokenizer_type rankmixer`，用户 5 个 Token，广告 2 个 Token
- `--emb_skip_threshold 1000000`（超高基数特征跳过 Embedding）
- `--num_workers 8`

**备用配置（Group 模式，注释中）**：
- 使用 `ns_groups.json` 分组，7 个用户组 + 4 个广告组
- 需要 `num_queries=1`（受 `d_model % T == 0` 约束）

---

#### `utils.py` — 工具函数

| 函数/类 | 说明 |
|---|---|
| `LogFormatter` | 自定义日志格式，输出绝对时间 + 相对耗时 |
| `create_logger()` | 初始化 root logger，同时写文件（DEBUG）和控制台（INFO） |
| `EarlyStopping` | 基于 AUC 的早停，自动保存最优 checkpoint |
| `set_seed()` | 固定所有随机种子（Python / NumPy / PyTorch / CUDA） |
| `sigmoid_focal_loss()` | Focal Loss 实现，缓解正负样本不平衡 |

---

#### `ns_groups.json` — NS Token 特征分组配置
定义用户特征和广告特征的分组方式（仅在 `--ns_tokenizer_type group` 时使用）。

**分组示例**：
- 用户侧 7 组（U1–U7）：按特征语义分组，如 `U1=[1,15]`（基础属性）、`U2=[48,49,89,90,91]`（活跃度）
- 广告侧 4 组（I1–I4）：如 `I1=[11,13]`（关键词）、`I2=[5,6,7,8,12]`（分类/主/计划）

> ⚠️ 文件注释说明这是**示例配置**，实际分组需根据自己的 schema 调整。

---

#### `test.py`（官方）— 数据读取示例
简单的 Parquet 读取脚本，展示数据集的基本信息和前 10 条记录。

---

### 根目录自定义文件

#### `train.py`（自定义）— sklearn 基线训练
使用 `scikit-learn` 实现的轻量级基线模型，无需 GPU，适合快速验证特征效果。

**包含两个模型**：
- **逻辑回归**（基线）：使用标准化后的全量特征
- **GBDT**（梯度提升树）：使用标量 + 序列统计特征，并输出 Top10 特征重要性

**特征处理**：
- 标量特征：空值填 -1，直接使用
- 用户 Embedding：展开为 256+320=576 维向量
- 序列特征：压缩为非零元素数量（有效行为次数）

---

## 快速开始

### 方式一：sklearn 基线（无需 GPU）

```bash
cd /Users/aolai/codes/race
pip install pyarrow pandas scikit-learn
python3 train.py
```

### 方式二：官方 PCVRHyFormer（需要 GPU + 完整数据）

```bash
pip install torch pyarrow pandas scikit-learn tqdm tensorboard
cd tx-ad-2026
bash run.sh \
    --data_dir /path/to/full_data \
    --ckpt_dir ./checkpoints \
    --log_dir ./logs
```

> ⚠️ 官方训练代码需要 `schema.json` 文件（当前 demo 数据集未包含），需从竞赛官网获取完整数据集。
