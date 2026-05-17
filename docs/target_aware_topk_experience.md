# Target-Aware TopK 改造经验文档

> 整理时间：2026-05-08  
> 适用项目：TAAC2026 广告转化率预估竞赛（pCVR）

---

## 一、背景与问题

### 官方基线结构（PCVRHyFormer）

- **架构**：NS Token（用户/物品侧特征归一化表示）+ Seq Token（行为序列）分离处理
- **序列编码器**：`LongerEncoder`，核心逻辑是先对行为序列做 **Recency TopK 截断**，取最近的 `top_k=64` 条，再送入 Transformer 编码
- **问题**：无论候选广告是什么，都直接取"最新的 64 条"——忽略了历史行为与目标广告之间的相关性

### 发现的双层截断问题

训练流程中存在**两层截断**，导致大量历史信息被提前丢弃：

1. **Dataset 层面**：`seq_max_lens` 参数在数据加载阶段就做了截断（如 `seq_b:256`），进入模型的序列最多只有 256 条
2. **LongerEncoder 层面**：TopK 截断再取最新 64 条

这意味着 target-aware 方法根本看不到被第一层截掉的历史行为。**修复方案**：将 `seq_max_lens` 改为接近各域 p90 的值，仅作为防 OOM 的安全边界。

---

## 二、核心改动

### 改动 1：Target-Aware TopK（SIM 风格的目标感知检索）

**文件**：`model.py`

**改动位置**：`LongerEncoder` 类，新增 `_gather_top_k_target_aware()` 方法

**核心逻辑**：
- 用目标广告的 NS embedding 均值（`item_ns.mean(dim=1)`，shape `(B, D)`）作为 **target_query**
- 与序列中每个 token 做点积，计算相关性分数
- 选出分数最高的 top_k 条，softmax 加权后送入 Transformer

```python
def _gather_top_k_target_aware(self, x, key_padding_mask, target_query):
    scores = torch.bmm(x, target_query.unsqueeze(-1)).squeeze(-1)  # (B, L)
    scores = scores.masked_fill(key_padding_mask, float('-inf'))
    topk_scores, indices = torch.topk(scores, self.top_k, dim=1, sorted=True)

    # 安全 softmax：padding 槽位先置 0 再 softmax，避免 NaN
    valid_len = (~key_padding_mask).sum(dim=1)
    actual_k = torch.clamp(valid_len, max=self.top_k)
    pad_count = self.top_k - actual_k
    pos_indices = torch.arange(self.top_k, device=x.device).unsqueeze(0)
    new_padding_mask = pos_indices < pad_count.unsqueeze(1)
    safe_scores = topk_scores.masked_fill(new_padding_mask, 0.0)
    relevance_weights = torch.softmax(safe_scores, dim=1)
    relevance_weights = relevance_weights * (~new_padding_mask).float()

    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)
    top_k_tokens = top_k_tokens * relevance_weights.unsqueeze(-1)
    return top_k_tokens, new_padding_mask, indices
```

**开关参数**：`--use_target_aware_topk`（默认关闭）

---

### 改动 2：Query Projection（target_query 语义增强）

**文件**：`model.py`

**改动位置**：`PCVRHyFormer.__init__()` + `forward()` + `predict()`

**动机**：直接用 `item_ns.mean()` 作为 query，早期 epoch embedding 还未训练好，query 信号弱；加一层 LayerNorm + Linear + SiLU 可以让模型学到更好的 query 表示空间。

**新增层**（仅当 `use_query_projection=True` 时实例化，避免无用参数）：

```python
if use_query_projection:
    self.query_proj = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.SiLU(),
    )
else:
    self.query_proj = None
```

**forward/predict 中的使用**：

```python
if self.use_target_aware_topk:
    target_query = item_ns.mean(dim=1)  # (B, D)
    if self.use_query_projection and self.query_proj is not None:
        target_query = self.query_proj(target_query)
else:
    target_query = None
```

**开关参数**：`--use_query_projection`（默认关闭，需配合 `--use_target_aware_topk` 使用）

---

### 改动 3：扩大 seq_max_lens 检索池

**文件**：`train.py`

**动机**：target-aware TopK 的检索质量取决于候选池大小。原来 `seq_b:256` 相当于从 256 条里选 64 条，扩大后能从更完整的历史里检索最相关的行为。

| 域 | 旧值 | 新值（推荐） | 说明 |
|---|---|---|---|
| seq_a | 256 | 128 | a 域序列较短，可适当缩小 |
| seq_b | 256 | 1000 | b 域序列较长，重点扩大 |
| seq_c | 512 | 512 | 保持不变 |
| seq_d | 512 | 2000 | d 域序列最长，重点扩大 |

> ⚠️ 注意：这个值只是防 OOM 的安全上限，**实际进入 Transformer 的永远是 top_k=64 条**（target-aware 选出来的），不会增加 Transformer 的计算量。但点积计算量会随序列长度线性增长，线上需要压测确认延迟。

---

## 三、实验结果对比

### 5 Epoch 快速验证（80/20 split）

| 版本 | Peak AUC | 最优 Epoch |
|---|---|---|
| 基线 Recency TopK（seq_b:256） | 0.9800 | E4 |
| Target-Aware TopK（seq_b:600） | 0.9719 | E5 |

**观察**：5 epoch 时 target-aware 略低，但 LogLoss 在 E5 为 0.2169（显著低于基线 0.2456），说明预测置信度更高。

---

### 15 Epoch 完整对比（patience=5）

| 版本 | 配置 | Peak AUC | 最低 LogLoss | Early Stop Epoch |
|---|---|---|---|---|
| 基线 v1（原始） | seq_b:256, bs=64 | 0.9821 | 0.1941 | E12 |
| 基线 v2（调参） | seq_b:256, dropout=0.0, bs=128 | 0.9813 | 0.2143 | E12 |
| Target-Aware v1 | topk, seq_b:600 | 0.9796 | 0.1975 | E14 |
| **Target-Aware v2（最优）** | topk + query_proj + seq_b:1000 | **0.9832** ✅ | **0.1839** ✅ | E15 |

**结论**：
- 方向2（query_projection）+ 方向3（更大 seq_max_lens）合并后，**Peak AUC 0.9832 首次超越基线**（+0.0011）
- LogLoss 全面领先，置信度提升明显
- target-aware 方法收敛较慢（需要更多 epoch 让 item embedding 学好），但最终效果更优且更稳定

---

## 四、关键经验与教训

### ✅ 正确的做法

1. **先解决截断问题再上 target-aware**：双层截断会让 target-aware 根本看不到有价值的历史，修复 `seq_max_lens` 是前提
2. **安全 softmax**：全 padding 序列时 topk_scores 全为 `-inf`，直接 softmax 会产生 NaN，必须先 mask 为 0 再 softmax，softmax 后再显式置 0
3. **用开关隔离特性**：`--use_target_aware_topk` 和 `--use_query_projection` 独立开关，方便消融实验
4. **query_projection 的正确位置**：在 target_query 传入 TopK 打分之前做，而不是在序列 token 上做

### ⚠️ 注意事项

1. **target-aware 需要更多 epoch 才能收敛**：早期 epoch item embedding 未训练好，query 信号弱，因此前几个 epoch 会落后于基线，patience 不能设太小（建议 ≥ 5）
2. **seq_max_lens 扩大对基线无效**：基线 Recency TopK 直接取最新的，扩大上限对它完全没有帮助，只对 target-aware 有意义
3. **线上内存压力**：seq_max_lens 扩大后，每条样本的 tensor 更大，实际数据量大时需要评估 OOM 风险，可配合梯度检查点或减小 batch_size
4. **demo 数据局限性**：本地 demo 只有 1000 条样本，序列实际长度可能已经被截断，扩大 seq_max_lens 上限在 demo 上的收益被低估——**线上完整数据效果应该更好**

---

## 五、线上调试建议

### 推荐配置（完整方案）

```bash
python train.py \
    --use_target_aware_topk \
    --use_query_projection \
    --seq_max_lens 'seq_a:128,seq_b:1000,seq_c:512,seq_d:2000' \
    --num_epochs 30 \
    --patience 5 \
    --seq_top_k 64
```

### 分阶段验证策略

```bash
# 阶段1：只开 target-aware，验证基础效果
--use_target_aware_topk --seq_max_lens 'seq_a:128,seq_b:1000,seq_c:512,seq_d:2000'

# 阶段2：再加 query_projection，看是否有增益
--use_target_aware_topk --use_query_projection --seq_max_lens 'seq_a:128,seq_b:1000,seq_c:512,seq_d:2000'

# 对照组（基线）
# 不传任何上述开关
```

### 线上 seq_max_lens 调优参考

线上真实数据的序列长度分布与 demo 不同，建议：
1. 先跑数据分析查看各域序列长度的 p50/p90/p99
2. 将 `seq_max_lens` 设为各域的 p90 值（超过 p90 的样本极少，设太大浪费显存）
3. 如果 OOM，优先缩小 `seq_d`（d 域序列最长，内存消耗最大）

---

## 六、文件改动汇总

| 文件 | 改动内容 |
|---|---|
| `model.py` | 新增 `_gather_top_k_target_aware()` 方法；`PCVRHyFormer.__init__()` 添加 `use_target_aware_topk` 和 `use_query_projection` 参数；添加 `query_proj` 层；`forward()` 和 `predict()` 中按开关构造 target_query |
| `train.py` | 添加 `--use_target_aware_topk` 和 `--use_query_projection` 命令行参数；修改 `seq_max_lens` 默认值；将两个开关传递给模型 |
