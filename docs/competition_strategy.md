# TAAC2026 赛题理解与技术思路

## 一、赛题核心

**赛题全称**：Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation

**核心问题**：能否用一套同构的 Recommendation Block，在一个架构里同时处理多域行为序列 + 多域非序列特征的交叉？

**评估指标**：AUC of ROC（同时有严格的推理延迟限制，超时提交直接作废）

**禁止**：模型集成/融合（Ensemble），只能单一架构

---

## 二、为什么要"统一"：双轨并行的瓶颈

### 传统双轨架构

过去推荐系统沿两条线独立发展：

| 轨道 | 代表模型 | 解决问题 |
|---|---|---|
| 特征交叉建模 | DeepFM、DCN、Wukong | 用户属性、广告属性等静态特征的组合关系 |
| 序列行为建模 | DIN、DIEN、SIM、TWIN | 用户历史行为轨迹的时序信号 |

两套网络各自输出向量，最后浅层 concat → MLP 融合，称为"双轨并行、后期融合"。

### GPU 时代的三大瓶颈

**① 显存无法共享**
```
特征交叉网络：独占显存 A
序列建模网络：独占显存 B
总显存 = A + B，两套参数无法复用
→ 想把模型做大，显存先爆
```

**② GPU 算子无法融合**
```
MLP（矩阵乘法）和 Transformer Attention（softmax + QKV）
是完全不同的计算模式，GPU 编译器无法将它们合并成一个高效 Kernel
→ 算力利用率低下
```

**③ 无法 Scaling**
```
大语言模型 Scaling：同构 Block × N，参数翻倍 → 效果幂律提升
推荐系统"Scaling"困境：
  - 特征交叉 MLP 加深超过 6 层基本无收益
  - 序列 Attention 是 O(L²) 复杂度，序列加长延迟暴涨
  - 两套异构网络无统一"堆叠单元"可规模化
```

### 统一架构的目标

```
输入特征 & 序列
      ↓
统一 Tokenization（统一变成 Token）
      ↓
Recommendation Block × N  ← 同构可堆叠，GPU 高效并行
      ↓
pCVR 预测头
```

---

## 三、官方基线 HyFormer 的设计

```
用户/广告特征 → NS Tokenizer → NS Tokens（非序列 Token）
行为序列      → Seq Tokenizer → Seq Tokens（序列 Token）
                      ↓
              MultiSeqHyFormerBlock × N
                      ↓
              Query Generator → Q Tokens
                      ↓
              分类头 → pCVR logit
```

**关键模块**：
- **NS Token**：RankMixerNSTokenizer 把所有非序列特征 Embedding 均匀切分为固定数量的 Token
- **Seq Token**：四个行为域（a/b/c/d）各自独立 Embedding + 投影
- **统一 Block**：Q/NS/Seq Token 一起进 Transformer 做 cross-attention

**基线局限性（即比赛机会点）**：
- NS Token 和 Seq Token 在 Block 内交互较浅（NS 和 Seq 之间无直接交叉）
- 多域序列之间跨域关联建模弱
- RankMixer 均匀切分未考虑特征语义分组

---

## 四、技术提分方向

### 方向 1：更好的 NS Tokenizer（难度：中）

**问题**：RankMixer 均匀切分，忽略特征语义关联

**思路**：
- 语义感知分组：按特征语义（用户画像类/行为统计类/广告属性类）分组，组内 self-attention 聚合后投影为 1 个 Token
- 层次化 Tokenization：组内聚合 → 组间交叉，两级 Token 结构
- 动态 Token 数量：根据特征信息量（vocab_size、方差）自适应分配

### 方向 2：多域序列的跨域交互（难度：中高）

**问题**：四个域序列独立处理，跨域行为关联被忽略

**思路**：
- 跨域 Co-Attention：domain_a 序列作为 K/V，domain_b 的 Q Token 去 attend，捕捉跨域兴趣迁移
- 时间对齐建模：四个域行为按时间排列成统一时间线
- 域感知位置编码：在时间桶基础上加入域 ID Embedding

### 方向 3：特征与序列的深度融合（难度：高，核心创新）

**问题**：NS Token 和 Seq Token 融合较浅，序列对特征的反向影响弱

**思路**：
- 双向 Cross-Attention：NS Token 更新 Q 的同时，也被 Seq Token 更新（用户行为修正用户画像 Token）
- Feature-Sequence Co-Evolution：每层 Block 里 NS/Seq Token 互相 attend，迭代精炼
- 类 PLE 结构：底层 Block 专注序列，高层 Block 专注特征交叉，中间层做融合

### 方向 4：时间建模深化（难度：中）

**思路**：
- 相对时间注意力偏置：在 attention score 加时间差相对偏置（类 ALiBi），替代固定 Embedding
- 时间衰减门控：序列 Token 权重乘时间衰减 `exp(-λ * Δt)`
- 多粒度时间建模：分钟级（即时兴趣）+ 天级（近期兴趣）+ 月级（长期偏好）

### 方向 5：高效长序列处理（难度：中，效率必选）

**思路**：
- Target-Aware 稀疏 Attention（类 SIM/ETA）：只取与目标广告最相关的 TopK 行为
- Hierarchical Pooling：时间窗口内 local pooling → global attention，O(N) → O(√N)
- 线性 Attention（Linformer/Performer）：O(L²) → O(L)

---

## 五、工程优化（满足延迟约束）

| 优化点 | 方法 |
|---|---|
| Embedding 压缩 | Hash Embedding + Compositional Embedding |
| 计算图优化 | `torch.compile` / TorchScript 算子融合 |
| 混合精度 | FP16/BF16 训练，Embedding 查表 INT8 |
| 批量推理 | Padding 按 bucket 分组，减少无效计算 |
| 稀疏优化器 | Adagrad（已实现）+ 高基数 Embedding 冷重启 |

---

## 六、最值得押注的创新方向

> **设计完全同构的 Recommendation Block，让 NS Token 和 Seq Token 在每一层做双向互相 attention（Co-Evolution），同时验证该架构下 AUC 随参数量的幂律增长（Scaling Law）**

该方向同时命中两个创新奖：
- **统一架构创新奖**（4.5 万美元）
- **Scaling Law 创新奖**（4.5 万美元）

两奖独立于排名，只要方法有原创性即可获奖，并有机会在 KDD Workshop 发表论文。

---

## 七、比赛时间线与策略

| 阶段 | 时间 | 目标 |
|---|---|---|
| 第一轮 | 4.24 - 5.23 | 跑通基线 → 特征工程实验 → NS Tokenizer 改进 → 进 TOP50 |
| 第二轮 | 5.25 - 6.24 | 数据量 ×10，验证 Scaling Law，引入高效长序列方案 |
| 颁奖 | 8.9 KDD 2026 | — |

**注意**：全程禁止 Ensemble，只能单一架构做到极致。

---

## 八、与自身业务的结合思考

| 业务洞察 | 具体方向 |
|---|---|
| 类别不平衡（点击:转化 ≈ 7:1） | Focal Loss（已实现），难样本挖掘（点击但未转化的负样本） |
| 跨域行为迁移 | 用用户在域 A 的丰富行为增强对域 B 的预估 |
| 延迟转化建模 | `label_time > timestamp`，存在行为到转化的时间差，用 DFM 避免 label leakage |
| 用户 Embedding 复用 | `user_dense_feats_61`（256维）/ `user_dense_feats_87`（320维）作为跨场景冷启动信号 |
