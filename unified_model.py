"""UnifiedSeqModel: 将特征抽象为虚拟商品访问事件，与真实行为序列统一建模。

核心思想：
    用户的静态特征（年龄、性别、职业等）被视为"用户在 t=0 时刻访问过的虚拟商品"。
    每个特征字段的每个取值 → 唯一虚拟 item_id，拼接到真实行为序列最前面。
    特征 token 和行为 token 共享同一套 Embedding 表，送入统一 Transformer。

三个可调节接口（对应三大挑战）：
    挑战1 - 特征值连续性：num_buckets
        控制连续特征值的分桶数量。越大越精细但稀疏度越高，越小越粗粒度但泛化性更好。
    挑战2 - 序列长度爆炸：max_feat_tokens
        控制特征虚拟 token 的最大数量。按特征字段重要性排序后截断，避免序列过长。
    挑战3 - 特征 token 时间位置：feat_pos_mode
        控制特征 token 的位置编码策略：
        - 'zero'      : 特征 token 位置编码全为 0（静态，无时序意义）
        - 'learnable' : 每个特征字段有独立可学习的位置 Embedding
        - 'prepend'   : 特征 token 拼在序列最前，使用与行为 token 相同的连续位置编码

feature_specs 格式说明（来自 dataset.FeatureSchema.entries）：
    List of (fid: int, col_offset: int, length: int)
    其中 col_offset 是该字段在 int_feats tensor 中的列起始索引，
    length == 1 表示标量特征，length > 1 表示多值特征（暂不虚拟化）。
    vocab_sizes 是对应 fid 的词表大小列表，与 feature_specs 一一对应。
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    ModelInput,
    TransformerEncoder,
    RotaryEmbedding,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FeatureAsItemTokenizer：把特征映射为虚拟商品访问事件
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureAsItemTokenizer(nn.Module):
    """把用户/广告的整型特征字段映射为虚拟 item_id，构成特征虚拟序列。

    每个特征字段 fid 的每个分桶值 bucket_v → 唯一虚拟 item_id：
        virtual_id = virtual_id_base[fid] + bucket_v

    其中 virtual_id_base 保证不同字段的 id 空间不重叠，
    且整个虚拟 id 空间相对于调用方传入的 virtual_id_start 偏移，
    避免与真实行为序列的 item_id 命名空间冲突。

    Args:
        feature_specs:   来自 dataset.FeatureSchema.entries，
                         格式 [(fid, col_offset, length), ...]，
                         只处理 length == 1 的标量字段。
        vocab_sizes:     与 feature_specs 等长的词表大小列表，
                         vocab_sizes[i] 是 feature_specs[i][0]（fid）的原始词表大小。
        virtual_id_start: 该 tokenizer 的虚拟 id 从此值开始分配，
                          调用方负责保证不同 tokenizer 之间无重叠。
        num_buckets:     【挑战1接口】连续/大基数特征的分桶数。
            - None：不分桶，直接使用原始特征值（需保证 0 < value < vocab_size）
            - int：把特征值均匀分桶到 [1, num_buckets]（0 保留给 padding）
        max_feat_tokens: 【挑战2接口】特征虚拟 token 的最大数量。
            - None：不限制，所有标量字段都生成 token
            - int：按字段顺序截断到前 max_feat_tokens 个字段
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        vocab_sizes: List[int],
        virtual_id_start: int,
        num_buckets: Optional[int],
        max_feat_tokens: Optional[int],
    ) -> None:
        super().__init__()

        self.num_buckets = num_buckets
        self.max_feat_tokens = max_feat_tokens

        # 只处理标量特征（length == 1）
        # col_offsets[i]：字段 i 在 int_feats tensor 中的列索引
        # id_bases[i]：字段 i 的虚拟 id 起始值（在全局 id 空间中）
        # orig_vocab_sizes[i]：字段 i 的原始词表大小（用于 num_buckets=None 时的上界 clamp）
        col_offsets: List[int] = []
        id_bases: List[int] = []
        orig_vocab_sizes: List[int] = []

        virtual_cursor = virtual_id_start
        for (fid, col_offset, length), vocab_size in zip(feature_specs, vocab_sizes):
            if length != 1:
                # 多值特征（数组型）暂不映射为虚拟 token
                continue
            effective_slots = num_buckets if num_buckets is not None else vocab_size
            col_offsets.append(col_offset)
            id_bases.append(virtual_cursor)
            orig_vocab_sizes.append(vocab_size)
            # 每个字段占用 effective_slots + 1 个 id 槽（0 为 padding）
            virtual_cursor += effective_slots + 1

        # 应用 max_feat_tokens 截断
        if max_feat_tokens is not None:
            col_offsets = col_offsets[:max_feat_tokens]
            id_bases = id_bases[:max_feat_tokens]
            orig_vocab_sizes = orig_vocab_sizes[:max_feat_tokens]

        self.num_feat_tokens: int = len(col_offsets)
        # 该 tokenizer 消耗的虚拟 id 总量（不含 virtual_id_start 之前的部分）
        self.virtual_id_consumed: int = virtual_cursor - virtual_id_start

        # 注册为 buffer，随模型保存，且不是可训练参数
        self.register_buffer(
            'col_offsets',
            torch.tensor(col_offsets, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            'id_bases',
            torch.tensor(id_bases, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            'orig_vocab_sizes',
            torch.tensor(orig_vocab_sizes, dtype=torch.long),
            persistent=True,
        )

        effective_slots = num_buckets if num_buckets is not None else 0
        logging.info(
            f"FeatureAsItemTokenizer: {self.num_feat_tokens} scalar feat tokens, "
            f"virtual_id_start={virtual_id_start}, "
            f"virtual_id_end={virtual_cursor}, "
            f"num_buckets={num_buckets}, max_feat_tokens={max_feat_tokens}"
        )

    def forward(self, int_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """把整型特征 tensor 转换为虚拟 item_id 序列（向量化实现）。

        Args:
            int_feats: (B, total_feat_dim) 整型特征 tensor，值为原始特征值

        Returns:
            virtual_ids: (B, num_feat_tokens) 虚拟 item_id，可直接送入 Embedding
            valid_mask:  (B, num_feat_tokens) bool，True 表示该 token 有效（非 padding）
        """
        if self.num_feat_tokens == 0:
            B = int_feats.shape[0]
            device = int_feats.device
            empty_ids = torch.zeros(B, 0, dtype=torch.long, device=device)
            empty_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
            return empty_ids, empty_mask

        # raw_values: (B, num_feat_tokens)，按 col_offsets 批量取列
        raw_values = int_feats[:, self.col_offsets]  # (B, K)

        # 有效性：原始值 > 0（0 表示缺失/padding）
        valid_mask = raw_values > 0  # (B, K) bool

        # 【挑战1接口】分桶处理
        if self.num_buckets is not None:
            # 均匀分桶：raw_value % num_buckets 落在 [0, num_buckets-1]，+1 后 [1, num_buckets]
            # 0 保留给 padding（缺失）
            bucket_values = (raw_values % self.num_buckets + 1).clamp(min=1, max=self.num_buckets)
        else:
            # 不分桶：直接使用原始值，但 clamp 到 [1, vocab_size-1]（防止越界）
            # orig_vocab_sizes: (K,) → broadcast to (B, K)
            upper = (self.orig_vocab_sizes - 1).unsqueeze(0)  # (1, K)
            bucket_values = raw_values.clamp(min=1).clamp_max(upper)

        # 加上字段基地址，得到全局虚拟 id
        # id_bases: (K,) → broadcast to (B, K)
        virtual_ids = self.id_bases.unsqueeze(0) + bucket_values  # (B, K)

        # 缺失位置置 0（padding_idx）
        virtual_ids = virtual_ids * valid_mask.long()  # (B, K)

        return virtual_ids, valid_mask


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 特征 Token 位置编码（挑战3接口）
# ═══════════════════════════════════════════════════════════════════════════════

class FeaturePositionEncoding(nn.Module):
    """【挑战3接口】特征 token 的位置编码策略。

    Args:
        num_feat_tokens: 特征虚拟 token 的数量
        d_model:         模型隐层维度
        feat_pos_mode:   位置编码模式
            - 'zero'      : 不加位置编码，特征 token 视为"无时序"的全局上下文
            - 'learnable' : 每个特征字段有独立的可学习位置 Embedding（推荐）
            - 'prepend'   : 使用 sinusoidal 固定位置编码，拼在行为序列之前
    """

    def __init__(
        self,
        num_feat_tokens: int,
        d_model: int,
        feat_pos_mode: str = 'learnable',
    ) -> None:
        super().__init__()
        assert feat_pos_mode in ('zero', 'learnable', 'prepend'), (
            f"feat_pos_mode 必须是 'zero'/'learnable'/'prepend'，当前: {feat_pos_mode}"
        )

        self.feat_pos_mode = feat_pos_mode
        self.num_feat_tokens = num_feat_tokens

        if num_feat_tokens == 0:
            return

        if feat_pos_mode == 'learnable':
            self.pos_emb = nn.Embedding(num_feat_tokens, d_model)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        elif feat_pos_mode == 'prepend':
            self.register_buffer(
                'pos_enc',
                self._build_sinusoidal(num_feat_tokens, d_model),
                persistent=False,
            )

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        """构建标准 sinusoidal 位置编码，shape (1, max_len, d_model)。"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        enc = torch.zeros(1, max_len, d_model)
        enc[0, :, 0::2] = torch.sin(position * div_term)
        # 当 d_model 为奇数时，cos 项比 sin 少一列
        enc[0, :, 1::2] = torch.cos(position * div_term[:enc.shape[2] // 2])
        return enc

    def forward(self, feat_tokens: torch.Tensor) -> torch.Tensor:
        """给特征 token 加位置编码。

        Args:
            feat_tokens: (B, num_feat_tokens, d_model)

        Returns:
            (B, num_feat_tokens, d_model)，加了位置编码
        """
        if self.num_feat_tokens == 0 or self.feat_pos_mode == 'zero':
            return feat_tokens

        if self.feat_pos_mode == 'learnable':
            positions = torch.arange(self.num_feat_tokens, device=feat_tokens.device)
            return feat_tokens + self.pos_emb(positions).unsqueeze(0)

        # 'prepend'：sinusoidal
        return feat_tokens + self.pos_enc[:, :self.num_feat_tokens, :].to(feat_tokens.device)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UnifiedSeqModel：统一序列 Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedSeqModel(nn.Module):
    """统一序列推荐模型：特征虚拟 token + 行为序列 token → 单一 Transformer。

    将用户/广告特征视为"虚拟商品访问事件"，和真实行为序列拼接后统一建模，
    彻底消除特征交叉模块和序列建模模块的边界。

    三大挑战的可调节接口：
        num_buckets    (挑战1): 特征值分桶数，控制连续特征的离散化粒度
        max_feat_tokens(挑战2): 特征虚拟 token 最大数量，控制序列长度
        feat_pos_mode  (挑战3): 特征 token 位置编码策略

    Args:
        user_feat_specs:    用户整型特征 schema，格式 [(fid, col_offset, length), ...]
                            与 dataset.user_int_schema.entries 对齐。
        user_vocab_sizes:   用户整型特征词表大小列表，与 user_feat_specs 等长。
        item_feat_specs:    广告整型特征 schema，格式同上，
                            与 dataset.item_int_schema.entries 对齐。
        item_vocab_sizes:   广告整型特征词表大小列表，与 item_feat_specs 等长。
        seq_vocab_sizes:    各行为域的词表大小，格式 {domain: [vs_per_fid, ...]}
                            其中 vs_per_fid 按侧信息字段顺序排列，
                            与 dataset.seq_domain_vocab_sizes 对齐。
        real_item_vocab_size: 真实行为 item_id 的词表大小（各域求和或取最大，
                              由调用方根据 seq_vocab_sizes 的第一个 fid 计算）。
        d_model:            模型隐层维度。
        emb_dim:            Embedding 维度（投影前）。
        num_heads:          注意力头数。
        num_layers:         Transformer 层数。
        hidden_mult:        FFN 隐层放大倍数。
        dropout_rate:       Dropout 比率。
        action_num:         分类头输出维度（1 = 二分类）。
        num_buckets:        【挑战1】特征值分桶数（None=不分桶，直接用原始值）。
        max_feat_tokens:    【挑战2】特征虚拟 token 最大数（None=不限制）。
        feat_pos_mode:      【挑战3】特征 token 位置编码模式。
        seq_max_len:        行为序列最大长度（用于 RoPE cache）。
        num_time_buckets:   时间分桶 Embedding 的桶数（0=不使用时间编码）。
    """

    def __init__(
        self,
        user_feat_specs: List[Tuple[int, int, int]],
        user_vocab_sizes: List[int],
        item_feat_specs: List[Tuple[int, int, int]],
        item_vocab_sizes: List[int],
        seq_vocab_sizes: Dict[str, List[int]],
        real_item_vocab_size: int,
        d_model: int = 64,
        emb_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        # ── 三大挑战的可调节接口 ──
        num_buckets: Optional[int] = 32,
        max_feat_tokens: Optional[int] = 30,
        feat_pos_mode: str = 'learnable',
        # ── 序列相关 ──
        seq_max_len: int = 256,
        num_time_buckets: int = 65,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.action_num = action_num
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_max_len = seq_max_len
        self.num_time_buckets = num_time_buckets

        # ── 虚拟 id 空间规划 ──
        # 布局：[0=padding] [1..real_item_vocab_size = 真实行为 item ids]
        #       [real_item_vocab_size+1 .. 用户特征虚拟 ids]
        #       [紧接其后 .. 广告特征虚拟 ids]
        # 0 固定为 padding_idx，不参与训练。
        user_virtual_id_start = real_item_vocab_size + 1

        # ── 用户侧特征 Tokenizer ──
        self.user_feat_tokenizer = FeatureAsItemTokenizer(
            feature_specs=user_feat_specs,
            vocab_sizes=user_vocab_sizes,
            virtual_id_start=user_virtual_id_start,
            num_buckets=num_buckets,
            max_feat_tokens=max_feat_tokens,
        )

        # 广告侧特征紧接用户虚拟 id 之后
        item_virtual_id_start = user_virtual_id_start + self.user_feat_tokenizer.virtual_id_consumed

        # ── 广告侧特征 Tokenizer ──
        self.item_feat_tokenizer = FeatureAsItemTokenizer(
            feature_specs=item_feat_specs,
            vocab_sizes=item_vocab_sizes,
            virtual_id_start=item_virtual_id_start,
            num_buckets=num_buckets,
            max_feat_tokens=max_feat_tokens,
        )

        self.num_feat_tokens = (
            self.user_feat_tokenizer.num_feat_tokens
            + self.item_feat_tokenizer.num_feat_tokens
        )

        # 统一词表大小：padding(1) + 真实 item ids + 用户虚拟 ids + 广告虚拟 ids
        unified_vocab_size = (
            item_virtual_id_start
            + self.item_feat_tokenizer.virtual_id_consumed
        )

        # ── 统一 Embedding 表（真实 item + 虚拟特征 id 共享）──
        self.unified_emb = nn.Embedding(unified_vocab_size, emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.unified_emb.weight.data)
        self.unified_emb.weight.data[0, :] = 0.0  # padding 向量固定为 0

        # ── 特征 token 位置编码（挑战3接口）──
        self.feat_pos_enc = FeaturePositionEncoding(
            num_feat_tokens=self.num_feat_tokens,
            d_model=d_model,
            feat_pos_mode=feat_pos_mode,
        )

        # ── 时间分桶 Embedding（行为序列的时间信号）──
        if num_time_buckets > 0:
            self.time_emb: Optional[nn.Embedding] = nn.Embedding(
                num_time_buckets, d_model, padding_idx=0
            )
            nn.init.xavier_normal_(self.time_emb.weight.data)
            self.time_emb.weight.data[0, :] = 0.0
        else:
            self.time_emb = None

        # ── Embedding 投影：emb_dim → d_model ──
        self.emb_proj = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── RoPE（为 TransformerEncoder 提供位置编码）──
        # head_dim = d_model // num_heads
        head_dim = d_model // num_heads
        total_seq_len = self.num_feat_tokens + seq_max_len * max(1, len(self.seq_domains))
        self.rope = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=total_seq_len + 64,  # 留 64 的余量
        )

        # ── 统一 Transformer 主干 ──
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

        # ── 分类头 ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logging.info(
            f"UnifiedSeqModel: unified_vocab={unified_vocab_size}, "
            f"num_feat_tokens={self.num_feat_tokens}, "
            f"num_buckets={num_buckets}, max_feat_tokens={max_feat_tokens}, "
            f"feat_pos_mode={feat_pos_mode}, total_params={total_params:,}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_feat_tokens(
        self,
        user_int_feats: torch.Tensor,
        item_int_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把用户/广告特征转换为虚拟 token Embedding 序列。

        Returns:
            feat_emb:  (B, num_feat_tokens, d_model)
            feat_mask: (B, num_feat_tokens)，True 表示 padding（无效 token）
        """
        user_vids, user_valid = self.user_feat_tokenizer(user_int_feats)
        item_vids, item_valid = self.item_feat_tokenizer(item_int_feats)

        # 拼接用户特征 token + 广告特征 token
        all_vids = torch.cat([user_vids, item_vids], dim=1)    # (B, num_feat_tokens)
        all_valid = torch.cat([user_valid, item_valid], dim=1)  # (B, num_feat_tokens)

        # 查统一 Embedding 表并投影
        feat_emb = self.unified_emb(all_vids)   # (B, K, emb_dim)
        feat_emb = self.emb_proj(feat_emb)       # (B, K, d_model)

        # 加位置编码（挑战3接口）
        feat_emb = self.feat_pos_enc(feat_emb)  # (B, K, d_model)

        # padding mask：无效位置置 True
        feat_mask = ~all_valid  # (B, K)
        return feat_emb, feat_mask

    def _embed_seq_domain(
        self,
        seq_ids: torch.Tensor,
        seq_time_buckets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把单个行为域的序列转换为 Embedding 序列。

        Args:
            seq_ids:          (B, S, L)，S=侧信息维度数，L=序列长度
            seq_time_buckets: (B, L)，时间分桶 id

        Returns:
            seq_emb:  (B, L, d_model)
            seq_mask: (B, L)，True 表示 padding
        """
        # 取第一个侧信息维度（主 item_id）作为 Embedding 查询键
        main_ids = seq_ids[:, 0, :]  # (B, L)

        emb = self.unified_emb(main_ids)   # (B, L, emb_dim)
        emb = self.emb_proj(emb)           # (B, L, d_model)

        # 加时间分桶 Embedding
        if self.time_emb is not None:
            emb = emb + self.time_emb(seq_time_buckets)  # (B, L, d_model)

        # padding mask：main_id == 0 的位置为 padding
        seq_mask = (main_ids == 0)  # (B, L)
        return emb, seq_mask

    def _build_unified_sequence(
        self,
        inputs: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建统一 token 序列：[特征虚拟 token | 行为序列 token]。

        Returns:
            unified_tokens: (B, K + sum(seq_lens), d_model)
            unified_mask:   (B, K + sum(seq_lens))，True = padding
        """
        # 特征虚拟 token
        feat_emb, feat_mask = self._embed_feat_tokens(
            inputs.user_int_feats, inputs.item_int_feats
        )

        # 各行为域序列 token
        seq_emb_list: List[torch.Tensor] = []
        seq_mask_list: List[torch.Tensor] = []
        for domain in self.seq_domains:
            seq_emb, seq_mask = self._embed_seq_domain(
                inputs.seq_data[domain],
                inputs.seq_time_buckets[domain],
            )
            seq_emb_list.append(seq_emb)
            seq_mask_list.append(seq_mask)

        if seq_emb_list:
            seq_emb_cat = torch.cat(seq_emb_list, dim=1)    # (B, sum_L, d_model)
            seq_mask_cat = torch.cat(seq_mask_list, dim=1)  # (B, sum_L)
            unified_tokens = torch.cat([feat_emb, seq_emb_cat], dim=1)
            unified_mask = torch.cat([feat_mask, seq_mask_cat], dim=1)
        else:
            # 无行为序列域时退化为仅特征 token
            unified_tokens = feat_emb
            unified_mask = feat_mask

        return unified_tokens, unified_mask

    # ──────────────────────────────────────────────────────────────────────────
    # 前向传播
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """统一序列建模的前向传播。

        Args:
            inputs: ModelInput，包含用户/广告特征和行为序列。

        Returns:
            logits: (B, action_num)
        """
        tokens, padding_mask = self._build_unified_sequence(inputs)
        # tokens:       (B, total_len, d_model)
        # padding_mask: (B, total_len)，True = 无效位置

        # 为 TransformerEncoder 的 RoPE 预计算 cos/sin
        seq_len = tokens.shape[1]
        rope_cos, rope_sin = self.rope(seq_len, device=tokens.device)

        for layer in self.transformer_layers:
            tokens, _ = layer(
                tokens,
                key_padding_mask=padding_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        # 取第一个 token 作为全局表示（特征 token 位于序列最前）
        # 当 num_feat_tokens > 0 时，tokens[:, 0, :] 是用户第一个特征 token，
        # 它参与了全局注意力，已聚合了所有 token 的信息。
        cls_repr = self.output_norm(tokens[:, 0, :])  # (B, d_model)

        logits = self.classifier(cls_repr)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理接口，同时返回 logits 和全局表示向量。

        Returns:
            logits:   (B, action_num)
            repr_vec: (B, d_model)，全局表示向量（可用于召回/蒸馏）
        """
        tokens, padding_mask = self._build_unified_sequence(inputs)
        seq_len = tokens.shape[1]
        rope_cos, rope_sin = self.rope(seq_len, device=tokens.device)

        for layer in self.transformer_layers:
            tokens, _ = layer(
                tokens,
                key_padding_mask=padding_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        cls_repr = self.output_norm(tokens[:, 0, :])
        logits = self.classifier(cls_repr)
        return logits, cls_repr
