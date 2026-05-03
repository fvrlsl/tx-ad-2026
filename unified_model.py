"""UnifiedSeqModel: 将特征抽象为虚拟商品访问事件，与真实行为序列统一建模。

核心思想：
    用户的静态特征（年龄、性别、职业等）被视为"用户在 t=0 时刻访问过的虚拟商品"。
    每个特征字段的每个取值 → 唯一虚拟 item_id，拼接到真实行为序列最前面，
    送入单一 Transformer 统一建模。

Embedding 架构（应对行为序列 vocab 达亿级的问题）：
    特征虚拟 id：独立小 Embedding 表，大小 = num_feat_tokens × (num_buckets+1)，完全可控。
    行为序列各域：在该域所有 sideinfo fid 中，选取首个 vocab_size ≤ max_seq_vocab 的
                 fid 作为主 item_id 建 Embedding；若没有满足条件的 fid，
                 该域不建 Embedding 表，仅用时间编码填充（零向量占位）。
    两侧均通过 emb_proj 投影到 d_model，拼接后送入共享 Transformer。

三个可调节接口（对应三大挑战）：
    挑战1 - 特征值连续性：num_buckets
    挑战2 - 序列长度爆炸：max_feat_tokens
    挑战3 - 特征 token 时间位置：feat_pos_mode

feature_specs 格式（来自 dataset.FeatureSchema.entries）：
    List of (fid: int, col_offset: int, length: int)
    col_offset 是该字段在 int_feats tensor 中的列起始索引。
    length == 1 表示标量特征，length > 1 为多值特征（暂不虚拟化）。
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model import ModelInput, TransformerEncoder, RotaryEmbedding


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FeatureAsItemTokenizer：把特征映射为虚拟商品访问事件
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureAsItemTokenizer(nn.Module):
    """把用户/广告的整型特征字段映射为紧凑虚拟 id 序列。

    虚拟 id 空间：字段 i 的 bucket_v → id_bases[i] + bucket_v
    总词表大小 = sum(effective_slots_per_field + 1)，与原始词表无关。

    Args:
        feature_specs:   [(fid, col_offset, length), ...]，只处理 length==1 的字段。
        vocab_sizes:     与 feature_specs 等长，各字段的原始词表大小
                         （num_buckets=None 时用于 clamp 上界）。
        num_buckets:     【挑战1接口】分桶数。None=不分桶直接用原始值；int=均匀分桶到 [1, num_buckets]。
        max_feat_tokens: 【挑战2接口】最大 token 数。None=不限制；int=按字段顺序截断。
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        vocab_sizes: List[int],
        num_buckets: Optional[int],
        max_feat_tokens: Optional[int],
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets

        col_offsets: List[int] = []
        id_bases: List[int] = []
        orig_vocab_sizes: List[int] = []

        cursor = 1  # 0 固定为 padding_idx，从 1 开始分配
        for (fid, col_offset, length), vocab_size in zip(feature_specs, vocab_sizes):
            if length != 1:
                continue
            effective_slots = num_buckets if num_buckets is not None else vocab_size
            col_offsets.append(col_offset)
            id_bases.append(cursor)
            orig_vocab_sizes.append(vocab_size)
            cursor += effective_slots + 1

        if max_feat_tokens is not None:
            col_offsets = col_offsets[:max_feat_tokens]
            id_bases = id_bases[:max_feat_tokens]
            orig_vocab_sizes = orig_vocab_sizes[:max_feat_tokens]

        self.num_feat_tokens: int = len(col_offsets)
        self.feat_vocab_size: int = cursor  # Embedding 表大小

        self.register_buffer('col_offsets', torch.tensor(col_offsets, dtype=torch.long), persistent=True)
        self.register_buffer('id_bases', torch.tensor(id_bases, dtype=torch.long), persistent=True)
        self.register_buffer('orig_vocab_sizes', torch.tensor(orig_vocab_sizes, dtype=torch.long), persistent=True)

        logging.info(
            f"FeatureAsItemTokenizer: {self.num_feat_tokens} scalar feat tokens, "
            f"feat_vocab_size={self.feat_vocab_size}, "
            f"num_buckets={num_buckets}, max_feat_tokens={max_feat_tokens}"
        )

    def forward(self, int_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """整型特征 tensor → 虚拟 id 序列（向量化）。

        Args:
            int_feats: (B, total_feat_dim)

        Returns:
            virtual_ids: (B, num_feat_tokens)，0=padding
            valid_mask:  (B, num_feat_tokens)，True=有效
        """
        if self.num_feat_tokens == 0:
            B, device = int_feats.shape[0], int_feats.device
            return (torch.zeros(B, 0, dtype=torch.long, device=device),
                    torch.zeros(B, 0, dtype=torch.bool, device=device))

        raw = int_feats[:, self.col_offsets]  # (B, K)
        valid = raw > 0

        if self.num_buckets is not None:
            bucket = (raw % self.num_buckets + 1).clamp(1, self.num_buckets)
        else:
            upper = (self.orig_vocab_sizes - 1).unsqueeze(0)
            bucket = raw.clamp(min=1).clamp_max(upper)

        vids = self.id_bases.unsqueeze(0) + bucket   # (B, K)
        vids = vids * valid.long()                    # 缺失位置置 0
        return vids, valid


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 特征 Token 位置编码（挑战3接口）
# ═══════════════════════════════════════════════════════════════════════════════

class FeaturePositionEncoding(nn.Module):
    """【挑战3接口】特征 token 的位置编码策略。

    feat_pos_mode:
        'zero'      : 不加位置编码（静态，无时序意义）
        'learnable' : 每个字段有独立可学习位置 Embedding（推荐）
        'prepend'   : sinusoidal 固定位置编码
    """

    def __init__(self, num_feat_tokens: int, d_model: int, feat_pos_mode: str = 'learnable') -> None:
        super().__init__()
        assert feat_pos_mode in ('zero', 'learnable', 'prepend'), \
            f"feat_pos_mode 须为 'zero'/'learnable'/'prepend'，当前: {feat_pos_mode}"
        self.feat_pos_mode = feat_pos_mode
        self.num_feat_tokens = num_feat_tokens
        if num_feat_tokens == 0:
            return
        if feat_pos_mode == 'learnable':
            self.pos_emb = nn.Embedding(num_feat_tokens, d_model)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        elif feat_pos_mode == 'prepend':
            self.register_buffer('pos_enc', self._sinusoidal(num_feat_tokens, d_model), persistent=False)

    @staticmethod
    def _sinusoidal(max_len: int, d: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        enc = torch.zeros(1, max_len, d)
        enc[0, :, 0::2] = torch.sin(pos * div)
        enc[0, :, 1::2] = torch.cos(pos * div[:d // 2])
        return enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_feat_tokens == 0 or self.feat_pos_mode == 'zero':
            return x
        if self.feat_pos_mode == 'learnable':
            idx = torch.arange(self.num_feat_tokens, device=x.device)
            return x + self.pos_emb(idx).unsqueeze(0)
        return x + self.pos_enc[:, :self.num_feat_tokens].to(x.device)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UnifiedSeqModel：统一序列 Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedSeqModel(nn.Module):
    """统一序列推荐模型：特征虚拟 token + 行为序列 token → 单一 Transformer。

    Args:
        user_feat_specs:   用户整型特征 schema，[(fid, col_offset, length), ...]
        user_vocab_sizes:  与 user_feat_specs 等长，各字段的原始词表大小。
        item_feat_specs:   广告整型特征 schema，格式同上。
        item_vocab_sizes:  与 item_feat_specs 等长。
        seq_vocab_sizes:   各行为域 sideinfo fid 的词表大小，
                           格式 {domain: [vs_fid0, vs_fid1, ...]}，
                           与 dataset.seq_domain_vocab_sizes 对齐。
        d_model:           Transformer 隐层维度。
        emb_dim:           Embedding 维度（投影前）。
        num_heads:         注意力头数。
        num_layers:        Transformer 层数。
        hidden_mult:       FFN 放大倍数。
        dropout_rate:      Dropout 比率。
        action_num:        分类头输出维度（1=二分类）。
        num_buckets:       【挑战1】特征值分桶数（None=不分桶）。
        max_feat_tokens:   【挑战2】特征虚拟 token 最大数（None=不限制）。
        feat_pos_mode:     【挑战3】特征 token 位置编码模式。
        seq_max_len:       行为序列最大长度（用于 RoPE cache 预热）。
        num_time_buckets:  时间分桶 Embedding 桶数（0=不使用）。
        max_seq_vocab:     行为序列主 item_id 的词表大小上限。对每个行为域，
                           在其 sideinfo fid 中选取首个 vocab_size ≤ max_seq_vocab
                           的 fid 建 Embedding；若无满足条件的 fid，该域用零向量
                           占位（仅保留时间编码）。默认 2_000_000。
    """

    def __init__(
        self,
        user_feat_specs: List[Tuple[int, int, int]],
        user_vocab_sizes: List[int],
        item_feat_specs: List[Tuple[int, int, int]],
        item_vocab_sizes: List[int],
        seq_vocab_sizes: Dict[str, List[int]],
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
        max_seq_vocab: int = 2_000_000,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.action_num = action_num
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_max_len = seq_max_len

        # ── 特征 Tokenizer（用户侧 + 广告侧）──
        self.user_feat_tokenizer = FeatureAsItemTokenizer(user_feat_specs, user_vocab_sizes, num_buckets, max_feat_tokens)
        self.item_feat_tokenizer = FeatureAsItemTokenizer(item_feat_specs, item_vocab_sizes, num_buckets, max_feat_tokens)
        self.num_feat_tokens = self.user_feat_tokenizer.num_feat_tokens + self.item_feat_tokenizer.num_feat_tokens

        # ── 特征虚拟 id Embedding 表（user/item 各自独立的小表）──
        def _make_feat_emb(tokenizer: FeatureAsItemTokenizer) -> Optional[nn.Embedding]:
            if tokenizer.num_feat_tokens == 0:
                return None
            emb = nn.Embedding(tokenizer.feat_vocab_size, emb_dim, padding_idx=0)
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0] = 0.0
            return emb

        self.user_feat_emb: Optional[nn.Embedding] = _make_feat_emb(self.user_feat_tokenizer)
        self.item_feat_emb: Optional[nn.Embedding] = _make_feat_emb(self.item_feat_tokenizer)

        # ── 行为序列各域 Embedding 表 ──
        # 对每个域的 sideinfo fid vocab 列表，选首个 vocab ≤ max_seq_vocab 的 fid 建表；
        # 没有合适 fid 时，该域 seq_slot_idx = -1（用零向量占位）。
        self.seq_embs = nn.ModuleDict()
        # seq_slot_idx[domain]: 选中的 sideinfo 槽位序号（即 seq_ids[:, slot, :]）
        self.seq_slot_idx: Dict[str, int] = {}

        for domain in self.seq_domains:
            vs_list = seq_vocab_sizes[domain]
            selected_slot = -1
            selected_vs = 0
            for slot, vs in enumerate(vs_list):
                if vs <= max_seq_vocab:
                    selected_slot = slot
                    selected_vs = vs
                    break

            self.seq_slot_idx[domain] = selected_slot
            if selected_slot >= 0:
                emb = nn.Embedding(selected_vs + 1, emb_dim, padding_idx=0)
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0] = 0.0
                self.seq_embs[domain] = emb
                logging.info(f"  seq_emb[{domain}]: slot={selected_slot}, vocab_size={selected_vs + 1}")
            else:
                logging.info(f"  seq_emb[{domain}]: 无满足 max_seq_vocab={max_seq_vocab} 的 fid，使用零向量占位")

        # ── 特征 token 位置编码（挑战3接口）──
        self.feat_pos_enc = FeaturePositionEncoding(self.num_feat_tokens, d_model, feat_pos_mode)

        # ── 时间分桶 Embedding ──
        if num_time_buckets > 0:
            self.time_emb: Optional[nn.Embedding] = nn.Embedding(num_time_buckets, d_model, padding_idx=0)
            nn.init.xavier_normal_(self.time_emb.weight.data)
            self.time_emb.weight.data[0] = 0.0
        else:
            self.time_emb = None

        # ── Embedding 投影：emb_dim → d_model（特征侧和序列侧共享）──
        self.emb_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))

        # ── RoPE ──
        head_dim = d_model // num_heads
        rope_max_len = self.num_feat_tokens + seq_max_len * max(1, len(self.seq_domains)) + 64
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=rope_max_len)

        # ── Transformer 主干 ──
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(d_model=d_model, num_heads=num_heads, hidden_mult=hidden_mult, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

        # ── 分类头 ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.SiLU(),
            nn.Dropout(dropout_rate), nn.Linear(d_model, action_num),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logging.info(
            f"UnifiedSeqModel: num_feat_tokens={self.num_feat_tokens}, "
            f"num_buckets={num_buckets}, max_feat_tokens={max_feat_tokens}, "
            f"feat_pos_mode={feat_pos_mode}, max_seq_vocab={max_seq_vocab}, "
            f"total_params={total_params:,}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _embed_feat_tokens(
        self,
        user_int_feats: torch.Tensor,
        item_int_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """用户/广告特征 → 虚拟 token Embedding 序列。

        Returns:
            feat_emb:  (B, num_feat_tokens, d_model)
            feat_mask: (B, num_feat_tokens)，True = padding
        """
        emb_list: List[torch.Tensor] = []
        valid_list: List[torch.Tensor] = []

        if self.user_feat_emb is not None:
            u_vids, u_valid = self.user_feat_tokenizer(user_int_feats)
            emb_list.append(self.emb_proj(self.user_feat_emb(u_vids)))
            valid_list.append(u_valid)

        if self.item_feat_emb is not None:
            i_vids, i_valid = self.item_feat_tokenizer(item_int_feats)
            emb_list.append(self.emb_proj(self.item_feat_emb(i_vids)))
            valid_list.append(i_valid)

        if not emb_list:
            B, device = user_int_feats.shape[0], user_int_feats.device
            return (torch.zeros(B, 0, self.d_model, device=device),
                    torch.zeros(B, 0, dtype=torch.bool, device=device))

        feat_emb = self.feat_pos_enc(torch.cat(emb_list, dim=1))
        feat_mask = ~torch.cat(valid_list, dim=1)
        return feat_emb, feat_mask

    def _embed_seq_domain(
        self,
        domain: str,
        seq_ids: torch.Tensor,
        seq_time_buckets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单个行为域 → Embedding 序列。

        Args:
            seq_ids:          (B, S, L)
            seq_time_buckets: (B, L)

        Returns:
            seq_emb:  (B, L, d_model)
            seq_mask: (B, L)，True = padding
        """
        B, S, L = seq_ids.shape
        device = seq_ids.device
        slot = self.seq_slot_idx[domain]

        if slot >= 0 and domain in self.seq_embs:
            emb_table = self.seq_embs[domain]
            main_ids = seq_ids[:, slot, :]  # (B, L)
            main_ids_clamped = main_ids.clamp(0, emb_table.num_embeddings - 1)
            emb = self.emb_proj(emb_table(main_ids_clamped))  # (B, L, d_model)
            seq_mask = (main_ids == 0)
        else:
            # 该域没有满足条件的 fid：用零向量占位，mask 由 seq_ids[:, 0, :] 决定
            emb = torch.zeros(B, L, self.d_model, device=device)
            seq_mask = (seq_ids[:, 0, :] == 0)

        if self.time_emb is not None:
            emb = emb + self.time_emb(seq_time_buckets)

        return emb, seq_mask

    def _build_unified_sequence(
        self, inputs: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建统一 token 序列：[特征虚拟 token | 行为序列 token]。"""
        feat_emb, feat_mask = self._embed_feat_tokens(
            inputs.user_int_feats, inputs.item_int_feats
        )

        seq_emb_list: List[torch.Tensor] = []
        seq_mask_list: List[torch.Tensor] = []
        for domain in self.seq_domains:
            emb, mask = self._embed_seq_domain(domain, inputs.seq_data[domain], inputs.seq_time_buckets[domain])
            seq_emb_list.append(emb)
            seq_mask_list.append(mask)

        if seq_emb_list:
            unified_tokens = torch.cat([feat_emb, *seq_emb_list], dim=1)
            unified_mask = torch.cat([feat_mask, *seq_mask_list], dim=1)
        else:
            unified_tokens, unified_mask = feat_emb, feat_mask

        return unified_tokens, unified_mask

    def _run_transformer(
        self, tokens: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """通过 Transformer 主干，返回输出 token 序列。"""
        rope_cos, rope_sin = self.rope(tokens.shape[1], device=tokens.device)
        for layer in self.transformer_layers:
            tokens, _ = layer(tokens, key_padding_mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
        return tokens

    # ──────────────────────────────────────────────────────────────────────────
    # 前向传播
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """前向传播，返回 logits (B, action_num)。"""
        tokens, mask = self._build_unified_sequence(inputs)
        tokens = self._run_transformer(tokens, mask)
        cls_repr = self.output_norm(tokens[:, 0, :])
        return self.classifier(cls_repr)

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理接口，返回 (logits, repr_vec)。"""
        tokens, mask = self._build_unified_sequence(inputs)
        tokens = self._run_transformer(tokens, mask)
        cls_repr = self.output_norm(tokens[:, 0, :])
        return self.classifier(cls_repr), cls_repr
