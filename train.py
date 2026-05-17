"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from unified_model import UnifiedSeqModel
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:128,seq_b:600,seq_c:512,seq_d:1200',
                        help='Per-domain sequence max length (safety OOM cap only). '
                             'When seq_encoder_type=longer, the real filtering is done '
                             'by target-aware TopK inside LongerEncoder, so this value '
                             'should be >= p90 of each domain to avoid pre-truncation '
                             'information loss. Data stats: '
                             'seq_a mean=38 p90=95, seq_b mean=569 p90=1393, '
                             'seq_c mean=449 p90=887, seq_d mean=1100 p90=2215. '
                             'Format: domain:length comma-separated, e.g. seq_a:128,seq_b:600')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='longer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention (O(T²), only suitable for short seq), '
                             'longer = Top-K most-recent tokens + self-attention (O(K²), default). '
                             'For seq_b/c/d whose p90 lengths are 1920/1109/3793, '
                             'transformer would be 200~800x slower than longer with K=64.')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=64,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer). '
                             'K=64 is the recommended default for long-seq domains '
                             '(seq_b p90=1920, seq_c p90=1109, seq_d p90=3793): '
                             'it captures the most recent 64 interactions while keeping '
                             'attention complexity O(64²) instead of O(T²). '
                             'Increase to K=128 if AUC drops >0.002 vs full sequence.')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--use_target_aware_topk', action='store_true', default=False,
                        help='Enable target-aware TopK in LongerEncoder: uses mean-pooled '
                             'item NS embedding as query to score and select the top_k most '
                             'relevant sequence tokens (SIM-style), instead of recency-based '
                             'truncation. Requires seq_encoder_type=longer. '
                             'When enabled, seq_max_lens should be set to >= p90 of each domain '
                             'so the full behaviour history is available for retrieval.')
    parser.add_argument('--use_query_projection', action='store_true', default=False,
                        help='Apply LayerNorm + Linear + SiLU projection to the target_query '
                             'vector before TopK scoring. Helps the model learn a better query '
                             'representation when use_target_aware_topk is enabled. '
                             'Only takes effect when --use_target_aware_topk is also set.')
    parser.add_argument('--pre_topk', type=int, default=256,
                        help='Two-stage TopK: recency pre-filter size before target-aware '
                             'fine-ranking. Set to 0 to disable pre-filtering and score '
                             'the full sequence (default: 256).')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Model variant selection.
    parser.add_argument('--model_type', type=str, default='hyformer',
                        choices=['hyformer', 'unified'],
                        help='Model variant: '
                             'hyformer = PCVRHyFormer (default), '
                             'unified  = UnifiedSeqModel (特征虚拟化统一序列建模)')

    # UnifiedSeqModel: 三大挑战的可调节接口（仅 --model_type=unified 时生效）.
    parser.add_argument('--num_buckets', type=int, default=32,
                        help='【挑战1】特征值分桶数，None 表示不分桶直接使用原始值 '
                             '(unified only, default=32)')
    parser.add_argument('--no_buckets', dest='num_buckets', action='store_const', const=None,
                        help='【挑战1】禁用分桶，等效于 --num_buckets=None')
    parser.add_argument('--max_feat_tokens', type=int, default=30,
                        help='【挑战2】特征虚拟 token 最大数量，超出部分按字段顺序截断 '
                             '(unified only, default=30)')
    parser.add_argument('--feat_pos_mode', type=str, default='learnable',
                        choices=['zero', 'learnable', 'prepend'],
                        help='【挑战3】特征 token 位置编码策略: '
                             'zero=不加位置编码, learnable=独立可学习(default), '
                             'prepend=sinusoidal固定编码 '
                             '(unified only)')
    parser.add_argument('--max_seq_vocab', type=int, default=2_000_000,
                        help='行为序列主 item_id 词表大小上限。对每个行为域，选取首个 '
                             'vocab_size <= 该值的 sideinfo fid 建 Embedding；若无满足的 '
                             'fid，该域用零向量占位（仅保留时间编码）。'
                             '(unified only, default=2_000_000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Mixed precision training.
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable AMP (Automatic Mixed Precision) training with fp16. '
                             'Uses torch.cuda.amp.autocast + GradScaler for ~1.5-2x speedup '
                             'on modern NVIDIA GPUs. Only effective on CUDA devices.')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    # ── Pre-training data analysis switch ─────────────────────────────────
    # 默认开启；使用 --no_data_analysis 可关闭
    parser.add_argument('--run_data_analysis', action='store_true', default=True,
                        help='Run full data analysis before training (default: on). '
                             'Generates report + optional offline export. '
                             'Use --no_data_analysis to disable.')
    parser.add_argument('--no_data_analysis', dest='run_data_analysis', action='store_false',
                        help='Disable pre-training data analysis.')
    parser.add_argument('--analysis_output_dir', type=str, default='data_analysis',
                        help='Output directory for analysis reports and offline exports '
                             '(default: <log_dir>/data_analysis/)')
    parser.add_argument('--analysis_export_rows', type=int, default=100_000,
                        help='Export the first N rows as an offline training parquet. '
                             '0 = skip export (default: 100000)')
    parser.add_argument('--analysis_max_rows', type=int, default=None,
                        help='Limit the number of rows analysed. '
                             'None = analyse all rows (default: None)')
    # ─────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH') or os.path.join(args.log_dir or '.', 'tf_events')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    # ---- Pre-training data analysis ----
    if args.run_data_analysis:
        import glob as _glob
        from data_analysis import run_analysis as _run_analysis

        parquet_files = sorted(_glob.glob(os.path.join(args.data_dir, '*.parquet')))
        if not parquet_files:
            logging.warning('[data_analysis] No parquet files found under data_dir, skipping.')
        else:
            analysis_data_path = parquet_files[0]
            analysis_output_dir = os.path.join(args.log_dir, args.analysis_output_dir)
            export_rows = args.analysis_export_rows if args.analysis_export_rows > 0 else None

            logging.info('=' * 60)
            logging.info('[data_analysis] Pre-training data analysis enabled')
            logging.info(f'[data_analysis]   data_path      : {analysis_data_path}')
            logging.info(f'[data_analysis]   schema_path    : {schema_path}')
            logging.info(f'[data_analysis]   output_dir     : {analysis_output_dir}')
            logging.info(f'[data_analysis]   export_rows    : {export_rows}')
            logging.info(f'[data_analysis]   max_rows       : {args.analysis_max_rows}')
            logging.info('=' * 60)

            _run_analysis(
                data_path=analysis_data_path,
                schema_path=schema_path,
                output_dir=analysis_output_dir,
                max_rows=args.analysis_max_rows,
                export_rows=export_rows,
            )
            logging.info(f'[data_analysis] Complete. Reports saved to: {analysis_output_dir}')

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    if args.model_type == 'unified':
        # ── UnifiedSeqModel：特征虚拟化统一序列建模 ──
        # feature_specs 直接使用 schema.entries 格式 [(fid, col_offset, length)]
        # vocab_sizes 使用 dataset 提供的每个 fid 的原始词表大小列表
        user_feat_specs = pcvr_dataset.user_int_schema.entries   # [(fid, col_offset, length)]
        item_feat_specs = pcvr_dataset.item_int_schema.entries

        # user_int_vocab_sizes 是按 tensor 列位置展开的列表，但 UnifiedSeqModel 只需
        # 每个字段（fid）对应一个 vocab_size；取每个字段 offset 处的第一个值即可。
        user_vocab_sizes = [
            pcvr_dataset.user_int_vocab_sizes[offset]
            for _, offset, _ in user_feat_specs
        ]
        item_vocab_sizes = [
            pcvr_dataset.item_int_vocab_sizes[offset]
            for _, offset, _ in item_feat_specs
        ]

        # seq_domain_vocab_sizes: {domain: [vs_per_sideinfo_fid, ...]}
        # 第一个元素是主 item_id 的 vocab_size（供 UnifiedSeqModel 内部各域独立 Embedding 使用）
        seq_domain_vocab_sizes = pcvr_dataset.seq_domain_vocab_sizes

        # seq_max_len：取各域最大截断长度，用于 RoPE cache 预热
        configured_seq_max_len = max(seq_max_lens.values()) if seq_max_lens else 256

        model = UnifiedSeqModel(
            user_feat_specs=user_feat_specs,
            user_vocab_sizes=user_vocab_sizes,
            item_feat_specs=item_feat_specs,
            item_vocab_sizes=item_vocab_sizes,
            seq_vocab_sizes=seq_domain_vocab_sizes,
            d_model=args.d_model,
            emb_dim=args.emb_dim,
            num_heads=args.num_heads,
            num_layers=args.num_hyformer_blocks,
            hidden_mult=args.hidden_mult,
            dropout_rate=args.dropout_rate,
            action_num=args.action_num,
            num_buckets=args.num_buckets,
            max_feat_tokens=args.max_feat_tokens,
            feat_pos_mode=args.feat_pos_mode,
            seq_max_len=configured_seq_max_len,
            num_time_buckets=NUM_TIME_BUCKETS if args.use_time_buckets else 0,
            max_seq_vocab=args.max_seq_vocab,
        ).to(args.device)

        total_params = sum(p.numel() for p in model.parameters())
        logging.info(
            f"UnifiedSeqModel created: "
            f"num_feat_tokens={model.num_feat_tokens}, "
            f"num_buckets={args.num_buckets}, "
            f"max_feat_tokens={args.max_feat_tokens}, "
            f"feat_pos_mode={args.feat_pos_mode}, "
            f"max_seq_vocab={args.max_seq_vocab}, "
            f"d_model={args.d_model}, "
            f"total_params={total_params:,}"
        )

        ckpt_params = {
            "layer": args.num_hyformer_blocks,
            "head": args.num_heads,
            "hidden": args.d_model,
        }

    else:
        # ── PCVRHyFormer（默认）──
        user_int_feature_specs = build_feature_specs(
            pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
        item_int_feature_specs = build_feature_specs(
            pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

        model_args = {
            "user_int_feature_specs": user_int_feature_specs,
            "item_int_feature_specs": item_int_feature_specs,
            "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
            "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
            "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
            "user_ns_groups": user_ns_groups,
            "item_ns_groups": item_ns_groups,
            "d_model": args.d_model,
            "emb_dim": args.emb_dim,
            "num_queries": args.num_queries,
            "num_hyformer_blocks": args.num_hyformer_blocks,
            "num_heads": args.num_heads,
            "seq_encoder_type": args.seq_encoder_type,
            "hidden_mult": args.hidden_mult,
            "dropout_rate": args.dropout_rate,
            "seq_top_k": args.seq_top_k,
            "seq_causal": args.seq_causal,
            "action_num": args.action_num,
            "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
            "rank_mixer_mode": args.rank_mixer_mode,
            "use_rope": args.use_rope,
            "rope_base": args.rope_base,
            "emb_skip_threshold": args.emb_skip_threshold,
            "seq_id_threshold": args.seq_id_threshold,
            "ns_tokenizer_type": args.ns_tokenizer_type,
            "user_ns_tokens": args.user_ns_tokens,
            "item_ns_tokens": args.item_ns_tokens,
            "use_target_aware_topk": args.use_target_aware_topk,
            "use_query_projection": args.use_query_projection,
            "pre_topk": args.pre_topk,
        }

        model = PCVRHyFormer(**model_args).to(args.device)

        num_sequences = len(pcvr_dataset.seq_domains)
        num_ns = model.num_ns
        T = args.num_queries * num_sequences + num_ns
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(
            f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, "
            f"d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}"
        )
        logging.info(f"User NS groups: {user_ns_groups}")
        logging.info(f"Item NS groups: {item_ns_groups}")
        logging.info(f"Total parameters: {total_params:,}")

        ckpt_params = {
            "layer": args.num_hyformer_blocks,
            "head": args.num_heads,
            "hidden": args.d_model,
        }

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        use_amp=args.use_amp,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
