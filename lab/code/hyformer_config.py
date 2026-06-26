import torch

class Config:
    train_path   = "../data/A2-Rec/train.csv"
    test_path    = "../data/A2-Rec/test.csv"
    user_path    = "../data/A2-Rec/user.csv"
    item_path    = "../data/A2-Rec/item.csv"
    output_path  = "submission_hyformer.csv"

    max_seq_len  = 50       # 行为序列最大长度
    emb_dim      = 32       # 每个特征域 embedding 维度
    repr_dim     = 128      # 主干隐层维度 D（GT / SeqToken / user_repr）

    # HyFormer 核心
    n_layers     = 2        # HyFormer 层数（QD+QB 交替次数）
    n_heads      = 4        # attention 头数
    n_gt         = 8        # Global Token 数量（= 用户特征域数）
    qb_n_heads   = 4        # Query Boosting self-attn 头数

    # Item Tower
    item_mlp_dims = [256, 128]
    dropout       = 0.2

    # 训练
    epochs        = 50
    batch_size    = 256
    lr            = 1e-3
    weight_decay  = 1e-5
    label_smooth  = 0.0
    aux_weight    = 0.05    # Masked Item Prediction 辅助损失权重
    mask_prob     = 0.15    # 序列 mask 概率
    mixup_alpha   = 0.1

    seed          = 42
    topk          = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"