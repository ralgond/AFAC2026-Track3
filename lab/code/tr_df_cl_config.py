import torch

class Config:
    train_path  = "../data/A2-Rec/train.csv"
    test_path   = "../data/A2-Rec/test.csv"
    user_path   = "../data/A2-Rec/user.csv"
    item_path   = "../data/A2-Rec/item.csv"
    output_path = "submission_contrastformer.csv"

    max_seq_len = 50
    emb_dim     = 32        # 每个特征域 embedding 维度 E
    repr_dim    = 128       # 主干隐层维度 D

    # HyFormer backbone
    n_layers    = 2
    n_heads     = 4
    n_gt        = 8         # Global Token 数量 = 用户特征域数
    qb_n_heads  = 4

    # Item/User DeepFM
    deep_dims   = [256, 128]   # Deep 分支 hidden dims
    dropout     = 0.2

    # 对比学习
    cl_weight   = 0.1          # NT-Xent 损失权重
    cl_temp     = 0.07         # 对比温度
    shuffle_prob = 0.3         # View-B 序列打乱比例

    # 辅助任务 Masked Item Prediction
    aux_weight  = 0.05
    mask_prob   = 0.15

    # 训练
    epochs      = 15
    batch_size  = 256
    lr          = 1e-3
    weight_decay = 1e-5
    label_smooth = 0.0
    mixup_alpha  = 0.1

    seed = 42
    topk = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"