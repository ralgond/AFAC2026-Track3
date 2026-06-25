class Config:
    train_path  = "../data/A2-Rec/train.csv"
    test_path   = "../data/A2-Rec/test.csv"
    user_path   = "../data/A2-Rec/user.csv"
    item_path   = "../data/A2-Rec/item.csv"
    output_path = "submission_e2e.csv"

    max_seq_len = 50
    emb_dim     = 32
    repr_dim    = 128
    n_heads     = 4
    mlp_dims    = [256, 128]
    dropout     = 0.2

    epochs      = 50
    batch_size  = 256
    lr          = 1e-3
    weight_decay= 1e-5
    seed        = 42
    topk        = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cl_weight = 0.2
    temp = 0.02