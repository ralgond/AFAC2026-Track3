"""
合并多个预测文件，对每个 test_idx 的 label 做多数投票（majority vote）。

用法：
    python merge_predictions.py

依赖：
    pip install pandas
"""

import pandas as pd
from collections import Counter

# ========== 1. 配置输入文件 ==========
files = [
    "predictions_gat.csv",
    "predictions_gcn.csv",
    "predictions_gin.csv",
    "predictions_residual.csv",
    "predictions_sage.csv",
]

output_file = "predictions_merged.csv"

# ========== 2. 读取所有文件 ==========
dfs = []
for f in files:
    df = pd.read_csv(f)
    # 确保列名统一
    assert "test_idx" in df.columns and "label" in df.columns, f"{f} 缺少必要列"
    dfs.append(df.set_index("test_idx")["label"])

# ========== 3. 按 test_idx 合并成一个宽表：每列是一个模型的预测 ==========
merged = pd.concat(dfs, axis=1)
merged.columns = [f"model_{i+1}" for i in range(len(files))]

# 检查是否所有文件的 test_idx 完全一致
if merged.isnull().any().any():
    missing = merged[merged.isnull().any(axis=1)]
    print("警告：以下 test_idx 在部分文件中缺失：")
    print(missing)

# ========== 4. 逐行多数投票 ==========
def majority_vote(row):
    counts = Counter(row.dropna().tolist())
    # most_common 按出现次数从高到低排序；若票数相同，取第一个（也可自定义平票规则）
    most_common = counts.most_common()
    max_count = most_common[0][1]
    # 找出所有票数并列最高的候选
    candidates = [label for label, cnt in most_common if cnt == max_count]
    if len(candidates) > 1:
        # 平票时的处理方式：这里简单取数值最小的（也可以改成随机、优先某个模型等）
        return sorted(candidates)[0]
    return candidates[0]

merged["label"] = merged.apply(majority_vote, axis=1)

# ========== 5. 输出结果 ==========
result = merged["label"].reset_index()
result.columns = ["test_idx", "label"]
result.to_csv(output_file, index=False)

print(f"合并完成，结果已保存到 {output_file}")
print(result.head())
