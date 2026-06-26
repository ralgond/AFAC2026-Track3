

PROMPT='''
## 你的角色
你是一名推荐系统模型调参专家。

## 简介
这个模型叫HyFormer，旨在解决传统推荐系统中“序列建模”和“特征交叉”两阶段分离导致的信息交互不足问题。简单来说，这个模型的核心逻辑是：不再先把用户行为序列压缩成一个向量，而是引入一组“全局令牌”（Global Tokens），让它们在每一层网络中都同时与用户的行为序列和用户画像特征进行交互。

为了让你能清晰地向 LLM 介绍这段代码，我将其拆解为核心架构总结和配置参数详解两部分。

### 代码核心架构总结
核心创新 (HyFormer Layer)：
Query Decoding (QD)：利用 Global Tokens 作为 Query，去“查询”用户的行为序列（Key/Value），让全局上下文直接感知序列信息。
Query Boosting (QB)：将 Global Tokens 与用户画像特征合并，进行 Self-Attention，实现跨特征域的深度交叉。

输入处理：
序列侧：将 Item ID 和 4 个 Item 特征（类别等）的 Embedding 均值相加，加上位置编码。
用户侧：使用 8 个用户特征域（如年龄、性别等）的 Embedding 来初始化 Global Tokens。

训练目标：
主任务：全量 Softmax 交叉熵损失（预测下一个点击的物品）。
辅助任务：Masked Item Prediction（类似 BERT 的掩码预测），帮助模型更好地理解序列语义。
推理方式：计算出 User Representation 后，与全量物品库的 Embedding 做矩阵乘法（Cosine Similarity）进行打分排序。

### 模型的参数
max_seq_len: 行为序列最大长度
emb_dim: 基础 Embedding 维度。每个稀疏特征（如 item_id, user_gender）映射成的向量长度。
repr_dim: 主干网络维度 (D)。模型内部处理（Global Tokens、Seq Tokens、User Representation）的隐藏层维度。
n_layers: HyFormer 层数。QD 和 QB 模块交替堆叠的次数，决定了模型的深度。
n_heads: 注意力头数。用于序列建模（QD 模块）的多头注意力机制的头数。
n_gt: Global Token 数量。这里设置为 8，对应代码中使用的 8 个用户特征域，每个特征域初始化一个 GT。
qb_n_heads: QB 模块注意力头数。用于 Query Boosting 阶段特征交叉的注意力头数。
dropout: Dropout 概率。用于防止过拟合，在全连接层和 Attention 输出后随机丢弃神经元。
batch_size:	批次大小。每次梯度更新使用的样本数量。
lr: 学习率。优化器（Adam）的初始学习率。
weight_decay: 权重衰减。L2 正则化系数，用于抑制模型复杂度。
label_smooth: 标签平滑。在计算 CrossEntropy 时使用，防止模型对预测结果过于自信，提高泛化能力。
aux_weight: 辅助任务权重。Masked Item Prediction 损失函数在总 Loss 中的占比。
mask_prob: 掩码概率。在训练序列时，随机将 15% 的物品 ID 替换为 MASK token，用于辅助任务训练。
mixup_alpha: Mixup 增强系数。用于数据增强，通过线性插值混合两个样本的特征和标签（Beta 分布参数）。

### 当前模型的参数值
{current_config}

    
## 模型的参数历史修改轨迹(history trajectory)
历史修改轨迹由多个节点组成，整体是一棵树，节点的有parent_id属性，id属性，is_leaf属性。parent_id为当前节点的父节点的id，is_leaf为true时表示该节点是一个叶子，也即是一条trajectory的tail。给定一个叶子节点id，通过追溯它的parent_id直到parent_id为-1，也即是根节点，可得到一个trajectory。
{history_trajectory}

## 目标
使模型计算出来的best_score尽可能地大，当前的最大best_score为{best_score}。

## 输出
- 请根据历史修改轨迹计算下一步的修改方案，注意只能修改一个参数, 且在一条trajectory上的parameter不能有重复。
- 生成的节点一定是叶子节点，它的父节点不一定是叶子节点(意味着你可以开辟新的trajectory)。
- 一条trajectory的长度不能大于7。
- 格式一定为合法的JSON字符串(不能是Markdown), 举个例子：{{"id":1, "parent_id":0, "is_leaf":true, "parameter":"max_seq_len", "modification":"50->30"}}。
'''

tune_config=''' max_seq_len  = 50
emb_dim      = 32
repr_dim     = 128
n_layers     = 2
n_heads      = 4
n_gt         = 8
qb_n_heads   = 4
dropout       = 0.2
batch_size    = 256
lr            = 1e-3
weight_decay  = 1e-5
label_smooth  = 0.0
aux_weight    = 0.05
mask_prob     = 0.15
mixup_alpha   = 0.1'''


import os
import json
import time
import shutil
import subprocess
from datetime import datetime
from openai import OpenAI

# 注意: 不同地域的base_url不通用（下方示例使用北京地域的 base_url）
# - 华北2（北京）: https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1，请将WorkspaceId替换为业务空间ID
# - 新加坡: https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
# - 德国（法兰克福）: https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/compatible-mode/v1
# - 日本（东京）: https://{WorkspaceId}.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1
# - 美国（弗吉尼亚）: https://dashscope-us.aliyuncs.com/compatible-mode/v1
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://llm-sctg3o0ri7j4gobl.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)


def modify_config(in_file_path, out_file_path, parameter, modification):
    from_value, to_value = modification.split('->')
    with open(out_file_path, "w+") as of:
        for line in open(in_file_path):
            if '=' in line:
                l = line.strip().split()
                k = l[0].strip()
                if k.strip() == parameter:
                    v = l[2].strip()
                    assert v.strip() == from_value, (v.strip(), from_value)
                    of.write("    "+k+" = "+to_value+"\n")
                else:
                    of.write(line)
            else:
                of.write(line)

def modify_config_redirect_submission(in_file_path, dir, out_file_path):
    with open(out_file_path, "w+") as of:
        for line in open(in_file_path):
            if '=' in line:
                # print(line.strip().split())
                l = line.strip().split()
                k = l[0]
                if k.strip() == 'output_path':
                    of.write("    " + k + " = " + F'''"{dir.split('/')[0]}/A2.csv"\n''')
                else:
                    of.write(line)
            else:
                of.write(line)
            

def get_trajectory_by_tail(tail_id, history_trajectory):
    node_d = {}
    for node in history_trajectory:
        node_d[node['id']] = node

    l = []
    tail = node_d[tail_id]
    while True:
        l.append(tail)
        if tail['parent_id'] < 0:
            break
        tail = node_d[tail['parent_id']]
    l.reverse()
    return l
    

def get_next_action(history_trajectory, best_score, config_path):
    config_filter_set = set()
    configs = tune_config.split("\n")
    for c in configs:
        if '=' in c:
            l = c.split()
            k = l[0]
            config_filter_set.add(k)

    current_config_l = []
    with open(config_path, "r") as inf:
        for line in inf:
            if '=' in line:
                l = line.strip().split()
                k = l[0]
                if k in config_filter_set:
                    current_config_l.append(k+" = "+l[2])
    current_config = '\n'.join(current_config_l)                           
    
    prompt = PROMPT.format(history_trajectory=history_trajectory, best_score=best_score, current_config=current_config)
    
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[{'role': 'user', 'content': prompt}]
    )
    json_str = completion.choices[0].message.content
    print(json_str)
    ret = json.loads(json_str)
    return ret, prompt


def run_experiment(parent_dir, dir, parameter=None, modification=None):
    if parent_dir is None:
        modify_config_redirect_submission("./code/hyformer_config.py", dir, f"{dir}/config1.py")
    else:
        modify_config_redirect_submission(f"{parent_dir}/config.py", dir, f"{dir}/config1.py")
    
    if parameter is not None:
        modify_config(f"{dir}/config1.py", f"{dir}/config.py", parameter, modification)
    else:
        shutil.copy(f"{dir}/config1.py", f"{dir}/config.py")
    shutil.copy("./code/hyformer.py", f"{dir}/model.py")
    result = subprocess.run(["python", f"{dir}/model.py"], capture_output=True, text=True)
    
    with open(f"{dir}/stdout.txt", "w+") as of:
        of.write(result.stdout)

    with open(f"{dir}/stderr.txt", "w+") as of:
        of.write(result.stderr)

    if result.returncode != 0:
        print("返回码:", result.returncode)
        os._exit(1)

    last_best_ndcg_at_10 = ''
    for line in result.stdout.split('\n'):
        if 'Final best NDCG@10: ' in line:
            last_best_ndcg_at_10 = line.strip().split('NDCG@10:')[-1]
    return float(last_best_ndcg_at_10)

def calc_best_score(history_trajectory):
    best_score = -1
    for node in history_trajectory:
        if node['best_score'] > best_score:
            best_score = node['best_score']
    return best_score

def run(dir, start_time):
    id = 0
    next_action = None
    history_trajectory_text = '''[]'''
    while time.time() - start_time < 3600 * 2 - 300:
        if id == 0:
            node_dir_path = f"{dir}/{id}"
            os.makedirs(node_dir_path, exist_ok=True)
        
            best_score = run_experiment(None, node_dir_path)
            print(best_score, "elapsed seconds:", time.time() - start_time)
            node = {
                'id': 0, 'parent_id': -1, 'is_leaf': True, 'parameter':"", 'modification':"", 'best_score': best_score
            }
            history_trajectory = json.loads(history_trajectory_text)
            history_trajectory.append(node)
            history_trajectory_text = json.dumps(history_trajectory)

            with open(f"{dir}/history_trajectory.json", "w+") as of:
                of.write(history_trajectory_text)

            next_action, prompt = get_next_action(history_trajectory_text, calc_best_score(history_trajectory), f"{dir}/{id}/config.py")
            with open(f"{dir}/{id}/prompt.txt", "w+") as of:
                of.write(prompt)       
            print(next_action)
            id = next_action['id']
        else:
            id = next_action['id']
            parent_id = next_action['parent_id']
            node_dir_path = f"{dir}/{id}"
            parent_node_dir_path = f"{dir}/{parent_id}"
            node_dir_path = f"{dir}/{id}"
            
            os.makedirs(node_dir_path, exist_ok=True)
            
            best_score = run_experiment(parent_node_dir_path, node_dir_path, next_action['parameter'], next_action['modification'])
            print(best_score)

            node = {
                'id': id, 
                'parent_id': parent_id, 
                'is_leaf': True, 
                'parameter':next_action['parameter'], 
                'modification':next_action['modification'], 
                'best_score': best_score
            }

            history_trajectory = json.loads(history_trajectory_text)
            history_trajectory.append(node)
            history_trajectory.sort(key=lambda x: x['id'])
            for node in history_trajectory:
                if node['id'] == parent_id:
                    node['is_leaf'] = False
            history_trajectory_text = json.dumps(history_trajectory)

            with open(f"{dir}/history_trajectory.json", "w+") as of:
                of.write(history_trajectory_text)

            next_action, prompt = get_next_action(history_trajectory_text, calc_best_score(history_trajectory), f"{dir}/{id}/config.py")
            print(next_action)
            with open(f"{dir}/{id}/prompt.txt", "w+") as of:
                of.write(prompt)
        
def main():
    start_time = time.time()
    root_dir = f"exp_{datetime.now().strftime("%Y%m%d-%H%M")}"
    run(root_dir, start_time)

    # root_dir = "exp_20260625-2034"
    history_trajectory_text = open(f'{root_dir}/history_trajectory.json').read()
    history_trajectory = json.loads(history_trajectory_text)
    best_score = -1
    best_id = 0
    for node in history_trajectory:
        if node['best_score'] > best_score:
            best_score = node['best_score']
            best_id = node['id']
    print("best_id:", best_id)
    trajectory = get_trajectory_by_tail(best_id, history_trajectory)
    print(trajectory)

    result = subprocess.run(["python", f"{root_dir}/{best_id}/model.py"], capture_output=True, text=True)
    print(result)
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    main()