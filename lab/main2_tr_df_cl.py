

PROMPT='''
## 你的角色
你是一名推荐系统模型调参专家。

## 简介
这份代码实现了一个名为 ContrastFormer 的推荐系统模型。你可以把它理解为在经典的序列推荐模型（如 HyFormer）基础上，引入了 DeepFM 结构和对比学习机制的增强版。
简单来说，它的核心目标是更精准地预测用户接下来会点击什么商品。为了达到这个目标，它做了三件关键的事情：
 - 增强物品（Item）表达：不再只看物品的 ID，而是用 DeepFM 结构同时捕捉物品的低阶特征（如类别）和高阶特征组合。
 - 增强用户（User）表达：在分析用户行为序列的同时，也用 DeepFM 分支处理用户的静态属性（如年龄、性别），让两者互补。
 - 引入对比学习：通过构造两种不同的用户行为视图（一个是随机遮盖，一个是随机打乱），让模型学习到的用户特征更鲁棒，减少噪声干扰。

## 模型的参数

### 基础模型参数
max_seq_len: 序列最大长度。模型最多只看用户最近的 50 个行为，超过的会被截断，不足的会被填充。
emb_dim: Embedding 维度。每个特征（如物品 ID、类别 ID）会被映射成一个 32 维的向量。
repr_dim: 主干隐层维度。这是模型内部处理信息的主要通道宽度（Transformer 的维度），比 Embedding 维度大，用于承载更复杂的特征交互。

### 网络结构参数
n_layers: HyFormer 层数。Transformer 编码器堆叠了 2 层。
n_heads: 注意力头数。在自注意力机制中，模型并行关注 4 个不同的子空间。
n_gt: Global Token 数量。这是 HyFormer 的特色，引入了 8 个全局 Token 来辅助提取用户兴趣，通常对应用户特征的域数量。
qb_n_heads: Query Boosting 头数。这是 HyFormer 中特定模块的注意力头数。
dropout: 丢弃率。训练时随机让部分神经元不工作，防止模型死记硬背（过拟合）。

###  损失函数与训练策略
cl_weight: 对比学习损失权重。总损失函数中，对比学习（Contrastive Learning）部分占 0.1 的比重。它用于拉近同一用户不同增强视图的距离。
cl_temp: 对比温度系数。用于调节对比学习中 Softmax 分布的平滑程度，数值越小，分布越尖锐。
shuffle_prob: 序列打乱概率。在构造对比学习的“视图 B”时，随机打乱序列中 30% 的 token 位置。
aux_weight: 辅助任务损失权重。Masked Item Prediction (MIP) 任务在总损失中的占比。
mask_prob: 遮盖概率。在构造“视图 A”时，随机遮盖（Mask）序列中 15% 的物品，让模型去猜被遮盖的是什么。
batch_size: 批大小。每次更新参数时，使用 256 个样本。
lr: 学习率。优化器（Adam）每次迈步的大小。
weight_decay: 权重衰减。L2 正则化系数，用于抑制模型参数过大，防止过拟合。
label_smooth: 标签平滑。设为 0 表示不使用。如果设为正数，可以防止模型对预测结果过于自信，增加泛化能力。
mixup_alpha: Mixup 系数。一种数据增强手段，通过线性插值混合两个样本的特征和标签。0.1 表示混合的强度。

## 当前模型的参数值
{current_config}

    
## 模型的参数历史修改轨迹(history trajectory)
历史修改轨迹由多个节点组成，整体是一棵树，节点的有parent_id属性，id属性，is_leaf属性。parent_id为当前节点的父节点的id，is_leaf为true时表示该节点是一个叶子，也即是一条trajectory的tail。给定一个叶子节点id，通过追溯它的parent_id直到parent_id为-1，也即是根节点，可得到一个trajectory。
{history_trajectory}

## 目标
使模型计算出来的best_score尽可能地大，当前的最大best_score为{best_score}。

## 输出
- 请根据历史修改轨迹计算下一步的修改方案，注意只能修改一个参数, 且在一条trajectory上的parameter不能有重复。
- 生成的节点一定是叶子节点，它的父节点不一定是叶子节点(意味着你可以开辟新的trajectory)。
- 格式一定为合法的JSON字符串(不能是Markdown), 举个例子：{{"id":1, "parent_id":0, "is_leaf":true, "parameter":"max_seq_len", "modification":"50->30"}}。
'''

tune_config='''max_seq_len = 50
emb_dim = 32
repr_dim = 128
n_layers = 2
n_heads = 4
n_gt = 8
qb_n_heads = 4
dropout = 0.2
cl_weight = 0.1
cl_temp = 0.07
shuffle_prob = 0.3
aux_weight = 0.05
mask_prob = 0.15
batch_size = 256
lr = 1e-3
weight_decay = 1e-5
label_smooth = 0.0
mixup_alpha = 0.1'''


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
        modify_config_redirect_submission("./code/tr_df_cl_config.py", dir, f"{dir}/config1.py")
    else:
        modify_config_redirect_submission(f"{parent_dir}/config.py", dir, f"{dir}/config1.py")
    
    if parameter is not None:
        modify_config(f"{dir}/config1.py", f"{dir}/config.py", parameter, modification)
    else:
        shutil.copy(f"{dir}/config1.py", f"{dir}/config.py")
    shutil.copy("./code/tr_df_cl.py", f"{dir}/model.py")
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

    with open("trajectory_B2.json", "w+") as of:
        of.write(json.dumps(trajectory))

    result = subprocess.run(["python", f"{root_dir}/{best_id}/model.py"], capture_output=True, text=True)
    print(result.returncode)
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    main()