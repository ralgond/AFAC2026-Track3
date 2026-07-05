PROMPT='''
## 你的角色
你是一名推荐系统模型调参专家。

## 简介
我有一个模型，实现了一个端到端的推荐系统模型，名为 DINDeepFME2E。它的核心特点是摒弃了传统的“召回-排序”两阶段模式，而是直接对全量商品进行打分和排序。
该模型巧妙地融合了 DIN (Deep Interest Network) 的用户兴趣提取能力和 DeepFM 的特征交叉能力。为了高效地对海量商品进行打分，它将模型拆分为“用户塔”（User Tower）和“商品塔”（Item Tower）。用户塔负责生成用户表征向量，商品塔则预先计算并缓存所有商品的表征向量。最终的推荐分数通过一次矩阵乘法（用户向量与所有商品向量的点积）即可得出，极大地提升了推理效率。

模型中的对比学习是一种自监督学习（Self-supervised Learning）方法，其核心目标是增强用户表征的鲁棒性（Robustness）。
简单来说，它希望模型学习到的用户兴趣向量，不应该因为用户行为序列中的一些微小、随机的扰动（比如偶尔点击了一个不相关的商品，或者数据记录有缺失）而发生剧烈变化。通过对比学习，模型能更好地抓住用户稳定、核心的兴趣。

经过专家认证，对比学习对于此模型有很大的帮助，和对比学习的参数应该尽早调试。

## 模型的参数
max_seq_len: 用户历史行为序列的最大长度。如果序列超过此长度会被截断，不足则用0填充。
emb_dim: 每个特征域（如用户类别、商品类别等）的嵌入（Embedding）向量的维度。
repr_dim: 用户塔和商品塔最终输出的表征（Representation）向量的维度。
n_heads: 用户兴趣提取模块中，多头自注意力机制（Multi-head Self-Attention）的头数。
dropout: Dropout 层的概率，用于防止模型过拟合。
batch_size: 训练时每个批次的样本数量。
lr: 优化器的初始学习率 (Learning Rate)。
weight_decay: 权重衰减系数，是 L2 正则化的一种形式，用于防止过拟合。
seed: 随机种子，用于保证实验结果的可复现性。
cl_weight: 对比学习（Contrastive Learning）损失函数的权重系数，用于平衡推荐任务和对比学习任务。
temp: 对比学习损失函数中使用的温度系数（Temperature），用于缩放相似度分数。

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
- modification字段的值格式为"from_value->to_value"，其中from_value一定得来自上面的“当前模型的参数值”，不能随意编造。
- n_heads必须是4的倍数，repr_dim必须是n_heads的倍数。
- 如果一个历史修改轨迹中的节点的子节点数量超过15，则该节点不能再作为父节点节点，你应该重新以根节点为父节点开始生成子节点。
- 生成节点时需要考虑兄弟节点，不要重复。
'''

tune_config='''max_seq_len = 50
emb_dim = 32
repr_dim = 128
n_heads = 4
dropout = 0.2
batch_size = 256
lr = 1e-3
weight_decay = 1e-5
seed = 42
cl_weight = 0.2  # 对比学习 Loss 的权重
temp = 0.02      # 温度系数'''


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

def get_config(in_file_path, parameter):
    for line in open(in_file_path):
        if '=' in line:
            l = line.strip().split()
            k = l[0].strip()
            if k.strip() == parameter:
                v = l[2].strip()
                return v
    return None

def modify_config(in_file_path, out_file_path, parameter, modification):
    from_value, to_value = modification.split('->')
    with open(out_file_path, "w+") as of:
        for line in open(in_file_path):
            if '=' in line:
                l = line.strip().split()
                k = l[0].strip()
                if k.strip() == parameter:
                    v = l[2].strip()
                    # assert v.strip() == from_value, (v.strip(), from_value)
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
        model="qwen3.7-plus",
        messages=[{'role': 'user', 'content': prompt}]
    )
    json_str = completion.choices[0].message.content
    print(json_str)
    ret = json.loads(json_str)
    return ret, prompt


def run_experiment(parent_dir, dir, parameter=None, modification=None):
    if parent_dir is None:
        modify_config_redirect_submission("./code/din_deepfm_e2e_cl_agu_config.py", dir, f"{dir}/config1.py")
    else:
        modify_config_redirect_submission(f"{parent_dir}/config.py", dir, f"{dir}/config1.py")
    
    if parameter is not None:
        modify_config(f"{dir}/config1.py", f"{dir}/config.py", parameter, modification)
    else:
        shutil.copy(f"{dir}/config1.py", f"{dir}/config.py")
    shutil.copy("./code/din_deepfm_e2e_cl_agu.py", f"{dir}/model.py")
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
    while time.time() - start_time < 3600 * 2 - 500:
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

    # root_dir = "exp_20260630-1245"
    
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

    with open(f"{root_dir}/trajectory_B2.json", "w+") as of:
        of.write(json.dumps(trajectory))

    final_path = f"{root_dir}/final"
    os.makedirs(final_path, exist_ok=True)
    shutil.copy(f"{root_dir}/{best_id}/config.py", f"{final_path}/config1.py")
    modify_config(f"{final_path}/config1.py", f"{final_path}/config2.py", "epochs", "5->50")
    modify_config(f"{final_path}/config2.py", f"{final_path}/config.py", "predict", "False->True")
    shutil.copy("./code/din_deepfm_e2e_cl_agu.py", f"{final_path}/model.py")
    
    result = subprocess.run(["python", f"{final_path}/model.py"], capture_output=True, text=True)
    print(result.returncode)
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    main()