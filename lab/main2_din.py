PROMPT='''
## 你的角色
你是一名推荐系统模型调参专家。

## 简介
模型是DIN(Deep Interest Network)

## 模型的参数
max_seq_len：每个用户历史序列截断/补齐的长度，这个值越大，模型能看到的历史越长。
emb_dim：核心 id embedding（用户id、物品id）的维度，同时也是最终 user_vec/item静态向量的输出维度。
side_emb_ratio：side特征（u_cat_*、i_cat_*、i_bucket_*）embedding维度相对于emb_dim的比例，实际维度为 max(1, int(emb_dim * side_emb_ratio))。比例越大，side特征表达能力越强。
dropout: Dropout 层的概率，用于防止模型过拟合。
batch_size: 训练时每个批次的样本数量。
lr: 优化器的初始学习率 (Learning Rate)。
seed: 随机种子，用于保证实验结果的可复现性。

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
- 格式一定为合法的JSON字符串(不能是Markdown,不能由```json开头), 举个例子：{{"id":1, "parent_id":0, "is_leaf":true, "parameter":"max_seq_len", "modification":"50->30"}}。
- modification字段的值格式为"from_value->to_value"，其中from_value一定得来自上面的“当前模型的参数值”，不能随意编造。
- 如果一个历史修改轨迹中的节点的子节点数量超过15，则该节点不能再作为父节点节点，你应该重新以根节点为父节点开始生成子节点。
- 生成节点时需要考虑兄弟节点，不要重复。
'''

tune_config='''max_seq_len = 50
emb_dim = 32
side_emb_ratio = 0.5
dropout = 0.2
batch_size = 256
lr = 1e-3
seed = 42'''


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
        modify_config_redirect_submission("./code/din_config.py", dir, f"{dir}/config1.py")
    else:
        modify_config_redirect_submission(f"{parent_dir}/config.py", dir, f"{dir}/config1.py")
    
    if parameter is not None:
        modify_config(f"{dir}/config1.py", f"{dir}/config.py", parameter, modification)
    else:
        shutil.copy(f"{dir}/config1.py", f"{dir}/config.py")
    shutil.copy("./code/din_model_notuemb.py", f"{dir}/model.py")
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
        if '[INFO] training done. best valid ndcg@10 =' in line:
            last_best_ndcg_at_10 = line.strip().split('ndcg@10 =')[-1]
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
    modify_config(f"{final_path}/config1.py", f"{final_path}/config.py", "predict", "False->True")
    shutil.copy("./code/din_model_notuemb.py", f"{final_path}/model.py")
    
    result = subprocess.run(["python", f"{final_path}/model.py"], capture_output=True, text=True)
    print(result.returncode)
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    main()