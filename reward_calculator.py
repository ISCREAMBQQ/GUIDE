import json

# --- 配置 ---
# 假设你的原始文件名为这个
INPUT_JSON_PATH = 'Graph/GUIDE_037_updated.json'
# 更新后的文件将保存到这里
OUTPUT_JSON_PATH = 'Graph/GUIDE_037_updated.json'


def calculate_reward(rating, review_count):
    """
    根据公式计算reward值。
    如果评分或评论数为None，则奖励为0。
    """
    if rating is None or review_count is None:
        return 0.0

    # 核心公式
    reward = (rating - 4.0) * review_count / 400.0

    # 限制范围在 [-4, 4]
    return max(-4.0, min(4.0, reward))


# --- 脚本主逻辑 ---

# 1. 读取原始JSON文件
try:
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        all_locations = json.load(f)
    print(f"成功从 '{INPUT_JSON_PATH}' 读取 {len(all_locations)} 个地点。")
except FileNotFoundError:
    print(f"错误：输入文件 '{INPUT_JSON_PATH}' 未找到。请检查文件名和路径。")
    exit()
except json.JSONDecodeError:
    print(f"错误：文件 '{INPUT_JSON_PATH}' 不是有效的JSON格式。")
    exit()

# 2. 第一轮：计算并更新所有地点的reward，并存入一个临时字典以便快速查找
updated_rewards = {}
for loc in all_locations:
    new_reward = calculate_reward(loc.get("rating"), loc.get("review_count"))
    loc['reward'] = new_reward  # 更新主条目的reward
    updated_rewards[loc['name']] = new_reward

print("\n第一轮完成：所有地点的 'reward' 已重新计算。")

# 3. 第二轮：更新所有边的 adjusted_value
for loc in all_locations:
    if 'neighbors' not in loc or 'neighbors_reward' not in loc:
        continue

    # 遍历当前地点的所有邻居
    for neighbor_name, distance in loc['neighbors'].items():
        # 获取邻居的新reward值，如果邻居不在数据中，则默认为0
        neighbor_reward = updated_rewards.get(neighbor_name, 0.0)

        # 计算新的 adjusted_value (cost)
        # New Cost = distance + (MaxReward - neighbor_reward)
        # MaxReward 是 4
        new_adjusted_value = distance + (4.0 - neighbor_reward)

        # 更新 neighbor_reward 字典中的 adjusted_value
        if neighbor_name in loc['neighbors_reward']:
            loc['neighbors_reward'][neighbor_name]['adjusted_value'] = round(new_adjusted_value, 4)
            # (可选) 同时更新邻居的reward信息以保持数据一致性
            loc['neighbors_reward'][neighbor_name]['reward'] = round(neighbor_reward, 4)
        else:
            # 如果邻居信息不存在，可以选择创建它
            print(f"警告：在 '{loc['name']}' 的 neighbors_reward 中未找到 '{neighbor_name}'，将跳过。")

print("第二轮完成：所有边的 'adjusted_value' 已基于新公式更新。")

# 4. 将更新后的数据写入新文件
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_locations, f, ensure_ascii=False, indent=2)

print(f"\n处理完成！修改后的数据已成功保存到 '{OUTPUT_JSON_PATH}'。")
print("现在所有的边权重（adjusted_value）都是非负的，可以安全地用于Dijkstra等算法。")