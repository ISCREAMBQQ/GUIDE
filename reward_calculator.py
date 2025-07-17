import json

# --- 配置 ---
# 假设你的原始文件名为这个
INPUT_JSON_PATH = 'Graph/GUIDE_037_with_urls&busy.json'
# 更新后的文件将保存到这里
OUTPUT_JSON_PATH = 'Graph/GUIDE038.json'


def calculate_reward(rating, review_count, busyness):
    """
    根据新的分层逻辑计算reward值。
    1. 如果评分或评论数为None，则奖励为0。
    2. 计算 (rating - 4.0) * review_count 的中间值。
    3. 根据中间值的大小决定最终reward。
    """
    if rating is None or review_count is None :
        return 0.0

    if not busyness:
        busyness = 0

    if not isinstance(busyness,int):
        busyness = float(busyness[:-1])

        # (3) 计算判断的中间值
    intermediate_value = ((rating - 4.0) * review_count) * (1 - busyness/100) / 500

    # (3) 根据中间值应用不同规则
    if intermediate_value > 16:
        reward = 5
    elif 4 <= intermediate_value <= 16:
        reward = 4.0
    else:
        # (1) 剩余的按真实值处理，使用新公式
        reward = intermediate_value

    return reward


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
    # 使用更新后的reward计算函数
    new_reward = calculate_reward(loc.get("rating"), loc.get("review_count"), loc.get('busyness'))
    loc['reward'] = new_reward
    updated_rewards[loc['name']] = new_reward

print("\n第一轮完成：所有地点的 'reward' 已根据新规则重新计算。")

# 3. 第二轮：更新所有边的 adjusted_value
for loc in all_locations:
    if 'neighbors' not in loc or 'neighbors_reward' not in loc:
        continue

        # 遍历当前地点的所有邻居
    for neighbor_name, distance in loc['neighbors'].items():
        # 获取邻居的新reward值，如果邻居不在数据中，则默认为0
        neighbor_reward = updated_rewards.get(neighbor_name, 0.0)

        # (2) 根据新公式计算 adjusted_value (cost)
        # New Cost = distance - neighbor_reward + 3.8
        new_adjusted_value = distance - neighbor_reward + 3.8

        # (4) 如果出现负数，统一设为 -0.0001
        if new_adjusted_value < 0:
            print(neighbor_name)
            new_adjusted_value = -0.01

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
print("注意：部分边权重（adjusted_value）可能被设置为-0.0001。标准的Dijkstra算法不适用于含负权重的图。")
