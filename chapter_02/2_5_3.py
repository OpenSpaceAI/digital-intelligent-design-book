# ==========================================
# 2.5.3 分治算法在探测器数据处理中的编程示例
# ==========================================

import random
import numpy as np

# ==========================================
# 前置依赖：基础归并排序算法 (引自 2.4.2.1 节)
# ==========================================
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# ==========================================
# 2.5.3.2 模拟数据生成
# ==========================================
# 模拟生成10000个星体亮度数据(浮点数,范围0~1000)
def generate_star_data(num_stars=10000):
    return [random.uniform(0, 1000) for _ in range(num_stars)]

# 生成样本数据
brightness_data = generate_star_data()
print("【基础数据处理】")
print("前10个亮度数据示例:\n", brightness_data[:10])

# ==========================================
# 2.5.3.4 排序处理并结合分析任务
# ==========================================
# 执行排序
sorted_brightness = merge_sort(brightness_data)

# 分析任务
top_k = 10
threshold = 900

# 提取亮度最强的前k个星体
brightest_stars = sorted_brightness[-top_k:]
print(f"\n亮度最强的前{top_k}个星体:\n", brightest_stars)

# 统计亮度大于阈值的星体数量
count_above_threshold = sum(1 for b in sorted_brightness if b > threshold)
print(f"\n亮度大于{threshold}的星体数量: {count_above_threshold}")

# ==========================================
# 2.5.3.5 扩展：将亮度与其他属性结合 (结构化数据排序)
# ==========================================
# 生成结构化数据(亮度、视速度)
def generate_structured_data(num_stars=10000):
    return [
        {"brightness": random.uniform(0, 1000),
         "velocity": random.uniform(-100, 100)}
        for _ in range(num_stars)
    ]

# 对结构化数据按指定属性(key)排序
def merge_sort_struct(data, key):
    if len(data) <= 1:
        return data
    mid = len(data) // 2
    left = merge_sort_struct(data[:mid], key)
    right = merge_sort_struct(data[mid:], key)
    return merge_struct(left, right, key)

def merge_struct(left, right, key):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][key] <= right[j][key]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
            
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 应用结构化排序
print("\n" + "="*40)
print("【扩展：结构化数据处理】")
structured_data = generate_structured_data()
sorted_structured = merge_sort_struct(structured_data, key="brightness")

print("亮度最强的前3个星体(包含视速度):")
for s in sorted_structured[-3:]:
    print(s)