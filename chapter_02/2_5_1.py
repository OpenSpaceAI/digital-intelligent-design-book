# ==========================================
# 2.5.1 基于矩阵运算的探测器姿态控制系统模拟编程示例
# ==========================================

import numpy as np

# ==========================================
# 2.5.1.2 设置初始姿态矩阵 
# ==========================================
# 初始姿态矩阵为单位矩阵，表示无旋转
initial_attitude = np.eye(3)
print("初始姿态矩阵:\n", initial_attitude)

# ==========================================
# 2.5.1.3 定义绕主轴的旋转操作 
# ==========================================
def rotate_x(theta):
    """绕X轴旋转的方向余弦矩阵"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotate_y(theta):
    """绕Y轴旋转的方向余弦矩阵"""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotate_z(theta):
    """绕Z轴旋转的方向余弦矩阵"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

# ==========================================
# 2.5.1.4 模拟复合姿态调整操作 
# ==========================================
# 角度转弧度
deg2rad = lambda deg: deg * np.pi / 180

# 定义旋转角度
theta_z = deg2rad(45)
theta_y = deg2rad(30)
theta_x = deg2rad(60)

# 分别计算各轴旋转矩阵
Rz = rotate_z(theta_z)
Ry = rotate_y(theta_y)
Rx = rotate_x(theta_x)

# 姿态更新：初始姿态乘以复合旋转矩阵
# 右乘表示相对于当前机体坐标系进行新的旋转
attitude_updated = initial_attitude @ Rz @ Ry @ Rx
print("\n更新后的姿态矩阵:\n", attitude_updated)

# ==========================================
# 2.5.1.5 验证姿态矩阵的性质与方向变化 
# ==========================================
# 检查是否正交: R^T * R ≈ I
orthogonality_check = attitude_updated.T @ attitude_updated
print("\n正交性验证(应接近单位矩阵):\n", orthogonality_check)

# 输出新的三个轴方向向量
# 矩阵列向量直接对应机体轴在惯性系中的方向
print("\n机体X轴方向:", attitude_updated[:, 0])
print("机体Y轴方向:", attitude_updated[:, 1])
print("机体Z轴方向:", attitude_updated[:, 2])