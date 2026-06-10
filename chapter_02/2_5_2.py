# ==========================================
# 2.5.2 基于贝叶斯定理的探测器故障诊断
# 依赖库安装: pip install pgmpy
# ==========================================

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ==========================================
# 2.5.2.3 构建贝叶斯网络结构
# ==========================================
# 定义网络结构 (边的方向表示单向因果关系)
model = BayesianNetwork([
    ('PowerFault', 'ThermalFault'),
    ('PowerFault', 'AttitudeFault'),
    ('ThermalFault', 'AttitudeFault')
])

# ==========================================
# 2.5.2.4 设定先验概率与条件概率
# 节点取值: 0(正常), 1(故障)
# ==========================================
# 1. Power 故障的先验概率
cpd_power = TabularCPD(variable='PowerFault', variable_card=2,
                       values=[[0.98],  # 正常
                               [0.02]]) # 故障

# 2. Thermal 故障的条件概率 (依赖于 Power)
cpd_thermal = TabularCPD(variable='ThermalFault', variable_card=2,
                         values=[
                             [0.99, 0.85],  # 正常
                             [0.01, 0.15]   # 故障
                         ],
                         evidence=['PowerFault'],
                         evidence_card=[2])

# 3. Attitude 故障的条件概率 (依赖于 Power 与 Thermal)
cpd_attitude = TabularCPD(variable='AttitudeFault', variable_card=2,
                          values=[
                              [0.99, 0.95, 0.90, 0.60],  # 正常
                              [0.01, 0.05, 0.10, 0.40]   # 故障
                          ],
                          evidence=['PowerFault', 'ThermalFault'],
                          evidence_card=[2, 2])

# ==========================================
# 2.5.2.5 加入模型并验证一致性
# ==========================================
# 加入 CPDs
model.add_cpds(cpd_power, cpd_thermal, cpd_attitude)

# 验证模型正确性 (检查概率和是否为1，结构是否合法)
assert model.check_model(), "贝叶斯网络模型构建存在错误！"

# ==========================================
# 2.5.2.6 模拟观测与故障后验概率计算
# ==========================================
# 创建推理器
infer = VariableElimination(model)

# 设置观测: 明确探测到姿态控制系统 (Attitude) 发生故障
evidence = {'AttitudeFault': 1}

# 计算后验概率
posterior_power = infer.query(variables=['PowerFault'], evidence=evidence)
posterior_thermal = infer.query(variables=['ThermalFault'], evidence=evidence)

print("【诊断报告】在观测到 Attitude 系统故障后：\n")
print("-> 电源系统(Power) 的状态概率分布:")
print(posterior_power)
print("\n-> 热控系统(Thermal) 的状态概率分布:")
print(posterior_thermal)