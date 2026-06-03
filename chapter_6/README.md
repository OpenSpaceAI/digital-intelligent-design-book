# 6.4 基于多层感知机的深空探测温度预测示例

本项目整理自专著第 6.4 节的编程示例，并结合修改反馈对代码缩进、空行、数据预处理流程和可复现性做了重构。项目目标是提供一套可以直接在 GitHub 开源的 Python 示例：从数据生成、缺失值处理、异常值剔除、归一化、模型训练、评估、可视化到敏感性分析，完整展示多层感知机在深空探测器温度预测任务中的基本用法。

## 项目结构

```text
chapter6_4_temperature_mlp/
  README.md
  requirements.txt
  scripts/
    01_generate_data.py
    02_preprocess_data.py
    03_explore_data.py
    04_train_model.py
    05_evaluate_model.py
    06_sensitivity_analysis.py
    07_tune_hyperparameters.py
    run_pipeline.py
  src/deepspace_temperature/
    config.py
    data.py
    preprocessing.py
    modeling.py
    evaluation.py
```

## 安装

建议使用 Python 3.10 或更高版本。

```bash
cd chapter6_4_temperature_mlp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux 或 macOS 可将激活命令替换为：

```bash
source .venv/bin/activate
```

## 一键运行

```bash
python scripts/run_pipeline.py
```

运行后会生成：

```text
data/raw/temperature_observations.csv
data/processed/train.csv
data/processed/test.csv
models/mlp_temperature_model.joblib
models/preprocess_bundle.joblib
reports/metrics.json
reports/figures/
```

## 分步运行

```bash
python scripts/01_generate_data.py
python scripts/02_preprocess_data.py
python scripts/03_explore_data.py
python scripts/04_train_model.py
python scripts/05_evaluate_model.py
python scripts/06_sensitivity_analysis.py
python scripts/07_tune_hyperparameters.py
```

## 示例任务说明

输入特征模拟深空探测场景中可能影响温度的因素，包括：

- `solar_flux`：太阳辐照强度
- `distance_au`：与太阳距离，单位 AU
- `orbital_radius`：轨道半径
- `albedo`：表面反照率
- `heater_power`：加热器功率
- `radiator_area`：散热面积
- `instrument_load`：仪器负载
- `attitude_angle`：姿态角

预测目标为：

- `temperature_c`：探测器关键部件温度，单位摄氏度

## 与修改意见的对应

反馈中第 242 页指出数据清洗代码存在排版和代码组织问题。本项目将该部分整理为 `src/deepspace_temperature/preprocessing.py`，核心改动包括：

- 用函数封装缺失值填充、异常值剔除、归一化和数据集划分；
- 保留并改进“均值填充”和“3σ 异常值剔除”的思路；
- 避免硬编码在单段脚本中，便于复用和测试；
- 所有脚本采用统一入口和清晰缩进，适合开源维护。

## 许可建议

如果用于 GitHub 开源，建议补充 `LICENSE` 文件，例如 MIT License 或 Apache-2.0 License。若书中原始文字、图表或数据不准备开源，请只发布本项目代码和合成数据生成逻辑。
