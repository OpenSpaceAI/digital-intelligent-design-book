# 星云图像分类 (Nebula Image Classification)

本项目是一个基于 PyTorch 框架构建的深度学习图像分类项目。主要是使用自己从头实现的 **ResNet-18** 模型，对星云（Nebulae）图像及其各个子类别进行分类。

## 简介

代码核心在 `chapter_7.py` 当中，主要包含以下功能和特点：
- **自定义数据加载**：借助 `CustomImageFolder` 和 `TransformSubset`，自动扫描有效图像数据集并过滤不支持的分类。
- **自动拆分验证集**：如果在数据文件夹中没有检测到独立的 `Validation` 验证集目录，它能够按 `80:20` 的比例自动划分训练集与验证集。
- **模型搭建**：不依赖预训练模型库，而是通过 `nn.LazyConv2d`、`nn.LazyLinear`、残差块（`Residual`模块）以及自定义的类似 Keras 列表层拼接封装结构的 `HybridSequential` 从头手工实现了 **ResNet-18** 网络。
- **验证与保存**：包含完整的训练、验证循环流程（附带有 `tqdm` 进度条），并在每个 Epoch 结束后判断，自动保存验证集准确率最高时的模型权重为 `best_model.pth`。

## 分类类别
模型支持并针对以下 5 种星云景象进行分类：
1. Dark Nebula (暗星云)
2. Emission Nebula (发射星云)
3. Planetary Nebula (行星状星云)
4. Reflection Nebula (反射星云)
5. Supernova Remnants (超新星遗迹)

## 环境依赖
运行该脚本前，请确保环境中已安装以下 Python 库：
- `torch`
- `torchvision`
- `tqdm`

## 数据集结构说明

默认配置下，代码会在以下相对路径读取数据:
`./data/helodrys_nebula_images/Nebulae`

推荐您的数据集目录结构如下组织：
```text
chapter_07/
├── chapter_7.py
└── data/
    └── helodrys_nebula_images/
        └── Nebulae/
            ├── Dark Nebula/
            │   ├── image1.jpg
            │   └── ...
            ├── Emission Nebula/
            ├── Planetary Nebula/
            ├── Reflection Nebula/
            ├── Supernova Remnants/
            └── Validation/ (可选)
                ├── Dark Nebula/
                └── ...
```
*备注: 如果未提供额外的 `Validation` 文件夹或校验遇到问题，脚本将自动在主类目文件夹上执行 `80:20` 的比例划分出独立训练/验证集。*

## 运行与使用

1. 确保安装好相关依赖以及准备好对应目录结构的数据集。
2. 在终端里，执行以下命令开始训练模型：
   ```bash
   python chapter_7.py
   ```
3. 训练过程会输出 Loss (损失) 和 Accuracy (准确率)，并在根目录生成最佳验证精度对应的权重文件 `best_model.pth`。
