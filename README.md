# CSI信号修复与预测模型 (PyTorch版)

本项目是一个基于PyTorch实现的深度学习模型，用于执行CSI（信道状态信息）的修复 (Inpainting) 和时序预测。模型利用 `t` 时刻的部分天线阵列数据，以及过去10个时间步的完整历史CSI序列，旨在重建 `t` 时刻完整的CSI信号。

## 数据格式说明

**CSI张量结构**: `[4, 2, 4, 117]`
- **第1维 (4)**: 设备数量 - 总共4个设备
- **第2维 (2)**: 接收天线矩阵的行数 
- **第3维 (4)**: 接收天线矩阵的列数
- **第4维 (117)**: 子载波数量

**模型任务**: 从选定设备的部分天线阵列位置预测所有设备的完整天线阵列CSI。当前配置为从单个天线阵列位置（2×4矩阵中的某一列）预测完整的2×4天线阵列数据。

**技术特点**: 采用Conv-LSTM混合架构处理时序依赖关系，支持多GPU并行训练。

---

## 项目结构

```
.
├── config.yaml               # 用于管理超参数和设置的主配置文件
├── datasets/
│   ├── espargos-....tfrecords  # 原始TFRecord数据文件
│   └── csi_data.h5           # 经过预处理后供PyTorch使用的HDF5数据文件
├── env/
│   └── zhangyang.yml         # 用于复现PyTorch环境的Conda配置文件
├── logs/                     # TensorBoard日志文件的默认输出目录
├── model_checkpoints/        # 训练好的模型检查点的默认输出目录
├── README.md                 # 本文件
├── src/
│   ├── train.py              # (PyTorch) 运行训练流程的主脚本
│   ├── model.py              # (PyTorch) 包含神经网络的结构定义
│   ├── utils.py              # (PyTorch) 数据加载、预处理及其他辅助函数
│   ├── losses.py             # (PyTorch) 损失函数定义
│   └── pre.py                # 将原始TFRecord转换为HDF5的预处理脚本
├── start_training.sh         # 用于启动训练的可执行脚本
└── training_visualizations/  # 训练过程中的可视化结果图片
```

---

## 如何使用 (完整流程)

请按照以下三个步骤来运行本项目。

### 第1步：环境设置

我们使用Conda来管理项目环境。

```bash

# 激活环境
conda activate zhangyang
```
*注意：如果您是第一次创建，可以使用 `conda env create -f env/zhangyang.yml`。*

### 第2步：数据预处理

在开始训练之前，您需要先运行一次预处理脚本。这个脚本会读取原始的 `.tfrecords` 文件，并生成一个PyTorch可以直接使用的高效 `.h5` 文件。

```bash
# 确保你已在 'zhangyang' 环境下
python src/pre.py
```
这个过程可能需要几分钟。完成后，您应该会在 `datasets/` 目录下看到一个名为 `csi_data.h5` 的新文件。**此步骤只需在数据未生成时执行一次。**

### 第3步：开始训练

完成以上步骤后，您可以开始训练模型。

```bash
# 直接执行启动脚本
./start_training.sh
```

该脚本会自动设置GPU，并调用 `src/train.py` 开始训练。训练进度会以进度条的形式显示在终端中。

---

## 查看结果


### 1. CSI 重建效果图

脚本会根据 `config.yaml` 中设置的频率，定期保存模型在验证集上的预测效果图。这些图片保存在 `training_visualizations/` 目录下，让您可以直观地看到模型的学习进展。
