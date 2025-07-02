# 基于图Transformer的鲁棒性停车位检测

本项目是论文 **《Attentional Graph Neural Network for Parking-Slot Detection》** (IEEE RA-L 2021) 官方开源项目 [gcn-parking-slot](https://github.com/Jiaolong/gcn-parking-slot) 的一个增强实现版本。核心创新在于将原始模型中的图注意力网络（GAT）升级为**图Transformer**，旨在利用其强大的全局关系建模能力，提升模型在复杂场景下的鲁棒性和检测性能。

## 核心改进：从GAT到图Transformer

原始模型开创性地使用图神经网络来学习停车位标记点之间的拓扑关系，避免了复杂的后处理步骤。其核心是图注意力网络（GAT），它在聚合信息时更侧重于节点的局部邻域。

然而，停车场的布局通常具有强烈的全局规律性（如车位等距、等向排列）。为了更好地利用这种全局上下文信息，我们引入了图Transformer。与GAT不同，图Transformer的自注意力机制能够一步到位地计算图中**所有节点对**之间的依赖关系，从而构建一个完全的全局感受野。

**优势：**
* **更强的全局建模能力**：能更好地理解整个停车场的宏观布局。
* **更佳的鲁棒性**：在面对标记点被部分遮挡或模糊等挑战时，能够根据全局信息进行更有效的推理，减少漏检和误检。

## 环境要求

本项目在以下环境中经过测试，建议您使用相同的版本以保证可复现性：
* **操作系统**: Ubuntu 20.04.5 LTS
* **Python**: 3.8.10
* **PyTorch**: 1.7.1
* **CUDA**: 11.0
* **GPU**: NVIDIA RTX 4090

其他依赖包已在 `requirements.txt` 文件中列出。

## 安装与配置

1.  **克隆项目仓库**
    ```bash
    git clone [您的项目仓库URL]
    cd [您的项目目录]
    ```

2.  **创建并激活conda环境 (推荐)**
    ```bash
    conda create -n park-trans python=3.8
    conda activate park-trans
    ```

3.  **安装PyTorch**
    请根据您的CUDA版本，从PyTorch官网安装对应的版本。例如：
    ```bash
    # For CUDA 11.0
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)
    ```

4.  **安装其他依赖**
    ```bash
    pip install -r requirements.txt
    ```

## 数据集准备

本项目使用公开数据集 **ps2.0**。请下载该数据集，并将其解压后按以下目录结构存放：

```
gcn-parking-slot/
├── data/
│   ├── ps2.0/
│   │   ├── images/
│   │   │   ├── 000001.jpg
│   │   │   ├── ...
│   │   ├── train.txt
│   │   ├── test.txt
│   │   └── annotations.json
...
```
`train.txt` 和 `test.txt` 文件包含了训练和测试图片的路径列表。

## 如何运行

您可以通过指定不同的配置文件来运行我们改进后的模型或原始的基线模型，方便进行性能对比。

### 训练模型

* **训练本文的图Transformer模型** (使用 `ps_transformer.yaml`)
    ```bash
    python train.py --config config/ps_transformer.yaml
    ```

* **训练原始的GAT基线模型** (使用 `ps_gat.yaml`)
    ```bash
    python train.py --config config/ps_gat.yaml
    ```

训练过程中的日志、模型权重和可视化结果将默认保存在 `output/` 目录下。

### 测试与评估

使用 `--evaluate` 标志来在测试集上评估已训练好的模型性能。请确保将 `--resume` 指向您希望评估的模型权重文件 (`.pth`)。

* **评估图Transformer模型**
    ```bash
    python train.py --config config/ps_transformer.yaml --evaluate --resume /path/to/your/transformer_checkpoint.pth
    ```

* **评估GAT基线模型**
    ```bash
    python train.py --config config/ps_gat.yaml --evaluate --resume /path/to/your/gat_checkpoint.pth
    ```

## 预期结果

通过将GAT替换为图Transformer，模型在公开数据集ps2.0上获得了性能提升。

| 方法                 | 精确率 (Precision) | 召回率 (Recall) |
| -------------------- | ------------------ | --------------- |
| Baseline (GAT)       | 99.56%             | 99.42%          |
| **Ours (Graph Transformer)** | **99.61%** | **99.55%** |


## 引用

如果您在您的研究中使用了本项目的思想或代码，请考虑引用原始论文：

```bibtex
@inproceedings{min2021attentional,
  title={Attentional graph neural network for parking-slot detection},
  author={Min, Chen and Xu, Jiaolong and Xiao, Liang and Zhao, Dawei and Nie, Yiming and Dai, Bin},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3445--3450},
  year={2021},
  organization={IEEE}
}
```

## 致谢
感谢原作者 **Chen Min, Jiaolong Xu** 等人出色的开创性工作以及他们对开源社区的无私贡献。
