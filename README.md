# PyTorch NLL 实验

此目录包含一个可重建的 CPU 实验，用于对照手工 gather、`torch.nn.functional.nll_loss` 和 `torch.nn.functional.cross_entropy` 的计算结果，并展示分类训练、填充 token 掩码和数值稳定性。

## 运行环境

- Python 3.10 或更高版本
- PyTorch 2.x
- Pillow 10.x 或更高版本（生成可提交的 PNG 曲线）

实验曾在 Python 3.12.7、PyTorch 2.12.0+cu126、Pillow 10.4.0 环境中运行。脚本固定使用 CPU，因此不要求 CUDA。

## 运行

在此目录运行：

```powershell
python -m pip install -r requirements.txt
python run_experiment.py
```

`requirements.txt` 给出兼容范围；`requirements-tested.txt` 记录本次实验使用的精确版本。使用其他依赖版本时，应重新运行脚本中的一致性检查。

脚本使用固定随机种子重新生成数据并写出结果，不需要下载数据集。运行后会生成：

- `data/synthetic_3class.csv`：带 train/test 标记的 360 条合成样本。
- `results/summary.json`：损失与梯度一致性、序列掩码、数值稳定性及训练评测指标。
- `results/training_metrics.csv`：每轮三种实现的训练损失。
- `figures/training_loss.png`：报告可直接插入的训练曲线。
- `figures/training_loss.svg`：由相同实验数据生成的矢量版本。

`run_experiment.py` 内置有针对性的断言：固定 logits 的三种损失及梯度相等；padding 目标不计入 token 平均；`log_softmax` 在极端 logits 下保持有限；等价损失从同一参数起点训练后保持一致。

## 目录结构

```text
作业1.5_PyTorch_NLL/
├── data/
├── figures/
├── results/
├── README.md
├── report.md
├── requirements.txt
└── run_experiment.py
```
