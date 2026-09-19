# 稀疏传播（B 层）：怎么开始

实现包是 **`nrde.engine`**。命令是 **`uv run nrde run simulate`**。

先有一份拟合好的 F，再在图上做

$$I = W r + I_{\mathrm{ext}}, \qquad r = F(I).$$

`W` 是连接，`F` 是 [上一步](pipeline.md) 写出的 npz。这一步每毫秒乘一次稀疏矩阵，再查表。它不更新 F，没有 epoch，也没有损失曲线。

**不接 TensorBoard，也不接 SwanLab。** 那两个工具记的是训练曲线。这里的标准输出是一行 JSON：`mean_rate` 和 `n_nodes`。α 标定（Shiu 0.275 mV 起点上的一个全局缩放）同样只产出一个系数，不接。

默认图是随机图，不是 1.1 GB 的 MaleCNS。全图加载是可选的规模测试，不是示例拟合的一部分。

## 1. 安装

```bash
uv sync --extra dev
```

CPU 参考实现是 NumPy / SciPy CSR，不需要 GPU。CLI 没有 `--device`。

已有 torch 时，不要用会重解析 torch 的 `uv sync --extra torch`。默认的 `uv sync` 是精确同步：这次解析里没有的包会被卸掉，已装的 GPU 版也可能被换成锁里的轮子。保留现有 torch：

```bash
uv sync --extra dev --inexact
```

`--inexact` 只保证「不删除环境里多出来的包」。它不会把另一份 CUDA 轮子升级成锁文件里的那一份。这次运行再加 `--no-sync`，uv 就完全不碰环境。

要在一台新机器上按情况安装，用 PyTorch 的索引，不要让默认的 PyPI extra 替你选：

```bash
# CPU
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
# 本机 CUDA 13（3080 Ti 这一档）
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
```

装完之后，日常同步继续用上面的 `uv sync --extra dev --inexact`。

## 2. 运行

用 [拟合 F](pipeline.md) 里的任意一份 npz。随机图 80 个点、50 步，用来确认 F 能被查到。

```bash
uv run nrde run simulate --fit /tmp/nrde_fit/lif_l0.npz --n 80 --steps 50
```

L1、L2、L1+L2 换成对应文件即可。速率模式用的是 f–I；LUT / SRM 留在脉冲模式，不改变这条命令的形式。

GPU 用已经装好的 torch，不在这条命令里重装：

```bash
NRDE_ENGINE=torch uv run --no-sync nrde run simulate --fit /tmp/nrde_fit/lif_l0.npz --n 80 --steps 50
```

`NRDE_ENGINE` 还可取 `numpy`（默认）或 `cuda`（与 `torch` 相同）。没装 torch 或没有 GPU 时不要设它。AVX-512 不属于这条命令；它只可能出现在拟合用的可选 Numba 核里。

## 3. 可选：真连接组

权重约 1.1 GB，文件名是仓库 `data/README.md` 里的那一套。`....` 不是路径。这是图，不是 F 的训练数据。全图大约 8 千万节点，需要十几 GiB 内存。

```bash
NRDE_ENGINE=torch uv run --no-sync nrde run simulate \
  --graph data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --annotations data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  --fit /tmp/nrde_fit/lif_l0.npz \
  --steps 100
```

还没有这两件 Feather 时先下载，再跑上面的仿真。下载和仿真分开。

```bash
uv run python scripts/download_malecns.py
```

类型绑定见 `configs/type_model_map.yaml`。α 起点见 `configs/alpha_defaults.yaml`（Shiu 0.275 mV）。
