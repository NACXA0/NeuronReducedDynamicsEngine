# NRDE — Neuron Reduced-Dynamics Engine

连接组驱动 · 类型可绑定降阶激活函数 · 可选具身接口。  
**不是**完整生物物理重建，**不**主张有意义行走行为（见 [RFC-001](docs/rfcs/RFC-001.md) 宣传口径）。

### 命名映射（D22）

| 角色 | 名称 | 说明 |
|---|---|---|
| 宣传 / 论文 / 仓库 | `Neuron Reduced-Dynamics Engine` / `NeuronReducedDynamicsEngine` | 产品全称；果蝇仅为实例预设 |
| 中文全称 | 神经元降阶动力学引擎 | — |
| PyPI | `neuron-reduced-dynamics-engine` | `pip install` 用（发布前查重，Q9） |
| Import / CLI | `nrde` | `import nrde` |
| 文内简称 | NRDE | — |
| 方法术语 | state approximation / 状态近似 | 术语层，非品牌层 |

egg-info 目录名由 setuptools 从 PyPI 名派生，**勿提交**。

## 产品面（A1 / A2 / B / C）

| | 🪰 A1 果蝇奇观 | 🧪 A2 果蝇预设 | 🔬 B 引擎 | 🏭 C 管线 |
|---|---|---|---|---|
| 给谁 | 想看果蝇跑起来的观众 | Python 果蝇研究者 / RL 快速开工 | 连接组研究者（v0.1 主用户） | 自有 ODE 的外部研究者 |
| 装什么 | 无需本地装（Colab / GIF / Release zip） | 仓库内 `uv sync --extra fly` | 仓库内 `uv sync` | 同左（可选 `--extra pysr`） |
| 5 分钟 | [Open in Colab](#) · GIF（M4） | `uv run nrde fetch fly` → `uv run nrde demo flygym` | `uv run nrde run simulate …` | `uv run nrde offline fit` |

> Demo 卡资产（GIF / Colab / 离线包）由 [A1 管线](docs/asset_pipeline.md) 随版本刷新（D21）；当前仓库以 **A2 路径** 为默认可跑入口，A1 链接在 M4 挂满。

本地开发**不要** `pip install` 进系统 Python。uv 把包装进仓库下的 `.venv`。命令一律 `uv run`，这样不必激活环境，也不会去系统 `PATH` 里找 `nrde`。

预训练是两步，不是一条命令。第一步拟合降阶函数 F，第二步用这张表做稀疏传播。传播不是再训练。

- **拟合 F**（L0、L1、L2，以及 L1+L2）：[怎么开始](docs/quickstart/pipeline.md)
- **稀疏传播**（`I = W r`，再查 F）：[怎么开始](docs/quickstart/engine.md)

旅程：[Demo A1/A2](docs/quickstart/demo.md) · [稀疏传播](docs/quickstart/engine.md) · [拟合 F](docs/quickstart/pipeline.md) · [ModelSpec](docs/model_spec_protocol.md)

### 许可证

本项目采用 Apache License 2.0 开源协议。完整内容见 [LICENSE](LICENSE)。

使用、复制、修改或分发本仓库代码时，均应遵守 Apache 2.0 条款。对本项目的贡献，默认视为在 Apache 2.0 许可下提交，除非另有单独书面约定。

### 安装

```bash
git clone https://github.com/NACXA0/NeuronReducedDynamicsEngine.git
cd NeuronReducedDynamicsEngine
uv sync --extra dev
```

### 运行

```bash
uv run nrde fetch fly --tier seed
uv run nrde demo flygym --steps 100 --headless
```

四种示例拟合（L0、L1、L2、L1+L2）的完整命令在 [拟合 F](docs/quickstart/pipeline.md)。下面只跑 L0，再用它做一次小图上的稀疏传播。不要写入 `artifacts/aCC.npz`。

```bash
mkdir -p /tmp/nrde_fit
uv run nrde offline fit \
  --model lif \
  --type-id lif_l0 \
  --I-min 0 \
  --I-max 0.8 \
  --n-I 16 \
  --out /tmp/nrde_fit/lif_l0.npz
```

```bash
uv run nrde run simulate --fit /tmp/nrde_fit/lif_l0.npz --n 80 --steps 50
```

```bash
uv run python examples/04_custom_model_pipeline.py
```

```python
import nrde
env = nrde.make("flygym-demo-v01")  # A2
nrde.offline(["fit", "--model", "lif", "--type-id", "demo"])  # C；内部包名 nrde.fitting
```

上面这段 Python 用 `uv run python` 执行，解释器仍是项目虚拟环境。

### 测试

```bash
uv run --extra dev pytest --cov=nrde --cov-fail-under=70 -m "not pysr and not flygym"
uv run --extra dev ruff check src tests examples scripts
```

## 安装矩阵（终稿四行）

| 身份 | 安装 | 面 |
|---|---|---|
| 只想看一眼（A1） | Colab / GIF / [Release 离线包](https://github.com/NACXA0/NeuronReducedDynamicsEngine/releases) | A1 |
| 果蝇预设开工（A2） | `uv sync --extra fly`，再 `uv run nrde fetch fly` | A2 |
| 连接组 / 自有模型（B/C） | `uv sync`（C 可选 `--extra pysr`） | B、C |
| 贡献者 | `uv sync --extra dev` | — |

Python：CI 主 lane **3.11**；开发可跟进 **3.14**（allow-failure lane，D13）。核心不强制 JAX/Julia。`--headless` 为 A1 渲染管线依赖（Q6 已决 / D21）。

### 从源码构建（完整开发）

终端用户用上表即可。若要**本地从源码开发 / 复现全部产品面**（extras、系统依赖、种子 artifacts、MaleCNS 数据集、A1 资产脚本），见独立清单：

→ **[从源码构建与完整开发环境](docs/from_source.md)**

安装、运行、测试的可复制命令见上面三节，以及 [从源码构建](docs/from_source.md)。

## 对比（定位）

| 项目 | 能力 | NRDE |
|---|---|---|
| FlyBrainLab / Neurokernel | 在线 HH / LIF | 主循环不跑 HH |
| ornata/fly | 连接组 + 通用激活 | F 由 ODE/SRM **拟合**得到 |
| Shiu et al. 2024 | 统一 LIF + 固定 W_syn | W_syn 作 α **初值**；类型绑定 F |
| `flybrain` GPU LIF | 166k 压力测试 | 非产品目标 |

## L0–L3

| 层 | 产物 | 运行时 | 示例 |
|---|---|---|---|
| L0 | r(I) | O(1) | [只拟合 f–I](docs/quickstart/pipeline.md#l0) |
| L1 | LUT (I, Δt) | O(1) | [加上 `--lut`](docs/quickstart/pipeline.md#l1) |
| L2 | SRM κ,η,θ | O(1)/核 | [加上 `--srm`](docs/quickstart/pipeline.md#l2) |
| L1+L2 | LUT 与 SRM 都在 | O(1)/核 | [两个开关一起](docs/quickstart/pipeline.md#l1l2) |
| L3 | FNO / Volterra | → v0.2 | 本版没有 |

PySR 仅为可选精化器：`uv sync --extra pysr`（禁止随主包装 Julia）。

## 结构说明

对用户呈现 **A1/A2/B/C**；开发者内部为八大块（[RFC-002](docs/rfcs/RFC-002.md)）。单仓 + extras，不拆仓。测试命令在上面的「测试」一节。
