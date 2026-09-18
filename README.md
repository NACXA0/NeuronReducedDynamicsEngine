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
| 装什么 | 无需本地装（Colab / GIF / Release zip） | `pip install "neuron-reduced-dynamics-engine[fly]"` | `pip install neuron-reduced-dynamics-engine` | 同左（可选 `[pysr]`） |
| 5 分钟 | [Open in Colab](#) · GIF（M4） | `nrde fetch fly` → `nrde demo flygym` | `nrde run simulate …` | `nrde offline fit` / `ModelSpec` |

> Demo 卡资产（GIF / Colab / 离线包）由 [A1 管线](docs/asset_pipeline.md) 随版本刷新（D21）；当前仓库以 **A2 路径** 为默认可跑入口，A1 链接在 M4 挂满。

### 5 分钟入口（A2 / B / C）

```bash
# A2 — 果蝇实例预设（Python）
pip install "neuron-reduced-dynamics-engine[fly]"
nrde fetch fly --tier seed
nrde demo flygym --steps 100 --headless

# B — 引擎
nrde offline fit --model adexp --type-id aCC
nrde run simulate --fit artifacts/aCC.npz --n 80 --steps 50

# C — 自己的模型
python examples/04_custom_model_pipeline.py
```

```python
import nrde
env = nrde.make("flygym-demo-v01")  # A2
nrde.offline(["fit", "--model", "lif", "--type-id", "demo"])  # C；内部包名 nrde.fitting
```

旅程：[Demo A1/A2](docs/quickstart/demo.md) · [引擎](docs/quickstart/engine.md) · [管线](docs/quickstart/pipeline.md) · [ModelSpec](docs/model_spec_protocol.md)

## 安装矩阵（终稿四行）

| 身份 | 安装 | 面 |
|---|---|---|
| 只想看一眼（A1） | Colab / GIF / [Release 离线包](https://github.com/NACXA0/NeuronReducedDynamicsEngine/releases) | A1 |
| 果蝇预设开工（A2） | `"neuron-reduced-dynamics-engine[fly]"` + `nrde fetch fly` | A2 |
| 连接组 / 自有模型（B/C） | `neuron-reduced-dynamics-engine`（C 可选 `[pysr]`） | B、C |
| 贡献者 | `".[dev]"` + `uv sync` | — |

Python：CI 主 lane **3.11**；开发可跟进 **3.14**（allow-failure lane，D13）。核心不强制 JAX/Julia。`--headless` 为 A1 渲染管线依赖（Q6 已决 / D21）。

### 从源码构建（完整开发）

终端用户用上表即可。若要**本地从源码开发 / 复现全部产品面**（extras、系统依赖、种子 artifacts、MaleCNS 数据集、A1 资产脚本），见独立清单：

→ **[从源码构建与完整开发环境](docs/from_source.md)**

```bash
git clone https://github.com/NACXA0/NeuronReducedDynamicsEngine.git
cd NeuronReducedDynamicsEngine
pip install -e ".[dev]"   # 或: uv sync --extra dev
pytest --cov=nrde --cov-fail-under=70 -m "not pysr and not flygym"
```

## 对比（定位）

| 项目 | 能力 | NRDE |
|---|---|---|
| FlyBrainLab / Neurokernel | 在线 HH / LIF | 主循环不跑 HH |
| ornata/fly | 连接组 + 通用激活 | F 由 ODE/SRM **拟合**得到 |
| Shiu et al. 2024 | 统一 LIF + 固定 W_syn | W_syn 作 α **初值**；类型绑定 F |
| `flybrain` GPU LIF | 166k 压力测试 | 非产品目标 |

## L0–L3

| 层 | 产物 | 运行时 |
|---|---|---|
| L0 | r(I) | O(1) |
| L1 | LUT (I, Δt) | O(1) |
| L2 | SRM κ,η,θ | O(1)/核 |
| L3 | FNO / Volterra | → v0.2 |

PySR 仅为可选精化器：`pip install 'neuron-reduced-dynamics-engine[pysr]'`（禁止随主包装 Julia）。

## 结构说明

对用户呈现 **A1/A2/B/C**；开发者内部为八大块（[RFC-002](docs/rfcs/RFC-002.md)）。单仓 + extras，不拆仓。

```bash
pytest --cov=nrde --cov-fail-under=70 -m "not pysr and not flygym"
```
