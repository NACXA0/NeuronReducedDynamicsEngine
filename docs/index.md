# NRDE documentation

安装矩阵（RFC-002 v1.1）：

| 身份 | 安装 | 面 |
|---|---|---|
| A1 果蝇奇观 | Colab / GIF / Release 离线包 | 非 Python 观众 |
| A2 果蝇预设 | `"neuron-reduced-dynamics-engine[fly]"` + `nrde fetch fly` | Python 果蝇研究者 |
| B 引擎 | `neuron-reduced-dynamics-engine` | 连接组研究者 |
| C 管线 | 同左（可选 `[pysr]`） | 自有 ModelSpec |

快速开始：[Demo（A1/A2）](quickstart/demo.md) · [拟合 F（L0 / L1 / L2）](quickstart/pipeline.md) · [稀疏传播](quickstart/engine.md) · [从源码构建](from_source.md)

Connectome in, type-bound reduced **F** (L0–L2) out. See the repository [README](https://github.com/NACXA0/NeuronReducedDynamicsEngine).

中文规格（历史快照，冲突以 RFC 为准）：

- [需求文档](需求文档.md) · [项目计划书](项目计划书.md) · [系统设计说明书](系统设计说明书.md) · [初始设想](初始设想.md)
- [RFC-001](rfcs/RFC-001.md) / [RFC-002](rfcs/RFC-002.md)

Public wording: **types may bind different F**. Not “N reconstructed cell types” and not meaningful walking.
