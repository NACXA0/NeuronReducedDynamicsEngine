# NRDE — Neuron Reduced-Dynamics Engine

连接组进，按细胞类型绑定的降阶激活函数 \(F\) 出。仿真主循环做稀疏传播并查 \(F\)，不再逐步积分 AdExp / HH。果蝇 MaleCNS 是当前实例预设，不是引擎的全部范围。

## 初衷

一边是 FlyBrainLab、Neurokernel 这类在线生物动力学仿真：每步求解单细胞微分方程，保真，但全脑规模太贵。另一边是 ornata/fly，以及 Shiu 等人那种全脑统一 LIF：只要连接表加一条激活曲线，快，但所有细胞共用同一个函数。

本项目要的是中间这条路。输入仍然只是连接表和细胞类型。离线把每一类神经元的 ODE 拟合成 \(F\)，运行时查表。比逐步积分轻，又比全脑一条 LIF 更接近原来的单胞动力学。这是工程上的抽象，用来把网络规模做上去，不是宣称已经重建了真实电生理，也不主张虚拟果蝇能走出有意义的行为。

全文在 [初始设想](docs/初始设想.md)。后来的范围和口径以 [RFC-001](docs/rfcs/RFC-001.md) 为准。

## 介绍

给定连接表和细胞类型，NRDE 离线拟合 \(F\)（L0 稳态 f–I、L1 查找表、L2 SRM 核），在线只做 \(I = Wr + I_{\mathrm{ext}}\)、\(r = F(I)\)。默认发布集在 `datasets/fly/`：aCC / T4 用同一张 AdExp 替身表，默认兴奋是 ExpLIF，默认抑制是 LIF，另有一条 249 节点、4622 边的附属屈肌运动神经元回路。

## 宣传

默认数据集上，相对在线 AdExp：NRDE 的发放率误差约 1.6 Hz，249 个细胞仿真 200 ms 大约快 122 倍。相对统一 LIF：墙钟同为十几毫秒，但 LIF 相对同一条 AdExp 曲线偏出约 110 Hz。

| 方法 | 相对 AdExp ODE 的 RMSE | \(R^2\) | 249 细胞 × 200 ms |
|---|---:|---:|---:|
| 在线 AdExp ODE | 0 | 1 | 1805.8 ms |
| NRDE \(F\) | 1.61 Hz | 0.994607 | 14.83 ms |
| 统一 LIF | 110.42 Hz | −24.25 | 11.98 ms |

数字锁定在 AMD Ryzen 7 5800X、NumPy 上；墙钟换机器会变。f–I 曲线、阻抗谱、协议和哈希在 [证据链](docs/reports/evidence.md)。

## 快速入门

本地开发不要 `pip install` 进系统 Python。uv 把包装进仓库下的 `.venv`，命令一律 `uv run`。

```bash
git clone https://github.com/NACXA0/NeuronReducedDynamicsEngine.git
cd NeuronReducedDynamicsEngine
uv sync
uv run --no-sync nrde fetch fly --tier pretrained
uv run --no-sync nrde run simulate \
  --fit datasets/fly/fits/aCC_l0.npz \
  --n 80 --steps 50
```

`fetch --tier pretrained` 校验 `datasets/fly/` 里 33 个表文件的 sha256。表已经拟合好；这条仿真只做稀疏传播，不是再训练。

## 文档目录

### 快速开始

| 文档 | 内容 |
|---|---|
| [果蝇 Demo](docs/quickstart/demo.md) | Colab / GIF，以及 `nrde demo flygym` |
| [拟合 F](docs/quickstart/pipeline.md) | 自己做 L0 / L1 / L2 / L1+L2 |
| [稀疏传播](docs/quickstart/engine.md) | 默认回路与全量 MaleCNS 上的 `nrde run simulate` |
| [从源码构建](docs/from_source.md) | extras、系统依赖、贡献者环境 |

### 数据与证据

| 文档 | 内容 |
|---|---|
| [默认数据集](datasets/fly/README.md) | `fits/`、`precision/`、`sparse/`、验收 JSON |
| [证据链](docs/reports/evidence.md) | 对比曲线、阻抗谱、协议、哈希、重算命令 |
| [验证报告索引](docs/reports/README.md) | M1：拟合、在线 vs 降阶、chirp |
| [数据源](docs/data_sources.md) | `fetch` 的 smoke / seed / pretrained / full |

### 使用与参考

| 文档 | 内容 |
|---|---|
| [ModelSpec](docs/model_spec_protocol.md) | 自有 ODE 怎么进管线 |
| [API](docs/api.md) | 公共符号 |
| [不适用清单](docs/known-inapproximable.md) | 降阶失败时的在线 ODE 回退 |
| [PySR 准入](docs/pysr_admission.md) | 可选符号回归；未通过则退场 |
| [贡献约定](CONTRIBUTING.md) | 命名、测试、许可证 |
| [Changelog](docs/CHANGELOG.md) | 版本记录 |

### 规格与决策

冲突时以 RFC 为准。中文规格是历史快照。

| 文档 | 内容 |
|---|---|
| [RFC-001](docs/rfcs/RFC-001.md) | 拟合分层、宣传口径 |
| [RFC-002](docs/rfcs/RFC-002.md) | 包结构与产品面 |
| [需求文档](docs/需求文档.md) | FR / NFR |
| [项目计划书](docs/项目计划书.md) | 里程碑 |
| [系统设计说明书](docs/系统设计说明书.md) | 两阶段架构 |
| [初始设想](docs/初始设想.md) | 初衷全文 |

许可证：[Apache License 2.0](LICENSE)。
